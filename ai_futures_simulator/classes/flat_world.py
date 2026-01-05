"""
FlatWorld and FlatStateDerivative classes for fast ODE integration.

These classes store state in a flat tensor for fast arithmetic operations
while providing the same attribute access interface as the nested World class.
"""

import torch
from torch import Tensor
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import fields

from classes.state_schema import (
    generate_state_schema,
    get_nested_attr,
    set_nested_attr,
    extract_non_state_fields,
    world_to_flat_tensor,
    flat_tensor_to_world,
    _parse_path,
)
from classes.tensor_dataclass import TensorDataclass

if TYPE_CHECKING:
    from classes.world.world import World


class ProxyBase:
    """Base class for proxy objects that map attribute access to tensor indices."""

    def __init__(self, state_tensor: Tensor, schema: Dict[str, int], prefix: str,
                 is_derivative: bool = False, template_obj: Any = None):
        """
        Args:
            state_tensor: The flat tensor containing state values
            schema: The schema mapping paths to indices
            prefix: The path prefix for this proxy (e.g., "nations[PRC].")
            is_derivative: If True, allow setting values (for derivatives)
            template_obj: The corresponding object from the template World (for non-state fields)
        """
        object.__setattr__(self, '_state', state_tensor)
        object.__setattr__(self, '_schema', schema)
        object.__setattr__(self, '_prefix', prefix)
        object.__setattr__(self, '_is_derivative', is_derivative)
        object.__setattr__(self, '_template_obj', template_obj)
        object.__setattr__(self, '_child_proxies', {})

    def _get_full_path(self, name: str) -> str:
        """Get the full path for an attribute."""
        return f"{self._prefix}{name}"

    def __getattr__(self, name: str) -> Any:
        # Avoid recursion on special attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        full_path = self._get_full_path(name)

        # Check if this is a direct state field in schema
        if full_path in self._schema:
            idx = self._schema[full_path]
            return self._state[idx]

        # Check if this is a parent of state fields - create child proxy
        child_prefix = f"{full_path}."
        has_children = any(p.startswith(child_prefix) for p in self._schema)

        if has_children:
            # Return cached proxy or create new one
            if name not in self._child_proxies:
                # Get template child object if available
                template_child = None
                if self._template_obj is not None and hasattr(self._template_obj, name):
                    template_child = getattr(self._template_obj, name)
                self._child_proxies[name] = ProxyBase(
                    self._state, self._schema, child_prefix, self._is_derivative, template_child
                )
            return self._child_proxies[name]

        # Fall back to template object for non-state fields (metrics, etc.)
        if self._template_obj is not None and hasattr(self._template_obj, name):
            return getattr(self._template_obj, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return

        full_path = self._get_full_path(name)

        if full_path in self._schema:
            idx = self._schema[full_path]
            if isinstance(value, Tensor):
                self._state[idx] = value.item() if value.numel() == 1 else value
            else:
                self._state[idx] = float(value)
        else:
            # May be setting a non-state attribute (metrics, etc.) - ignore for derivatives
            if self._is_derivative:
                pass  # Silently ignore non-state field assignments in derivatives
            else:
                raise AttributeError(f"Cannot set attribute '{name}' - not in state schema")

    def _set_frozen_field(self, name: str, value: Any) -> None:
        """Compatibility method for frozen dataclass pattern."""
        self.__setattr__(name, value)


class DictProxy:
    """Proxy for dict-like access to nested entities (nations, developers, etc.)."""

    def __init__(self, state_tensor: Tensor, schema: Dict[str, int], prefix: str,
                 keys: List[str], is_derivative: bool = False, template_dict: Dict = None):
        self._state = state_tensor
        self._schema = schema
        self._prefix = prefix
        self._keys = keys
        self._is_derivative = is_derivative
        self._template_dict = template_dict or {}
        self._item_proxies = {}

    def __getitem__(self, key: str) -> ProxyBase:
        if key not in self._keys:
            raise KeyError(f"Key '{key}' not found")

        if key not in self._item_proxies:
            item_prefix = f"{self._prefix}[{key}]."
            template_obj = self._template_dict.get(key)
            self._item_proxies[key] = ProxyBase(
                self._state, self._schema, item_prefix, self._is_derivative, template_obj
            )
        return self._item_proxies[key]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return self._keys

    def values(self):
        return [self[k] for k in self._keys]

    def items(self):
        return [(k, self[k]) for k in self._keys]

    def get(self, key: str, default=None):
        if key in self._keys:
            return self[key]
        return default


class ListProxy:
    """Proxy for list-like access to indexed items (e.g., operating_compute)."""

    def __init__(self, state_tensor: Tensor, schema: Dict[str, int], prefix: str,
                 length: int, is_derivative: bool = False):
        self._state = state_tensor
        self._schema = schema
        self._prefix = prefix
        self._length = length
        self._is_derivative = is_derivative
        self._item_proxies = {}

    def __getitem__(self, index: int) -> ProxyBase:
        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} out of range [0, {self._length})")

        if index not in self._item_proxies:
            item_prefix = f"{self._prefix}[{index}]."
            self._item_proxies[index] = ProxyBase(
                self._state, self._schema, item_prefix, self._is_derivative
            )
        return self._item_proxies[index]

    def __iter__(self):
        return (self[i] for i in range(self._length))

    def __len__(self):
        return self._length


class FlatWorld:
    """
    World state backed by a flat tensor for fast ODE integration.

    Provides the same attribute access interface as World but stores all
    state values in a single flat tensor. Arithmetic operations become
    fast tensor ops instead of nested dataclass cloning.
    """

    def __init__(self, state_tensor: Tensor, schema: Dict[str, int],
                 metadata: Dict[str, Any], template_world: 'World'):
        """
        Args:
            state_tensor: The flat tensor containing all state values
            schema: Mapping from path strings to tensor indices
            metadata: Non-state fields needed for reconstruction
            template_world: Reference World instance for structure
        """
        self._state = state_tensor
        self._schema = schema
        self._metadata = metadata
        self._template = template_world
        self._proxies = {}

    @classmethod
    def from_world(cls, world: 'World') -> 'FlatWorld':
        """
        Convert a nested World to flat representation.

        Args:
            world: The World instance to convert

        Returns:
            A FlatWorld with the same state
        """
        schema = generate_state_schema(world)
        state_tensor = world_to_flat_tensor(world, schema)
        metadata = extract_non_state_fields(world)
        return cls(state_tensor, schema, metadata, world)

    def to_world(self) -> 'World':
        """
        Convert back to nested World structure.

        Returns:
            A new World instance with state values from this FlatWorld
        """
        return flat_tensor_to_world(self._state, self._schema, self._template)

    def _get_dict_proxy(self, name: str, keys: List[str], template_dict: Dict = None) -> DictProxy:
        """Get or create a DictProxy for a dict field."""
        if name not in self._proxies:
            self._proxies[name] = DictProxy(
                self._state, self._schema, name, keys, is_derivative=False, template_dict=template_dict
            )
        return self._proxies[name]

    @property
    def current_time(self) -> Tensor:
        """Get current simulation time."""
        if 'current_time' in self._schema:
            return self._state[self._schema['current_time']]
        return self._template.current_time

    @current_time.setter
    def current_time(self, value: Any):
        if 'current_time' in self._schema:
            if isinstance(value, Tensor):
                self._state[self._schema['current_time']] = value.item() if value.numel() == 1 else value
            else:
                self._state[self._schema['current_time']] = float(value)

    @property
    def nations(self) -> DictProxy:
        keys = self._metadata.get('nations_keys', [])
        template_dict = self._template.nations if self._template else None
        return self._get_dict_proxy('nations', keys, template_dict)

    @property
    def coalitions(self) -> DictProxy:
        keys = self._metadata.get('coalitions_keys', [])
        template_dict = self._template.coalitions if self._template else None
        return self._get_dict_proxy('coalitions', keys, template_dict)

    @property
    def ai_software_developers(self) -> DictProxy:
        keys = self._metadata.get('ai_software_developers_keys', [])
        template_dict = self._template.ai_software_developers if self._template else None
        return self._get_dict_proxy('ai_software_developers', keys, template_dict)

    @property
    def ai_policies(self) -> DictProxy:
        keys = self._metadata.get('ai_policies_keys', [])
        template_dict = self._template.ai_policies if self._template else None
        return self._get_dict_proxy('ai_policies', keys, template_dict)

    @property
    def black_projects(self) -> DictProxy:
        keys = self._metadata.get('black_projects_keys', [])
        template_dict = self._template.black_projects if self._template else None
        return self._get_dict_proxy('black_projects', keys, template_dict)

    @property
    def perceptions(self) -> DictProxy:
        keys = self._metadata.get('perceptions_keys', [])
        template_dict = self._template.perceptions if self._template else None
        return self._get_dict_proxy('perceptions', keys, template_dict)

    # Fast arithmetic operations (no cloning!)
    def __add__(self, other: 'FlatWorld') -> 'FlatWorld':
        """Element-wise addition of state tensors."""
        if not isinstance(other, FlatWorld):
            return NotImplemented
        return FlatWorld(
            self._state + other._state,
            self._schema,
            self._metadata,
            self._template
        )

    def __mul__(self, scalar: float) -> 'FlatWorld':
        """Scalar multiplication."""
        return FlatWorld(
            self._state * scalar,
            self._schema,
            self._metadata,
            self._template
        )

    def __rmul__(self, scalar: float) -> 'FlatWorld':
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def to_state_tensor(self) -> Tensor:
        """Return the underlying state tensor (for compatibility)."""
        return self._state

    @classmethod
    def from_state_tensor(cls, state_tensor: Tensor, template: 'FlatWorld') -> 'FlatWorld':
        """Create a FlatWorld from a state tensor using another as template."""
        return cls(
            state_tensor,
            template._schema,
            template._metadata,
            template._template
        )

    @classmethod
    def zeros(cls, template: 'FlatWorld') -> 'FlatWorld':
        """Create a zero-initialized FlatWorld with same structure as template."""
        return cls(
            torch.zeros_like(template._state),
            template._schema,
            template._metadata,
            template._template
        )

    def _clone(self) -> 'FlatWorld':
        """Create a shallow clone (shares schema/metadata, clones tensor)."""
        return FlatWorld(
            self._state.clone(),
            self._schema,
            self._metadata,
            self._template
        )

    def _get_state_fields(self) -> list:
        """
        Get state fields for compatibility with TensorDataclass interface.

        Returns list of (path, value) tuples.
        """
        return [(path, self._state[idx]) for path, idx in self._schema.items()]

    def _set_state_field(self, path: str, value: Any) -> None:
        """Set a state field by path for compatibility."""
        if path in self._schema:
            idx = self._schema[path]
            if isinstance(value, Tensor):
                self._state[idx] = value.item() if value.numel() == 1 else value
            else:
                self._state[idx] = float(value)


class FlatStateDerivative:
    """
    Derivative wrapper for flat tensor state.

    Provides the same interface as StateDerivative but operates on flat tensors.
    Used by updaters to accumulate derivative contributions.
    """

    def __init__(self, state_tensor: Tensor, schema: Dict[str, int],
                 metadata: Dict[str, Any], template_world: 'World'):
        """
        Args:
            state_tensor: The flat tensor for derivative values
            schema: Mapping from path strings to tensor indices
            metadata: Non-state fields (same as FlatWorld)
            template_world: Reference World for structure
        """
        self._state = state_tensor
        self._schema = schema
        self._metadata = metadata
        self._template = template_world
        self._proxies = {}

    @classmethod
    def zeros(cls, template: 'FlatWorld') -> 'FlatStateDerivative':
        """Create a zero-initialized derivative with same structure as template."""
        return cls(
            torch.zeros_like(template._state),
            template._schema,
            template._metadata,
            template._template
        )

    def _get_dict_proxy(self, name: str, keys: List[str]) -> DictProxy:
        """Get or create a DictProxy for a dict field."""
        if name not in self._proxies:
            self._proxies[name] = DictProxy(
                self._state, self._schema, name, keys, is_derivative=True
            )
        return self._proxies[name]

    @property
    def current_time(self) -> Tensor:
        if 'current_time' in self._schema:
            return self._state[self._schema['current_time']]
        return torch.tensor(0.0)

    @current_time.setter
    def current_time(self, value: Any):
        if 'current_time' in self._schema:
            if isinstance(value, Tensor):
                self._state[self._schema['current_time']] = value.item() if value.numel() == 1 else value
            else:
                self._state[self._schema['current_time']] = float(value)

    @property
    def nations(self) -> DictProxy:
        keys = self._metadata.get('nations_keys', [])
        return self._get_dict_proxy('nations', keys)

    @property
    def coalitions(self) -> DictProxy:
        keys = self._metadata.get('coalitions_keys', [])
        return self._get_dict_proxy('coalitions', keys)

    @property
    def ai_software_developers(self) -> DictProxy:
        keys = self._metadata.get('ai_software_developers_keys', [])
        return self._get_dict_proxy('ai_software_developers', keys)

    @property
    def ai_policies(self) -> DictProxy:
        keys = self._metadata.get('ai_policies_keys', [])
        return self._get_dict_proxy('ai_policies', keys)

    @property
    def black_projects(self) -> DictProxy:
        keys = self._metadata.get('black_projects_keys', [])
        return self._get_dict_proxy('black_projects', keys)

    @property
    def perceptions(self) -> DictProxy:
        keys = self._metadata.get('perceptions_keys', [])
        return self._get_dict_proxy('perceptions', keys)

    def to_state_tensor(self) -> Tensor:
        """Return the underlying derivative tensor."""
        return self._state

    def __add__(self, other) -> 'FlatStateDerivative':
        """Element-wise addition of derivatives."""
        # Handle FlatStateDerivative directly
        if isinstance(other, FlatStateDerivative):
            return FlatStateDerivative(
                self._state + other._state,
                self._schema,
                self._metadata,
                self._template
            )
        # Handle StateDerivative by converting to tensor
        if hasattr(other, 'to_state_tensor'):
            other_tensor = other.to_state_tensor()
            return FlatStateDerivative(
                self._state + other_tensor,
                self._schema,
                self._metadata,
                self._template
            )
        return NotImplemented

    def __radd__(self, other):
        """Support sum() by handling 0 + FlatStateDerivative."""
        if other == 0:
            return self
        # For non-zero, delegate to __add__
        return self.__add__(other)

    def __mul__(self, scalar: float) -> 'FlatStateDerivative':
        """Scalar multiplication."""
        return FlatStateDerivative(
            self._state * scalar,
            self._schema,
            self._metadata,
            self._template
        )

    def __rmul__(self, scalar: float) -> 'FlatStateDerivative':
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    # Compatibility property - wraps self as world-like for StateDerivative.world pattern
    @property
    def world(self) -> 'FlatStateDerivative':
        """For compatibility with code that accesses deriv.world."""
        return self
