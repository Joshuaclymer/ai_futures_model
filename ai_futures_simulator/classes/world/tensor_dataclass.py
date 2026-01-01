"""
TensorDataclass base class for differentiable world state.

Provides automatic tensor packing/unpacking for ODE integration,
supporting nested dataclasses with is_state metadata.
"""

import torch
from torch import Tensor
from dataclasses import dataclass, fields, field
from typing import TypeVar, Type, Any, get_type_hints, get_origin, get_args

T = TypeVar('T', bound='TensorDataclass')

# Global cache for state field paths and shapes
# Key: (class_type, structure_key) -> list of (path, shape, numel)
_STATE_FIELD_CACHE: dict = {}

# Global cache for shallow clone recipes
# Key: (class_type, structure_key) -> CloneRecipe
_CLONE_RECIPE_CACHE: dict = {}

# Global cache for structure keys
# Key: id(obj) -> (structure_key, is_valid_for_id)
# We use object id as a quick check, but validate structure hasn't changed
_STRUCTURE_KEY_CACHE: dict = {}


def is_tensor_dataclass(obj: Any) -> bool:
    """Check if an object is a TensorDataclass instance."""
    return isinstance(obj, TensorDataclass)


def is_tensor_dataclass_type(cls: Type) -> bool:
    """Check if a class is a TensorDataclass subclass."""
    try:
        return issubclass(cls, TensorDataclass)
    except TypeError:
        return False


def _deep_clone(obj: Any) -> Any:
    """
    Deep clone an object, properly handling tensors with gradients.

    Unlike copy.deepcopy(), this uses tensor.clone() which preserves
    gradient computation capability.
    """
    if isinstance(obj, Tensor):
        return obj.clone()
    elif is_tensor_dataclass(obj):
        return obj._clone()
    elif isinstance(obj, dict):
        return {k: _deep_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_clone(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_clone(item) for item in obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # For other objects, try to copy or return as-is
        try:
            import copy
            return copy.copy(obj)
        except Exception:
            return obj


@dataclass
class TensorDataclass:
    """
    Base class for dataclasses that can be converted to/from tensors.

    Fields marked with metadata={'is_state': True} are treated as state
    variables that get packed into tensors for ODE integration.

    Fields marked with metadata={'is_state': False} or without is_state
    metadata are treated as metrics (derived quantities).

    Supports nested TensorDataclass instances - state fields are recursively
    collected from nested objects.
    """

    def _get_structure_key(self) -> tuple:
        """
        Get a hashable key representing the structure of this object.

        Used for caching state field paths. The key captures the class type
        and any dict keys that affect the structure. Results are cached per-instance.
        """
        # Check for cached structure key
        cached = getattr(self, '_cached_structure_key', None)
        if cached is not None:
            return cached

        key_parts = [type(self).__name__]
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, dict):
                # Include dict keys in structure key
                key_parts.append((f.name, tuple(sorted(value.keys()))))
            elif is_tensor_dataclass(value):
                key_parts.append((f.name, value._get_structure_key()))

        result = tuple(key_parts)
        # Cache on instance (bypass frozen dataclass)
        object.__setattr__(self, '_cached_structure_key', result)
        return result

    def _get_state_field_paths(self) -> list[tuple[str, tuple, int]]:
        """
        Get cached state field paths, shapes, and sizes.

        Returns list of (path, shape, numel) tuples. This is cached based on
        the object structure to avoid repeated traversal.
        """
        structure_key = self._get_structure_key()
        cache_key = (type(self), structure_key)

        if cache_key in _STATE_FIELD_CACHE:
            return _STATE_FIELD_CACHE[cache_key]

        # Compute paths and shapes
        paths_and_shapes = []
        state_fields = self._get_state_fields()
        for path, value in state_fields:
            if isinstance(value, Tensor):
                paths_and_shapes.append((path, tuple(value.shape), value.numel()))
            else:
                paths_and_shapes.append((path, (), 1))

        _STATE_FIELD_CACHE[cache_key] = paths_and_shapes
        return paths_and_shapes

    def _get_state_fields(self) -> list[tuple[str, Any]]:
        """
        Recursively collect all state field values.

        Returns list of (path, value) tuples where path is dot-separated.
        """
        state_values = []

        for f in fields(self):
            value = getattr(self, f.name)
            is_state = f.metadata.get('is_state', False)

            if is_state:
                if isinstance(value, Tensor):
                    state_values.append((f.name, value))
                elif isinstance(value, (int, float)):
                    state_values.append((f.name, torch.tensor(float(value))))
                elif is_tensor_dataclass(value):
                    # Recurse into nested TensorDataclass that's marked as state
                    nested_states = value._get_state_fields()
                    for nested_path, nested_value in nested_states:
                        state_values.append((f"{f.name}.{nested_path}", nested_value))
            elif is_tensor_dataclass(value):
                # Recurse into nested TensorDataclass (not marked as state, but may contain state)
                nested_states = value._get_state_fields()
                for nested_path, nested_value in nested_states:
                    state_values.append((f"{f.name}.{nested_path}", nested_value))
            elif isinstance(value, dict):
                # Handle dict of TensorDataclass (e.g., dict[str, AIProject])
                for key, item in value.items():
                    if is_tensor_dataclass(item):
                        nested_states = item._get_state_fields()
                        for nested_path, nested_value in nested_states:
                            state_values.append((f"{f.name}[{key}].{nested_path}", nested_value))

        return state_values

    def _set_frozen_field(self, name: str, value: Any) -> None:
        """
        Set a field on a frozen dataclass, bypassing immutability protection.

        This is the canonical way to set init=False fields after construction
        on frozen TensorDataclass instances. Use this instead of calling
        object.__setattr__ directly.

        Args:
            name: The field name to set
            value: The value to assign
        """
        object.__setattr__(self, name, value)

    def _clone(self: T) -> T:
        """
        Create a deep clone of this object, properly handling tensors.

        Uses tensor.clone() instead of copy.deepcopy() to preserve gradients.
        Only passes fields with init=True to the constructor; sets init=False
        fields after construction.
        """
        kwargs = {}
        post_init_fields = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.init:
                kwargs[f.name] = _deep_clone(value)
            else:
                post_init_fields.append((f.name, _deep_clone(value)))

        result = type(self)(**kwargs)

        # Set fields that have init=False after construction
        for name, value in post_init_fields:
            result._set_frozen_field(name, value)

        return result

    def _set_state_field(self, path: str, value: Tensor) -> None:
        """Set a state field value given its path."""
        parts = path.replace(']', '').split('[')  # Handle dict indexing

        obj = self
        for i, part in enumerate(parts[:-1]):
            if '.' in part:
                subparts = part.split('.')
                for sp in subparts:
                    obj = getattr(obj, sp)
            else:
                attr = getattr(obj, part, None)
                if attr is None:
                    # This is a dict key
                    continue
                if isinstance(attr, dict):
                    # Next part is the key
                    key = parts[i + 1].split('.')[0]
                    obj = attr[key]
                    parts[i + 1] = '.'.join(parts[i + 1].split('.')[1:]) if '.' in parts[i + 1] else ''
                else:
                    obj = attr

        # Handle the final part
        final_part = parts[-1]
        if '.' in final_part:
            subparts = final_part.split('.')
            for sp in subparts[:-1]:
                if sp:  # Skip empty strings
                    obj = getattr(obj, sp)
            final_attr = subparts[-1]
        else:
            final_attr = final_part

        if final_attr:
            setattr(obj, final_attr, value)

    def to_state_tensor(self) -> Tensor:
        """
        Pack all is_state=True fields into a 1D tensor.

        Recursively traverses nested TensorDataclass instances.
        """
        state_fields = self._get_state_fields()
        if not state_fields:
            return torch.tensor([])

        tensors = []
        for path, value in state_fields:
            if isinstance(value, Tensor):
                tensors.append(value.flatten())
            else:
                tensors.append(torch.tensor([float(value)]))

        return torch.cat(tensors)

    @classmethod
    def from_state_tensor(cls: Type[T], tensor: Tensor, template: T) -> T:
        """
        Unpack a 1D tensor back into state fields.

        Args:
            tensor: The packed state tensor
            template: An existing instance to use as a template for structure

        Returns:
            A new instance with state fields populated from the tensor
        """
        # Use fast path with cached field info
        return cls.from_state_tensor_fast(tensor, template)

    @classmethod
    def from_state_tensor_fast(cls: Type[T], tensor: Tensor, template: T) -> T:
        """
        Fast version of from_state_tensor using cached paths and shallow cloning.

        Instead of deep cloning the entire template and then setting state fields,
        this method:
        1. Uses cached state field paths from the template
        2. Creates a shallow clone (shares non-state data with template)
        3. Only allocates new tensors for state fields

        This is ~5x faster than the original from_state_tensor.
        """
        # Get cached paths and shapes from template
        paths_and_shapes = template._get_state_field_paths()

        if not paths_and_shapes:
            return template._clone()

        # Create shallow clone - share structure but prepare to update state fields
        result = template._shallow_clone()

        # Unpack tensor into state fields using cached info
        idx = 0
        for path, shape, numel in paths_and_shapes:
            if shape:  # Non-empty shape means it's a tensor
                new_value = tensor[idx:idx + numel].reshape(shape)
            else:  # Scalar
                new_value = tensor[idx]
            idx += numel

            result._set_state_field(path, new_value)

        return result

    def _get_clone_recipe(self) -> tuple:
        """
        Get a cached recipe for shallow cloning this object.

        Returns a tuple of:
        - init_field_recipes: list of (name, action, dict_keys_or_none)
          where action is 'ref', 'shallow_clone', or 'dict_shallow_clone'
        - post_init_field_recipes: list of (name, action)
        """
        structure_key = self._get_structure_key()
        cache_key = (type(self), structure_key)

        if cache_key in _CLONE_RECIPE_CACHE:
            return _CLONE_RECIPE_CACHE[cache_key]

        init_recipes = []
        post_init_recipes = []

        for f in fields(self):
            value = getattr(self, f.name)
            is_state = f.metadata.get('is_state', False)

            if f.init:
                if is_state and isinstance(value, Tensor):
                    init_recipes.append((f.name, 'ref', None))
                elif is_tensor_dataclass(value):
                    init_recipes.append((f.name, 'shallow_clone', None))
                elif isinstance(value, dict) and value and is_tensor_dataclass(next(iter(value.values()), None)):
                    init_recipes.append((f.name, 'dict_shallow_clone', tuple(value.keys())))
                else:
                    init_recipes.append((f.name, 'ref', None))
            else:
                if is_tensor_dataclass(value):
                    post_init_recipes.append((f.name, 'shallow_clone'))
                else:
                    post_init_recipes.append((f.name, 'ref'))

        recipe = (tuple(init_recipes), tuple(post_init_recipes))
        _CLONE_RECIPE_CACHE[cache_key] = recipe
        return recipe

    def _shallow_clone(self: T) -> T:
        """
        Create a shallow clone that shares non-tensor data with the original.

        This is faster than _clone() because it doesn't deep copy tensors
        or nested structures. State fields will be overwritten anyway.
        Uses a cached recipe for even faster execution.
        """
        init_recipes, post_init_recipes = self._get_clone_recipe()

        kwargs = {}
        for name, action, extra in init_recipes:
            value = getattr(self, name)
            if action == 'ref':
                kwargs[name] = value
            elif action == 'shallow_clone':
                kwargs[name] = value._shallow_clone()
            elif action == 'dict_shallow_clone':
                kwargs[name] = {k: v._shallow_clone() for k, v in value.items()}

        result = type(self)(**kwargs)

        for name, action in post_init_recipes:
            value = getattr(self, name)
            if action == 'shallow_clone':
                result._set_frozen_field(name, value._shallow_clone())
            else:
                result._set_frozen_field(name, value)

        # Copy cached structure key to avoid recomputation
        cached_key = getattr(self, '_cached_structure_key', None)
        if cached_key is not None:
            object.__setattr__(result, '_cached_structure_key', cached_key)

        return result

    @classmethod
    def zeros(cls: Type[T], template: T = None) -> T:
        """
        Create a zero-initialized instance.

        If template is provided, creates zeros with same structure.
        Otherwise creates default instance with zero tensors.
        """
        if template is not None:
            result = template._clone()
            state_fields = result._get_state_fields()
            for path, value in state_fields:
                if isinstance(value, Tensor):
                    result._set_state_field(path, torch.zeros_like(value))
                else:
                    result._set_state_field(path, torch.tensor(0.0))
            return result

        # Without template, create default instance
        # Subclasses should override this for proper initialization
        raise NotImplementedError(
            "zeros() without template requires subclass implementation"
        )

    def __add__(self: T, other: T) -> T:
        """Element-wise addition of state fields."""
        if not isinstance(other, type(self)):
            return NotImplemented

        result = self._clone()
        self_states = self._get_state_fields()
        other_states = other._get_state_fields()

        for (path, self_val), (_, other_val) in zip(self_states, other_states):
            if isinstance(self_val, Tensor) and isinstance(other_val, Tensor):
                result._set_state_field(path, self_val + other_val)
            elif isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
                result._set_state_field(path, torch.tensor(float(self_val) + float(other_val)))

        return result

    def __mul__(self: T, scalar: float) -> T:
        """Scalar multiplication of state fields."""
        result = self._clone()
        state_fields = result._get_state_fields()

        for path, value in state_fields:
            if isinstance(value, Tensor):
                result._set_state_field(path, value * scalar)
            else:
                result._set_state_field(path, torch.tensor(float(value) * scalar))

        return result

    def __rmul__(self: T, scalar: float) -> T:
        """Right scalar multiplication."""
        return self.__mul__(scalar)
