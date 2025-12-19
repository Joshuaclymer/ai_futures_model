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
                # Recurse into nested TensorDataclass
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

    def _clone(self: T) -> T:
        """
        Create a deep clone of this object, properly handling tensors.

        Uses tensor.clone() instead of copy.deepcopy() to preserve gradients.
        """
        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            kwargs[f.name] = _deep_clone(value)
        return type(self)(**kwargs)

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
        result = template._clone()
        state_fields = result._get_state_fields()

        if not state_fields:
            return result

        idx = 0
        for path, original_value in state_fields:
            if isinstance(original_value, Tensor):
                numel = original_value.numel()
                new_value = tensor[idx:idx + numel].reshape(original_value.shape)
                idx += numel
            else:
                new_value = tensor[idx]
                idx += 1

            result._set_state_field(path, new_value)

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
