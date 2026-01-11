"""
Dynamic schema generator for flat tensor state representation.

Recursively walks through World's state fields to build an index mapping
from path strings to tensor indices.
"""

import torch
from torch import Tensor
from dataclasses import fields, is_dataclass
from typing import Dict, Any, List

from classes.tensor_dataclass import is_tensor_dataclass


def generate_state_schema(template_world: 'World') -> Dict[str, int]:
    """
    Recursively walk through World's state fields and assign indices.

    Returns dict mapping "path.to.field" -> index in flat tensor.
    Only includes fields marked with metadata={'is_state': True}.

    Args:
        template_world: A World instance to use as template for structure

    Returns:
        Dictionary mapping path strings to tensor indices
    """
    schema = {}
    index = [0]  # Use list for mutable counter in recursion

    def recurse(obj: Any, prefix: str = ""):
        """Recursively traverse dataclass fields to build schema."""
        if not is_dataclass(obj):
            return

        for f in fields(obj):
            value = getattr(obj, f.name)
            path = f"{prefix}{f.name}" if prefix else f.name
            is_state = f.metadata.get('is_state', False)

            if is_state:
                if isinstance(value, Tensor):
                    # Single tensor state field
                    numel = value.numel()
                    if numel == 1:
                        schema[path] = index[0]
                        index[0] += 1
                    else:
                        # Multi-element tensor - store start index and size
                        for i in range(numel):
                            schema[f"{path}[{i}]"] = index[0]
                            index[0] += 1
                elif isinstance(value, (int, float)):
                    # Scalar state field
                    schema[path] = index[0]
                    index[0] += 1
                elif is_tensor_dataclass(value):
                    # Nested TensorDataclass marked as state - recurse into it
                    recurse(value, f"{path}.")
                # Note: Lists are NOT traversed when is_state=True, matching TensorDataclass._get_state_fields()
                # This is because the original implementation doesn't pack list items as state
            elif is_tensor_dataclass(value):
                # Nested TensorDataclass not marked as state - may contain state fields
                recurse(value, f"{path}.")
            elif isinstance(value, dict):
                # Handle dict of TensorDataclass (e.g., Dict[str, Nation])
                for key, item in value.items():
                    if is_tensor_dataclass(item):
                        recurse(item, f"{path}[{key}].")
            elif isinstance(value, list):
                # Handle list of TensorDataclass not marked as state but containing state
                for i, item in enumerate(value):
                    if is_tensor_dataclass(item):
                        recurse(item, f"{path}[{i}].")

    recurse(template_world)
    return schema


def get_nested_attr(obj: Any, path: str) -> Any:
    """
    Get a nested attribute value given its path string.

    Handles paths like:
    - "compute.prc_stock"
    - "nations[PRC].compute_stock.functional_tpp_h100e"
    - "black_projects[bp1].operating_compute[0].functional_tpp_h100e"

    Args:
        obj: The root object to traverse
        path: Dot-separated path with optional bracket notation for dict/list access

    Returns:
        The value at the specified path
    """
    current = obj

    # Parse the path into segments
    segments = _parse_path(path)

    for segment in segments:
        if isinstance(segment, str):
            # Attribute access
            current = getattr(current, segment)
        elif isinstance(segment, tuple):
            # Index access (dict key or list index)
            attr_name, key = segment
            if attr_name:
                current = getattr(current, attr_name)
            # Try dict access first, then list/tuple
            if isinstance(current, dict):
                current = current[key]
            elif isinstance(current, (list, tuple)):
                current = current[int(key)]
            else:
                # Tensor indexing
                current = current[int(key)]

    return current


def set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """
    Set a nested attribute value given its path string.

    Args:
        obj: The root object to modify
        path: Dot-separated path with optional bracket notation
        value: The value to set
    """
    segments = _parse_path(path)

    # Navigate to parent
    current = obj
    for segment in segments[:-1]:
        if isinstance(segment, str):
            current = getattr(current, segment)
        elif isinstance(segment, tuple):
            attr_name, key = segment
            if attr_name:
                current = getattr(current, attr_name)
            if isinstance(current, dict):
                current = current[key]
            elif isinstance(current, (list, tuple)):
                current = current[int(key)]
            else:
                current = current[int(key)]

    # Set the final value
    final = segments[-1]
    if isinstance(final, str):
        # Use _set_frozen_field if available (for frozen dataclasses)
        if hasattr(current, '_set_frozen_field'):
            current._set_frozen_field(final, value)
        else:
            setattr(current, final, value)
    elif isinstance(final, tuple):
        attr_name, key = final
        if attr_name:
            parent = getattr(current, attr_name)
        else:
            parent = current
        if isinstance(parent, dict):
            parent[key] = value
        elif isinstance(parent, (list, tuple)):
            parent[int(key)] = value
        elif isinstance(parent, Tensor):
            parent[int(key)] = value


def _parse_path(path: str) -> List[Any]:
    """
    Parse a path string into segments.

    Returns a list where:
    - Plain strings represent attribute access
    - Tuples (attr_name, key) represent indexed access

    Examples:
        "foo.bar" -> ["foo", "bar"]
        "nations[PRC].compute" -> [("nations", "PRC"), "compute"]
        "items[0].value" -> [("items", "0"), "value"]
    """
    segments = []
    current = ""
    i = 0

    while i < len(path):
        char = path[i]

        if char == '.':
            if current:
                segments.append(current)
                current = ""
            i += 1
        elif char == '[':
            # Find closing bracket
            j = path.index(']', i)
            key = path[i+1:j]
            if current:
                segments.append((current, key))
                current = ""
            else:
                # Bracket follows another bracket
                segments.append(("", key))
            i = j + 1
        else:
            current += char
            i += 1

    if current:
        segments.append(current)

    return segments


def extract_non_state_fields(world: 'World') -> Dict[str, Any]:
    """
    Extract non-state fields from a World instance.

    These are fields that don't participate in ODE integration but are needed
    for reconstruction (e.g., entity IDs, parameters, metadata).

    Returns a dict that can be used with create_world_from_metadata().
    """

    # For now, return a shallow copy of structure-defining fields
    # The flat tensor will hold state values, metadata holds structure
    metadata = {
        'coalitions_keys': list(world.coalitions.keys()),
        'nations_keys': list(world.nations.keys()),
        'ai_software_developers_keys': list(world.ai_software_developers.keys()),
        'ai_policies_keys': list(world.ai_policies.keys()),
        'black_projects_keys': list(world.black_projects.keys()),
        'perceptions_keys': list(world.perceptions.keys()),
    }

    # Store entity IDs and other non-state info
    for nation_id, nation in world.nations.items():
        metadata[f'nations[{nation_id}].id'] = nation.id

    for dev_id, dev in world.ai_software_developers.items():
        metadata[f'ai_software_developers[{dev_id}].id'] = dev.id
        # Store the length of operating_compute list
        metadata[f'ai_software_developers[{dev_id}].operating_compute_len'] = len(dev.operating_compute)

    for bp_id, bp in world.black_projects.items():
        metadata[f'black_projects[{bp_id}].id'] = bp.id
        metadata[f'black_projects[{bp_id}].operating_compute_len'] = len(bp.operating_compute)

    return metadata


def world_to_flat_tensor(world: 'World', schema: Dict[str, int]) -> Tensor:
    """
    Convert a World instance to a flat tensor using the given schema.

    Args:
        world: The World instance to convert
        schema: The schema mapping paths to indices

    Returns:
        A 1D tensor containing all state values
    """
    state_tensor = torch.zeros(len(schema), dtype=torch.float64)

    for path, idx in schema.items():
        value = get_nested_attr(world, path)
        if isinstance(value, Tensor):
            state_tensor[idx] = value.item() if value.numel() == 1 else value.flatten()[0]
        else:
            state_tensor[idx] = float(value)

    return state_tensor


def flat_tensor_to_world(state_tensor: Tensor, schema: Dict[str, int], template_world: 'World') -> 'World':
    """
    Update a World instance's state fields from a flat tensor.

    Args:
        state_tensor: The flat tensor containing state values
        schema: The schema mapping paths to indices
        template_world: The World instance to update (will be cloned)

    Returns:
        A new World instance with updated state values
    """
    # Clone the template to avoid modifying it
    world = template_world._clone()

    for path, idx in schema.items():
        value = state_tensor[idx]
        # Get the original to check type
        original = get_nested_attr(template_world, path)
        if isinstance(original, Tensor):
            set_nested_attr(world, path, value.reshape(original.shape))
        else:
            set_nested_attr(world, path, value.item())

    return world
