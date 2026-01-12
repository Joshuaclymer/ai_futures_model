"""Serialization utilities for converting simulation objects to JSON."""

import math


def tensor_to_value(t):
    """Convert a tensor to a JSON-serializable value."""
    if t is None:
        return None
    if hasattr(t, 'item'):
        # Check if it's a single-element tensor
        if hasattr(t, 'numel') and t.numel() == 1:
            val = t.item()
            if not math.isfinite(val):
                return None
            return val
        # Multi-element tensor - convert to list
        if hasattr(t, 'tolist'):
            return t.tolist()
        # Fallback for numpy arrays
        if hasattr(t, '__len__') and len(t) > 1:
            return [tensor_to_value(x) for x in t]
        # Single element without numel (numpy scalar)
        try:
            val = t.item()
            if not math.isfinite(val):
                return None
            return val
        except (ValueError, TypeError):
            return float(t) if isinstance(t, (int, float)) else str(t)
    return float(t) if isinstance(t, (int, float)) else t


def serialize_dataclass(obj, depth=0):
    """
    Recursively serialize a dataclass (including TensorDataclass) to a dict.
    Handles nested dataclasses, dicts, lists, and tensors.
    """
    if depth > 10:  # Prevent infinite recursion
        return str(obj)

    if obj is None:
        return None

    # Handle tensors
    if hasattr(obj, 'item'):
        return tensor_to_value(obj)

    # Handle primitive types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle lists
    if isinstance(obj, (list, tuple)):
        return [serialize_dataclass(item, depth + 1) for item in obj]

    # Handle dicts
    if isinstance(obj, dict):
        return {k: serialize_dataclass(v, depth + 1) for k, v in obj.items()}

    # Handle dataclasses
    from dataclasses import fields, is_dataclass
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = serialize_dataclass(value, depth + 1)
        return result

    # Fallback: convert to string
    return str(obj)


def serialize_world(world) -> dict:
    """Serialize a World object to a JSON-serializable dict."""
    return serialize_dataclass(world)


def serialize_simulation_result(result) -> dict:
    """
    Serialize a SimulationTrajectory to a JSON-serializable dict.
    Returns the raw trajectory of World objects.
    """
    return {
        'times': [t.item() for t in result.times],
        'trajectory': [serialize_world(world) for world in result.trajectory],
        'params': serialize_dataclass(result.params),
    }
