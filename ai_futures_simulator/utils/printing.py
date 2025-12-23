"""
Printing utilities for the AI Futures Simulator.
"""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from classes.simulation_primitives import SimulationResult


def tensor_to_python(obj: Any) -> Any:
    """Recursively convert tensors and dataclasses to JSON-serializable types."""
    # Handle tensors
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except:
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)

    # Handle dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: tensor_to_python(v) for k, v in asdict(obj).items()}

    # Handle dicts
    if isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}

    # Handle lists
    if isinstance(obj, list):
        return [tensor_to_python(v) for v in obj]

    # Handle tuples
    if isinstance(obj, tuple):
        return [tensor_to_python(v) for v in obj]

    return obj


def print_simulation_summary(result: "SimulationResult", output_file: str = "simulation_output.json"):
    """Convert entire simulation result to JSON and save to file."""
    import sys

    trajectory = result.trajectory
    total = len(trajectory)

    # Progress bar for conversion
    print("Converting results to JSON...", end=" ", flush=True)

    worlds = []
    for i, world in enumerate(trajectory):
        worlds.append(tensor_to_python(world))
        # Simple progress indicator
        if (i + 1) % 10 == 0 or i == total - 1:
            pct = int((i + 1) / total * 100)
            sys.stdout.write(f"\rConverting results to JSON... {pct}%")
            sys.stdout.flush()

    print()  # newline

    output = {
        "params": tensor_to_python(result.params),
        "trajectory": worlds,
    }

    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to: {output_path.absolute()}")
