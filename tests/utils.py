"""
Test utilities for AI Futures Model golden data tests.

Contains field mappings, tolerances, and assertion helpers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Path configuration
TESTS_DIR = Path(__file__).parent
GOLDEN_DATA_DIR = TESTS_DIR / "golden_data"

# =============================================================================
# Tolerance Configuration
# =============================================================================
# All tolerance values used in tests should be defined here as named constants.
# This ensures consistency and makes it easy to adjust tolerances globally.

# Relative tolerances for comparing trajectories
TRAJECTORY_RTOL = 1e-3      # 0.1% - for ODE-derived trajectory values
MILESTONE_RTOL = 1e-3       # for milestone times
STRICT_RTOL = 1e-10         # for exact comparisons (constants)

# Absolute tolerances for numerical comparisons
NUMERICAL_ZERO = 1e-10      # threshold for treating values as zero
FLOAT32_ATOL = 1e-6         # absolute tolerance for float32 precision

# Time-related tolerances
TIME_SYNC_ATOL = 0.01       # tolerance for World.current_time vs times array (years)
YEAR_RANGE_ATOL = 1.0       # tolerance for simulation start/end year checks

# Property test tolerances
MONOTONICITY_ATOL = 1e-10   # tolerance for monotonicity checks
BOUNDS_ATOL = 1e-10         # tolerance for bounds checks (e.g., [0, 1])

# Threshold constants
COMPLIANCE_THRESHOLD = 0.9  # minimum fraction for "usually" assertions (90%)
REPRODUCIBILITY_DECIMALS = 10  # decimal places for reproducibility checks

# Expected data ranges (from input_data.csv)
EXPECTED_INPUT_DATA_START_YEAR = 2017.0
EXPECTED_INPUT_DATA_END_YEAR = 2200.0

# Test configuration
SHORT_SIMULATION_EVAL_POINTS = 10  # number of eval points for short simulation tests


# Field mapping from production API (camelCase) to PyTorch model (snake_case)
# Format: production_field -> pytorch_field
FIELD_MAPPING = {
    # Core trajectory fields
    "year": "year",
    "progress": "progress",
    "automationFraction": "automation_fraction",
    "aiResearchTaste": "ai_research_taste",
    "effectiveCompute": "effective_compute",  # Note: might be computed differently
    "researchStock": "research_stock",
    "horizonLength": "horizon_length",
    "softwareProgressRate": "software_progress_rate",
    "experimentCapacity": "experiment_capacity",
    # Additional metrics
    "codingLabor": "coding_labor",
    "serialCodingLabor": "serial_coding_labor",
    "researchEffort": "research_effort",
}

# Reverse mapping (PyTorch -> production)
REVERSE_FIELD_MAPPING = {v: k for k, v in FIELD_MAPPING.items()}


def load_golden_data(filename: str) -> Dict[str, Any]:
    """Load a golden data file."""
    filepath = GOLDEN_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Golden data file not found: {filepath}\n"
            f"Run 'python -m scripts.generate_golden_data' to generate golden data."
        )
    with open(filepath, "r") as f:
        return json.load(f)


def get_golden_time_series(golden_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract time series from golden data."""
    return golden_data["time_series"]


def get_golden_years(golden_data: Dict[str, Any]) -> np.ndarray:
    """Extract years array from golden data."""
    time_series = get_golden_time_series(golden_data)
    return np.array([p["year"] for p in time_series])


def get_golden_field(golden_data: Dict[str, Any], field: str) -> np.ndarray:
    """
    Extract a field from golden data time series.

    Args:
        golden_data: Golden data dictionary
        field: Field name (production API format, e.g., "automationFraction")

    Returns:
        NumPy array of field values over time
    """
    time_series = get_golden_time_series(golden_data)
    return np.array([p.get(field) for p in time_series])


def get_golden_parameters(golden_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters used to generate this golden data."""
    return golden_data.get("_metadata", {}).get("parameters", {})


def get_golden_time_range(golden_data: Dict[str, Any]) -> List[int]:
    """Extract time range used to generate this golden data."""
    return golden_data.get("_metadata", {}).get("time_range", [2015, 2050])


def extract_trajectory_field(trajectory, pytorch_field: str) -> np.ndarray:
    """
    Extract a field from a PyTorch simulation trajectory.

    Args:
        trajectory: SimulationTrajectory object from PyTorch model
        pytorch_field: Field name in PyTorch format (e.g., "automation_fraction")

    Returns:
        NumPy array of field values over time
    """
    values = []
    for world in trajectory.trajectory:
        # Navigate nested world structure to find the field
        value = get_world_field(world, pytorch_field)
        if value is not None:
            values.append(float(value))
        else:
            values.append(np.nan)
    return np.array(values)


def get_world_field(world, field_name: str) -> Optional[float]:
    """
    Get a field value from a World object.

    The World object has a nested structure:
    - World.current_time
    - World.ai_software_developers[id].ai_software_progress.{field}
    - World.black_projects[id].ai_software_progress.{field}
    """
    # Check current_time for year
    if field_name == "year" and hasattr(world, "current_time"):
        val = world.current_time
        return float(val.item() if hasattr(val, "item") else val)

    # Check direct attributes on world
    if hasattr(world, field_name):
        val = getattr(world, field_name)
        if val is not None:
            return float(val.item() if hasattr(val, "item") else val)

    # Check ai_software_developers for software progress fields
    if hasattr(world, "ai_software_developers") and world.ai_software_developers:
        for dev_id, developer in world.ai_software_developers.items():
            if hasattr(developer, "ai_software_progress") and developer.ai_software_progress is not None:
                sw = developer.ai_software_progress
                if hasattr(sw, field_name):
                    val = getattr(sw, field_name)
                    if val is not None:
                        return float(val.item() if hasattr(val, "item") else val)

    # Check black_projects for software progress fields
    if hasattr(world, "black_projects") and world.black_projects:
        for proj_id, project in world.black_projects.items():
            if hasattr(project, "ai_software_progress") and project.ai_software_progress is not None:
                sw = project.ai_software_progress
                if hasattr(sw, field_name):
                    val = getattr(sw, field_name)
                    if val is not None:
                        return float(val.item() if hasattr(val, "item") else val)

    return None


def assert_arrays_close(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = TRAJECTORY_RTOL,
    atol: float = 0.0,
    field_name: str = "",
) -> None:
    """
    Assert two arrays are close within tolerances.

    Args:
        actual: Actual values from PyTorch model
        expected: Expected values from golden data
        rtol: Relative tolerance
        atol: Absolute tolerance
        field_name: Name of field being compared (for error messages)
    """
    # Filter out NaN values from both arrays (at matching positions)
    mask = ~(np.isnan(actual) | np.isnan(expected))
    actual_clean = actual[mask]
    expected_clean = expected[mask]

    if len(actual_clean) == 0:
        raise ValueError(f"No valid (non-NaN) values to compare for field: {field_name}")

    try:
        np.testing.assert_allclose(
            actual_clean,
            expected_clean,
            rtol=rtol,
            atol=atol,
            err_msg=f"Mismatch in field: {field_name}",
        )
    except AssertionError as e:
        # Add more diagnostic info
        abs_diff = np.abs(actual_clean - expected_clean)
        rel_diff = np.where(expected_clean != 0, abs_diff / np.abs(expected_clean), abs_diff)
        max_rel_diff = np.max(rel_diff)
        max_abs_diff = np.max(abs_diff)
        max_diff_idx = np.argmax(rel_diff)

        raise AssertionError(
            f"{e}\n"
            f"Max relative diff: {max_rel_diff:.6f} (at index {max_diff_idx})\n"
            f"Max absolute diff: {max_abs_diff:.6f}\n"
            f"Values at max diff: actual={actual_clean[max_diff_idx]:.6f}, "
            f"expected={expected_clean[max_diff_idx]:.6f}"
        )


def assert_monotonic(values: np.ndarray, field_name: str = "", non_decreasing: bool = True) -> None:
    """Assert that an array is monotonic (non-decreasing or non-increasing)."""
    if non_decreasing:
        violations = np.where(np.diff(values) < -MONOTONICITY_ATOL)[0]
        if len(violations) > 0:
            raise AssertionError(
                f"Field {field_name} is not monotonically non-decreasing. "
                f"Violations at indices: {violations[:5]}... (showing first 5)"
            )
    else:
        violations = np.where(np.diff(values) > MONOTONICITY_ATOL)[0]
        if len(violations) > 0:
            raise AssertionError(
                f"Field {field_name} is not monotonically non-increasing. "
                f"Violations at indices: {violations[:5]}... (showing first 5)"
            )


def assert_bounded(values: np.ndarray, lower: float, upper: float, field_name: str = "") -> None:
    """Assert that all values are within bounds."""
    violations_low = np.where(values < lower - BOUNDS_ATOL)[0]
    violations_high = np.where(values > upper + BOUNDS_ATOL)[0]

    if len(violations_low) > 0 or len(violations_high) > 0:
        min_val = np.min(values)
        max_val = np.max(values)
        raise AssertionError(
            f"Field {field_name} out of bounds [{lower}, {upper}]. "
            f"Actual range: [{min_val:.6f}, {max_val:.6f}]"
        )


def assert_finite(values: np.ndarray, field_name: str = "") -> None:
    """Assert that all values are finite (not NaN or Inf)."""
    non_finite = ~np.isfinite(values)
    if np.any(non_finite):
        non_finite_indices = np.where(non_finite)[0]
        raise AssertionError(
            f"Field {field_name} contains non-finite values at indices: "
            f"{non_finite_indices[:10]}... (showing first 10)"
        )


def list_golden_data_files() -> List[str]:
    """List all available golden data files."""
    if not GOLDEN_DATA_DIR.exists():
        return []
    return [f.name for f in GOLDEN_DATA_DIR.glob("*.json")]


def has_golden_data() -> bool:
    """Check if golden data has been generated."""
    return len(list_golden_data_files()) > 0
