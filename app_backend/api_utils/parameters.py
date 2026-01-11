"""Parameter conversion and loading utilities."""

import sys
import copy
from pathlib import Path
from dataclasses import fields, is_dataclass
from typing import Any

# Add ai_futures_simulator subdirectory to path for imports
# Structure: ai_futures_simulator/app_backend/api_utils/parameters.py
#            ai_futures_simulator/ai_futures_simulator/parameters/...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator"))

from parameters.model_parameters import ModelParameters
from parameters.classes import (
    SimulationParameters,
    SimulationSettings,
)

# Default YAML config path
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator" / "parameters" / "default_parameters.yaml"

# Cache the model parameters (loaded once at startup)
_cached_model_params = None


def get_model_params():
    """Get cached model parameters, loading from YAML if needed."""
    global _cached_model_params
    if _cached_model_params is None:
        _cached_model_params = ModelParameters.from_yaml(DEFAULT_CONFIG_PATH)
    return _cached_model_params


def _get_nested_attr(obj: Any, path: str) -> Any:
    """Get a nested attribute using dot notation (e.g., 'software_r_and_d.rho_coding_labor')."""
    parts = path.split('.')
    for part in parts:
        obj = getattr(obj, part)
    return obj


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute using dot notation (e.g., 'software_r_and_d.rho_coding_labor')."""
    parts = path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _deep_copy_dataclass(obj: Any) -> Any:
    """Create a mutable deep copy of a dataclass hierarchy."""
    if not is_dataclass(obj):
        return copy.deepcopy(obj)

    kwargs = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if is_dataclass(value):
            kwargs[f.name] = _deep_copy_dataclass(value)
        else:
            kwargs[f.name] = copy.deepcopy(value)
    return type(obj)(**kwargs)


# Parameters that need special transformation before being set
PARAM_TRANSFORMS = {
    # Frontend sends ac_time_horizon_minutes in log10 scale if value < 1000
    'software_r_and_d.ac_time_horizon_minutes': lambda v: 10 ** v if isinstance(v, (int, float)) and v < 1000 else v,
}


def _build_dataclass_with_overrides(default_obj: Any, overrides: dict, prefix: str = '') -> Any:
    """
    Recursively build a dataclass from defaults with overrides applied.

    Args:
        default_obj: The default dataclass instance to use as template
        overrides: Dict of param_path -> value (e.g., {"rho_coding_labor": 0.5})
        prefix: Current path prefix for nested dataclasses

    Returns:
        A new dataclass instance with overrides applied
    """
    if not is_dataclass(default_obj):
        return copy.deepcopy(default_obj)

    kwargs = {}
    for f in fields(default_obj):
        field_path = f"{prefix}.{f.name}" if prefix else f.name
        default_value = getattr(default_obj, f.name)

        if is_dataclass(default_value):
            # Recursively process nested dataclass
            kwargs[f.name] = _build_dataclass_with_overrides(default_value, overrides, field_path)
        elif field_path in overrides:
            # Direct override for this field
            value = overrides[field_path]
            # Apply any transform if needed
            if field_path in PARAM_TRANSFORMS:
                value = PARAM_TRANSFORMS[field_path](value)
            kwargs[f.name] = value
        else:
            # Use default
            kwargs[f.name] = copy.deepcopy(default_value)

    return type(default_obj)(**kwargs)


def frontend_params_to_simulation_params(frontend_params: dict, time_range: list) -> SimulationParameters:
    """
    Convert frontend parameter format to SimulationParameters.

    Frontend sends parameters using full dot-notation paths matching the backend structure
    (e.g., 'software_r_and_d.rho_coding_labor', 'compute.USComputeParameters.total_us_compute_annual_growth_rate').
    """
    import logging
    logger = logging.getLogger(__name__)

    # Load defaults from YAML
    default_model_params = ModelParameters.from_yaml(DEFAULT_CONFIG_PATH)
    default_sim_params = default_model_params.sample()

    # Build settings from time_range (handled separately since it comes from time_range, not frontend_params)
    start_year = int(time_range[0]) if time_range else 2024
    end_year = float(time_range[1]) if len(time_range) > 1 else 2040.0
    n_eval_points = frontend_params.get('n_eval_points', 100)
    settings = SimulationSettings(
        simulation_start_year=start_year,
        simulation_end_year=end_year,
        n_eval_points=n_eval_points,
        ode_rtol=1.0e-3,
        ode_atol=1.0e-6,
        ode_max_step=0.1,
    )

    # Frontend params are already full paths - just copy them
    normalized_params = {}
    unrecognized_params = []
    for key, value in frontend_params.items():
        # Skip settings params (handled above)
        if key in ('n_eval_points',):
            continue
        normalized_params[key] = value

        # Warn if parameter doesn't look like a full path (no dots)
        if '.' not in key:
            unrecognized_params.append(key)

    if unrecognized_params:
        logger.warning(f"[frontend_params_to_simulation_params] Parameters without dot-notation path (may be unrecognized): {unrecognized_params}")

    # Build each top-level parameter section with overrides
    software_r_and_d = _build_dataclass_with_overrides(
        default_sim_params.software_r_and_d, normalized_params, 'software_r_and_d'
    )
    compute = _build_dataclass_with_overrides(
        default_sim_params.compute, normalized_params, 'compute'
    )
    datacenter_and_energy = _build_dataclass_with_overrides(
        default_sim_params.datacenter_and_energy, normalized_params, 'datacenter_and_energy'
    )
    policy = _build_dataclass_with_overrides(
        default_sim_params.policy, normalized_params, 'policy'
    )

    return SimulationParameters(
        settings=settings,
        software_r_and_d=software_r_and_d,
        compute=compute,
        datacenter_and_energy=datacenter_and_energy,
        policy=policy,
    )
