"""
pytest configuration and fixtures for AI Futures Model tests.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add project paths for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai_futures_simulator"))

from tests.utils import (
    load_golden_data,
    get_golden_parameters,
    get_golden_time_range,
    has_golden_data,
    list_golden_data_files,
    GOLDEN_DATA_DIR,
)


# Mark for tests that require golden data
requires_golden_data = pytest.mark.skipif(
    not has_golden_data(),
    reason="Golden data not generated. Run: python -m scripts.generate_golden_data"
)


@pytest.fixture
def golden_trajectory():
    """Load the default golden trajectory data."""
    return load_golden_data("default_trajectory.json")


@pytest.fixture
def golden_fast_progress():
    """Load the fast progress scenario golden data."""
    return load_golden_data("trajectory_fast_progress.json")


@pytest.fixture
def golden_slow_progress():
    """Load the slow progress scenario golden data."""
    return load_golden_data("trajectory_slow_progress.json")


@pytest.fixture
def computed_trajectory(golden_trajectory):
    """
    Run the PyTorch model with the same parameters as the default golden data.

    Returns a SimulationTrajectory object.
    """
    from ai_futures_simulator import AIFuturesSimulator
    from parameters.model_parameters import ModelParameters

    # Load default parameters
    params_path = PROJECT_ROOT / "ai_futures_simulator" / "parameters" / "default_parameters.yaml"
    model_params = ModelParameters.from_yaml(params_path)

    # Get time range from golden data
    time_range = get_golden_time_range(golden_trajectory)

    # Override simulation settings to match golden data
    # Note: The production API may use different default settings,
    # so we need to align our settings to match
    if model_params.params.settings is not None:
        model_params.params.settings.simulation_start_year = time_range[0]
        model_params.params.settings.simulation_end_year = float(time_range[1])

    # Create simulator and run
    simulator = AIFuturesSimulator(model_parameters=model_params)
    result = simulator.run_modal_simulation()

    return result


@pytest.fixture
def computed_trajectory_for_scenario(request):
    """
    Parametrized fixture for running PyTorch model with scenario parameters.

    Use with indirect=True to pass golden data to this fixture.
    """
    golden_data = request.param

    from ai_futures_simulator import AIFuturesSimulator
    from parameters.model_parameters import ModelParameters

    params_path = PROJECT_ROOT / "ai_futures_simulator" / "parameters" / "default_parameters.yaml"
    model_params = ModelParameters.from_yaml(params_path)

    # Get parameters and time range from golden data
    golden_params = get_golden_parameters(golden_data)
    time_range = get_golden_time_range(golden_data)

    # Apply golden data parameters to model
    # This needs to map production API parameter names to PyTorch parameter names
    apply_golden_params_to_model(model_params, golden_params)

    if model_params.params.settings is not None:
        model_params.params.settings.simulation_start_year = time_range[0]
        model_params.params.settings.simulation_end_year = float(time_range[1])

    simulator = AIFuturesSimulator(model_parameters=model_params)
    result = simulator.run_modal_simulation()

    return result, golden_data


def apply_golden_params_to_model(model_params, golden_params):
    """
    Apply parameters from golden data to PyTorch ModelParameters.

    Maps production API parameter names to PyTorch model parameter structure.
    """
    if not golden_params:
        return  # Empty params means use defaults

    # Parameter mapping from production API to PyTorch model
    PARAM_MAPPING = {
        # Software R&D parameters
        "present_doubling_time": ("software_r_and_d", "present_doubling_time"),
        "rho_coding_labor": ("software_r_and_d", "rho_coding_labor"),
        "rho_serial_coding_labor": ("software_r_and_d", "rho_serial_coding_labor"),
        "serial_share": ("software_r_and_d", "serial_share"),
        "training_compute_weight": ("software_r_and_d", "training_compute_weight"),
        "experiment_weight": ("software_r_and_d", "experiment_weight"),
        "inference_weight": ("software_r_and_d", "inference_weight"),
        # Add more mappings as needed
    }

    for param_name, value in golden_params.items():
        if param_name in PARAM_MAPPING:
            group, attr = PARAM_MAPPING[param_name]
            param_group = getattr(model_params.params, group, None)
            if param_group is not None and hasattr(param_group, attr):
                setattr(param_group, attr, value)


@pytest.fixture
def model_parameters():
    """Load default model parameters."""
    from parameters.model_parameters import ModelParameters

    params_path = PROJECT_ROOT / "ai_futures_simulator" / "parameters" / "default_parameters.yaml"
    return ModelParameters.from_yaml(params_path)


@pytest.fixture
def simulator(model_parameters):
    """Create a simulator instance with default parameters."""
    from ai_futures_simulator import AIFuturesSimulator

    return AIFuturesSimulator(model_parameters=model_parameters)


# Utility function for parametrized tests
def get_all_golden_scenarios():
    """Get all available golden data scenarios for parametrization."""
    files = list_golden_data_files()
    return [f for f in files if f.startswith("trajectory_") or f == "default_trajectory.json"]
