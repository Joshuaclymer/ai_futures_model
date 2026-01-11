"""Core simulation execution utilities."""

import sys
import time
import logging
from pathlib import Path

# Add ai_futures_simulator subdirectory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator"))

from ai_futures_simulator import AIFuturesSimulator
from .parameters import (
    frontend_params_to_simulation_params,
    get_model_params,
)
from .serialization import serialize_simulation_result

logger = logging.getLogger(__name__)


def run_simulation_internal(frontend_params: dict, time_range: list) -> dict:
    """
    Internal function to run simulation and return serialized raw result.

    This is the single source of truth for running simulations.
    Both /api/run-simulation and /api/run-sw-progress-simulation use this.

    Returns:
        dict with keys: success, times, trajectory, params, generation_time_seconds
    """
    # Convert frontend params to simulation params
    sim_params = frontend_params_to_simulation_params(frontend_params, time_range)

    # Create simulator (uses cached model params)
    model_params = get_model_params()
    simulator = AIFuturesSimulator(model_parameters=model_params)

    # Run simulation with the converted parameters
    start_time = time.perf_counter()
    result = simulator.run_simulation(params=sim_params)
    elapsed = time.perf_counter() - start_time
    logger.info(f"[simulation] Completed in {elapsed:.3f}s")

    # Serialize the raw result
    return {
        'success': True,
        **serialize_simulation_result(result),
        'generation_time_seconds': elapsed,
    }
