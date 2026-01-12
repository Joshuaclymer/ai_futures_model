"""
Simulation runner for black project Monte Carlo simulations.

Runs the actual AIFuturesSimulator to generate simulation results.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Add ai_futures_simulator subdirectory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "ai_futures_simulator"))

from ai_futures_simulator import AIFuturesSimulator
from parameters.model_parameters import ModelParameters

logger = logging.getLogger(__name__)


def run_black_project_simulations(
    frontend_params: dict,
    num_simulations: int = 100,
    ai_slowdown_start_year: float = 2030.0,
    end_year: float = 2037.0,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulations using the actual AIFuturesSimulator.

    Args:
        frontend_params: Parameters from frontend UI
        num_simulations: Number of Monte Carlo simulations to run
        ai_slowdown_start_year: Year when the AI slowdown agreement takes effect.
            Detection times are measured from this year.
        end_year: Year when simulation ends.

    Returns results dict with:
    - simulation_results: List of SimulationTrajectory objects
    - ai_slowdown_start_year: When agreement takes effect
    - end_year: When simulation ends
    """
    start_time = time.perf_counter()

    # Load model parameters from YAML - use shared default config
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "ai_futures_simulator" / "parameters" / "default_parameters.yaml"
    logger.info(f"[black-project] Loading config from: {config_path}")

    try:
        model_params = ModelParameters.from_yaml(config_path)
        logger.info("[black-project] Model parameters loaded successfully")
    except Exception as e:
        logger.exception(f"[black-project] Failed to load model parameters: {e}")
        raise

    # For black project page: disable software progress updates (only compute matters)
    model_params.params.software_r_and_d.update_software_progress = False
    logger.info("[black-project] Set update_software_progress=False for black project simulations")

    logger.info(f"[black-project] Agreement year: {ai_slowdown_start_year}, End year: {end_year}")

    # Override simulation_end_year from time_range (YAML default is 2040)
    # Keep simulation_start_year from YAML (needs historical data to initialize)
    start_year = getattr(model_params.params.settings, 'simulation_start_year', 2026)
    model_params.params.settings.simulation_end_year = end_year
    # Calculate n_eval_points to maintain 0.1-year resolution for full simulation
    n_years = end_year - start_year
    model_params.params.settings.n_eval_points = int(n_years * 10) + 1
    logger.info(f"[black-project] Updated settings: start_year={start_year}, end_year={end_year}, n_eval_points={model_params.params.settings.n_eval_points}")

    # Create simulator
    simulator = AIFuturesSimulator(model_parameters=model_params)

    # Run Monte Carlo simulations
    logger.info(f"[black-project] Running {num_simulations} simulations...")

    simulation_results = simulator.run_simulations(num_simulations=num_simulations)

    elapsed = time.perf_counter() - start_time
    logger.info(f"[black-project] Completed {len(simulation_results)} simulations in {elapsed:.2f}s")

    return {
        'simulation_results': simulation_results,
        'ai_slowdown_start_year': ai_slowdown_start_year,
        'end_year': end_year,
        'model_params': model_params,
    }
