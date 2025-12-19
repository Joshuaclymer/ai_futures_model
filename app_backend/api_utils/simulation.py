"""Core simulation execution utilities."""

import sys
import time
import math
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_futures_simulator import AIFuturesSimulator
from .parameters import (
    frontend_params_to_simulation_params,
    get_model_params,
    DEVELOPER_ID,
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
    logger.info(f"[run_simulation_internal] Simulation completed in {elapsed:.3f}s")

    # Serialize the raw result
    return {
        'success': True,
        **serialize_simulation_result(result),
        'generation_time_seconds': elapsed,
    }


def extract_sw_progress_from_raw(raw_result: dict) -> dict:
    """
    Extract software progress time series from raw simulation result.

    This transforms the raw World trajectory into the format expected
    by the frontend for software progress visualization.

    Args:
        raw_result: Output from run_simulation_internal()

    Returns:
        dict with time_series, milestones, exp_capacity_params, horizon_params
    """
    times = raw_result['times']
    trajectory = raw_result['trajectory']
    params = raw_result['params']

    time_series = []

    for i, world in enumerate(trajectory):
        year = times[i]

        # Get the developer's AI software progress
        ai_devs = world.get('ai_software_developers', {})
        developer = ai_devs.get(DEVELOPER_ID)
        if developer is None and ai_devs:
            # Fallback: get first developer
            developer = next(iter(ai_devs.values()))

        if developer is None:
            continue

        prog = developer.get('ai_software_progress', {})

        # Helper to safely get value with default
        def safe_get(d, key, default=0.0):
            val = d.get(key)
            if val is None:
                return default
            if not math.isfinite(val):
                return None
            return val

        # Extract metrics
        point = {
            'year': year,
            'progress': safe_get(prog, 'progress'),
            'researchStock': safe_get(prog, 'research_stock'),
            'automationFraction': safe_get(prog, 'automation_fraction'),
            'aiCodingLaborMultiplier': safe_get(prog, 'ai_coding_labor_multiplier', 1.0),
            'aiSwProgressMultRefPresentDay': safe_get(prog, 'ai_sw_progress_mult_ref_present_day', 1.0),
            'softwareProgressRate': safe_get(prog, 'software_progress_rate'),
            'researchEffort': safe_get(prog, 'research_effort'),
            'codingLabor': safe_get(prog, 'coding_labor'),
            'serialCodingLabor': safe_get(prog, 'serial_coding_labor'),
            'aiResearchTaste': safe_get(prog, 'ai_research_taste'),
            'aggregateResearchTaste': safe_get(prog, 'aggregate_research_taste', 1.0),
        }

        # Optional metrics
        horizon = prog.get('horizon_length')
        if horizon is not None:
            point['horizonLength'] = horizon if math.isfinite(horizon) else None

        # Compute effective compute (training compute + software progress)
        base_compute_ooms = 24  # Baseline ~10^24 FLOP
        if point['progress'] is not None:
            point['effectiveCompute'] = base_compute_ooms + point['progress']
        else:
            point['effectiveCompute'] = None

        time_series.append(point)

    # Get software R&D params for milestones
    sw_params = params.get('software_r_and_d', {})

    # Build milestones
    milestones = {
        'AC': {
            'interpolation_type': 'linear',
            'metric': 'progress',
            'target': sw_params.get('progress_at_aa') or 100.0,
        }
    }

    return {
        'success': True,
        'time_series': time_series,
        'milestones': milestones,
        'exp_capacity_params': {
            'rho': sw_params.get('rho_experiment_capacity'),
            'alpha': sw_params.get('alpha_experiment_capacity'),
            'experiment_compute_exponent': sw_params.get('experiment_compute_exponent'),
        },
        'horizon_params': None,
    }
