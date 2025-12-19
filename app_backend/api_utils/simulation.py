"""Core simulation execution utilities."""

import sys
import time
import math
import logging
from pathlib import Path

# Add ai_futures_simulator subdirectory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator"))

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


def compute_horizon_from_progress(progress: float, present_horizon: float,
                                   present_doubling_time: float,
                                   doubling_difficulty_growth_factor: float,
                                   anchor_progress: float = 0.0) -> float:
    """
    Compute horizon length from progress value using the decaying doubling time formula.

    This matches the frontend's computeHorizonFromProgress function.

    Args:
        progress: Current progress value
        present_horizon: Horizon at present_day (H_0)
        present_doubling_time: Time constant (T_0)
        doubling_difficulty_growth_factor: Growth factor (used to derive A_0)
        anchor_progress: Anchor progress for shifted form

    Returns:
        Horizon length in minutes
    """
    H_0 = present_horizon
    T_0 = present_doubling_time
    A_0 = 1 - doubling_difficulty_growth_factor  # Decay parameter

    # Handle special case where A_0 is zero (no decay)
    if A_0 == 0:
        return H_0 * math.pow(2, progress / T_0)

    # Safety checks
    if T_0 <= 0 or H_0 <= 0 or A_0 >= 1:
        return H_0  # fallback to base horizon

    # Apply progress shifting (shifted form)
    progress_adjusted = progress - anchor_progress

    # Calculate base term: (1 - A_0 * progressAdjusted / T_0)
    base_term = max(1 - A_0 * progress_adjusted / T_0, 1e-12)

    # Calculate log denominator: ln(1 - A_0)
    log_denominator = math.log(max(1 - A_0, 1e-12))

    # Calculate exponent: ln(2) / ln(1 - A_0)
    exponent = math.log(2) / log_denominator

    # Final horizon calculation: H_0 * (base_term)^exponent
    result = H_0 * math.pow(base_term, exponent)

    return result if (math.isfinite(result) and result > 0) else H_0


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

    # Get horizon parameters for computing horizon from progress
    sw_params = params.get('software_r_and_d', {})
    present_horizon = sw_params.get('present_horizon', 26.0)  # minutes
    present_doubling_time = sw_params.get('present_doubling_time', 0.458)
    doubling_difficulty_growth_factor = sw_params.get('doubling_difficulty_growth_factor', 0.92)

    time_series = []

    # Helper to safely get value with default
    def safe_get(d, key, default=0.0):
        val = d.get(key)
        if val is None:
            return default
        if not math.isfinite(val):
            return None
        return val

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

        # Extract metrics
        progress = safe_get(prog, 'progress')
        # Use 'progress_rate' field (the simulator uses this name, not 'software_progress_rate')
        progress_rate = safe_get(prog, 'progress_rate')
        if progress_rate == 0.0:
            # Fallback to software_progress_rate if progress_rate is 0
            progress_rate = safe_get(prog, 'software_progress_rate')

        # Get serial coding labor and multiplier from World state
        serial_coding_labor = safe_get(prog, 'serial_coding_labor')
        serial_coding_labor_multiplier = safe_get(prog, 'serial_coding_labor_multiplier', 1.0)
        ai_coding_labor_mult = safe_get(prog, 'ai_coding_labor_multiplier', 1.0)

        # Get input time series from World state (interpolated values)
        human_labor = safe_get(prog, 'human_labor', 0.0)
        inference_compute = safe_get(prog, 'inference_compute')
        experiment_compute = safe_get(prog, 'experiment_compute')

        # Get experiment capacity from World state
        experiment_capacity = safe_get(prog, 'experiment_capacity')

        point = {
            'year': year,
            'progress': progress,
            'researchStock': safe_get(prog, 'research_stock'),
            'automationFraction': safe_get(prog, 'automation_fraction'),
            'aiCodingLaborMultiplier': ai_coding_labor_mult,
            'aiSwProgressMultRefPresentDay': safe_get(prog, 'ai_sw_progress_mult_ref_present_day', 1.0),
            'softwareProgressRate': progress_rate,
            'researchEffort': safe_get(prog, 'research_effort'),
            'codingLabor': safe_get(prog, 'coding_labor'),
            'serialCodingLabor': serial_coding_labor,
            'serialCodingLaborMultiplier': serial_coding_labor_multiplier,
            'humanLabor': human_labor,
            'inferenceCompute': inference_compute,
            'experimentCompute': experiment_compute,
            'experimentCapacity': experiment_capacity,
            'aiResearchTaste': safe_get(prog, 'ai_research_taste'),
            'aggregateResearchTaste': safe_get(prog, 'aggregate_research_taste', 1.0),
        }

        # Compute horizon from progress if not provided by simulation
        horizon = prog.get('horizon_length')
        if horizon is not None and math.isfinite(horizon):
            point['horizonLength'] = horizon
        elif progress is not None:
            # Compute horizon from progress using the formula
            computed_horizon = compute_horizon_from_progress(
                progress, present_horizon, present_doubling_time,
                doubling_difficulty_growth_factor, anchor_progress=0.0
            )
            point['horizonLength'] = computed_horizon if math.isfinite(computed_horizon) else None
        else:
            point['horizonLength'] = None

        # Compute effective compute (training compute + software progress)
        base_compute_ooms = 24  # Baseline ~10^24 FLOP
        if progress is not None:
            point['effectiveCompute'] = base_compute_ooms + progress
        else:
            point['effectiveCompute'] = None

        time_series.append(point)

    # Compute training_compute and software_efficiency from trajectory data
    # training_compute_growth_rate = progress_rate - software_progress_rate
    # training_compute[i] = cumulative integral via trapezoidal rule
    # software_efficiency[i] = progress[i] - initial_progress - training_compute[i]
    if len(time_series) > 1:
        initial_progress = time_series[0].get('progress', 0.0) or 0.0
        training_compute = [0.0]  # Start at 0

        for i in range(1, len(time_series)):
            dt = time_series[i]['year'] - time_series[i-1]['year']

            # Get progress_rate and software_progress_rate
            pr_curr = time_series[i].get('softwareProgressRate') or 0.0
            pr_prev = time_series[i-1].get('softwareProgressRate') or 0.0

            # For overall progress rate, we can compute from progress difference
            prog_curr = time_series[i].get('progress') or 0.0
            prog_prev = time_series[i-1].get('progress') or 0.0
            overall_rate = (prog_curr - prog_prev) / dt if dt > 0 else 0.0

            # training_compute_growth_rate = overall_rate - software_progress_rate
            tc_rate_prev = overall_rate - pr_prev
            tc_rate_curr = overall_rate - pr_curr

            # Trapezoidal integration
            tc_increment = 0.5 * (tc_rate_prev + tc_rate_curr) * dt
            training_compute.append(training_compute[-1] + tc_increment)

        # Add training_compute and software_efficiency to each point
        for i, point in enumerate(time_series):
            tc = training_compute[i]
            progress = point.get('progress') or 0.0
            sw_eff = progress - initial_progress - tc

            point['trainingCompute'] = tc if math.isfinite(tc) else None
            point['softwareEfficiency'] = sw_eff if math.isfinite(sw_eff) else None

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
