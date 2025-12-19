"""Core simulation execution utilities."""

import sys
import time
import math
import logging
import numpy as np
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

# Constants for training compute normalization (from reference model model_config.py)
TRAINING_COMPUTE_REFERENCE_YEAR = 2025.13
TRAINING_COMPUTE_REFERENCE_OOMS = 26.54

# Milestone SD thresholds (from model_config.py in reference model)
TOP_RESEARCHER_SD = 3.090232306167813  # 99.9th percentile


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
        # Get the software-only progress rate (not the overall rate which includes training compute)
        software_progress_rate = safe_get(prog, 'software_progress_rate')
        # Also get overall progress rate for reference
        overall_progress_rate = safe_get(prog, 'progress_rate')

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

        # Get training_compute_growth_rate and ai_research_taste_sd from World state
        training_compute_growth_rate = safe_get(prog, 'training_compute_growth_rate', 0.0)
        ai_research_taste_sd = safe_get(prog, 'ai_research_taste_sd', 0.0)

        point = {
            'year': year,
            'progress': progress,
            'researchStock': safe_get(prog, 'research_stock'),
            'automationFraction': safe_get(prog, 'automation_fraction'),
            'aiCodingLaborMultiplier': ai_coding_labor_mult,
            'aiSwProgressMultRefPresentDay': safe_get(prog, 'ai_sw_progress_mult_ref_present_day', 1.0),
            'softwareProgressRate': software_progress_rate,
            'overallProgressRate': overall_progress_rate,
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
            'trainingComputeGrowthRate': training_compute_growth_rate,
            'aiResearchTasteSd': ai_research_taste_sd,
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

        # effectiveCompute will be computed after training_compute normalization
        point['effectiveCompute'] = None

        time_series.append(point)

    # Compute training_compute from the stored training_compute_growth_rate time series
    # This matches the reference model's approach in metrics_computation.py
    if len(time_series) > 1:
        initial_progress = time_series[0].get('progress', 0.0) or 0.0
        years = np.array([p['year'] for p in time_series])
        tc_growth_rates = np.array([p.get('trainingComputeGrowthRate', 0.0) or 0.0 for p in time_series])

        # Trapezoidal integration of training_compute_growth_rate
        training_compute = np.zeros(len(time_series))
        if len(time_series) > 1:
            dt = np.diff(years)
            avg_growth_rates = (tc_growth_rates[:-1] + tc_growth_rates[1:]) / 2.0
            increments = avg_growth_rates * dt
            training_compute[1:] = np.cumsum(increments)

        # Normalize training_compute at reference year (like reference model)
        tc_at_ref = float(np.interp(TRAINING_COMPUTE_REFERENCE_YEAR, years, training_compute))
        training_compute = training_compute - tc_at_ref + TRAINING_COMPUTE_REFERENCE_OOMS

        # Compute software_efficiency (raw, before normalization)
        # software_efficiency = progress - initial_progress - training_compute_raw
        # But training_compute is already normalized, so we need to un-normalize it first
        tc_at_start_normalized = training_compute[0]

        # Raw software efficiency (before normalization)
        progress_arr = np.array([p.get('progress', 0.0) or 0.0 for p in time_series])
        software_efficiency_raw = progress_arr - initial_progress - (training_compute - tc_at_start_normalized)

        # Normalize software_efficiency at reference year (like reference model)
        sw_eff_at_ref = float(np.interp(TRAINING_COMPUTE_REFERENCE_YEAR, years, software_efficiency_raw))
        software_efficiency = software_efficiency_raw - sw_eff_at_ref

        # Compute effective_compute = training_compute + software_efficiency
        effective_compute = training_compute + software_efficiency

        # Normalize effective_compute at reference year (like reference model)
        eff_at_ref = float(np.interp(TRAINING_COMPUTE_REFERENCE_YEAR, years, effective_compute))
        effective_compute = effective_compute - eff_at_ref + TRAINING_COMPUTE_REFERENCE_OOMS

        # Add training_compute, software_efficiency, and effective_compute to each point
        for i, point in enumerate(time_series):
            tc = training_compute[i]
            sw_eff = software_efficiency[i]
            eff_comp = effective_compute[i]

            point['trainingCompute'] = tc if math.isfinite(tc) else None
            point['softwareEfficiency'] = sw_eff if math.isfinite(sw_eff) else None
            point['effectiveCompute'] = eff_comp if math.isfinite(eff_comp) else None

    # Get software R&D params for milestones
    sw_params = params.get('software_r_and_d', {})

    # Build milestones - include AC, SAR, SIAR, TED-AI, ASI
    progress_at_aa = sw_params.get('progress_at_aa') or 100.0
    strat_ai_m2b = sw_params.get('strat_ai_m2b', 2.0)
    ted_ai_m2b = sw_params.get('ted_ai_m2b', 3.0)

    milestones = {
        'AC': {
            'interpolation_type': 'linear',
            'metric': 'progress',
            'target': progress_at_aa,
        },
        'SAR-level-experiment-selection-skill': {
            'interpolation_type': 'exponential',
            'metric': 'ai_research_taste_sd',
            'target': TOP_RESEARCHER_SD,
            'requires_ac': True,
        },
        'SIAR-level-experiment-selection-skill': {
            'interpolation_type': 'exponential',
            'metric': 'ai_research_taste_sd',
            'target': TOP_RESEARCHER_SD * 3.0,
            'requires_ac': True,
        },
        'TED-AI': {
            'interpolation_type': 'exponential',
            'metric': 'ai_research_taste_sd',
            'target': TOP_RESEARCHER_SD * (1.0 + ted_ai_m2b),
        },
        'ASI': {
            'interpolation_type': 'exponential',
            'metric': 'ai_research_taste_sd',
            'target': TOP_RESEARCHER_SD * (1.0 + ted_ai_m2b + 2.0),
        },
    }

    # Compute milestone times and progress_multiplier by interpolation
    years_arr = np.array([p['year'] for p in time_series])
    progress_arr = np.array([p.get('progress', 0) or 0 for p in time_series])
    ai_research_taste_sd_arr = np.array([p.get('aiResearchTasteSd', 0) or 0 for p in time_series])
    ai_sw_mult_arr = np.array([p.get('aiSwProgressMultRefPresentDay', 1) or 1 for p in time_series])

    for milestone_name, milestone in milestones.items():
        metric = milestone.get('metric')
        target = milestone.get('target')

        if metric == 'progress':
            metric_arr = progress_arr
        elif metric == 'ai_research_taste_sd':
            metric_arr = ai_research_taste_sd_arr
        else:
            continue

        # Check if target is reached within trajectory
        if len(metric_arr) > 0 and metric_arr[-1] >= target:
            if metric_arr[0] >= target:
                # Target already reached at start
                milestone_time = years_arr[0]
            else:
                # Interpolate to find crossing time
                try:
                    milestone_time = float(np.interp(target, metric_arr, years_arr))
                except Exception:
                    milestone_time = None

            if milestone_time is not None:
                milestone['time'] = milestone_time
                # Compute progress_multiplier at milestone time
                try:
                    progress_mult = float(np.interp(milestone_time, years_arr, ai_sw_mult_arr))
                    milestone['progress_multiplier'] = progress_mult
                except Exception:
                    pass

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
