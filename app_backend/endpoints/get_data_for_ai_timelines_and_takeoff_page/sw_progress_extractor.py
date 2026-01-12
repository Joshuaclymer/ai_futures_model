"""Software progress extraction utilities for the sw_progress endpoint."""

import math
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Developer ID in the simulator
DEVELOPER_ID = "us_frontier_lab"

# Constants for training compute normalization (from reference model model_config.py)
TRAINING_COMPUTE_REFERENCE_YEAR = 2025.13
TRAINING_COMPUTE_REFERENCE_OOMS = 26.54

# Milestone SD thresholds (from model_config.py in reference model)
TOP_RESEARCHER_SD = 3.090232306167813  # 99.9th percentile


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


def _safe_get(d: dict, key: str, default=0.0):
    """Safely get a value from a dict, returning default if None or non-finite."""
    val = d.get(key)
    if val is None:
        return default
    if not math.isfinite(val):
        return None
    return val


def _extract_time_series_point(year: float, world: dict, horizon_params: dict) -> dict | None:
    """
    Extract a single time series data point from a World snapshot.

    Args:
        year: The simulation year
        world: The World state dict
        horizon_params: Dict with present_horizon, present_doubling_time, doubling_difficulty_growth_factor

    Returns:
        dict with time series metrics, or None if no developer found
    """
    ai_devs = world.get('ai_software_developers', {})
    developer = ai_devs.get(DEVELOPER_ID)
    if developer is None and ai_devs:
        developer = next(iter(ai_devs.values()))

    if developer is None:
        return None

    prog = developer.get('ai_software_progress', {})
    progress = _safe_get(prog, 'progress')

    # Get compute inputs from developer (not ai_software_progress)
    human_labor = developer.get('human_ai_capability_researchers', 0.0)
    inference_compute = developer.get('ai_r_and_d_inference_compute_tpp_h100e', 0.0)
    experiment_compute = developer.get('ai_r_and_d_training_compute_tpp_h100e', 0.0)
    frontier_training_compute = developer.get('frontier_training_compute_tpp_h100e', 0.0)
    training_compute_growth_rate = developer.get('training_compute_growth_rate', 0.0)

    point = {
        'year': year,
        'progress': progress,
        'researchStock': _safe_get(prog, 'research_stock'),
        'automationFraction': _safe_get(prog, 'automation_fraction'),
        'aiCodingLaborMultiplier': _safe_get(prog, 'ai_coding_labor_multiplier', 1.0),
        'aiSwProgressMultRefPresentDay': _safe_get(prog, 'ai_sw_progress_mult_ref_present_day', 1.0),
        'softwareProgressRate': _safe_get(prog, 'software_progress_rate'),
        'overallProgressRate': _safe_get(prog, 'progress_rate'),
        'researchEffort': _safe_get(prog, 'research_effort'),
        'codingLabor': _safe_get(prog, 'coding_labor'),
        'serialCodingLabor': _safe_get(prog, 'serial_coding_labor'),
        'serialCodingLaborMultiplier': _safe_get(prog, 'serial_coding_labor_multiplier', 1.0),
        'humanLabor': human_labor if human_labor and math.isfinite(human_labor) else 0.0,
        'inferenceCompute': inference_compute if inference_compute and math.isfinite(inference_compute) else 0.0,
        'experimentCompute': experiment_compute if experiment_compute and math.isfinite(experiment_compute) else 0.0,
        'frontierTrainingCompute': frontier_training_compute if frontier_training_compute and math.isfinite(frontier_training_compute) else 0.0,
        'experimentCapacity': _safe_get(prog, 'experiment_capacity'),
        'aiResearchTaste': _safe_get(prog, 'ai_research_taste'),
        'aggregateResearchTaste': _safe_get(prog, 'aggregate_research_taste', 1.0),
        'trainingComputeGrowthRate': training_compute_growth_rate if training_compute_growth_rate and math.isfinite(training_compute_growth_rate) else 0.0,
        'aiResearchTasteSd': _safe_get(prog, 'ai_research_taste_sd', 0.0),
        'effectiveCompute': None,
    }

    # Compute horizon from progress if not provided
    horizon = prog.get('horizon_length')
    if horizon is not None and math.isfinite(horizon):
        point['horizonLength'] = horizon
    elif progress is not None:
        computed_horizon = compute_horizon_from_progress(
            progress,
            horizon_params['present_horizon'],
            horizon_params['present_doubling_time'],
            horizon_params['doubling_difficulty_growth_factor'],
            anchor_progress=0.0
        )
        point['horizonLength'] = computed_horizon if math.isfinite(computed_horizon) else None
    else:
        point['horizonLength'] = None

    return point


def _compute_training_and_efficiency(time_series: list) -> None:
    """
    Compute training_compute, software_efficiency, and effective_compute.
    Updates time_series points in place.

    Uses trapezoidal integration of training_compute_growth_rate and
    normalizes values at the reference year.
    """
    if len(time_series) <= 1:
        return

    initial_progress = time_series[0].get('progress', 0.0) or 0.0
    years = np.array([p['year'] for p in time_series])
    tc_growth_rates = np.array([p.get('trainingComputeGrowthRate', 0.0) or 0.0 for p in time_series])

    # Trapezoidal integration of training_compute_growth_rate
    training_compute = np.zeros(len(time_series))
    dt = np.diff(years)
    avg_growth_rates = (tc_growth_rates[:-1] + tc_growth_rates[1:]) / 2.0
    increments = avg_growth_rates * dt
    training_compute[1:] = np.cumsum(increments)

    # Normalize training_compute at reference year
    tc_at_ref = float(np.interp(TRAINING_COMPUTE_REFERENCE_YEAR, years, training_compute))
    training_compute = training_compute - tc_at_ref + TRAINING_COMPUTE_REFERENCE_OOMS

    # Compute software_efficiency (raw, before normalization)
    tc_at_start_normalized = training_compute[0]
    progress_arr = np.array([p.get('progress', 0.0) or 0.0 for p in time_series])
    software_efficiency_raw = progress_arr - initial_progress - (training_compute - tc_at_start_normalized)

    # Normalize software_efficiency at reference year
    sw_eff_at_ref = float(np.interp(TRAINING_COMPUTE_REFERENCE_YEAR, years, software_efficiency_raw))
    software_efficiency = software_efficiency_raw - sw_eff_at_ref

    # Compute effective_compute = training_compute + software_efficiency
    effective_compute = training_compute + software_efficiency

    # Normalize effective_compute at reference year
    eff_at_ref = float(np.interp(TRAINING_COMPUTE_REFERENCE_YEAR, years, effective_compute))
    effective_compute = effective_compute - eff_at_ref + TRAINING_COMPUTE_REFERENCE_OOMS

    # Update time series points
    for i, point in enumerate(time_series):
        point['trainingCompute'] = training_compute[i] if math.isfinite(training_compute[i]) else None
        point['softwareEfficiency'] = software_efficiency[i] if math.isfinite(software_efficiency[i]) else None
        point['effectiveCompute'] = effective_compute[i] if math.isfinite(effective_compute[i]) else None


def _compute_progress_at_aa(time_series: list, ac_time_horizon_minutes: float) -> float:
    """
    Compute progress_at_aa by finding where horizon reaches AC threshold.

    AC (Automated Coder) is defined as when the AI's time horizon reaches
    ac_time_horizon_minutes (typically 12,000,000 minutes).

    Args:
        time_series: List of time series points with 'progress' and 'horizonLength'
        ac_time_horizon_minutes: AC threshold in minutes

    Returns:
        Progress value at AC, or 100.0 as fallback
    """
    if not time_series:
        return 100.0

    # Find where horizon reaches the AC threshold
    # Horizon increases with progress, so we're looking for where horizon >= threshold
    for point in time_series:
        horizon = point.get('horizonLength')
        progress = point.get('progress')
        if horizon is not None and progress is not None:
            if math.isfinite(horizon) and math.isfinite(progress):
                if horizon >= ac_time_horizon_minutes:
                    return progress

    # Fallback: if horizon never reaches threshold in trajectory, use last progress value
    # or 100.0 if no valid data
    for point in reversed(time_series):
        progress = point.get('progress')
        if progress is not None and math.isfinite(progress):
            return progress

    return 100.0


def _build_milestones(sw_params: dict, time_series: list = None) -> dict:
    """Build the milestones dictionary with targets for AC, SAR, SIAR, TED-AI, ASI."""
    # Try to get progress_at_aa from params first
    progress_at_aa = sw_params.get('progress_at_aa')

    # If not set (None), compute from time series using horizon threshold
    if progress_at_aa is None and time_series:
        ac_time_horizon_minutes = sw_params.get('ac_time_horizon_minutes')
        if ac_time_horizon_minutes is None:
            ac_time_horizon_minutes = 12000000.0  # Default fallback
        progress_at_aa = _compute_progress_at_aa(time_series, ac_time_horizon_minutes)
        logger.debug(f"Computed progress_at_aa from horizon: {progress_at_aa:.2f} (threshold: {ac_time_horizon_minutes})")
    elif progress_at_aa is None:
        progress_at_aa = 100.0  # Final fallback
        logger.debug(f"Using fallback progress_at_aa: {progress_at_aa}")

    ted_ai_m2b = sw_params.get('ted_ai_m2b', 3.0)

    return {
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


def _log_interp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """Interpolate in log space for exponentially growing values."""
    # Ensure positive values for log
    fp_safe = np.maximum(fp, 1e-10)
    log_fp = np.log(fp_safe)
    log_result = np.interp(x, xp, log_fp)
    return float(np.exp(log_result))


def _compute_milestone_times(milestones: dict, time_series: list) -> None:
    """
    Compute milestone times and progress_multiplier by interpolation.
    Updates milestones dict in place.
    """
    if not time_series:
        return

    years_arr = np.array([p['year'] for p in time_series])
    progress_arr = np.array([p.get('progress', 0) or 0 for p in time_series])
    ai_research_taste_sd_arr = np.array([p.get('aiResearchTasteSd', 0) or 0 for p in time_series])
    ai_sw_mult_arr = np.array([p.get('aiSwProgressMultRefPresentDay', 1) or 1 for p in time_series])

    for milestone in milestones.values():
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
                milestone_time = years_arr[0]
            else:
                try:
                    milestone_time = float(np.interp(target, metric_arr, years_arr))
                except Exception:
                    milestone_time = None

            if milestone_time is not None:
                milestone['time'] = milestone_time
                try:
                    # Use log-space interpolation for exponentially growing values
                    milestone['progress_multiplier'] = _log_interp(milestone_time, years_arr, ai_sw_mult_arr)
                except Exception:
                    pass


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

    sw_params = params.get('software_r_and_d', {})
    horizon_params = {
        'present_horizon': sw_params.get('present_horizon', 26.0),
        'present_doubling_time': sw_params.get('present_doubling_time', 0.458),
        'doubling_difficulty_growth_factor': sw_params.get('doubling_difficulty_growth_factor', 0.92),
    }

    # Extract time series points from trajectory
    time_series = []
    for i, world in enumerate(trajectory):
        point = _extract_time_series_point(times[i], world, horizon_params)
        if point is not None:
            time_series.append(point)

    # Compute training_compute, software_efficiency, effective_compute
    _compute_training_and_efficiency(time_series)

    # Build milestones and compute their times
    # Pass time_series to compute progress_at_aa from horizon data if not in params
    milestones = _build_milestones(sw_params, time_series)
    _compute_milestone_times(milestones, time_series)

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
