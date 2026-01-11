#!/usr/bin/env python3
"""
Metrics Computation Module

Handles the computation of comprehensive metrics for each time point in the trajectory.
This module was extracted from _impl.py to improve modularity and enable future vectorization.
"""

import numpy as np
import time
import logging
import model_config as cfg
from typing import Dict, List, Tuple, Any, Optional

from .utils import _log_interp, should_reraise
from .types import TimeSeriesData
from .parameters import Parameters
from .progress_rate import (
    compute_research_effort,
    compute_software_progress_rate,
    compute_overall_progress_rate,
    compute_automation_fraction,
    compute_ai_research_taste,
    compute_aggregate_research_taste,
    progress_rate_at_time,
)
from .ces_functions import compute_coding_labor_deprecated
from .automation_model import AutomationModel

logger = logging.getLogger(__name__)


def compute_mrts_exp_compute_human_labor(
    experiment_compute: np.ndarray,
    human_labor: np.ndarray,
    alpha: float,
    rho: float,
    gamma: float,
    parallel_penalty: float,
) -> np.ndarray:
    """
    Compute the MRTS (Marginal Rate of Technical Substitution) between
    experiment compute and human labor (human-only, no AI contribution).

    For the CES experiment capacity function:
        ExpCap = [α · (C^γ)^ρ + (1-α) · L^ρ]^(1/ρ)

    where L = H^pp (human-only serial coding labor), the MRTS tells us how many
    units of experiment compute one additional human is worth at the margin
    (i.e., keeping research output constant).

    MRTS_{H,C} = MP_H / MP_C = [(1-α) · pp / (α · γ)] · H^(pp·ρ-1) / C^(γρ-1)

    Args:
        experiment_compute: Experiment compute budget C (array)
        human_labor: Human labor H, i.e. L_HUMAN (array)
        alpha: alpha_experiment_capacity - weight on compute in CES
        rho: rho_experiment_capacity - substitution parameter
        gamma: experiment_compute_exponent - discounting exponent on compute
        parallel_penalty: pp - exponent converting human count to serial labor

    Returns:
        MRTS in units of "experiment compute per human" (array).
        Positive values indicate how much experiment compute one human provides.
    """
    pp = parallel_penalty

    # Handle edge cases to avoid division by zero
    if abs(alpha) < 1e-18 or abs(gamma) < 1e-18:
        return np.zeros_like(experiment_compute)

    # Coefficient: (1-alpha) * pp / (alpha * gamma)
    coeff = ((1.0 - alpha) * pp) / (alpha * gamma)

    # Exponents for the ratio
    h_exp = pp * rho - 1.0
    c_exp = gamma * rho - 1.0

    # MRTS = coeff * H^h_exp / C^c_exp
    # Handle potential numerical issues
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        h_term = np.power(human_labor, h_exp)
        c_term = np.power(experiment_compute, c_exp)

        # Avoid division by zero
        mrts = np.where(
            (c_term > 0) & np.isfinite(c_term) & np.isfinite(h_term),
            coeff * h_term / c_term,
            0.0
        )

    # Replace non-finite values with 0
    mrts = np.where(np.isfinite(mrts), mrts, 0.0)

    return mrts


def compute_metrics_loop(
    times: np.ndarray,
    progress_values: np.ndarray,
    research_stock_values: np.ndarray,
    data: TimeSeriesData,
    params: Parameters,
    initial_progress: float,
    present_day_human_labor: float,
    present_day_inference_compute: float,
    present_day_experiment_compute: float,
    present_day_research_stock: float,
    present_day_sw_progress_rate: float,
    taste_distribution: Any,
    horizon_trajectory: Optional[Any],
    takeoff_start_human_only_stats: Optional[Dict[str, float]],
    get_capability: callable,
) -> Tuple[Dict[str, List[float]], Dict[str, float], Dict[str, float]]:
    """
    Compute comprehensive metrics for each time point in the trajectory using vectorized operations.

    Args:
        times: Array of time points
        progress_values: Array of progress values at each time point
        research_stock_values: Array of research stock values at each time point
        data: Time series input data
        params: Model parameters
        initial_progress: Initial progress value
        present_day_human_labor: Human labor at present day
        present_day_inference_compute: Inference compute at present day
        present_day_experiment_compute: Experiment compute at present day
        present_day_research_stock: Research stock at present day
        present_day_sw_progress_rate: Software progress rate at present day
        taste_distribution: Taste distribution object
        horizon_trajectory: Optional horizon trajectory function
        takeoff_start_human_only_stats: Optional stats for takeoff start
        get_capability: Function to compute capability from training compute and software efficiency

    Returns:
        Tuple of (metrics_dict, profile_times_dict, timing_meta_dict)
    """
    _t_metrics_loop_start = time.perf_counter()
    _c_metrics_loop_start = time.process_time()

    n_points = len(times)

    # Pre-compute takeoff start resources if needed
    if takeoff_start_human_only_stats is not None:
        takeoff_start_research_stock = takeoff_start_human_only_stats['research_stock']
    else:
        takeoff_start_research_stock = None

    # ========== STEP 1: Pre-interpolate all time series data ==========
    _t0 = time.perf_counter()
    # Use precomputed logs from TimeSeriesData to avoid redundant log() calls
    if data.can_use_log_L_HUMAN:
        L_HUMAN = np.exp(np.interp(times, data.time, data.log_L_HUMAN))
    else:
        L_HUMAN = np.interp(times, data.time, data.L_HUMAN)

    if data.can_use_log_inference_compute:
        inference_compute = np.exp(np.interp(times, data.time, data.log_inference_compute))
    else:
        inference_compute = np.interp(times, data.time, data.inference_compute)

    if data.can_use_log_experiment_compute:
        experiment_compute = np.exp(np.interp(times, data.time, data.log_experiment_compute))
    else:
        experiment_compute = np.interp(times, data.time, data.experiment_compute)

    training_compute_growth_rate = data.get_training_compute_growth_rate(times)
    _dt_interp = time.perf_counter() - _t0

    # ========== STEP 2: Compute automation fraction ==========
    _t0 = time.perf_counter()
    automation_fraction = params.automation_model.get_automation_fraction(progress_values)
    _dt_automation = time.perf_counter() - _t0

    # ========== STEP 3: Compute research taste ==========
    _t0 = time.perf_counter()
    _t_taste_start = _t0

    # Compute AI research taste
    ai_research_taste = compute_ai_research_taste(progress_values, params)
    _dt_taste_compute_ai = time.perf_counter() - _t0

    # Get SD and quantile for each taste value (vectorized)
    _t0 = time.perf_counter()
    # get_sd_of_taste and get_quantile_of_taste support array inputs
    ai_research_taste_sd = taste_distribution.get_sd_of_taste(ai_research_taste)
    # Replace non-finite values with 0.0
    ai_research_taste_sd = np.where(np.isfinite(ai_research_taste_sd), ai_research_taste_sd, 0.0)
    _dt_taste_get_sd = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    ai_research_taste_quantile = taste_distribution.get_quantile_of_taste(ai_research_taste)
    # Replace non-finite values with 0.0
    ai_research_taste_quantile = np.where(np.isfinite(ai_research_taste_quantile), ai_research_taste_quantile, 0.0)
    _dt_taste_get_quantile = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste, params.taste_distribution)
    _dt_taste_aggregate = time.perf_counter() - _t0

    _dt_taste = time.perf_counter() - _t_taste_start

    # ========== STEP 4: Compute coding labor ==========
    _t0 = time.perf_counter()

    if getattr(params, 'coding_labor_mode', 'simple_ces') == 'optimal_ces':
        H = L_HUMAN.astype(np.float64)
        C = inference_compute.astype(np.float64)
        logE = (np.log(cfg.BASE_FOR_SOFTWARE_LOM) * progress_values).astype(np.float64)

        try:
            automation_model = params.automation_model
            L_opt = automation_model.coding_labor_optimal_ces(H, C, logE, params)
            L_opt_present_resources = automation_model.coding_labor_optimal_ces(
                present_day_human_labor, present_day_inference_compute, logE, params
            )

            if takeoff_start_human_only_stats is not None:
                L_opt_takeoff_start = automation_model.coding_labor_optimal_ces(
                    takeoff_start_human_only_stats['human_labor'],
                    takeoff_start_human_only_stats['inference_compute'],
                    logE,
                    params
                )

            if L_opt is None or not np.all(np.isfinite(L_opt)):
                logger.error(f"L_opt contains invalid values")
                raise ValueError("L_opt is None or contains non-finite values")

            coding_labor = L_opt
            coding_labor_with_present_resources = L_opt_present_resources
            serial_coding_labor = (L_opt ** params.parallel_penalty) * params.coding_labor_normalization
            serial_coding_labor_with_present_resources = (L_opt_present_resources ** params.parallel_penalty) * params.coding_labor_normalization

            if takeoff_start_human_only_stats is not None:
                serial_coding_labor_takeoff_start = (L_opt_takeoff_start ** params.parallel_penalty) * params.coding_labor_normalization

        except Exception as e:
            if should_reraise(e):
                raise
            raise AssertionError(f"Falling back to simple CES in vectorized metrics due to optimal_ces error: {e}")
    else:
        # Simple CES mode
        serial_coding_labor = compute_coding_labor_deprecated(
            automation_fraction, inference_compute, L_HUMAN,
            params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
        )
        # For deprecated mode, extract coding_labor (before parallel penalty)
        if params.parallel_penalty != 0:
            coding_labor = (serial_coding_labor / params.coding_labor_normalization) ** (1.0 / params.parallel_penalty)
        else:
            coding_labor = serial_coding_labor / params.coding_labor_normalization

        # Compute with present resources
        serial_coding_labor_with_present_resources = compute_coding_labor_deprecated(
            automation_fraction, np.full_like(inference_compute, present_day_inference_compute),
            np.full_like(L_HUMAN, present_day_human_labor),
            params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
        )
        if params.parallel_penalty != 0:
            coding_labor_with_present_resources = (serial_coding_labor_with_present_resources / params.coding_labor_normalization) ** (1.0 / params.parallel_penalty)
        else:
            coding_labor_with_present_resources = serial_coding_labor_with_present_resources / params.coding_labor_normalization

        if takeoff_start_human_only_stats is not None:
            serial_coding_labor_takeoff_start = compute_coding_labor_deprecated(
                automation_fraction,
                np.full_like(inference_compute, takeoff_start_human_only_stats['inference_compute']),
                np.full_like(L_HUMAN, takeoff_start_human_only_stats['human_labor']),
                params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
            )

    # Ensure finite values
    coding_labor = np.where(np.isfinite(coding_labor), coding_labor, 0.0)
    coding_labor_with_present_resources = np.where(np.isfinite(coding_labor_with_present_resources), coding_labor_with_present_resources, 0.0)
    serial_coding_labor = np.where(np.isfinite(serial_coding_labor), serial_coding_labor, 0.0)
    _dt_coding_labor = time.perf_counter() - _t0

    # ========== STEP 5: Compute research effort and discounted experiment compute ==========
    _t0 = time.perf_counter()
    discounted_exp_compute = np.power(experiment_compute, params.experiment_compute_exponent)
    discounted_exp_compute = np.where(np.isfinite(discounted_exp_compute), discounted_exp_compute, 0.0)

    research_effort = compute_research_effort(
        experiment_compute, serial_coding_labor,
        params.alpha_experiment_capacity, params.rho_experiment_capacity,
        params.experiment_compute_exponent, aggregate_research_taste
    )

    research_effort_present_resources = compute_research_effort(
        np.full(n_points, present_day_experiment_compute),
        serial_coding_labor_with_present_resources,
        params.alpha_experiment_capacity, params.rho_experiment_capacity,
        params.experiment_compute_exponent, aggregate_research_taste
    )

    if takeoff_start_human_only_stats is not None:
        research_effort_takeoff_start_resources = compute_research_effort(
            np.full(n_points, takeoff_start_human_only_stats['experiment_compute']),
            serial_coding_labor_takeoff_start,
            params.alpha_experiment_capacity, params.rho_experiment_capacity,
            params.experiment_compute_exponent, aggregate_research_taste
        )
    _dt_research_effort = time.perf_counter() - _t0

    # ========== STEP 6: Compute experiment capacity ==========
    _t0 = time.perf_counter()
    exp_capacity = np.where(
        aggregate_research_taste > 0,
        research_effort / aggregate_research_taste,
        0.0
    )
    exp_capacity = np.where(np.isfinite(exp_capacity), exp_capacity, 0.0)

    if params.rho_experiment_capacity < -1e-4:
        exp_cap_with_infinite_compute = (
            (1 - params.alpha_experiment_capacity) ** (1 / params.rho_experiment_capacity) * serial_coding_labor
        )
        exp_cap_with_infinite_labor = (
            params.alpha_experiment_capacity ** (1 / params.rho_experiment_capacity) * discounted_exp_compute
        )
        exp_cap_mult_with_infinite_labor = np.where(
            exp_capacity > 0,
            exp_cap_with_infinite_labor / exp_capacity,
            0.0
        )
        exp_cap_mult_with_infinite_compute = np.where(
            exp_capacity > 0,
            exp_cap_with_infinite_compute / exp_capacity,
            0.0
        )
    else:
        exp_cap_mult_with_infinite_labor = np.zeros(n_points)
        exp_cap_mult_with_infinite_compute = np.zeros(n_points)
    _dt_exp_capacity = time.perf_counter() - _t0

    # ========== STEP 7: Compute software progress rate ==========
    _t0 = time.perf_counter()
    software_progress_rate = compute_software_progress_rate(
        research_stock_values, research_effort, params.r_software
    )

    # Handle plan_a_mode overrides
    if params.plan_a_mode:
        plan_a_mask = times >= params.plan_a_start_time
        software_progress_rate = np.where(
            plan_a_mask,
            params.main_project_software_progress_rate,
            software_progress_rate
        )

    software_progress_rate_present_resources = compute_software_progress_rate(
        np.full(n_points, present_day_research_stock),
        research_effort_present_resources,
        params.r_software
    )

    if takeoff_start_human_only_stats is not None:
        software_rate_takeoff_start = compute_software_progress_rate(
            np.full(n_points, takeoff_start_research_stock),
            research_effort_takeoff_start_resources,
            params.r_software
        )

    software_progress_rate = np.where(np.isfinite(software_progress_rate), software_progress_rate, 0.0)
    software_progress_rate_present_resources = np.where(np.isfinite(software_progress_rate_present_resources), software_progress_rate_present_resources, 0.0)
    _dt_software_rate = time.perf_counter() - _t0

    # ========== STEP 8: Compute training compute via direct interpolation ==========
    _t0 = time.perf_counter()
    # Interpolate absolute training compute values from input data
    training_compute_abs = np.interp(times, data.time, data.training_compute)
    # Make relative (starts at 0) to match previous behavior
    training_compute = training_compute_abs - training_compute_abs[0]
    training_compute = np.where(np.isfinite(training_compute), training_compute, 0.0)

    # Software efficiency
    software_efficiency = progress_values - initial_progress - training_compute
    software_efficiency = np.where(np.isfinite(software_efficiency), software_efficiency, 0.0)

    # Effective compute
    effective_compute = training_compute + software_efficiency
    _dt_training_compute = time.perf_counter() - _t0

    # ========== STEP 9: Compute capability (ECI) - vectorized ==========
    _t0 = time.perf_counter()
    # get_capability uses simple arithmetic that works with numpy arrays:
    # effective_compute = training_compute + software_efficiency
    # capability = CAPABILITY_POINTS_PER_OOM * (effective_compute - TRAINING_COMPUTE_REFERENCE_OOMS) + CAPABILITY_REFERENCE_SCORE
    capability_eci = get_capability(training_compute, software_efficiency)
    capability_eci = np.where(np.isfinite(capability_eci), capability_eci, 0.0)
    _dt_capability = time.perf_counter() - _t0

    # ========== STEP 10: Compute horizon lengths - vectorized ==========
    _t0 = time.perf_counter()
    if horizon_trajectory is not None:
        # horizon_trajectory already supports array inputs via np.atleast_1d()
        try:
            horizon_lengths = horizon_trajectory(progress_values)
            # Handle case where horizon_trajectory returns scalar for array input
            horizon_lengths = np.atleast_1d(horizon_lengths)
            # Ensure correct shape
            if horizon_lengths.shape != progress_values.shape:
                horizon_lengths = np.broadcast_to(horizon_lengths, progress_values.shape).copy()
            # Filter non-finite and negative values
            horizon_lengths = np.where(
                np.isfinite(horizon_lengths) & (horizon_lengths >= 0),
                horizon_lengths,
                0.0
            )
        except Exception as e:
            if should_reraise(e):
                raise
            horizon_lengths = np.zeros(n_points)
    else:
        horizon_lengths = np.zeros(n_points)
    _dt_horizon = time.perf_counter() - _t0

    # ========== STEP 11: Compute human-only metrics ==========
    _t0 = time.perf_counter()
    human_only_serial_coding_labor = L_HUMAN ** params.parallel_penalty
    human_only_aggregate_research_taste = compute_aggregate_research_taste(
        np.zeros(n_points),  # No AI research taste
        params.taste_distribution
    )
    human_only_research_effort = compute_research_effort(
        experiment_compute, human_only_serial_coding_labor,
        params.alpha_experiment_capacity, params.rho_experiment_capacity,
        params.experiment_compute_exponent, human_only_aggregate_research_taste
    )
    human_only_research_effort = np.where(np.isfinite(human_only_research_effort), human_only_research_effort, 0.0)

    human_only_software_progress_rate = compute_software_progress_rate(
        research_stock_values, human_only_research_effort, params.r_software
    )
    human_only_software_progress_rate = np.where(np.isfinite(human_only_software_progress_rate), human_only_software_progress_rate, 0.0)

    human_only_progress_rate = human_only_software_progress_rate + training_compute_growth_rate
    human_only_progress_rate = np.where(np.isfinite(human_only_progress_rate), human_only_progress_rate, 0.0)

    # Compute MRTS between experiment compute and human labor (human-only case)
    mrts_exp_compute_human_labor = compute_mrts_exp_compute_human_labor(
        experiment_compute,
        L_HUMAN,
        params.alpha_experiment_capacity,
        params.rho_experiment_capacity,
        params.experiment_compute_exponent,
        params.parallel_penalty,
    )
    _dt_human_only = time.perf_counter() - _t0

    # ========== STEP 12: Compute AI-only coding labor ==========
    _t0 = time.perf_counter()
    # logE = log(BASE_FOR_SOFTWARE_LOM) * progress
    logE = np.log(cfg.BASE_FOR_SOFTWARE_LOM) * progress_values
    ai_only_coding_labor = params.automation_model.compute_ai_only_coding_labor(
        inference_compute, logE, params, automation_fraction
    )
    ai_only_coding_labor = np.where(np.isfinite(ai_only_coding_labor), ai_only_coding_labor, 0.0)
    _dt_ai_only_coding_labor = time.perf_counter() - _t0

    # ========== STEP 13: Compute labor contributions and multipliers ==========
    _t0 = time.perf_counter()
    human_labor_contribution = L_HUMAN
    ai_labor_contribution = np.maximum(0.0, coding_labor - human_labor_contribution)

    if params.parallel_penalty and params.parallel_penalty != 0:
        ai_coding_labor_multiplier = np.where(
            ai_labor_contribution > 0,
            coding_labor / human_labor_contribution,
            1.0
        )
        ai_coding_labor_mult_ref_present_day = np.where(
            ai_labor_contribution > 0,
            coding_labor_with_present_resources / present_day_human_labor,
            1.0
        )
        # Serial coding labor multiplier: ratio of serial coding labor to human-only serial coding labor
        serial_coding_labor_multiplier = np.where(
            human_only_serial_coding_labor > 0,
            serial_coding_labor / human_only_serial_coding_labor,
            1.0
        )
    else:
        ai_coding_labor_multiplier = np.zeros(n_points)
        ai_coding_labor_mult_ref_present_day = np.zeros(n_points)
        serial_coding_labor_multiplier = np.ones(n_points)

    ai_sw_progress_mult_ref_present_day = np.where(
        present_day_sw_progress_rate > 0,
        software_progress_rate_present_resources / present_day_sw_progress_rate,
        0.0
    )

    if takeoff_start_human_only_stats is not None:
        takeoff_progress_multiplier = np.where(
            takeoff_start_human_only_stats['research_effort'] > 0,
            research_effort_takeoff_start_resources / takeoff_start_human_only_stats['research_effort'],
            0.0
        )
    else:
        takeoff_progress_multiplier = np.zeros(n_points)
    _dt_labor_contrib = time.perf_counter() - _t0

    # ========== STEP 14: Compute progress_rates (overall rates) ==========
    _t0 = time.perf_counter()
    progress_rates = software_progress_rate + training_compute_growth_rate
    progress_rates = np.where(np.isfinite(progress_rates), progress_rates, 0.0)
    _dt_progress_rate = time.perf_counter() - _t0

    # ========== STEP 15: Compute optimal training run length ==========
    # Optimal training run length = 1 / (experiment compute growth rate + software efficiency growth rate)
    # where experiment compute growth rate = d/dt(ln(experiment_compute))
    # Note: software_progress_rate is in OOMs/year, convert to e-foldings/year by multiplying by ln(10)
    _t0 = time.perf_counter()
    # Compute experiment compute growth rate as d/dt(ln(experiment_compute))
    log_experiment_compute = np.log(experiment_compute)
    experiment_compute_growth_rate = np.gradient(log_experiment_compute, times)
    experiment_compute_growth_rate = np.where(np.isfinite(experiment_compute_growth_rate), experiment_compute_growth_rate, 0.0)

    # Convert software progress rate from OOMs/year to e-foldings/year
    software_progress_rate_efoldings = software_progress_rate * np.log(10)
    total_growth_rate = experiment_compute_growth_rate + software_progress_rate_efoldings
    optimal_training_run_length = np.where(
        total_growth_rate > 0,
        1.0 / total_growth_rate,
        np.inf
    )
    optimal_training_run_length = np.where(np.isfinite(optimal_training_run_length), optimal_training_run_length, 0.0)
    _dt_optimal_training_run_length = time.perf_counter() - _t0

    # ========== Assemble results ==========
    _dt_metrics_loop = time.perf_counter() - _t_metrics_loop_start
    _dt_metrics_loop_cpu = time.process_time() - _c_metrics_loop_start
    _avg_iter_ms = (1000.0 * _dt_metrics_loop / n_points) if n_points > 0 else float('nan')

    logger.info(f"Timing: vectorized metrics processed {n_points} points in {_dt_metrics_loop:.3f}s (avg {_avg_iter_ms:.3f} ms/iter)")
    logger.info(f"=== Vectorized Metrics Profiling Breakdown ===")
    logger.info(f"  {'input_interp':20s}: {_dt_interp:7.3f}s ({100.0 * _dt_interp / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_interp / n_points:6.3f} ms/iter")
    logger.info(f"  {'automation':20s}: {_dt_automation:7.3f}s ({100.0 * _dt_automation / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_automation / n_points:6.3f} ms/iter")
    logger.info(f"  {'taste':20s}: {_dt_taste:7.3f}s ({100.0 * _dt_taste / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_taste / n_points:6.3f} ms/iter")
    logger.info(f"    ↳ {'taste_compute_ai':18s}: {_dt_taste_compute_ai:7.3f}s ({100.0 * _dt_taste_compute_ai / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_taste_compute_ai / n_points:6.3f} ms/iter")
    logger.info(f"    ↳ {'taste_get_sd':18s}: {_dt_taste_get_sd:7.3f}s ({100.0 * _dt_taste_get_sd / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_taste_get_sd / n_points:6.3f} ms/iter")
    logger.info(f"    ↳ {'taste_get_quantile':18s}: {_dt_taste_get_quantile:7.3f}s ({100.0 * _dt_taste_get_quantile / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_taste_get_quantile / n_points:6.3f} ms/iter")
    logger.info(f"    ↳ {'taste_aggregate':18s}: {_dt_taste_aggregate:7.3f}s ({100.0 * _dt_taste_aggregate / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_taste_aggregate / n_points:6.3f} ms/iter")
    logger.info(f"  {'coding_labor':20s}: {_dt_coding_labor:7.3f}s ({100.0 * _dt_coding_labor / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_coding_labor / n_points:6.3f} ms/iter")
    logger.info(f"  {'research_effort':20s}: {_dt_research_effort:7.3f}s ({100.0 * _dt_research_effort / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_research_effort / n_points:6.3f} ms/iter")
    logger.info(f"  {'exp_capacity':20s}: {_dt_exp_capacity:7.3f}s ({100.0 * _dt_exp_capacity / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_exp_capacity / n_points:6.3f} ms/iter")
    logger.info(f"  {'software_rate':20s}: {_dt_software_rate:7.3f}s ({100.0 * _dt_software_rate / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_software_rate / n_points:6.3f} ms/iter")
    logger.info(f"  {'training_compute':20s}: {_dt_training_compute:7.3f}s ({100.0 * _dt_training_compute / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_training_compute / n_points:6.3f} ms/iter")
    logger.info(f"  {'capability':20s}: {_dt_capability:7.3f}s ({100.0 * _dt_capability / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_capability / n_points:6.3f} ms/iter")
    logger.info(f"  {'horizon':20s}: {_dt_horizon:7.3f}s ({100.0 * _dt_horizon / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_horizon / n_points:6.3f} ms/iter")
    logger.info(f"  {'human_only':20s}: {_dt_human_only:7.3f}s ({100.0 * _dt_human_only / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_human_only / n_points:6.3f} ms/iter")
    logger.info(f"  {'ai_only_coding_labor':20s}: {_dt_ai_only_coding_labor:7.3f}s ({100.0 * _dt_ai_only_coding_labor / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_ai_only_coding_labor / n_points:6.3f} ms/iter")
    logger.info(f"  {'labor_contrib':20s}: {_dt_labor_contrib:7.3f}s ({100.0 * _dt_labor_contrib / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_labor_contrib / n_points:6.3f} ms/iter")
    logger.info(f"  {'progress_rate':20s}: {_dt_progress_rate:7.3f}s ({100.0 * _dt_progress_rate / _dt_metrics_loop:5.1f}%) avg {1000.0 * _dt_progress_rate / n_points:6.3f} ms/iter")

    total_profiled = (_dt_interp + _dt_automation + _dt_taste + _dt_coding_labor +
                     _dt_research_effort + _dt_exp_capacity + _dt_software_rate +
                     _dt_training_compute + _dt_capability + _dt_horizon +
                     _dt_human_only + _dt_ai_only_coding_labor + _dt_labor_contrib + _dt_progress_rate)
    _unaccounted = _dt_metrics_loop - total_profiled
    logger.info(f"  {'unaccounted':20s}: {_unaccounted:7.3f}s ({100.0 * _unaccounted / _dt_metrics_loop:5.1f}%)")
    logger.info("=" * 45)

    metrics = {
        'progress_rates': progress_rates.tolist(),
        'research_efforts': research_effort.tolist(),
        'automation_fractions': automation_fraction.tolist(),
        'ai_research_tastes': ai_research_taste.tolist(),
        'ai_research_taste_sds': ai_research_taste_sd.tolist(),
        'ai_research_taste_quantiles': ai_research_taste_quantile.tolist(),
        'aggregate_research_tastes': aggregate_research_taste.tolist(),
        'coding_labors': coding_labor.tolist(),
        'coding_labors_with_present_resources': coding_labor_with_present_resources.tolist(),
        'serial_coding_labors': serial_coding_labor.tolist(),
        'software_progress_rates': software_progress_rate.tolist(),
        'software_progress_rates_present_resources': software_progress_rate_present_resources.tolist(),
        'software_efficiency': software_efficiency.tolist(),
        'human_only_research_efforts': human_only_research_effort.tolist(),
        'human_only_software_progress_rates': human_only_software_progress_rate.tolist(),
        'human_only_progress_rates': human_only_progress_rate.tolist(),
        'mrts_exp_compute_human_labor': mrts_exp_compute_human_labor.tolist(),
        'ai_labor_contributions': ai_labor_contribution.tolist(),
        'human_labor_contributions': human_labor_contribution.tolist(),
        'ai_coding_labor_multipliers': ai_coding_labor_multiplier.tolist(),
        'ai_coding_labor_mult_ref_present_day': ai_coding_labor_mult_ref_present_day.tolist(),
        'serial_coding_labor_multipliers': serial_coding_labor_multiplier.tolist(),
        'ai_sw_progress_mult_ref_present_day': ai_sw_progress_mult_ref_present_day.tolist(),
        'takeoff_progress_multipliers': takeoff_progress_multiplier.tolist(),
        'discounted_exp_compute': discounted_exp_compute.tolist(),
        'horizon_lengths': horizon_lengths.tolist(),
        'effective_compute': effective_compute.tolist(),
        'training_compute': training_compute.tolist(),
        'training_compute_growth_rate': training_compute_growth_rate.tolist(),
        'capability_ecis': capability_eci.tolist(),
        'experiment_capacity': exp_capacity.tolist(),
        'exp_cap_mult_with_infinite_labor': exp_cap_mult_with_infinite_labor.tolist(),
        'exp_cap_mult_with_infinite_compute': exp_cap_mult_with_infinite_compute.tolist(),
        'ai_only_coding_labors': ai_only_coding_labor.tolist(),
        'optimal_training_run_length': optimal_training_run_length.tolist(),
    }

    _profile_times = {
        'input_interp': float(_dt_interp),
        'automation': float(_dt_automation),
        'taste': float(_dt_taste),
        'taste_compute_ai': float(_dt_taste_compute_ai),
        'taste_get_sd': float(_dt_taste_get_sd),
        'taste_get_quantile': float(_dt_taste_get_quantile),
        'taste_aggregate': float(_dt_taste_aggregate),
        'coding_labor': float(_dt_coding_labor),
        'research_effort': float(_dt_research_effort),
        'exp_capacity': float(_dt_exp_capacity),
        'software_rate': float(_dt_software_rate),
        'training_compute': float(_dt_training_compute),
        'capability': float(_dt_capability),
        'horizon': float(_dt_horizon),
        'human_only': float(_dt_human_only),
        'ai_only_coding_labor': float(_dt_ai_only_coding_labor),
        'labor_contrib': float(_dt_labor_contrib),
        'progress_rate': float(_dt_progress_rate),
    }

    timing_meta = {
        'metrics_loop_num_points': int(n_points),
        'metrics_loop_avg_iter_ms': float(_avg_iter_ms) if np.isfinite(_avg_iter_ms) else None,
        'metrics_loop_profile_wall_seconds': _profile_times,
        'metrics_loop_profile_wall_seconds_total': float(total_profiled),
        'metrics_loop_profile_wall_seconds_unaccounted': float(_unaccounted),
    }

    timing_wall_cpu = {
        'wall': _dt_metrics_loop,
        'cpu': _dt_metrics_loop_cpu,
    }

    return metrics, timing_wall_cpu, timing_meta
