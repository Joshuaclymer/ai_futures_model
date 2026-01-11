#!/usr/bin/env python3
"""
Progress Rate Functions

This module contains functions for computing progress rates in the AI progress model,
including research effort, software progress, automation fraction, and the main
progress_rate_at_time function that serves as the RHS of the ODE system.
"""

import numpy as np
from typing import List, TYPE_CHECKING
import logging

from . import model_config as cfg
from .data_types import TimeSeriesData
from .utils import should_reraise
from .ces_functions import _ces_function, compute_coding_labor_deprecated
from .taste_distribution import (
    TasteDistribution,
    compute_ai_research_taste as _compute_ai_research_taste_core,
)

if TYPE_CHECKING:
    from .automation_model import AutomationModel

logger = logging.getLogger(__name__)


def compute_research_effort(
    experiment_compute,
    serial_coding_labor,
    alpha_experiment_capacity,
    rho,
    experiment_compute_exponent,
    aggregate_research_taste=cfg.AGGREGATE_RESEARCH_TASTE_BASELINE
):
    """
    CES combination of compute and cognitive work to determine research stock growth rate.
    Handles both scalar and array inputs.

    Args:
        experiment_compute: Experiment compute budget (scalar or array)
        serial_coding_labor: Output from cognitive work (scalar or array)
        alpha_experiment_capacity: Weight on experiment compute [0,1] (scalar or array)
        rho: Standard substitution parameter in (-inf, 1] (scalar or array)
        experiment_compute_exponent: Discounting factor (scalar or array)
        aggregate_research_taste: Multiplier for research effectiveness (scalar or array, default 1.0)

    Returns:
        Research stock growth rate RS'(t) (scalar if all inputs scalar, array otherwise)
    """
    # Track if input was scalar
    input_was_scalar = (np.ndim(experiment_compute) == 0 and np.ndim(serial_coding_labor) == 0 and
                       np.ndim(alpha_experiment_capacity) == 0 and np.ndim(rho) == 0 and
                       np.ndim(experiment_compute_exponent) == 0)

    # Ensure all inputs are arrays
    experiment_compute = np.atleast_1d(experiment_compute)
    serial_coding_labor = np.atleast_1d(serial_coding_labor)
    alpha_experiment_capacity = np.atleast_1d(alpha_experiment_capacity)
    rho = np.atleast_1d(rho)
    experiment_compute_exponent = np.atleast_1d(experiment_compute_exponent)

    if aggregate_research_taste is None or (np.ndim(aggregate_research_taste) == 0 and aggregate_research_taste == cfg.AGGREGATE_RESEARCH_TASTE_BASELINE):
        aggregate_research_taste = np.ones_like(experiment_compute) * cfg.AGGREGATE_RESEARCH_TASTE_BASELINE
    else:
        aggregate_research_taste = np.atleast_1d(aggregate_research_taste)

    # Broadcast to common shape
    shape = np.broadcast_shapes(
        experiment_compute.shape, serial_coding_labor.shape,
        alpha_experiment_capacity.shape, rho.shape,
        experiment_compute_exponent.shape, aggregate_research_taste.shape
    )

    # Broadcast all arrays
    experiment_compute = np.broadcast_to(experiment_compute, shape)
    serial_coding_labor = np.broadcast_to(serial_coding_labor, shape)
    alpha_experiment_capacity = np.broadcast_to(alpha_experiment_capacity, shape)
    rho = np.broadcast_to(rho, shape)
    experiment_compute_exponent = np.broadcast_to(experiment_compute_exponent, shape)
    aggregate_research_taste = np.broadcast_to(aggregate_research_taste, shape)

    # Initialize result
    result = np.zeros(shape, dtype=np.float64)

    # Input validation
    valid = (np.isfinite(experiment_compute) &
             np.isfinite(serial_coding_labor) &
             np.isfinite(alpha_experiment_capacity) &
             np.isfinite(rho) &
             np.isfinite(experiment_compute_exponent) &
             np.isfinite(aggregate_research_taste) &
             (experiment_compute >= 0) &
             (serial_coding_labor >= 0) &
             (aggregate_research_taste >= 0))

    if not np.any(valid):
        return float(result[0]) if input_was_scalar else result

    # Clamp experiment_compute_exponent
    ece = np.clip(experiment_compute_exponent[valid],
                  cfg.experiment_compute_exponent_CLIP_MIN,
                  cfg.experiment_compute_exponent_CLIP_MAX)

    # Clamp alpha_experiment_capacity
    alpha = np.clip(alpha_experiment_capacity[valid], 0.0, 1.0)

    # Apply discounting factor to experiment compute
    discounted_experiment_compute = np.power(experiment_compute[valid], ece)

    # Use the CES function (now handles both scalar and array)
    rate = _ces_function(
        discounted_experiment_compute,
        serial_coding_labor[valid],
        alpha,
        rho[valid]
    )

    # Cap extremely large rates
    rate = np.clip(rate, 0, cfg.MAX_RESEARCH_EFFORT)

    # Apply aggregate research taste multiplier
    final_rate = rate * aggregate_research_taste[valid]

    # Apply final cap
    final_rate = np.clip(final_rate, 0, cfg.MAX_RESEARCH_EFFORT)

    result[valid] = final_rate

    # Return scalar if input was scalar, array otherwise
    return float(result[0]) if input_was_scalar else result


def compute_software_progress_rate(research_stock, research_effort, r_software):
    """
    Compute software progress rate using research stock and labor.
    Handles both scalar and array inputs.
    S(t) = RS'(t) / RS(t) * r_software

    Args:
        research_stock: Current research stock RS(t) (scalar or array)
        research_effort: Current research stock rate RS'(t) (scalar or array)
        r_software: Software progress share parameter s [0.1,10] (scalar or array)

    Returns:
        Software progress rate (scalar if all inputs scalar, array otherwise)
    """
    # Track if input was scalar
    input_was_scalar = (np.ndim(research_stock) == 0 and np.ndim(research_effort) == 0 and
                       np.ndim(r_software) == 0)

    # Ensure all inputs are arrays
    research_stock = np.atleast_1d(research_stock)
    research_effort = np.atleast_1d(research_effort)
    r_software = np.atleast_1d(r_software)

    # Broadcast to common shape
    shape = np.broadcast_shapes(research_stock.shape, research_effort.shape, r_software.shape)
    research_stock = np.broadcast_to(research_stock, shape)
    research_effort = np.broadcast_to(research_effort, shape)
    r_software = np.broadcast_to(r_software, shape)

    # Initialize result
    result = np.zeros(shape, dtype=np.float64)

    # Input validation
    valid = (np.isfinite(research_stock) &
             np.isfinite(research_effort) &
             np.isfinite(r_software) &
             (research_stock > 0) &
             (r_software >= 0.01))

    if not np.any(valid):
        return float(result[0]) if input_was_scalar else result

    # Compute software progress rate
    software_progress_rate = research_effort[valid] / research_stock[valid]

    # Validate intermediate results
    valid_spr = np.isfinite(software_progress_rate) & (software_progress_rate >= 0)

    # Apply software progress share multiplier
    final_rate = np.zeros(np.sum(valid), dtype=np.float64)
    final_rate[valid_spr] = software_progress_rate[valid_spr] * r_software[valid][valid_spr]

    # Convert to base-for-software-lom
    rate_at_base = final_rate / np.log(cfg.BASE_FOR_SOFTWARE_LOM)

    # Final validation
    valid_final = np.isfinite(rate_at_base) & (rate_at_base >= 0)

    # Store results
    indices = np.where(valid)[0]
    result[indices[valid_final]] = rate_at_base[valid_final]

    # Return scalar if input was scalar, array otherwise
    return float(result[0]) if input_was_scalar else result


def compute_overall_progress_rate(software_progress_rate: float, training_compute_growth_rate: float) -> float:
    """
    Sum of software and training progress rates

    Args:
        software_progress_rate: Rate from software development (already adjusted by software share)
        training_compute_growth_rate: Training compute budget

    Returns:
        Overall progress rate (sum of software and hardware contributions)
    """
    return software_progress_rate + training_compute_growth_rate


def compute_automation_fraction(cumulative_progress: float, automation_model: "AutomationModel") -> float:
    """
    Interpolate automation fraction based on cumulative progress.

    Uses the automation model's interpolation schedule (linear or logistic).
    Extrapolates beyond the anchors and clips the result to [0, 1].

    Args:
        cumulative_progress: Current cumulative progress.
        automation_model: AutomationModel instance for computing automation fraction.

    Returns:
        Automation fraction in [0, 1].
    """
    return automation_model.get_automation_fraction(cumulative_progress)


def compute_ai_research_taste(
    cumulative_progress: float,
    taste_distribution: TasteDistribution,
    ai_research_taste_slope: float,
    progress_at_aa: float,
    ai_research_taste_at_coding_automation_anchor_sd: float,
) -> float:
    """
    Compute AI research taste based on cumulative progress.

    Args:
        cumulative_progress: Current cumulative progress.
        taste_distribution: TasteDistribution instance.
        ai_research_taste_slope: Slope of AI research taste vs progress.
        progress_at_aa: Progress at the coding automation anchor.
        ai_research_taste_at_coding_automation_anchor_sd: SD of taste at anchor.

    Returns:
        AI research taste value.
    """
    return _compute_ai_research_taste_core(
        cumulative_progress=cumulative_progress,
        taste_distribution=taste_distribution,
        slope=ai_research_taste_slope,
        anchor_progress=progress_at_aa,
        anchor_sd=ai_research_taste_at_coding_automation_anchor_sd,
        max_sd=cfg.AI_RESEARCH_TASTE_MAX_SD,
    )


def compute_aggregate_research_taste(ai_research_taste, taste_distribution: TasteDistribution):
    """
    Compute aggregate research taste using a taste distribution with a clip-and-keep floor.
    Handles both scalar and array inputs.

    The research taste of individuals follows the taste_distribution. A hard floor F
    (given by `ai_research_taste`) lifts every draw below F up to F while leaving the
    upper tail unchanged, i.e. we consider Y = max(T, F).

    Args:
        ai_research_taste: Floor value(s) F for research taste (scalar or array)
        taste_distribution: Pre-initialized TasteDistribution object

    Returns:
        Mean taste after clipping, E[max(T, F)] (scalar if input scalar, array otherwise)
    """
    # Track if input was scalar
    input_was_scalar = np.ndim(ai_research_taste) == 0

    # Ensure ai_research_taste is an array
    ai_research_taste = np.atleast_1d(ai_research_taste)

    # Input validation
    valid = np.isfinite(ai_research_taste)
    result = np.full_like(ai_research_taste, cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK, dtype=np.float64)

    if not np.any(valid):
        return float(result[0]) if input_was_scalar else result

    # Clamp negative values to 0
    valid_values = np.where(ai_research_taste[valid] < 0, 0.0, ai_research_taste[valid])

    try:
        # Use the get_mean_with_floor method (will be updated to handle both scalar and array)
        result[valid] = taste_distribution.get_mean_with_floor(valid_values)
    except Exception as e:
        # Don't swallow timeout/interrupt exceptions - let them propagate
        if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
            raise
        logger.warning(f"Error computing aggregate research taste: {e}")
        # Fallback: return max of floor and baseline for each value
        result[valid] = np.maximum(valid_values, cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)

    # Return scalar if input was scalar, array otherwise
    return float(result[0]) if input_was_scalar else result


def progress_rate_at_time(
    t: float,
    state: List[float],
    time_series_data: TimeSeriesData,
    # CES parameters
    rho_coding_labor: float,
    parallel_penalty: float,
    coding_labor_normalization: float,
    alpha_experiment_capacity: float,
    rho_experiment_capacity: float,
    experiment_compute_exponent: float,
    r_software: float,
    # Automation model
    automation_model: "AutomationModel",
    # Taste parameters
    taste_distribution: TasteDistribution,
    ai_research_taste_slope: float,
    progress_at_aa: float,
    ai_research_taste_at_coding_automation_anchor_sd: float,
    # Mode flags
    human_only: bool = False,
) -> List[float]:
    """
    Compute instantaneous rates for both progress and research stock.
    This is the RHS of the coupled differential equation system.

    Args:
        t: Current time
        state: [cumulative_progress, research_stock]
        time_series_data: Input time series
        rho_coding_labor: CES substitution parameter for coding labor
        parallel_penalty: Parallel penalty exponent
        coding_labor_normalization: Normalization constant for coding labor
        alpha_experiment_capacity: Weight on experiment compute
        rho_experiment_capacity: CES substitution parameter for experiment capacity
        experiment_compute_exponent: Exponent on experiment compute
        r_software: Software progress rate parameter
        automation_model: AutomationModel instance
        taste_distribution: TasteDistribution instance
        ai_research_taste_slope: Slope of AI research taste
        progress_at_aa: Progress at coding automation anchor
        ai_research_taste_at_coding_automation_anchor_sd: SD of taste at anchor
        human_only: If True, compute human-only trajectory (no automation)

    Returns:
        [dP/dt, dRS/dt] - rates for both state variables
    """
    # Input validation
    if not np.isfinite(t):
        logger.warning(f"Non-finite time input: {t}")
        return [0.0, 0.0]

    if len(state) != 2:
        logger.warning(f"Invalid state vector length: {len(state)}, expected 2")
        return [0.0, 0.0]

    cumulative_progress, research_stock = state

    if not np.isfinite(cumulative_progress):
        logger.warning(f"Invalid cumulative progress: {cumulative_progress}")
        cumulative_progress = 0.0

    if not np.isfinite(research_stock) or research_stock <= 0:
        logger.warning(f"Invalid research stock: {research_stock}")

    # Validate time is within reasonable bounds
    time_min, time_max = time_series_data.time.min(), time_series_data.time.max()
    if t < time_min - cfg.TIME_EXTRAPOLATION_WINDOW or t > time_max + cfg.TIME_EXTRAPOLATION_WINDOW:
        logger.warning(f"Time {t} far outside data range [{time_min}, {time_max}]")

    try:
        # Interpolate time series data to time t
        if time_series_data.can_use_log_L_HUMAN:
            L_HUMAN = np.exp(np.interp(t, time_series_data.time, time_series_data.log_L_HUMAN))
        else:
            L_HUMAN = np.interp(t, time_series_data.time, time_series_data.L_HUMAN)

        if time_series_data.can_use_log_inference_compute:
            inference_compute = np.exp(np.interp(t, time_series_data.time, time_series_data.log_inference_compute))
        else:
            inference_compute = np.interp(t, time_series_data.time, time_series_data.inference_compute)

        if time_series_data.can_use_log_experiment_compute:
            experiment_compute = np.exp(np.interp(t, time_series_data.time, time_series_data.log_experiment_compute))
        else:
            experiment_compute = np.interp(t, time_series_data.time, time_series_data.experiment_compute)

        training_compute_growth_rate = time_series_data.get_training_compute_growth_rate(t)

        # Validate interpolated values
        if not all(np.isfinite([L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate])):
            logger.warning(f"Non-finite interpolated values at t={t}")
            return [0.0, 0.0]

        # Ensure non-negative values
        L_HUMAN = max(0.0, L_HUMAN)
        inference_compute = max(0.0, inference_compute)
        experiment_compute = max(0.0, experiment_compute)
        training_compute_growth_rate = max(0.0, training_compute_growth_rate)

        if human_only:
            aggregate_research_taste = 1.0
            coding_labor = compute_coding_labor_deprecated(
                None, inference_compute, L_HUMAN,
                rho_coding_labor, parallel_penalty, coding_labor_normalization,
                human_only=True
            )
        else:
            # Compute automation fraction from cumulative progress
            automation_fraction = compute_automation_fraction(cumulative_progress, automation_model)
            if not (0 <= automation_fraction <= 1):
                logger.warning(f"Invalid automation fraction {automation_fraction} at progress {cumulative_progress}")
                automation_fraction = np.clip(automation_fraction, 0.0, 1.0)

            # Compute AI research taste and aggregate research taste
            ai_research_taste = compute_ai_research_taste(
                cumulative_progress,
                taste_distribution,
                ai_research_taste_slope,
                progress_at_aa,
                ai_research_taste_at_coding_automation_anchor_sd,
            )
            aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste, taste_distribution)

            # Compute coding labor
            coding_labor = compute_coding_labor_deprecated(
                automation_fraction, inference_compute, L_HUMAN,
                rho_coding_labor, parallel_penalty, coding_labor_normalization
            )

        if not np.isfinite(coding_labor) or coding_labor < 0:
            logger.warning(f"Invalid cognitive output: {coding_labor}")
            return [0.0, 0.0]

        # Compute research effort
        research_effort = compute_research_effort(
            experiment_compute, coding_labor,
            alpha_experiment_capacity, rho_experiment_capacity, experiment_compute_exponent,
            aggregate_research_taste
        )

        if not np.isfinite(research_effort) or research_effort < 0:
            logger.warning(f"Invalid research stock rate: {research_effort}")
            return [0.0, 0.0]

        # Compute software progress rate
        software_progress_rate = compute_software_progress_rate(
            research_stock, research_effort, r_software
        )

        if not np.isfinite(software_progress_rate) or software_progress_rate < 0:
            logger.warning(f"Invalid software progress rate: {software_progress_rate}")
            return [0.0, 0.0]

        # Compute overall progress rate (dP/dt)
        overall_rate = compute_overall_progress_rate(
            software_progress_rate, training_compute_growth_rate
        )

        # Final validation
        if not np.isfinite(overall_rate) or overall_rate < 0:
            logger.warning(f"Invalid overall progress rate: {overall_rate}")
            return [0.0, 0.0]

        # Cap extremely large rates to prevent numerical issues
        if overall_rate > cfg.MAX_NORMALIZED_PROGRESS_RATE:
            logger.warning(f"Very large progress rate {overall_rate}, capping to {cfg.MAX_NORMALIZED_PROGRESS_RATE}")
            overall_rate = cfg.MAX_NORMALIZED_PROGRESS_RATE

        logger.debug(f"t={t:.2f}, progress={cumulative_progress:.3f}, research_stock={research_stock:.3f}, "
                    f"dP/dt={overall_rate:.3f}, dRS/dt={research_effort:.3f}")

        return [overall_rate, research_effort]

    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
            raise
        logger.error(f"Error computing rates at t={t}, state={state}: {e}")
        return [0.0, 0.0]
