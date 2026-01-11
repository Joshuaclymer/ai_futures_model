#!/usr/bin/env python3
"""
Progress Modeling Script

Models AI progress over time using nested CES production functions with
feedback loops between automation fraction and cumulative progress.

This is the implementation module. For external use, import from the
progress_model package instead of this module directly.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union, NamedTuple
from scipy import optimize, integrate, interpolate
from collections import OrderedDict
import logging
import time
import model_config as cfg
import yaml
import copy
from datetime import datetime

# Import utilities from the new modular structure
from .utils import _coerce_float_scalar, _gauss_hermite_expectation, should_reraise
from .types import TimeSeriesData, AnchorConstraint, InitialConditions
from .ces_functions import (
    _ces_function,
    compute_coding_labor_deprecated,
    compute_rho_from_asymptotes,
    compute_experiment_compute_exponent_from_anchor,
    compute_alpha_experiment_capacity_from_asymptotes,
    compute_exp_capacity_params_from_anchors,
)
from .taste_distribution import (
    TasteDistribution,
    get_or_create_taste_distribution,
    _taste_distribution_cache,
    _TASTE_CACHE_MAX_SIZE,
    compute_ai_research_taste as _compute_ai_research_taste_core,
)
from .automation_model import AutomationModel, aut_frac_from_swe_multiplier, solve_lower_anchor_via_automation_model
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
from .utils import _log_interp
from .integration import (
    _find_exponential_crossing_time,
    calculate_initial_research_stock,
    compute_initial_conditions,
    setup_model,
    integrate_progress,
    integrate_progress_human_only,
)
from .metrics_computation import compute_metrics_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Global cache for benchmark_results.yaml to avoid repeated file I/O
_BENCHMARK_DATA_CACHE = None


# TasteDistribution and get_or_create_taste_distribution moved to taste_distribution.py


def _load_benchmark_data():
    """
    Load benchmark_results.yaml with module-level caching.

    Returns:
        Dict containing benchmark data, or empty dict if file not found/error.
    """
    global _BENCHMARK_DATA_CACHE

    if _BENCHMARK_DATA_CACHE is None:
        try:
            with open('benchmark_results.yaml', 'r') as f:
                _BENCHMARK_DATA_CACHE = yaml.safe_load(f)
            logger.debug("Loaded benchmark_results.yaml into cache")
        except FileNotFoundError:
            logger.warning("benchmark_results.yaml file not found")
            _BENCHMARK_DATA_CACHE = {}
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Error reading benchmark_results.yaml: {e}")
            _BENCHMARK_DATA_CACHE = {}

    return _BENCHMARK_DATA_CACHE


# _log_interp moved to utils.py
# _find_exponential_crossing_time, calculate_initial_research_stock, compute_initial_conditions,
# setup_model, integrate_progress moved to integration.py
# aut_frac_from_swe_multiplier, solve_lower_anchor_via_automation_model moved to automation_model.py


class ProgressModel:
    """Main class for AI progress modeling"""
    
    def __init__(self, params: Parameters, time_series_data: TimeSeriesData):
        """
        Time series data is from the capabilities input spreadsheet.
        """
        self.params = params

        # For now: overriding blacksite start time to be the same as plan A start time.
        self.params.blacksite_start_time = self.params.plan_a_start_time

        self.data = time_series_data
        self.human_only_results = {}
        self.results = {}
        self.horizon_trajectory = None
        
        # Initialize flags for horizon trajectory anchor fix
        self._horizon_uses_shifted_form = False
        self._horizon_params = None

        # Use taste distribution from params (initialized in Parameters.__post_init__)
        self.taste_distribution = self.params.taste_distribution
    def freeze_time_series_at_time(self, freeze_time: float):
        """
        Freeze time series data at a specified time.

        Human labor, experiment compute, and inference compute remain unchanged up to freeze_time,
        then stay constant at their freeze_time values thereafter.

        Training compute is unchanged up to freeze_time, then stays constant (zero growth rate)
        after freeze_time.

        Args:
            freeze_time: Time at which to freeze the time series (decimal year)
        """
        # Get the freeze_time values using log interpolation
        freeze_human_labor = _log_interp(freeze_time, self.data.time, self.data.L_HUMAN)
        freeze_experiment_compute = _log_interp(freeze_time, self.data.time, self.data.experiment_compute)
        freeze_inference_compute = _log_interp(freeze_time, self.data.time, self.data.inference_compute)
        freeze_training_compute = np.interp(freeze_time, self.data.time, self.data.training_compute)

        insertion_idx = np.searchsorted(self.data.time, freeze_time)

        time_with_freeze = np.insert(self.data.time, insertion_idx, freeze_time)
        human_labor_with_freeze = np.insert(self.data.L_HUMAN, insertion_idx, freeze_human_labor)
        experiment_compute_with_freeze = np.insert(self.data.experiment_compute, insertion_idx, freeze_experiment_compute)
        inference_compute_with_freeze = np.insert(self.data.inference_compute, insertion_idx, freeze_inference_compute)
        training_compute_with_freeze = np.insert(self.data.training_compute, insertion_idx, freeze_training_compute)

        # Find where to insert the freeze_time point
        # Insert freeze_time + epsilon to ensure proper ordering and transition
        epsilon = 1e-6  # Small epsilon to ensure freeze_time + epsilon comes after freeze_time
        freeze_time_after = freeze_time + epsilon

        # Find the insertion point (first index where time > freeze_time)
        new_insertion_idx = insertion_idx + 1

        # Create new arrays with the freeze point inserted
        new_time = np.insert(time_with_freeze, new_insertion_idx, freeze_time_after)
        new_L_HUMAN = np.insert(human_labor_with_freeze, new_insertion_idx, freeze_human_labor)
        new_experiment_compute = np.insert(experiment_compute_with_freeze, new_insertion_idx, freeze_experiment_compute)
        new_inference_compute = np.insert(inference_compute_with_freeze, new_insertion_idx, freeze_inference_compute)
        # Training compute stays constant after freeze_time (same value = zero growth rate)
        new_training_compute = np.insert(training_compute_with_freeze, new_insertion_idx, freeze_training_compute)

        # Find indices where time >= freeze_time_after (including the newly inserted point)
        freeze_mask = new_time >= freeze_time_after

        # Set frozen values for human labor, experiment compute, and inference compute
        new_L_HUMAN[freeze_mask] = freeze_human_labor
        new_experiment_compute[freeze_mask] = freeze_experiment_compute
        new_inference_compute[freeze_mask] = freeze_inference_compute
        # Training compute stays constant at freeze value (zero growth rate)
        new_training_compute[freeze_mask] = freeze_training_compute

        # Create new TimeSeriesData object
        self.data = TimeSeriesData(
            time=new_time,
            L_HUMAN=new_L_HUMAN,
            inference_compute=new_inference_compute,
            experiment_compute=new_experiment_compute,
            training_compute=new_training_compute,
        )
        
        logger.info(f"Frozen time series at time {freeze_time}:")
        logger.info(f"  Human labor: {freeze_human_labor:.6f}")
        logger.info(f"  Experiment compute: {freeze_experiment_compute:.6f}")
        logger.info(f"  Inference compute: {freeze_inference_compute:.6f}")
        logger.info(f"  Training compute: {freeze_training_compute:.6f} (frozen)")
        logger.info(f"  Inserted freeze point at time {freeze_time_after:.6f}")

    def find_sos_milestone_time(
        self,
        times: np.ndarray,
        progress_values: np.ndarray,
        research_stock_values: np.ndarray,
        milestone: str
    ) -> Optional[float]:
        """
        Find the time when the specified SOS milestone is reached.

        Args:
            times: Array of time points from integration
            progress_values: Array of progress values from integration
            research_stock_values: Array of research stock values from integration
            milestone: The milestone name (e.g., 'AC', 'AI2027-SC', 'AIR-5x')

        Returns:
            Time when milestone is reached, or None if not reached
        """
        if milestone == 'AC':
            # AC uses progress directly
            if progress_values[-1] > self.params.progress_at_aa:
                return float(np.interp(self.params.progress_at_aa, progress_values, times))
            return None

        # For other milestones, we need to compute metrics
        # Get present-day reference values
        present_day_human_labor = _log_interp(self.params.present_day, self.data.time, self.data.L_HUMAN)
        present_day_inference_compute = _log_interp(self.params.present_day, self.data.time, self.data.inference_compute)
        present_day_experiment_compute = _log_interp(self.params.present_day, self.data.time, self.data.experiment_compute)
        present_day_progress = np.interp(self.params.present_day, times, progress_values)
        present_day_research_stock = _log_interp(self.params.present_day, times, research_stock_values)

        # Compute present-day software progress rate for reference
        present_day_automation_fraction = compute_automation_fraction(present_day_progress, self.params)
        present_day_ai_taste = compute_ai_research_taste(present_day_progress, self.params)
        present_day_aggregate_taste = compute_aggregate_research_taste(present_day_ai_taste, self.params.taste_distribution)

        # Compute present-day coding labor
        if getattr(self.params, 'coding_labor_mode', 'simple_ces') == 'optimal_ces':
            H = float(present_day_human_labor)
            C = float(present_day_inference_compute)
            logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * present_day_progress)
            automation_model = self.params.automation_model
            present_day_coding_labor = automation_model.coding_labor_optimal_ces(H, C, logE, self.params)
            present_day_serial_coding_labor = float((present_day_coding_labor ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
        else:
            present_day_serial_coding_labor = compute_coding_labor_deprecated(
                present_day_automation_fraction, present_day_inference_compute, present_day_human_labor,
                self.params.rho_coding_labor, self.params.parallel_penalty, self.params.coding_labor_normalization
            )

        present_day_research_effort = compute_research_effort(
            present_day_experiment_compute, present_day_serial_coding_labor,
            self.params.alpha_experiment_capacity, self.params.rho_experiment_capacity,
            self.params.experiment_compute_exponent, present_day_aggregate_taste
        )
        present_day_sw_progress_rate = compute_software_progress_rate(
            present_day_research_stock, present_day_research_effort, self.params.r_software
        )

        # Determine which metric and target we need
        if milestone == 'AI2027-SC':
            target = (30 ** (1 / self.params.parallel_penalty)) * 30 * cfg.PARALLEL_LABOR_MULT_BETWEEN_AVERAGE_AND_TOP_FOR_AI2027_SC
            metric_name = 'ai_coding_labor_mult_ref_present_day'
        elif milestone.startswith('AIR-'):
            # Extract multiplier from milestone name (e.g., 'AIR-5x' -> 5)
            mult_str = milestone.replace('AIR-', '').replace('x', '')
            target = float(mult_str)
            metric_name = 'ai_sw_progress_mult_ref_present_day'
        else:
            logger.warning(f"Unknown SOS milestone: {milestone}, falling back to AC")
            if progress_values[-1] > self.params.progress_at_aa:
                return float(np.interp(self.params.progress_at_aa, progress_values, times))
            return None

        # Compute the relevant metric for each time point
        metric_values = []
        for i, (t, progress, rs) in enumerate(zip(times, progress_values, research_stock_values)):
            L_HUMAN = _log_interp(t, self.data.time, self.data.L_HUMAN)
            inference_compute = _log_interp(t, self.data.time, self.data.inference_compute)
            experiment_compute = _log_interp(t, self.data.time, self.data.experiment_compute)

            # Compute automation fraction and taste
            ai_taste = compute_ai_research_taste(progress, self.params)
            aggregate_taste = compute_aggregate_research_taste(ai_taste, self.params.taste_distribution)

            # Compute coding labor
            if getattr(self.params, 'coding_labor_mode', 'simple_ces') == 'optimal_ces':
                H = float(L_HUMAN)
                C = float(inference_compute)
                logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * progress)
                automation_model = self.params.automation_model
                coding_labor = automation_model.coding_labor_optimal_ces(H, C, logE, self.params)
                coding_labor_with_present_resources = automation_model.coding_labor_optimal_ces(
                    present_day_human_labor, present_day_inference_compute, logE, self.params
                )
                serial_coding_labor = float((coding_labor ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
                serial_coding_labor_with_present_resources = float((coding_labor_with_present_resources ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
            else:
                automation_fraction = compute_automation_fraction(progress, self.params)
                serial_coding_labor = compute_coding_labor_deprecated(
                    automation_fraction, inference_compute, L_HUMAN,
                    self.params.rho_coding_labor, self.params.parallel_penalty, self.params.coding_labor_normalization
                )
                serial_coding_labor_with_present_resources = compute_coding_labor_deprecated(
                    automation_fraction, present_day_inference_compute, present_day_human_labor,
                    self.params.rho_coding_labor, self.params.parallel_penalty, self.params.coding_labor_normalization
                )
                coding_labor_with_present_resources = serial_coding_labor_with_present_resources

            if metric_name == 'ai_coding_labor_mult_ref_present_day':
                # ai_coding_labor_mult_ref_present_day = coding_labor_with_present_resources / present_day_human_labor
                metric_val = coding_labor_with_present_resources / present_day_human_labor if present_day_human_labor > 0 else 1.0
            else:  # ai_sw_progress_mult_ref_present_day
                # Compute software progress rate with present-day resources
                research_effort_present_resources = compute_research_effort(
                    present_day_experiment_compute, serial_coding_labor_with_present_resources,
                    self.params.alpha_experiment_capacity, self.params.rho_experiment_capacity,
                    self.params.experiment_compute_exponent, aggregate_taste
                )
                software_rate_present_resources = compute_software_progress_rate(
                    present_day_research_stock, research_effort_present_resources, self.params.r_software
                )
                metric_val = software_rate_present_resources / present_day_sw_progress_rate if present_day_sw_progress_rate > 0 else 0.0

            metric_values.append(metric_val if np.isfinite(metric_val) else 0.0)

        # Find crossing time using exponential interpolation
        crossing_time = _find_exponential_crossing_time(
            np.asarray(times, dtype=float),
            np.asarray(metric_values, dtype=float),
            float(target)
        )

        return crossing_time

    def initiate_fab_foom(self, foom_time: float):
        """
        Initiate fab foom.
        
        Args:
            foom_time: Time at which to initiate fab foom (decimal year)
        """
        foom_human_labor = _log_interp(foom_time, self.data.time, self.data.L_HUMAN)
        foom_experiment_compute = _log_interp(foom_time, self.data.time, self.data.experiment_compute)
        foom_inference_compute = _log_interp(foom_time, self.data.time, self.data.inference_compute)
        # Get training compute at foom_time (linear interp since already OOMs)
        foom_training_compute = np.interp(foom_time, self.data.time, self.data.training_compute)

        # Get the target growth rate from parameters
        growth_rate = self.params.main_project_training_compute_growth_rate

        insertion_idx = np.searchsorted(self.data.time, foom_time)

        time_with_foom = np.insert(self.data.time, insertion_idx, foom_time)
        human_labor_with_foom = np.insert(self.data.L_HUMAN, insertion_idx, foom_human_labor)
        experiment_compute_with_foom = np.insert(self.data.experiment_compute, insertion_idx, foom_experiment_compute)
        inference_compute_with_foom = np.insert(self.data.inference_compute, insertion_idx, foom_inference_compute)
        training_compute_with_foom = np.insert(self.data.training_compute, insertion_idx, foom_training_compute)

        # Find where to insert the foom_time point
        # Insert foom_time + epsilon to ensure proper ordering and transition
        epsilon = 1e-6  # Small epsilon to ensure foom_time + epsilon comes after foom_time
        foom_time_after = foom_time + epsilon

        # Find the insertion point (first index where time > foom_time)
        new_insertion_idx = insertion_idx + 1

        # Create new arrays with the foom point inserted
        new_time = np.insert(time_with_foom, new_insertion_idx, foom_time_after)
        new_L_HUMAN = np.insert(human_labor_with_foom, new_insertion_idx, foom_human_labor)
        new_experiment_compute = np.insert(experiment_compute_with_foom, new_insertion_idx, foom_experiment_compute)
        new_inference_compute = np.insert(inference_compute_with_foom, new_insertion_idx, foom_inference_compute)
        new_training_compute = np.insert(training_compute_with_foom, new_insertion_idx, foom_training_compute)

        # Find indices where time >= foom_time (including both inserted points and all future points)
        foom_mask = new_time >= foom_time

        # Make experiment compute, inference compute, and training compute grow at growth_rate (in OOMs/year)
        # For times >= foom_time, values grow as: value(t) = value_0 * 10^(growth_rate * (t - foom_time))
        time_delta = new_time[foom_mask] - foom_time
        new_experiment_compute[foom_mask] = foom_experiment_compute * (10 ** (growth_rate * time_delta))
        new_inference_compute[foom_mask] = foom_inference_compute * (10 ** (growth_rate * time_delta))
        # Training compute grows linearly in OOMs (constant growth rate)
        new_training_compute[foom_mask] = foom_training_compute + growth_rate * time_delta

        # Create new TimeSeriesData object
        self.data = TimeSeriesData(
            time=new_time,
            L_HUMAN=new_L_HUMAN,
            inference_compute=new_inference_compute,
            experiment_compute=new_experiment_compute,
            training_compute=new_training_compute
        )

        logger.info(f"Initiated fab foom at time {foom_time}:")
        logger.info(f"  Human labor: {foom_human_labor:.6f}")
        logger.info(f"  Experiment compute: {foom_experiment_compute:.6f} (growing at {growth_rate:.6f} OOMs/year from foom_time)")
        logger.info(f"  Inference compute: {foom_inference_compute:.6f} (growing at {growth_rate:.6f} OOMs/year from foom_time)")
        logger.info(f"  Training compute growth rate: {growth_rate:.6f} OOMs/year from foom_time")
    
    def estimate_horizon_trajectory(self, human_only_times: np.ndarray, human_only_progress: np.ndarray, anchor_progress_rate: float):
        """
        Estimate horizon trajectory by fitting to log(p80_horizon_length) vs progress.
        Uses provided human-only progress trajectory to get progress values at model release dates.
        
        The functional form depends on horizon_extrapolation_type:
        - "exponential": linear regression on log(horizon) vs progress
        - "decaying doubling time": decaying doubling time functional form
        
        Args:
            human_only_times: Time array from human-only trajectory computation
            human_only_progress: Progress array from human-only trajectory computation
            anchor_progress_rate: Progress rate at anchor time (progress units per time unit),
                                 used to convert present_doubling_time from time units to progress units
            
        Returns:
            Function that maps progress to horizon length
        """

        # Determine if we need benchmark data for autofitting
        # Benchmark data is only needed if any manual parameters are missing
        needs_benchmark_data = False

        if self.params.horizon_extrapolation_type == "exponential":
            # For exponential: need data if present_horizon or present_doubling_time is None
            if self.params.present_horizon is None or self.params.present_doubling_time is None:
                needs_benchmark_data = True
        elif self.params.horizon_extrapolation_type == "decaying doubling time":
            # For decaying doubling time: need data if any of the three params is None
            if (self.params.present_horizon is None or
                self.params.present_doubling_time is None or
                self.params.doubling_difficulty_growth_factor is None):
                needs_benchmark_data = True

        # Initialize benchmark data variables (will be populated if needed)
        progress_values = np.array([])
        horizon_values = np.array([])

        if needs_benchmark_data:
            # Load METR benchmark data from cache
            benchmark_data = _load_benchmark_data()
            if not benchmark_data:
                logger.error("benchmark_results.yaml could not be loaded")
                return None

            # Extract (progress, horizon) pairs from METR data
            progress_horizon_pairs = []

            for model_name, model_info in benchmark_data['results'].items():
                # Convert release date to decimal year
                release_date_obj = model_info['release_date']
                try:
                    # Handle both string and date objects
                    if isinstance(release_date_obj, str):
                        release_date = datetime.strptime(release_date_obj, '%Y-%m-%d').date()
                    else:
                        release_date = release_date_obj

                    decimal_year = release_date.year + (release_date.timetuple().tm_yday - 1) / 365.25
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Could not parse release date for {model_name}: {release_date_obj} ({e})")
                    continue

                # Interpolate progress value at the release date using human-only trajectory results
                if decimal_year >= human_only_times.min() and decimal_year <= human_only_times.max():
                    interpolated_progress = np.interp(decimal_year, human_only_times, human_only_progress)
                elif decimal_year < human_only_times.min():
                    interpolated_progress = human_only_progress[0]
                else:
                    interpolated_progress = human_only_progress[-1]

                # Extract p80_horizon_length from metrics
                if 'metrics' in model_info and 'p80_horizon_length' in model_info['metrics']:
                    p80_data = model_info['metrics']['p80_horizon_length']
                    p80_estimate = p80_data.get('estimate')

                    if p80_estimate is not None and p80_estimate > 0:  # Must be positive for log transform
                        progress_horizon_pairs.append((interpolated_progress, p80_estimate))

            if len(progress_horizon_pairs) < 2:
                logger.error("Not enough valid (progress, horizon) pairs for regression")
                return None

            # Convert to arrays for regression
            progress_values = np.array([pair[0] for pair in progress_horizon_pairs])
            horizon_values = np.array([pair[1] for pair in progress_horizon_pairs])
        
        # Fit functional form based on horizon_extrapolation_type
        log_horizon_values = np.log(horizon_values)
        
        # Define fitting functions
        def linear_func(x, a, b):
            return a * x + b
        
        def decaying_doubling_time_func(t, H_0, A_0, T_0):
            """Decaying doubling time function with numerical safeguards.
            Supports A_0 in (-inf, 1), excluding A_0 == 0. Handles both accelerating (A_0>0) and decelerating (A_0<0) cases.
            """
            try:
                # Handle scalar vs array inputs
                is_scalar = np.isscalar(t)
                t_arr = np.atleast_1d(t)
                
                # Ensure parameters are within valid ranges
                # A_0 must be < 1 and != 0; H_0 and T_0 must be > 0
                if T_0 <= 0 or H_0 <= 0 or A_0 >= 1 or A_0 == 0:
                    fallback = np.full_like(t_arr, np.log(1e12))
                    return fallback[0] if is_scalar else fallback
                
                # Calculate the base term (1 - A_0 * t / T_0)
                base_term = 1 - A_0 * t_arr / T_0
                # Clamp to small positive to avoid domain errors
                base_term = np.maximum(base_term, 1e-12)
                
                # Calculate the exponent
                log_denominator = np.log(1 - A_0)
                if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                    fallback = np.full_like(t_arr, np.log(1e12))
                    return fallback[0] if is_scalar else fallback
                exponent = np.log(2) / log_denominator
                
                # Calculate the result
                result = H_0 * (base_term ** exponent)
                
                # Replace only invalid elements (not the entire array)
                result = np.where(
                    np.isfinite(result) & (result > 0),
                    result,
                    1e12
                )

                log_result = np.log(result)
                return log_result[0] if is_scalar else log_result
                
            except (ValueError, ZeroDivisionError, OverflowError):
                t_arr = np.atleast_1d(t)
                fallback = np.full_like(t_arr, np.log(1e12))
                return fallback[0] if np.isscalar(t) else fallback
        
        try:
            if self.params.horizon_extrapolation_type == "exponential":
                # Check if manual parameters are provided
                if self.params.present_horizon is not None:
                    # Use manual fitting with anchor point
                    # Get progress at present_day
                    anchor_progress = np.interp(self.params.present_day, human_only_times, human_only_progress)
                    
                    # If present_doubling_time is provided, use it to calculate slope
                    if self.params.present_doubling_time is not None:
                        # Convert present_doubling_time from time units to progress units
                        # doubling_time_in_progress_units = doubling_time_in_time_units * progress_rate
                        doubling_time_in_progress_units = self.params.present_doubling_time * anchor_progress_rate
                        # slope = log(2) / doubling_time (in progress units)
                        slope = np.log(2) / doubling_time_in_progress_units
                    else:
                        # Optimize slope using data, but fix intercept using anchor point
                        def fit_slope_only(slope_val):
                            intercept_val = np.log(self.params.present_horizon) - slope_val * anchor_progress
                            predicted = linear_func(progress_values, slope_val, intercept_val)
                            return np.sum((log_horizon_values - predicted)**2)
                        
                        result = optimize.minimize_scalar(fit_slope_only, bounds=(-10, 10), method='bounded')
                        slope = result.x
                    
                    # Calculate intercept from anchor point: log(present_horizon) = slope * anchor_progress + intercept
                    intercept = np.log(self.params.present_horizon) - slope * anchor_progress
                    
                    logger.info(f"Manual exponential horizon trajectory: log(horizon) = {slope:.6f} * progress + {intercept:.6f}")
                    logger.info(f"Using anchor point: time={self.params.present_day}, progress={anchor_progress:.4f}, horizon={self.params.present_horizon:.4f}")
                    if self.params.present_doubling_time is not None:
                        logger.info(f"Anchor doubling time (time units): {self.params.present_doubling_time:.4f}, converted to progress units: {doubling_time_in_progress_units:.4f} (using progress rate: {anchor_progress_rate:.4f})")
                else:
                    # Use automatic curve fitting
                    popt, pcov = optimize.curve_fit(linear_func, progress_values, log_horizon_values)
                    slope, intercept = popt
                    
                    logger.info(f"Fitted exponential horizon trajectory: log(horizon) = {slope:.6f} * progress + {intercept:.6f}")
                    logger.info(f"R-squared: {1 - np.sum((log_horizon_values - linear_func(progress_values, *popt))**2) / np.sum((log_horizon_values - np.mean(log_horizon_values))**2):.4f}")
                
                # Create horizon trajectory function: progress -> horizon
                def horizon_trajectory(progress):
                    """Map progress to horizon length using fitted exponential model"""
                    return np.exp(slope * progress + intercept)
                
                # Calculate progress level where horizon reaches the target horizon
                # Target depends on include_gap (formerly benchmarks_and_gaps_mode)
                _include_gap_flag = False
                try:
                    _inc = getattr(self.params, 'include_gap', 'no gap')
                    if isinstance(_inc, str):
                        _include_gap_flag = _inc.strip().lower() == 'gap'
                    else:
                        _include_gap_flag = bool(_inc)
                except Exception as e:
                    if should_reraise(e):
                        raise
                    _include_gap_flag = False
                target_horizon = self.params.ac_time_horizon_minutes
                if target_horizon > 0:
                    try:
                        # Solve: target_horizon = exp(slope * progress + intercept)
                        # Therefore: progress = (log(target_horizon) - intercept) / slope
                        calculated_progress_at_aa = (np.log(target_horizon) - intercept) / slope
                        # If in gap-included mode, add the gap (specified in anchor-progress-years)
                        # Convert anchor-progress-years to progress units using anchor_progress_rate
                        if _include_gap_flag:
                            try:
                                gap_anchor_years = float(self.params.gap_years)
                                gap_progress_units = float(anchor_progress_rate) * gap_anchor_years
                            except Exception as e:
                                if should_reraise(e):
                                    raise
                                gap_progress_units = float(self.params.gap_years)
                            calculated_progress_at_aa = calculated_progress_at_aa + gap_progress_units
                            try:
                                year_label = int(self.params.present_day) if getattr(self.params, 'present_day', None) is not None else 'anchor'
                            except Exception as e:
                                if should_reraise(e):
                                    raise
                                year_label = 'anchor'
                            logger.info(
                                f"Gap-included mode: using ac_time_horizon_minutes {self.params.ac_time_horizon_minutes} and "
                                f"adding gap {self.params.gap_years} {year_label}-progress-years (~{gap_progress_units:.6f} progress units)"
                            )
                        self.params.progress_at_aa = calculated_progress_at_aa
                        logger.info(f"Progress level at target horizon ({target_horizon} min): {calculated_progress_at_aa:.4f}")
                    except (ValueError, ZeroDivisionError) as e:
                        logger.warning(f"Could not calculate progress at ac_time_horizon_minutes: {e}")
                        self.params.progress_at_aa = None
            
            elif self.params.horizon_extrapolation_type == "decaying doubling time":
                # Determine approach based on whether present_doubling_time is specified
                # If present_doubling_time is specified, we MUST use the shifted function approach
                # to ensure the doubling time at the anchor point equals present_doubling_time
                if self.params.present_doubling_time is not None or self.params.present_horizon is not None:
                    # Use shifted function approach
                    self._horizon_uses_shifted_form = True
                    # Get progress at present_day
                    anchor_progress = np.interp(self.params.present_day, human_only_times, human_only_progress)
                    
                    # Determine what parameters we have and what we need to optimize
                    params_to_optimize = []
                    fixed_params = {}
                    
                    # Handle H_0 (present_horizon)
                    if self.params.present_horizon is not None:
                        fixed_params['H_0'] = self.params.present_horizon
                    else:
                        params_to_optimize.append('H_0')
                    
                    # Handle T_0 (present_doubling_time)
                    if self.params.present_doubling_time is not None:
                        # Convert present_doubling_time from time units to progress units
                        # doubling_time_in_progress_units = doubling_time_in_time_units * progress_rate
                        doubling_time_in_progress_units = self.params.present_doubling_time * anchor_progress_rate
                        fixed_params['T_0'] = doubling_time_in_progress_units
                    else:
                        params_to_optimize.append('T_0')
                    
                    # Handle A_0 (doubling_difficulty_growth_factor converted to decay_rate)
                    if self.params.doubling_difficulty_growth_factor is not None:
                        fixed_params['A_0'] = 1.0 - self.params.doubling_difficulty_growth_factor
                    else:
                        params_to_optimize.append('A_0')
                    
                    if len(params_to_optimize) == 0:
                        # All parameters specified (Case 8)
                        H_0 = self.params.present_horizon
                        T_0 = doubling_time_in_progress_units
                        A_0 = 1.0 - self.params.doubling_difficulty_growth_factor
                        logger.info(f"Manual decaying doubling time: All parameters specified")
                    elif len(params_to_optimize) == 1:
                        # Optimize one parameter
                        param_name = params_to_optimize[0]
                        
                        def fit_single_param(param_val):
                            if param_name == 'A_0' and (param_val >= 1 or abs(param_val) < 1e-6):
                                return 1e6
                            if param_name in ['H_0', 'T_0'] and param_val <= 0:
                                return 1e6
                            
                            # Reconstruct full parameter set
                            param_dict = fixed_params.copy()
                            param_dict[param_name] = param_val
                            
                            H_0_val = param_dict['H_0']
                            T_0_val = param_dict['T_0']
                            A_0_val = param_dict['A_0']
                            
                            # Use shifted function: horizon(t) = H_0 * (1 - A_0 * (t - anchor_progress) / T_0)^exponent
                            def shifted_func(t_vals):
                                try:
                                    t_shifted = t_vals - anchor_progress
                                    base_term = 1 - A_0_val * t_shifted / T_0_val
                                    # Clamp to small positive to avoid domain errors
                                    base_term = np.maximum(base_term, 1e-12)
                                    
                                    # Guard against A_0 near 0 or >=1
                                    log_denominator = np.log(1 - A_0_val)
                                    if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                                        return np.full_like(t_vals, np.log(1e12))
                                    exponent = np.log(2) / log_denominator
                                    result = H_0_val * (base_term ** exponent)
                                    
                                    # Replace only invalid elements
                                    result = np.where(
                                        np.isfinite(result) & (result > 0),
                                        result,
                                        1e12
                                    )

                                    return np.log(result)
                                except Exception as e:
                                    if should_reraise(e):
                                        raise
                                    logger.warning(f"Error in shifted_func: {e}")
                                    return np.full_like(t_vals, np.log(1e12))

                            predicted = shifted_func(progress_values)
                            return np.sum((log_horizon_values - predicted)**2)

                        # Set bounds based on parameter type
                        if param_name == 'A_0':
                            bounds = (-0.999, 0.999)
                            initial_guess = 0.05
                        elif param_name == 'H_0':
                            bounds = (1e-6, 1e6)
                            initial_guess = 0.00001
                        elif param_name == 'T_0':
                            bounds = (1e-6, 100.0)
                            initial_guess = 1.35
                        
                        result = optimize.minimize_scalar(fit_single_param, bounds=bounds, method='bounded')
                        
                        # Extract optimized parameters
                        param_dict = fixed_params.copy()
                        param_dict[param_name] = result.x
                        
                        H_0 = param_dict['H_0']
                        T_0 = param_dict['T_0']
                        A_0 = param_dict['A_0']
                        
                        logger.info(f"Optimized {param_name}: {result.x:.6f}")
                    else:
                        # Optimize multiple parameters
                        def fit_multiple_params(params_array):
                            # Reconstruct full parameter set
                            param_dict = fixed_params.copy()
                            for i, param_name in enumerate(params_to_optimize):
                                param_dict[param_name] = params_array[i]
                            
                            H_0_val = param_dict['H_0']
                            T_0_val = param_dict['T_0']
                            A_0_val = param_dict['A_0']
                            
                            if A_0_val >= 1 or abs(A_0_val) < 1e-6 or H_0_val <= 0 or T_0_val <= 0:
                                return 1e6
                            
                            # Use shifted function
                            def shifted_func(t_vals):
                                try:
                                    t_shifted = t_vals - anchor_progress
                                    base_term = 1 - A_0_val * t_shifted / T_0_val
                                    # Clamp to small positive to avoid domain errors
                                    base_term = np.maximum(base_term, 1e-12)
                                    
                                    log_denominator = np.log(1 - A_0_val)
                                    if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                                        return np.full_like(t_vals, np.log(1e12))
                                    exponent = np.log(2) / log_denominator
                                    result = H_0_val * (base_term ** exponent)
                                    
                                    # Replace only invalid elements
                                    result = np.where(
                                        np.isfinite(result) & (result > 0),
                                        result,
                                        1e12
                                    )

                                    return np.log(result)
                                except Exception as e:
                                    if should_reraise(e):
                                        raise
                                    logger.warning(f"Error in shifted_func: {e}")
                                    return np.full_like(t_vals, np.log(1e12))

                            predicted = shifted_func(progress_values)
                            return np.sum((log_horizon_values - predicted)**2)

                        # Set up bounds and initial guesses
                        bounds = []
                        p0 = []
                        for param_name in params_to_optimize:
                            if param_name == 'A_0':
                                bounds.append((-0.999, 0.999))
                                p0.append(0.05)
                            elif param_name == 'H_0':
                                bounds.append((1e-6, 1e6))
                                p0.append(0.00001)
                            elif param_name == 'T_0':
                                bounds.append((1e-6, 100.0))
                                p0.append(1.35)
                        
                        result = optimize.minimize(fit_multiple_params, p0, bounds=bounds, method='L-BFGS-B')
                        
                        # Extract optimized parameters
                        param_dict = fixed_params.copy()
                        for i, param_name in enumerate(params_to_optimize):
                            param_dict[param_name] = result.x[i]
                        
                        H_0 = param_dict['H_0']
                        T_0 = param_dict['T_0']
                        A_0 = param_dict['A_0']
                        
                        logger.info(f"Optimized parameters: {', '.join([f'{name}={param_dict[name]:.6f}' for name in params_to_optimize])}")
                    
                    logger.info(f"Manual decaying doubling time horizon trajectory: H_0={H_0:.6f}, A_0={A_0:.6f}, T_0={T_0:.6f}")
                    logger.info(f"Using anchor point: time={self.params.present_day}, progress={anchor_progress:.4f}")
                    if self.params.present_horizon is not None:
                        logger.info(f"Anchor horizon: {self.params.present_horizon:.4f}")
                    if self.params.present_doubling_time is not None:
                        logger.info(f"Anchor doubling time (time units): {self.params.present_doubling_time:.4f}, converted to progress units: {doubling_time_in_progress_units:.4f} (using progress rate: {anchor_progress_rate:.4f})")
                    
                    # Store anchor_progress for use in horizon_trajectory function
                    anchor_progress_for_trajectory = anchor_progress
                else:
                    # Use automatic curve fitting (Cases 1-2: no anchor parameters specified)
                    self._horizon_uses_shifted_form = False
                    # These specific values are somewhat important for the optimization
                    H_0_init = 0.00001  
                    A_0_init = 0.05
                    T_0_init = 1.35
                    
                    popt, pcov = optimize.curve_fit(
                        decaying_doubling_time_func, 
                        progress_values, 
                        log_horizon_values,
                        p0=[H_0_init, A_0_init, T_0_init],
                        bounds=([1e-6, -0.999, 1e-6], [np.inf, 0.999, np.inf])  # Allow negative A_0 but exclude 0 implicitly via fit
                    )
                    H_0, A_0, T_0 = popt
                    
                    logger.info(f"Fitted decaying doubling time horizon trajectory: H_0={H_0:.6f}, A_0={A_0:.6f}, T_0={T_0:.6f}")
                    logger.info(f"R-squared: {1 - np.sum((log_horizon_values - decaying_doubling_time_func(progress_values, *popt))**2) / np.sum((log_horizon_values - np.mean(log_horizon_values))**2):.4f}")
                    
                    # For automatic fitting, no anchor_progress shift is used
                    anchor_progress_for_trajectory = None
                
                # Create horizon trajectory function: progress -> horizon
                def horizon_trajectory(progress):
                    """Map progress to horizon length using fitted decaying doubling time model"""
                    try:
                        # Handle scalar vs array inputs
                        is_scalar = np.isscalar(progress)
                        progress_arr = np.atleast_1d(progress)
                        
                        # Ensure parameters are within valid ranges
                        if T_0 <= 0 or H_0 <= 0 or A_0 >= 1 or A_0 == 0:
                            fallback = np.full_like(progress_arr, 1e12)
                            return fallback[0] if is_scalar else fallback
                        
                        # Use shifted form if we're in the manual parameter case (anchor_progress_for_trajectory is set)
                        if anchor_progress_for_trajectory is not None:
                            # Shifted form: horizon(t) = H_0 * (1 - A_0 * (t - anchor_progress) / T_0)^exponent
                            progress_shifted = progress_arr - anchor_progress_for_trajectory
                            base_term = 1 - A_0 * progress_shifted / T_0
                        else:
                            # Original form: horizon(t) = H_0 * (1 - A_0 * t / T_0)^exponent
                            base_term = 1 - A_0 * progress_arr / T_0
                        
                        # Clamp to small positive to avoid domain errors
                        base_term = np.maximum(base_term, 1e-12)
                        
                        # Calculate the exponent
                        log_denominator = np.log(1 - A_0)
                        if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                            fallback = np.full_like(progress_arr, 1e12)
                            return fallback[0] if is_scalar else fallback
                        
                        exponent = np.log(2) / log_denominator
                        
                        # Calculate the result
                        result = H_0 * (base_term ** exponent)
                        
                        # Replace only invalid elements (not the entire array)
                        result = np.where(
                            np.isfinite(result) & (result > 0),
                            result,
                            1e12
                        )

                        return result[0] if is_scalar else result
                        
                    except (ValueError, ZeroDivisionError, OverflowError):
                        progress_arr = np.atleast_1d(progress)
                        fallback = np.full_like(progress_arr, 1e12)
                        return fallback[0] if np.isscalar(progress) else fallback
                
                # Calculate progress level where horizon reaches the target horizon
                # Target depends on include_gap (formerly benchmarks_and_gaps_mode)
                _include_gap_flag = False
                try:
                    _inc = getattr(self.params, 'include_gap', 'no gap')
                    if isinstance(_inc, str):
                        _include_gap_flag = _inc.strip().lower() == 'gap'
                    else:
                        _include_gap_flag = bool(_inc)
                except Exception as e:
                    if should_reraise(e):
                        raise
                    _include_gap_flag = False
                target_horizon = self.params.ac_time_horizon_minutes
                if target_horizon > 0:
                    try:
                        # Add numerical safeguards
                        if T_0 <= 0 or H_0 <= 0 or A_0 >= 1 or A_0 == 0:
                            logger.warning("Invalid parameters for progress_at_aa calculation")
                            self.params.progress_at_aa = None
                        elif target_horizon <= 0:
                            logger.warning("Invalid ac_time_horizon_minutes for calculation")
                            self.params.progress_at_aa = None
                        else:
                            # Check if the ratio is valid
                            ratio = target_horizon / H_0
                            if ratio <= 0:
                                logger.warning("Invalid ratio for progress_at_aa calculation")
                                self.params.progress_at_aa = None
                            else:
                                log_ratio = np.log(1-A_0) / np.log(2)
                                if not np.isfinite(log_ratio):
                                    logger.warning("Invalid log ratio for progress_at_aa calculation")
                                    self.params.progress_at_aa = None
                                else:
                                    ratio_term = ratio ** log_ratio
                                    if not np.isfinite(ratio_term):
                                        logger.warning("Invalid ratio_term for progress_at_aa calculation")
                                        self.params.progress_at_aa = None
                                    else:
                                        # Use shifted form if we're in the manual parameter case
                                        if anchor_progress_for_trajectory is not None:
                                            # Shifted form: ac_time_horizon_minutes = H_0 * (1 - A_0 * (progress - anchor_progress) / T_0)^exponent
                                            # progress = anchor_progress + T_0 * (1 - (ac_time_horizon_minutes / H_0)^(log(1-A_0)/log(2))) / A_0
                                            calculated_progress_at_aa = anchor_progress_for_trajectory + T_0 * (1 - ratio_term) / A_0
                                        else:
                                            # Original form: ac_time_horizon_minutes = H_0 * (1 - A_0 * progress / T_0)^exponent
                                            # progress = T_0 * (1 - (ac_time_horizon_minutes / H_0)^(log(1-A_0)/log(2))) / A_0
                                            calculated_progress_at_aa = T_0 * (1 - ratio_term) / A_0

                                        # If in gap-included mode, add the gap (specified in anchor-progress-years)
                                        # Convert anchor-progress-years to progress units using anchor_progress_rate
                                        if _include_gap_flag:
                                            try:
                                                gap_anchor_years = float(self.params.gap_years)
                                                gap_progress_units = float(anchor_progress_rate) * gap_anchor_years
                                            except Exception as e:
                                                if should_reraise(e):
                                                    raise
                                                gap_progress_units = float(self.params.gap_years)
                                            calculated_progress_at_aa = calculated_progress_at_aa + gap_progress_units
                                            try:
                                                year_label = int(self.params.present_day) if getattr(self.params, 'present_day', None) is not None else 'anchor'
                                            except Exception as e:
                                                if should_reraise(e):
                                                    raise
                                                year_label = 'anchor'
                                            logger.info(
                                                f"Gap-included mode: using ac_time_horizon_minutes {self.params.ac_time_horizon_minutes} and "
                                                f"adding gap {self.params.gap_years} {year_label}-progress-years (~{gap_progress_units:.6f} progress units)"
                                            )
                                        
                                        if not np.isfinite(calculated_progress_at_aa):
                                            logger.warning("Invalid progress_at_aa result")
                                            self.params.progress_at_aa = None
                                        else:
                                            self.params.progress_at_aa = calculated_progress_at_aa
                                            logger.info(f"Progress level at target horizon ({target_horizon} min): {calculated_progress_at_aa:.4f}")
                    except (ValueError, ZeroDivisionError, OverflowError) as e:
                        logger.warning(f"Could not calculate progress at ac_time_horizon_minutes: {e}")
                        self.params.progress_at_aa = None
            
            else:
                logger.error(f"Unknown horizon_extrapolation_type: {self.params.horizon_extrapolation_type}")
                return None
            
            # Store the function and parameters for later use
            self.horizon_trajectory = horizon_trajectory
            
            # Store parameters needed for anchor update in shifted form cases
            if hasattr(self, '_horizon_uses_shifted_form') and self._horizon_uses_shifted_form:
                self._horizon_params = {
                    'H_0': H_0,
                    'A_0': A_0, 
                    'T_0': T_0,
                    'original_anchor_progress': anchor_progress_for_trajectory
                }
            
            return horizon_trajectory

        except Exception as e:
            if should_reraise(e):
                raise
            logger.error(f"Error fitting horizon trajectory: {e}")
            return None

    def _update_horizon_trajectory_anchor(self, new_anchor_progress: float):
        """
        Update the horizon trajectory function with a new anchor progress value.
        This fixes the Case 2 issue where the fitted anchor progress differs from 
        the actual integrated progress at present_day.
        """
        if not hasattr(self, '_horizon_params') or not self._horizon_params:
            logger.warning("Cannot update horizon trajectory: no stored parameters")
            return
            
        # Extract stored parameters
        H_0 = self._horizon_params['H_0']
        A_0 = self._horizon_params['A_0'] 
        T_0 = self._horizon_params['T_0']
        
        # Update the stored anchor progress
        self._horizon_params['original_anchor_progress'] = new_anchor_progress
        
        # Create new horizon trajectory function with updated anchor progress
        def horizon_trajectory(progress):
            """Map progress to horizon length using fitted decaying doubling time model with updated anchor"""
            try:
                # Handle scalar vs array inputs
                is_scalar = np.isscalar(progress)
                progress_arr = np.atleast_1d(progress)
                
                # Ensure parameters are within valid ranges
                # Allow A_0 < 0, disallow A_0 == 0 and A_0 >= 1
                if T_0 <= 0 or H_0 <= 0 or A_0 >= 1 or A_0 == 0:
                    fallback = np.full_like(progress_arr, 1e12)
                    return fallback[0] if is_scalar else fallback
                
                # Use shifted form with updated anchor progress
                progress_shifted = progress_arr - new_anchor_progress
                base_term = 1 - A_0 * progress_shifted / T_0
                # Clamp to small positive to avoid domain errors
                base_term = np.maximum(base_term, 1e-12)
                
                # Calculate the exponent
                log_denominator = np.log(1 - A_0)
                if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                    fallback = np.full_like(progress_arr, 1e12)
                    return fallback[0] if is_scalar else fallback
                
                exponent = np.log(2) / log_denominator
                
                # Calculate the result
                result = H_0 * (base_term ** exponent)
                
                # Replace only invalid elements (not the entire array)
                result = np.where(
                    np.isfinite(result) & (result > 0),
                    result,
                    1e12
                )

                return result[0] if is_scalar else result

            except (ValueError, ZeroDivisionError, OverflowError):
                progress_arr = np.atleast_1d(progress)
                fallback = np.full_like(progress_arr, 1e12)
                return fallback[0] if np.isscalar(progress) else fallback

        # Replace the horizon trajectory function
        self.horizon_trajectory = horizon_trajectory
        logger.info(f"Updated horizon trajectory anchor progress to {new_anchor_progress:.6f}")

    def compute_metr_mse(self) -> Optional[float]:
        """
        Compute Mean Squared Error between model predictions and METR benchmark data.

        Returns MSE in log-space: MSE = mean((log(predicted_horizon) - log(actual_horizon))^2)
        Returns None if METR data cannot be loaded or horizon trajectory is not available.
        """
        # Check if horizon trajectory is available
        if not hasattr(self, 'horizon_trajectory') or self.horizon_trajectory is None:
            logger.warning("Cannot compute METR MSE: horizon trajectory not available")
            return None

        # Check if results are available
        if not hasattr(self, 'results') or not self.results:
            logger.warning("Cannot compute METR MSE: model results not available")
            return None

        # Load METR benchmark data from cache
        benchmark_data = _load_benchmark_data()
        if not benchmark_data:
            logger.warning("Cannot compute METR MSE: benchmark_results.yaml could not be loaded")
            return None

        # Get human-only trajectory for progress interpolation
        if not hasattr(self, 'human_only_results') or not self.human_only_results:
            logger.warning("Cannot compute METR MSE: human-only results not available")
            return None

        human_only_times = self.human_only_results['times']
        human_only_progress = self.human_only_results['progress']

        # Extract (progress, horizon) pairs from METR data
        squared_errors = []

        for model_name, model_info in benchmark_data['results'].items():
            # Convert release date to decimal year
            release_date_obj = model_info['release_date']
            try:
                # Handle both string and date objects
                if isinstance(release_date_obj, str):
                    release_date = datetime.strptime(release_date_obj, '%Y-%m-%d').date()
                else:
                    release_date = release_date_obj

                decimal_year = release_date.year + (release_date.timetuple().tm_yday - 1) / 365.25
            except (ValueError, AttributeError) as e:
                logger.debug(f"Could not parse release date for {model_name}: {release_date_obj} ({e})")
                continue

            # Interpolate progress value at the release date using human-only trajectory
            if decimal_year >= human_only_times.min() and decimal_year <= human_only_times.max():
                interpolated_progress = np.interp(decimal_year, human_only_times, human_only_progress)
            elif decimal_year < human_only_times.min():
                interpolated_progress = human_only_progress[0]
            else:
                interpolated_progress = human_only_progress[-1]

            # Extract p80_horizon_length from metrics
            if 'metrics' in model_info and 'p80_horizon_length' in model_info['metrics']:
                p80_data = model_info['metrics']['p80_horizon_length']
                actual_horizon = p80_data.get('estimate')

                if actual_horizon is not None and actual_horizon > 0:
                    # Get predicted horizon from model
                    try:
                        predicted_horizon = self.horizon_trajectory(interpolated_progress)

                        # Ensure predicted_horizon is valid
                        if np.isfinite(predicted_horizon) and predicted_horizon > 0:
                            # Compute squared error in log space
                            log_error = np.log(predicted_horizon) - np.log(actual_horizon)
                            squared_errors.append(log_error ** 2)
                        else:
                            logger.debug(f"Invalid predicted horizon for {model_name}: {predicted_horizon}")
                    except Exception as e:
                        if should_reraise(e):
                            raise
                        logger.debug(f"Error predicting horizon for {model_name}: {e}")
                        continue

        if len(squared_errors) == 0:
            logger.warning("Cannot compute METR MSE: no valid prediction-observation pairs")
            return None

        # Compute MSE
        mse = np.mean(squared_errors)
        logger.info(f"METR MSE (log-space): {mse:.6f} over {len(squared_errors)} data points")

        return mse

    def compute_human_only_trajectory(self, time_range: List[float], initial_progress: Optional[float] = None, use_ode: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute human-only progress over specified time range.

        Args:
            time_range: [start_time, end_time]
            initial_progress: Initial progress value (default 0.0)
            use_ode: If True, use ODE solver for higher accuracy (needed for r_software calibration).
                     If False (default), use fast vectorized integration.
        """
        if initial_progress is None:
            initial_progress = 0.0  # Use a reasonable default value

        # Create human-only params for initial conditions computation
        human_only_params = copy.deepcopy(self.params)
        human_only_params.human_only = True

        # Compute initial research stock (still need this for the starting point)
        initial_conditions = compute_initial_conditions(self.data, human_only_params, initial_progress)
        initial_research_stock_val = initial_conditions.research_stock
        logger.info(f"HUMAN-ONLY::: initial_research_stock_val: {initial_research_stock_val}")

        if use_ode:
            # Use ODE solver for higher accuracy (needed for r_software calibration)
            times, progress_values, research_stock_values = integrate_progress(
                time_range, initial_progress, initial_research_stock_val, self.data, human_only_params
            )
            # Compute derived metrics via loop
            research_efforts = []
            sw_progress_rates = []
            for i, (t, p, rs) in enumerate(zip(times, progress_values, research_stock_values)):
                state = [p, rs]
                progress_rate, research_effort = progress_rate_at_time(t, state, self.data, human_only_params)
                research_efforts.append(research_effort)
                sw_progress_rate = compute_software_progress_rate(rs, research_effort, human_only_params.r_software)
                sw_progress_rates.append(sw_progress_rate)
            research_efforts = np.array(research_efforts)
            sw_progress_rates = np.array(sw_progress_rates)
        else:
            # Use fast vectorized integration (no ODE solver needed for human-only mode)
            times, progress_values, research_stock_values, research_efforts, sw_progress_rates = integrate_progress_human_only(
                time_range, initial_progress, initial_research_stock_val, self.data, human_only_params
            )

        # Compute progress rates (vectorized)
        training_compute_growth_rate = self.data.get_training_compute_growth_rate(times)
        progress_rates = sw_progress_rates + training_compute_growth_rate
        
        # Anchor stats at params.present_day
        present_day = human_only_params.present_day
        present_day_progress = np.interp(present_day, times, progress_values)
        present_day_progress_rate = np.interp(present_day, times, progress_rates)
        reference_sw_progress_rate = np.interp(cfg.SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR, times, sw_progress_rates)
        present_day_sw_progress_rate = np.interp(present_day, times, sw_progress_rates)
        present_day_research_effort = _log_interp(present_day, times, np.array(research_efforts))
        present_day_research_stock = _log_interp(present_day, times, np.array(research_stock_values))
        # Interpolate human and AI labor at anchor time using log-space when positive
        if np.all(self.data.L_HUMAN > 0):
            present_day_human_labor = _log_interp(present_day, self.data.time, self.data.L_HUMAN)
        else:
            present_day_human_labor = np.interp(present_day, self.data.time, self.data.L_HUMAN)
        if np.all(self.data.inference_compute > 0):
            present_day_inference_compute = _log_interp(present_day, self.data.time, self.data.inference_compute)
        else:
            present_day_inference_compute = np.interp(present_day, self.data.time, self.data.inference_compute)
        if np.all(self.data.experiment_compute > 0):
            present_day_experiment_compute = _log_interp(present_day, self.data.time, self.data.experiment_compute)
        else:
            present_day_experiment_compute = np.interp(present_day, self.data.time, self.data.experiment_compute)
        self.human_only_results = {
            'times': times,
            'progress': progress_values,
            'research_stock': research_stock_values,
            'progress_rates': progress_rates,
            'research_efforts': research_efforts,
            'sw_progress_rates': sw_progress_rates,
            'reference_sw_progress_rate': reference_sw_progress_rate,
            'anchor_stats': {
                'progress': present_day_progress,
                'progress_rate': present_day_progress_rate,
                'sw_progress_rate': present_day_sw_progress_rate,
                'experiment_compute': present_day_experiment_compute,
                'human_labor': present_day_human_labor,
                'inference_compute': present_day_inference_compute,
                'research_effort': present_day_research_effort,
                'research_stock': present_day_research_stock
            },
            'input_time_series': {
                'time': self.data.time,
                'L_HUMAN': self.data.L_HUMAN,
                'inference_compute': self.data.inference_compute,
                'experiment_compute': self.data.experiment_compute,
                'training_compute': self.data.training_compute,
                'training_compute_growth_rate': self.data.get_training_compute_growth_rate(self.data.time)
            }
        }
        
        return times, progress_values, research_stock_values
        
    def get_capability(self, training_compute, software_efficiency):
        effective_compute_val = training_compute + software_efficiency
        return cfg.CAPABILITY_POINTS_PER_OOM * (effective_compute_val - cfg.TRAINING_COMPUTE_REFERENCE_OOMS) + cfg.CAPABILITY_REFERENCE_SCORE
    def compute_progress_trajectory(self, time_range: List[float], initial_progress: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute progress over specified time range with comprehensive metrics
        
        Args:
            time_range: [start_time, end_time]
            initial_progress: Initial progress (defaults to 0.0)
        
        Returns:
            Tuple of (times, cumulative_progress_values, research_stock_values)
        """
        _fn_start_time = time.perf_counter()
        _fn_start_cpu_time = time.process_time()
        _timing_wall: Dict[str, float] = {}
        _timing_cpu: Dict[str, float] = {}
        _timing_meta: Dict[str, Any] = {}
        # first, compute exp capacity params from asymptotes and anchors
        if not self.params.direct_input_exp_cap_ces_params:
            # Use present_day for reference instead of fixed REFERENCE_YEAR (2024.8)
            # Note: Uses human labor only, not accounting for automation
            present_day = self.params.present_day
            ref_exp_compute = _log_interp(present_day, self.data.time, self.data.experiment_compute)
            ref_coding_labor = _log_interp(present_day, self.data.time, self.data.L_HUMAN)
            logger.info(f"ref_exp_compute: {ref_exp_compute}, ref_coding_labor: {ref_coding_labor}")
            rho, alpha_experiment_capacity, experiment_compute_exponent = compute_exp_capacity_params_from_anchors(
                self.params.inf_labor_asymptote,
                self.params.inf_compute_asymptote,
                (cfg.REFERENCE_COMPUTE_CHANGE, self.params.compute_anchor_exp_cap),
                (cfg.REFERENCE_LABOR_CHANGE, self.params.labor_anchor_exp_cap),
                ref_exp_compute, ref_coding_labor,
                self.params.parallel_penalty
            )
            self.params.rho_experiment_capacity = rho
            self.params.alpha_experiment_capacity = alpha_experiment_capacity
            self.params.experiment_compute_exponent = experiment_compute_exponent
            logger.info(f"computed exp capacity params: rho: {rho}, alpha: {alpha_experiment_capacity}, experiment_compute_exponent: {experiment_compute_exponent}")
        else:
            logger.info(f"using direct input exp capacity params: rho: {self.params.rho_experiment_capacity}, alpha: {self.params.alpha_experiment_capacity}, experiment_compute_exponent: {self.params.experiment_compute_exponent}")
        
        # hackily handle doubling_difficulty_growth_factor = 1 case (equivalent to decay_rate = 0)
        if self.params.doubling_difficulty_growth_factor == 1.0:
            logger.info(f"doubling_difficulty_growth_factor is 1.0 (decay_rate = 0), setting to exponential")
            self.params.horizon_extrapolation_type = "exponential"
        
        # NVDA to the moon
        if self.params.plan_a_mode:
            self.initiate_fab_foom(self.params.plan_a_start_time)
        
        # next compute human-only trajectory
        # Need to do this for various reasons, e.g. to auto-fit the METR trajectory you need to the (effective compute, horizon) pairs (we assume no automation)
        # Also if we want to specify current doubling time or gap size in years rather than progress units, need to know how much EC was increased in present day
        # First, run a SHORT human-only trajectory just to get anchor stats for r_software calibration
        # Only need values at reference_year (2024) and present_day (~2025.6), so end 1 year after present_day
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        short_end_time = min(self.params.present_day + 1.0, time_range[1])
        short_time_range = [time_range[0], short_end_time]
        self.compute_human_only_trajectory(short_time_range, initial_progress)
        _timing_wall['human_only_trajectory_first'] = time.perf_counter() - _t0
        _timing_cpu['human_only_trajectory_first'] = time.process_time() - _c0

        # Then, scale r_software so that sw_progress_rate at anchor time is 1
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        logger.info(f"reference year: {cfg.SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR}")
        logger.info(f"reference sw_progress_rate: {self.human_only_results['reference_sw_progress_rate']}")
        logger.info(f"desired software_progress_rate_at_reference_year: {self.params.software_progress_rate_at_reference_year}")
        self.params.r_software = self.params.software_progress_rate_at_reference_year * self.params.r_software/self.human_only_results['reference_sw_progress_rate']
        logger.info(f"new r_software: {self.params.r_software}")
        _timing_wall['human_only_rescale_r_software'] = time.perf_counter() - _t0
        _timing_cpu['human_only_rescale_r_software'] = time.process_time() - _c0

        # Finally, recompute the human-only trajectory with the new r_software
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        human_only_times, human_only_progress, _ = self.compute_human_only_trajectory(time_range, initial_progress)
        _timing_wall['human_only_trajectory_second'] = time.perf_counter() - _t0
        _timing_cpu['human_only_trajectory_second'] = time.process_time() - _c0
        logger.info(f"new reference sw_progress_rate: {self.human_only_results['reference_sw_progress_rate']}")
        _timing_wall['human_only_total'] = (
            _timing_wall['human_only_trajectory_first']
            + _timing_wall['human_only_rescale_r_software']
            + _timing_wall['human_only_trajectory_second']
        )
        _timing_cpu['human_only_total'] = (
            _timing_cpu['human_only_trajectory_first']
            + _timing_cpu['human_only_rescale_r_software']
            + _timing_cpu['human_only_trajectory_second']
        )
        logger.info(f"Timing: human-only trajectory computed in {_timing_wall['human_only_total']:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        
        # Store the various facts about the present day, assuming no automation
        present_day = self.params.present_day
        present_day_progress = self.human_only_results['anchor_stats']['progress']
        present_day_progress_rate = self.human_only_results['anchor_stats']['progress_rate']
        present_day_sw_progress_rate = self.human_only_results['anchor_stats']['sw_progress_rate']
        present_day_research_effort = self.human_only_results['anchor_stats']['research_effort']
        present_day_research_stock = self.human_only_results['anchor_stats']['research_stock']
        present_day_human_labor = self.human_only_results['anchor_stats']['human_labor']
        present_day_inference_compute = self.human_only_results['anchor_stats']['inference_compute']
        present_day_experiment_compute = self.human_only_results['anchor_stats']['experiment_compute']
        logger.info(f"present_day_human_labor: {present_day_human_labor}, present_day_inference_compute: {present_day_inference_compute}")

        # estimate horizon trajectory from METR data using human-only trajectory
        _t_horizon_est_start = time.perf_counter()
        _c_horizon_est_start = time.process_time()
        try:
            anchor_progress_rate = self.human_only_results['anchor_stats']['progress_rate']
            self.estimate_horizon_trajectory(human_only_times, human_only_progress, anchor_progress_rate)
            _dt_horizon_est = time.perf_counter() - _t_horizon_est_start
            _dt_horizon_est_cpu = time.process_time() - _c_horizon_est_start
            _timing_wall['horizon_trajectory_estimation'] = _dt_horizon_est
            _timing_cpu['horizon_trajectory_estimation'] = _dt_horizon_est_cpu
            logger.info(f"Timing: horizon trajectory estimation completed in {_dt_horizon_est:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed to estimate horizon trajectory: {e}")
            _dt_horizon_est = time.perf_counter() - _t_horizon_est_start
            _dt_horizon_est_cpu = time.process_time() - _c_horizon_est_start
            _timing_wall['horizon_trajectory_estimation'] = _dt_horizon_est
            _timing_cpu['horizon_trajectory_estimation'] = _dt_horizon_est_cpu
            _timing_meta['horizon_trajectory_estimation_failed'] = True
            logger.info(f"Timing: horizon trajectory estimation failed after {_dt_horizon_est:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
            self.horizon_trajectory = None

        # Convert AI research taste slope if using "SDs per progress-year" mode
        # This must be done after computing anchor_progress_rate but before the main trajectory
        # Store original slope for display purposes
        self._original_taste_slope = self.params.ai_research_taste_slope
        if self.params.taste_schedule_type == "SDs per progress-year":
            try:
                anchor_progress_rate = self.human_only_results['anchor_stats']['progress_rate']
                if anchor_progress_rate is not None and np.isfinite(anchor_progress_rate) and anchor_progress_rate > 0:
                    # Convert from SDs/progress-year to SDs/progress-unit by multiplying by (1 year / anchor_progress_rate progress-units)
                    # which simplifies to dividing by anchor_progress_rate
                    original_slope = self.params.ai_research_taste_slope
                    converted_slope = original_slope / anchor_progress_rate
                    logger.info(f"Converting taste slope from {original_slope:.6f} SDs/progress-year to {converted_slope:.6f} SDs/progress-unit (anchor rate: {anchor_progress_rate:.6f})")
                    self.params.ai_research_taste_slope = converted_slope
                else:
                    logger.warning(f"Invalid anchor_progress_rate for taste slope conversion: {anchor_progress_rate}")
            except Exception as e:
                if should_reraise(e):
                    raise
                logger.warning(f"Failed to convert taste slope for progress-year mode: {e}")

        # Convert coding automation efficiency slope to OOMs/progress-unit
        # This conversion is always applied (OOMs/progress-year → OOMs/OOM)
        self._original_automation_efficiency_slope = self.params.coding_automation_efficiency_slope
        try:
            anchor_progress_rate = self.human_only_results['anchor_stats']['progress_rate']
            if anchor_progress_rate is not None and np.isfinite(anchor_progress_rate) and anchor_progress_rate > 0:
                # Convert from OOMs/progress-year to OOMs/progress-unit by dividing by anchor_progress_rate
                original_eff_slope = self.params.coding_automation_efficiency_slope
                converted_eff_slope = original_eff_slope / anchor_progress_rate
                logger.info(f"Converting automation efficiency slope from {original_eff_slope:.6f} OOMs/progress-year to {converted_eff_slope:.6f} OOMs/progress-unit (anchor rate: {anchor_progress_rate:.6f})")
                self.params.coding_automation_efficiency_slope = converted_eff_slope
            else:
                logger.warning(f"Invalid anchor_progress_rate for automation efficiency slope conversion: {anchor_progress_rate}")
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed to convert automation efficiency slope: {e}")

        # compute automation fraction at anchor time by solving for lower anchor via AutomationModel
        _t_anchor_solver_start = time.perf_counter()
        _c_anchor_solver_start = time.process_time()
        anchor_aut_frac = solve_lower_anchor_via_automation_model(
            self.params.swe_multiplier_at_present_day,
            float(present_day_progress),
            float(present_day_human_labor),
            float(present_day_inference_compute),
            self.params,
        )
        _dt_anchor_solver = time.perf_counter() - _t_anchor_solver_start
        _dt_anchor_solver_cpu = time.process_time() - _c_anchor_solver_start
        _timing_wall['automation_anchor_solver'] = _dt_anchor_solver
        _timing_cpu['automation_anchor_solver'] = _dt_anchor_solver_cpu
        logger.info(f"Timing: lower anchor via AutomationModel solved in {_dt_anchor_solver:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        logger.info(f"calculated anchor automation fraction (via AM solver): {anchor_aut_frac} from swe_multiplier_at_present_day: {self.params.swe_multiplier_at_present_day} and present_day: {present_day}")
        automation_anchors = {
            present_day_progress: anchor_aut_frac,
            self.params.progress_at_aa: self.params.automation_fraction_at_coding_automation_anchor
        }
        logger.info(f"Automation anchors: {automation_anchors}")
        self.params.automation_anchors = automation_anchors

        # STORE AUTOMATION MODEL INSTANCE IN PARAMS HERE
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        self.params.automation_model = AutomationModel(self.params)
        _timing_wall['automation_model_init'] = time.perf_counter() - _t0
        _timing_cpu['automation_model_init'] = time.process_time() - _c0

        # Use utility function to compute initial conditions with the correct parameters
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        initial_conditions = compute_initial_conditions(self.data, self.params, initial_progress)
        _timing_wall['initial_conditions'] = time.perf_counter() - _t0
        _timing_cpu['initial_conditions'] = time.process_time() - _c0
        initial_research_effort_val = initial_conditions.research_effort
        initial_research_stock_val = initial_conditions.research_stock
        logger.info(f"ACTUAL::: initial_research_stock_val: {initial_research_stock_val}, initial_research_effort_val: {initial_research_effort_val}")

        # Below gives you at each time what is the effectige compute value and what is the research stock value. It runs the whole model.
        # With just time -> effective compute and research stock, you can compute all the other metrics.
        _t_integrate_start = time.perf_counter()
        _c_integrate_start = time.process_time()
        times, progress_values, research_stock_values = integrate_progress(time_range, initial_progress, initial_research_stock_val, self.data, self.params)
        _dt_integrate = time.perf_counter() - _t_integrate_start
        _dt_integrate_cpu = time.process_time() - _c_integrate_start
        _timing_wall['integrate_progress'] = _dt_integrate
        _timing_cpu['integrate_progress'] = _dt_integrate_cpu
        logger.info(f"Timing: integrate_progress completed in {_dt_integrate:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        
        # SIE MODE
        takeoff_start_human_only_stats = None
        if self.params.sos_mode:
            # Find the time when the selected SOS milestone is reached
            sos_milestone = getattr(self.params, 'sos_start_milestone', 'AC')
            full_automation_time = self.find_sos_milestone_time(
                times, progress_values, research_stock_values, sos_milestone
            )

            if full_automation_time is not None:
                takeoff_start_human_only_stats = {
                    'time': full_automation_time,
                    'human_labor': _log_interp(full_automation_time, self.data.time, self.data.L_HUMAN),
                    'inference_compute': _log_interp(full_automation_time, self.data.time, self.data.inference_compute),
                    'experiment_compute': _log_interp(full_automation_time, self.data.time, self.data.experiment_compute),
                    'research_stock': _log_interp(full_automation_time, np.array(times), np.array(self.human_only_results['research_stock'])),
                    'research_effort': _log_interp(full_automation_time, np.array(times), np.array(self.human_only_results['research_efforts'])),
                }
                logger.info(f"SIE mode freeze time ({sos_milestone}): {full_automation_time}")
                self.freeze_time_series_at_time(full_automation_time)
                times, progress_values, research_stock_values = integrate_progress(time_range, initial_progress, initial_research_stock_val, self.data, self.params)
                takeoff_start_research_stock = _log_interp(full_automation_time, times, research_stock_values)
                self.human_only_results['takeoff_start_stats'] = takeoff_start_human_only_stats
            else:
                logger.info(f"SIE mode enabled but milestone {sos_milestone} not reached")
        # Fix for Case 2 anchor horizon blowup: Update anchor_progress after ODE integration
        # This ensures that the horizon at present_day matches the specified present_horizon value
        if (self.horizon_trajectory is not None and 
            hasattr(self, '_horizon_uses_shifted_form') and self._horizon_uses_shifted_form and
            self.params.present_horizon is not None):
            
            # Recompute the actual progress at present_day from the integrated trajectory
            actual_present_day_progress = np.interp(self.params.present_day, times, progress_values)
            
            # Update the horizon trajectory function with the corrected anchor progress
            logger.info(f"Updating present_day progress from fitted value ({present_day_progress:.6f}) to actual integrated value: {actual_present_day_progress:.6f}")
            self._update_horizon_trajectory_anchor(actual_present_day_progress)
        

        # Calculate all metrics in a single pass to avoid redundancy
        logger.info("Computing metrics")
        metrics, timing_wall_cpu, timing_meta_metrics = compute_metrics_loop(
            times=times,
            progress_values=progress_values,
            research_stock_values=research_stock_values,
            data=self.data,
            params=self.params,
            initial_progress=initial_progress,
            present_day_human_labor=present_day_human_labor,
            present_day_inference_compute=present_day_inference_compute,
            present_day_experiment_compute=present_day_experiment_compute,
            present_day_research_stock=present_day_research_stock,
            present_day_sw_progress_rate=present_day_sw_progress_rate,
            taste_distribution=self.taste_distribution,
            horizon_trajectory=self.horizon_trajectory,
            takeoff_start_human_only_stats=takeoff_start_human_only_stats,
            get_capability=self.get_capability,
        )

        # Extract metrics from the returned dictionary
        progress_rates = metrics['progress_rates']
        research_efforts = metrics['research_efforts']
        automation_fractions = metrics['automation_fractions']
        ai_research_tastes = metrics['ai_research_tastes']
        ai_research_taste_sds = metrics['ai_research_taste_sds']
        ai_research_taste_quantiles = metrics['ai_research_taste_quantiles']
        aggregate_research_tastes = metrics['aggregate_research_tastes']
        coding_labors = metrics['coding_labors']
        coding_labors_with_present_resources = metrics['coding_labors_with_present_resources']
        serial_coding_labors = metrics['serial_coding_labors']
        software_progress_rates = metrics['software_progress_rates']
        software_progress_rates_present_resources = metrics['software_progress_rates_present_resources']
        software_efficiency = metrics['software_efficiency']
        human_only_research_efforts = metrics['human_only_research_efforts']
        human_only_software_progress_rates = metrics['human_only_software_progress_rates']
        human_only_progress_rates = metrics['human_only_progress_rates']
        ai_labor_contributions = metrics['ai_labor_contributions']
        human_labor_contributions = metrics['human_labor_contributions']
        ai_coding_labor_multipliers = metrics['ai_coding_labor_multipliers']
        ai_coding_labor_mult_ref_present_day = metrics['ai_coding_labor_mult_ref_present_day']
        serial_coding_labor_multipliers = metrics['serial_coding_labor_multipliers']
        ai_sw_progress_mult_ref_present_day = metrics['ai_sw_progress_mult_ref_present_day']
        takeoff_progress_multipliers = metrics['takeoff_progress_multipliers']
        discounted_exp_compute = metrics['discounted_exp_compute']
        horizon_lengths = metrics['horizon_lengths']
        effective_compute = metrics['effective_compute']
        training_compute = metrics['training_compute']
        capability_ecis = metrics['capability_ecis']
        experiment_capacity = metrics['experiment_capacity']
        exp_cap_mult_with_infinite_labor = metrics['exp_cap_mult_with_infinite_labor']
        exp_cap_mult_with_infinite_compute = metrics['exp_cap_mult_with_infinite_compute']
        ai_only_coding_labors = metrics['ai_only_coding_labors']
        mrts_exp_compute_human_labor = metrics['mrts_exp_compute_human_labor']
        training_compute_growth_rate = metrics['training_compute_growth_rate']
        optimal_training_run_length = metrics['optimal_training_run_length']

        # Update timing information
        _timing_wall['metrics_loop'] = timing_wall_cpu['wall']
        _timing_cpu['metrics_loop'] = timing_wall_cpu['cpu']
        _timing_meta.update(timing_meta_metrics)
        # Calculate time when superhuman coder level is reached
        aa_time = None
        if self.params.progress_at_aa is not None:
            # Find the time when progress reaches progress_at_aa
            sc_progress_target = self.params.progress_at_aa
            
            # Check if SC is reached within the trajectory
            if progress_values[-1] >= sc_progress_target:
                # Find the exact time by interpolation
                if progress_values[0] >= sc_progress_target:
                    # SC level already reached at start
                    aa_time = times[0]
                else:
                    # Interpolate to find when progress crosses sc_progress_target
                    try:
                        aa_time = np.interp(sc_progress_target, progress_values, times)
                    except Exception as e:
                        if should_reraise(e):
                            raise
                        logger.warning(f"Error interpolating SC time: {e}")
                        aa_time = None

                logger.info(f"AC level ({sc_progress_target:.3f}) reached at time {aa_time:.3f}")
            else:
                logger.info(f"AC level ({sc_progress_target:.3f}) not reached within trajectory (final progress: {progress_values[-1]:.3f})")
        
        # Calculate software progress multiplier at SC
        if aa_time is not None:
            self.sc_sw_multiplier = _log_interp(aa_time, times, np.asarray(ai_sw_progress_mult_ref_present_day, dtype=float))
        else:
            sc_sw_multiplier = None
        
        # Compute the time when ai_coding_labor_mult_ref_present_day first reaches the required threshold
        # using exponential (log-space) interpolation between adjacent samples.
        ai2027_sc_time = None
        try:
            ai2027_sc_required_mult = (30 ** (1 / self.params.parallel_penalty)) * 30 * cfg.PARALLEL_LABOR_MULT_BETWEEN_AVERAGE_AND_TOP_FOR_AI2027_SC
            ai2027_sc_time = _find_exponential_crossing_time(
                np.asarray(times, dtype=float),
                np.asarray(ai_coding_labor_mult_ref_present_day, dtype=float),
                float(ai2027_sc_required_mult),
            )
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Error computing ai2027_sc_time: {e}")

        # Calculate progress rate at anchor time
        present_day = self.params.present_day
        anchor_progress_rate = None
        if present_day is not None:
            # Check if anchor time is within our trajectory
            if times[0] <= present_day <= times[-1]:
                anchor_progress_rate = np.interp(present_day, times, progress_rates)
                logger.info(f"Progress rate at anchor time ({present_day:.3f}): {anchor_progress_rate:.6f}")
            else:
                logger.warning(f"Anchor time {present_day:.3f} is outside trajectory range [{times[0]:.3f}, {times[-1]:.3f}]")

        # Compute instantaneous doubling time at the anchor (years)
        instantaneous_anchor_doubling_time_years = None
        try:
            if (self.horizon_trajectory is not None and
                anchor_progress_rate is not None and np.isfinite(anchor_progress_rate) and anchor_progress_rate > 0):
                anchor_progress_value = self.human_only_results['anchor_stats']['progress']
                progress = float(anchor_progress_value)
                # Numerical derivative of ln(horizon) with respect to progress at anchor
                eps = 1e-6 * max(1.0, abs(progress))
                if eps == 0:
                    eps = 1e-6
                H_p = self.horizon_trajectory(progress)
                H_p_eps = self.horizon_trajectory(progress + eps)
                if (np.isfinite(H_p) and np.isfinite(H_p_eps) and H_p > 0 and H_p_eps > 0):
                    dlnH_dprogress = (np.log(H_p_eps) - np.log(H_p)) / eps
                    if np.isfinite(dlnH_dprogress) and dlnH_dprogress > 0:
                        instantaneous_anchor_doubling_time_years = float(np.log(2) / (dlnH_dprogress * anchor_progress_rate))
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed computing instantaneous doubling time at anchor: {e}")

        # Compute AI research taste slope in SD per anchor-progress-year (SD/year at anchor)
        ai_taste_slope_per_anchor_progress_year = None
        ai_taste_slope_per_effective_oom = None
        try:
            if anchor_progress_rate is not None and np.isfinite(anchor_progress_rate):
                if self.params.taste_schedule_type == "SDs per progress-year":
                    # For progress-year mode, use original input value for progress-year display
                    ai_taste_slope_per_anchor_progress_year = float(self._original_taste_slope)
                    # And compute effective OOM value using converted slope
                    ai_taste_slope_per_effective_oom = float(self.params.ai_research_taste_slope)
                else:
                    # For effective OOM mode, compute progress-year display from effective OOM input
                    ai_taste_slope_per_effective_oom = float(self.params.ai_research_taste_slope)
                    ai_taste_slope_per_anchor_progress_year = float(self.params.ai_research_taste_slope) * float(anchor_progress_rate)
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed computing taste slope conversions: {e}")

        # Compute automation efficiency slope in OOMs per anchor-progress-year (OOMs/year at anchor)
        automation_efficiency_slope_per_anchor_progress_year = None
        automation_efficiency_slope_per_effective_oom = None
        try:
            if anchor_progress_rate is not None and np.isfinite(anchor_progress_rate):
                # Always show original input (which is in OOMs/progress-year)
                automation_efficiency_slope_per_anchor_progress_year = float(self._original_automation_efficiency_slope)
                # And the converted value (in OOMs/OOM)
                automation_efficiency_slope_per_effective_oom = float(self.params.coding_automation_efficiency_slope)
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed computing automation efficiency slope conversions: {e}")

        # Compute top taste percentile metrics for display
        top_taste_percentile = self.params.top_percentile
        top_taste_value = None
        top_taste_num_sds = None
        f_multiplier_per_sd = None
        slope_times_log_f = None
        m_over_beta = None
        sie_uplift_doubling_ratio = None

        try:
            # Get taste value at top percentile (e.g., 99.9th percentile if top_percentile = 0.999)
            top_taste_value = self.taste_distribution.get_taste_at_quantile(top_taste_percentile)
            
            # Get how many standard deviations this represents
            top_taste_num_sds = self.taste_distribution.get_sd_of_taste(top_taste_value)
            
            # Compute f: multiplier per standard deviation
            # f = median_to_top_taste_multiplier^(1/num_sds)
            if top_taste_num_sds is not None and np.isfinite(top_taste_num_sds) and top_taste_num_sds > 0:
                f_multiplier_per_sd = self.params.median_to_top_taste_multiplier ** (1.0 / top_taste_num_sds)
                
                # Compute s * log10(f), where s is ai_research_taste_slope (SDs per effective OOM)
                if (ai_taste_slope_per_effective_oom is not None and
                    np.isfinite(ai_taste_slope_per_effective_oom) and
                    f_multiplier_per_sd > 0):
                    slope_times_log_f = ai_taste_slope_per_effective_oom * np.log10(f_multiplier_per_sd)

            # Compute m_over_beta (m/β) and sie_uplift_doubling_ratio
            # m = slope_times_log_f, β = beta_software = 1/r_software
            beta_software_local = 1.0 / self.params.r_software if self.params.r_software != 0 else None
            if (slope_times_log_f is not None and beta_software_local is not None and
                    np.isfinite(slope_times_log_f) and np.isfinite(beta_software_local) and
                    beta_software_local > 0):
                m_over_beta = slope_times_log_f / beta_software_local
                # sie_uplift_doubling_ratio = 2^((β/m) - 1)
                if slope_times_log_f > 0:
                    sie_uplift_doubling_ratio = 2 ** ((beta_software_local / slope_times_log_f) - 1)
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed computing top taste metrics: {e}")

        # RESCALE SOFTWARE EFFICIENCY TO REFLECT PRESENT-DAY BASELINE
        software_efficiency = software_efficiency - np.interp(cfg.TRAINING_COMPUTE_REFERENCE_YEAR, times, software_efficiency)
        training_compute = training_compute - np.interp(cfg.TRAINING_COMPUTE_REFERENCE_YEAR, times, training_compute) + cfg.TRAINING_COMPUTE_REFERENCE_OOMS
        effective_compute = effective_compute - np.interp(cfg.TRAINING_COMPUTE_REFERENCE_YEAR, times, effective_compute) + cfg.TRAINING_COMPUTE_REFERENCE_OOMS
        try:
            capability_ecis = capability_ecis - np.interp(cfg.TRAINING_COMPUTE_REFERENCE_YEAR, times, capability_ecis) + cfg.CAPABILITY_REFERENCE_SCORE
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed rescaling ECI: {e}")

        # Compute doubling times for ai_sw_progress_mult_ref_present_day
        # Doubling time = log(2) / (d log(value) / dt)
        ai_sw_uplift_doubling_times = None
        ai_sw_uplift_doubling_time_of_doubling_times = None
        try:
            sw_uplift = np.asarray(ai_sw_progress_mult_ref_present_day, dtype=float)
            times_arr = np.asarray(times, dtype=float)
            if len(sw_uplift) > 1 and np.all(sw_uplift > 0):
                # Compute log of the uplift values
                log_sw_uplift = np.log(sw_uplift)
                # Compute derivative of log values using central differences
                # d(log(y))/dt at each point
                d_log_dt = np.gradient(log_sw_uplift, times_arr)
                # Doubling time = log(2) / (d log(y) / dt)
                # Handle division by zero and negative growth rates
                with np.errstate(divide='ignore', invalid='ignore'):
                    doubling_times = np.log(2) / d_log_dt
                    # Replace inf/nan with a large number or None-like indicator
                    doubling_times = np.where(np.isfinite(doubling_times) & (doubling_times > 0),
                                              doubling_times, np.nan)
                ai_sw_uplift_doubling_times = doubling_times.tolist()

                # Compute doubling time of the doubling times
                # This measures how fast the doubling times are shrinking
                # Filter to finite positive doubling times
                valid_mask = np.isfinite(doubling_times) & (doubling_times > 0)
                if np.sum(valid_mask) > 2:
                    valid_times = times_arr[valid_mask]
                    valid_dt = doubling_times[valid_mask]
                    log_valid_dt = np.log(valid_dt)
                    d_log_dt_dt = np.gradient(log_valid_dt, valid_times)
                    # Compute the median "halving time" of the doubling times
                    # (negative because doubling times should be decreasing)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        halving_times = -np.log(2) / d_log_dt_dt
                    finite_halving = halving_times[np.isfinite(halving_times) & (halving_times > 0)]
                    if len(finite_halving) > 0:
                        ai_sw_uplift_doubling_time_of_doubling_times = float(np.median(finite_halving))
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed computing ai_sw_uplift doubling times: {e}")

        _t_results_assembly_start = time.perf_counter()
        _c_results_assembly_start = time.process_time()
        # Store comprehensive results
        self.results = {
            'times': times,
            'progress': progress_values,
            'research_stock': research_stock_values,
            'automation_fraction': automation_fractions,
            'ai_research_taste': ai_research_tastes,
            'ai_research_taste_sd': ai_research_taste_sds,
            'ai_research_taste_quantile': ai_research_taste_quantiles,
            'aggregate_research_taste': aggregate_research_tastes,
            'progress_rates': progress_rates,
            'research_efforts': research_efforts,
            'coding_labors': coding_labors,
            'serial_coding_labors': serial_coding_labors,
            'coding_labors_with_present_resources': coding_labors_with_present_resources,
            'software_progress_rates': software_progress_rates,
            'software_efficiency': software_efficiency,
            'human_only_progress_rates': human_only_progress_rates,
            'ai_labor_contributions': ai_labor_contributions,
            'human_labor_contributions': human_labor_contributions,
            'ai_coding_labor_multipliers': ai_coding_labor_multipliers,
            'ai_coding_labor_mult_ref_present_day': ai_coding_labor_mult_ref_present_day,
            'serial_coding_labor_multipliers': serial_coding_labor_multipliers,
            'ai_sw_progress_mult_ref_present_day': ai_sw_progress_mult_ref_present_day,
            'takeoff_progress_multipliers': takeoff_progress_multipliers,
            'discounted_exp_compute': discounted_exp_compute,
            'horizon_lengths': horizon_lengths,
            'effective_compute': effective_compute,
            'capability_ecis': capability_ecis,
            'training_compute': training_compute,
            'experiment_capacity': experiment_capacity,
            'aa_time': aa_time,  # Time when superhuman coder level is reached
            'sc_progress_level': self.params.progress_at_aa,  # Progress level for SC
            'sc_sw_multiplier': self.sc_sw_multiplier if hasattr(self, 'sc_sw_multiplier') else None,  # Software progress multiplier at SC
            'ai2027_sc_time': ai2027_sc_time,  # Time when @AI2027 SC condition is met
            'present_day': present_day,  # Anchor time for manual horizon fitting
            'anchor_progress_rate': anchor_progress_rate,  # Progress rate at anchor time
            'instantaneous_anchor_doubling_time_years': instantaneous_anchor_doubling_time_years,  # Instantaneous doubling time of horizon at anchor (years)
            'ai_research_taste_slope_per_anchor_progress_year': ai_taste_slope_per_anchor_progress_year,  # SD per anchor-progress-year
            'ai_research_taste_slope_per_effective_oom': ai_taste_slope_per_effective_oom,  # SD per effective OOM
            'automation_efficiency_slope_per_anchor_progress_year': automation_efficiency_slope_per_anchor_progress_year,  # OOMs per anchor-progress-year
            'automation_efficiency_slope_per_effective_oom': automation_efficiency_slope_per_effective_oom,  # OOMs per effective OOM
            'input_time_series': {
                'time': self.data.time,
                'L_HUMAN': self.data.L_HUMAN,
                'inference_compute': self.data.inference_compute,
                'experiment_compute': self.data.experiment_compute,
                'training_compute': self.data.training_compute,
                'training_compute_growth_rate': self.data.get_training_compute_growth_rate(self.data.time)
            },
            'exp_capacity_params': {
                'rho': self.params.rho_experiment_capacity,
                'alpha': self.params.alpha_experiment_capacity,
                'experiment_compute_exponent': self.params.experiment_compute_exponent,
            },
            'r_software': self.params.r_software,  # Calibrated r_software value
            'beta_software': 1.0 / self.params.r_software if self.params.r_software != 0 else None,  # Beta (inverse of r_software)
            'top_taste_percentile': top_taste_percentile,  # Top percentile (e.g., 0.01 for 99th percentile)
            'top_taste_num_sds': top_taste_num_sds,  # Number of SDs the top percentile represents
            'f_multiplier_per_sd': f_multiplier_per_sd,  # Multiplier per standard deviation
            'slope_times_log_f': slope_times_log_f,  # s * log10(f), where s is SDs per effective OOM
            'm_over_beta': m_over_beta,  # m/β ratio (>1 means superexponential growth from taste alone)
            'sie_uplift_doubling_ratio': sie_uplift_doubling_ratio,  # 2^((β/m) - 1), approx ratio of successive doubling times
            'ai_sw_uplift_doubling_times': ai_sw_uplift_doubling_times,  # Doubling times for ai_sw_progress_mult_ref_present_day
            'ai_sw_uplift_doubling_time_of_doubling_times': ai_sw_uplift_doubling_time_of_doubling_times,  # Median halving time of doubling times
            'exp_cap_mult_with_infinite_labor': exp_cap_mult_with_infinite_labor,
            'exp_cap_mult_with_infinite_compute': exp_cap_mult_with_infinite_compute,
            'ai_only_coding_labors': ai_only_coding_labors,
            'mrts_exp_compute_human_labor': mrts_exp_compute_human_labor,
            'training_compute_growth_rate': training_compute_growth_rate,
            'optimal_training_run_length': optimal_training_run_length,
        }
        _timing_wall['results_assembly'] = time.perf_counter() - _t_results_assembly_start
        _timing_cpu['results_assembly'] = time.process_time() - _c_results_assembly_start

        _t0 = time.perf_counter()
        _c0 = time.process_time()
        self.results['milestones'] = self.compute_milestones()
        _timing_wall['milestones'] = time.perf_counter() - _t0
        _timing_cpu['milestones'] = time.process_time() - _c0

        # Compute METR MSE
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        metr_mse = self.compute_metr_mse()
        self.results['metr_mse'] = metr_mse
        _timing_wall['metr_mse'] = time.perf_counter() - _t0
        _timing_cpu['metr_mse'] = time.process_time() - _c0

        self.results['timing'] = {
            'wall_time_seconds_total': float(time.perf_counter() - _fn_start_time),
            'cpu_time_seconds_total': float(time.process_time() - _fn_start_cpu_time),
            'wall_time_breakdown_seconds': _timing_wall,
            'cpu_time_breakdown_seconds': _timing_cpu,
            'meta': _timing_meta,
        }

        # logger.info(f"Computed trajectory from {time_range[0]} to {time_range[1]}")
        # logger.info(f"Progress: {progress_values[0]:.3f} -> {progress_values[-1]:.3f}")
        # logger.info(f"Research Stock: {research_stock_values[0]:.3f} -> {research_stock_values[-1]:.3f}")
        # logger.info(f"Automation: {automation_fractions[0]:.3f} -> {automation_fractions[-1]:.3f}")
        if self.params.plan_a_mode:
            # Import here to avoid circular dependency
            from .blacksite import BlacksiteProgressModel
            blacksite_params = copy.deepcopy(self.params)
            blacksite_data = copy.deepcopy(self.data)
            blacksite_params.is_blacksite = True
            self.blacksite_model = BlacksiteProgressModel(blacksite_params, blacksite_data, self)
            self.blacksite_model.compute_progress_trajectory([self.params.blacksite_start_time, self.data.time[-1]])
            self.blacksite_results = self.blacksite_model.results
        if self.params.show_blacksite:
            self.results = self.blacksite_model.results
        return times, progress_values, research_stock_values
    
    def compute_milestones(self):
        """
        Compute milestones for the model.
        """
        # First, determine if SAR experiment selection would be achieved before AC
        # This affects how we calculate TED-AI and ASI targets
        ac_time = self.results.get('aa_time')
        sar_exp_selection_time_unconstrained = None
        sd_at_ac = None

        if ac_time is not None and 'ai_research_taste_sd' in self.results:
            # Find when SAR experiment selection skill would be achieved (ignoring AC requirement)
            sar_exp_selection_time_unconstrained = _find_exponential_crossing_time(
                np.asarray(self.results['times'], dtype=float),
                np.asarray(self.results['ai_research_taste_sd'], dtype=float),
                float(cfg.TOP_RESEARCHER_SD)
            )

            # Get the SD level at AC time
            sd_at_ac = _log_interp(
                ac_time,
                self.results['times'],
                np.asarray(self.results['ai_research_taste_sd'], dtype=float)
            )

        # Determine baseline SD for TED-AI calculation
        # If SAR exp selection would happen before AC, use SD at AC; otherwise use TOP_RESEARCHER_SD
        use_sar_baseline = (
            sar_exp_selection_time_unconstrained is not None and
            ac_time is not None and
            sar_exp_selection_time_unconstrained < ac_time
        )

        if use_sar_baseline:
            ted_ai_baseline_sd = sd_at_ac
        else:
            ted_ai_baseline_sd = cfg.TOP_RESEARCHER_SD

        milestones = {
            'AC': {
                'metric': 'progress',
                'target': self.results['sc_progress_level'],
                'interpolation_type': 'linear',            },
            'AI2027-SC': {
                'metric': 'ai_coding_labor_mult_ref_present_day',
                'target': (30 ** (1 / self.params.parallel_penalty)) * 30 * cfg.PARALLEL_LABOR_MULT_BETWEEN_AVERAGE_AND_TOP_FOR_AI2027_SC,
                'interpolation_type': 'exponential',
            },
            'AIR-5x': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 5,
                'interpolation_type': 'exponential',
                'progress_multiplier': 5,
            },
            'AIR-25x': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 25,
                'interpolation_type': 'exponential',
                'progress_multiplier': 25
            },
            'AIR-250x': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 250,
                'interpolation_type': 'exponential',
                'progress_multiplier': 250
            },
            'AIR-2000x': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 2000,
                'interpolation_type': 'exponential',
                'progress_multiplier': 2000
            },
            'AIR-10000x': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 10000,
                'interpolation_type': 'exponential',
                'progress_multiplier': 10000
            },
            'SAR-level-experiment-selection-skill': {
                'metric': 'ai_research_taste_sd',
                'target': cfg.TOP_RESEARCHER_SD,  # SD of top researcher (99.9th percentile)
                'interpolation_type': 'exponential',
                'requires_ac': True  # SAR also requires achieving AC coding requirement
            },
            'SIAR-level-experiment-selection-skill': {
                'metric': 'ai_research_taste_sd',
                'target': cfg.TOP_RESEARCHER_SD * 3.0,  # 3x the SD of top researcher
                'interpolation_type': 'exponential',
                'requires_ac': True  # SIAR also requires achieving AC coding requirement
            },
            'STRAT-AI': {
                'metric': 'ai_research_taste_sd',
                'target': cfg.TOP_RESEARCHER_SD * (1.0 + self.params.strat_ai_m2b),  # Linearly spaced in SD
                'interpolation_type': 'exponential'
            },
            'TED-AI': {
                'metric': 'ai_research_taste_sd',
                'target': ted_ai_baseline_sd + cfg.TOP_RESEARCHER_SD * self.params.ted_ai_m2b,  # M2B jumps measured in TOP_RESEARCHER_SD units
                'interpolation_type': 'exponential'
            },
            'ASI': {
                'metric': 'ai_research_taste_sd',
                'target': ted_ai_baseline_sd + cfg.TOP_RESEARCHER_SD * (self.params.ted_ai_m2b + 2.0),  # 2 M2Bs above TED-AI
                'interpolation_type': 'exponential'
            },
            'Maximum possible experiment selection skill': {
                'metric': 'ai_research_taste_sd',
                'target': cfg.AI_RESEARCH_TASTE_MAX_SD,
                'interpolation_type': 'exponential'
            }
        }

        for milestone_name, milestone in milestones.items():
            # Special case: Use pre-computed aa_time for AC milestone to ensure consistency
            if milestone_name == 'AC' and self.results.get('aa_time') is not None:
                time = self.results['aa_time']
            elif milestone['interpolation_type'] == 'exponential':
                time = _find_exponential_crossing_time(
                    np.asarray(self.results['times'], dtype=float),
                    np.asarray(self.results[milestone['metric']], dtype=float),
                    float(milestone['target'])
                )
            else:
                time = np.interp(milestone['target'], self.results[milestone['metric']], self.results['times'])
                if np.interp(time, self.results['times'], self.results[milestone['metric']]) < milestone['target'] - 1e-4:
                    time = None

            # Check if milestone also requires AC coding requirement
            if time is not None and milestone.get('requires_ac', False):
                # Get AC time (from aa_time)
                ac_time = self.results.get('aa_time')
                if ac_time is not None:
                    # SAR can only be achieved at the later of the two requirements
                    time = max(time, ac_time)
                else:
                    # If AC is never achieved, SAR cannot be achieved either
                    time = None

            if time is not None:
                milestone['time'] = time
                if 'progress_multiplier' not in milestone:
                    milestone['progress_multiplier'] = _log_interp(milestone['time'], self.results['times'], np.asarray(self.results['ai_sw_progress_mult_ref_present_day'], dtype=float))
                if 'effective_compute_ooms' not in milestone:
                    milestone['effective_compute_ooms'] = np.interp(milestone['time'], self.results['times'], np.asarray(self.results['effective_compute'], dtype=float))
                milestone['research_effort'] = _log_interp(milestone['time'], self.results['times'], np.asarray(self.results['research_efforts'], dtype=float))
                milestone['research_stock'] = _log_interp(milestone['time'], self.results['times'], np.asarray(self.results['research_stock'], dtype=float))
                if self.results['takeoff_progress_multipliers']:
                    milestone['takeoff_progress_multiplier'] = _log_interp(milestone['time'], self.results['times'], np.asarray(self.results['takeoff_progress_multipliers'], dtype=float))
        milestones_list = list(zip(milestones.keys(), milestones.values()))
        # Filter out milestones without time and sort by time
        milestones_list = [(name, milestone) for name, milestone in milestones_list if 'time' in milestone]
        milestones_list.sort(key=lambda x: x[1]['time'])
        for i in range(len(milestones_list) - 1):
            this_RS = milestones_list[i][1]['research_stock']
            next_RS = milestones_list[i+1][1]['research_stock']
            if 'takeoff_start_stats' in self.human_only_results:
                milestones[milestones_list[i][0]]['human_only_years_to_next_milestone'] = (next_RS - this_RS) / self.human_only_results['takeoff_start_stats']['research_effort']

        return milestones
    
    def evaluate_anchor_constraint(self, constraint: AnchorConstraint) -> float:
        """
        Evaluate an anchor constraint using pre-computed metrics.
        
        Args:
            constraint: Constraint specification
            
        Returns:
            Error (model_value - target_value), normalized if target != 0
        """
        if not self.results:
            raise ValueError("No results available. Run compute_progress_trajectory first.")
        
        # Cache frequently accessed arrays to avoid repeated dictionary lookups
        times = self.results['times']
        
        # Extract condition (only one allowed)
        conditions = constraint.conditions
        if len(conditions) != 1:
            raise ValueError("Only one condition is allowed")
        
        condition_key, condition_value = next(iter(conditions.items()))
        
        # Optimize condition processing with direct lookup table
        if condition_key == 'time':
            # Already optimized with binary search
            if condition_value <= times[0]:
                time_idx = 0
            elif condition_value >= times[-1]:
                time_idx = len(times) - 1
            else:
                time_idx = np.searchsorted(times, condition_value)
                if time_idx > 0:
                    # Choose closer of the two neighboring points
                    if abs(times[time_idx] - condition_value) > abs(times[time_idx-1] - condition_value):
                        time_idx = time_idx - 1
        else:
            # For all other conditions, use optimized search
            condition_array = self.results.get(condition_key)
            if condition_array is None:
                # Handle input time series conditions
                if condition_key in ['L_HUMAN', 'inference_compute', 'experiment_compute']:
                    input_series = self.results['input_time_series'][condition_key]
                    # Interpolate to model times for comparison
                    condition_array = np.interp(times, self.results['input_time_series']['time'], input_series)
                else:
                    raise ValueError(f"Unknown condition key: {condition_key}")
            
            # Use optimized closest value search
            time_idx = self._find_closest_index(condition_array, condition_value)
            
            # Check if condition value is reachable (with tolerance)
            if abs(condition_array[time_idx] - condition_value) > 0.01:
                logger.warning(f"{condition_key} never reaches condition value")
                return 0.0

        # Direct array access for target variable (avoid dictionary lookup)
        target_key = constraint.target_variable
        if target_key == 'progress_rate':
            model_value = self.results['progress_rates'][time_idx]
        elif target_key == 'automation_fraction':
            model_value = self.results['automation_fraction'][time_idx]
        elif target_key == 'coding_labor':
            model_value = self.results['coding_labors'][time_idx]
        else:
            raise ValueError(f"Unknown target variable: {target_key}")
        
        # Optimized error calculation
        if constraint.target_value != 0:
            error = (model_value - constraint.target_value) / abs(constraint.target_value)
            # Use faster clipping if available
            error = max(-cfg.RELATIVE_ERROR_CLIP, min(cfg.RELATIVE_ERROR_CLIP, error))
        else:
            error = model_value - constraint.target_value
        
        logger.debug(f"Constraint evaluation: target={target_key}, model_value={model_value:.6f}, target_value={constraint.target_value:.6f}, error={error:.6f}")
        
        return error
    
    def _find_closest_index(self, array: np.ndarray, target_value: float) -> int:
        """
        Optimized method to find closest array index to target value.
        
        Uses binary search when array is sorted, otherwise falls back to linear search.
        Could be further optimized by checking monotonicity once and caching the result.
        """
        # Quick check if array is sorted (monotonic)
        if len(array) > 1:
            is_increasing = np.all(array[1:] >= array[:-1])
            is_decreasing = np.all(array[1:] <= array[:-1])
            
            if is_increasing:
                # Use binary search for increasing array
                idx = np.searchsorted(array, target_value)
                if idx == 0:
                    return 0
                elif idx == len(array):
                    return len(array) - 1
                else:
                    # Choose closer of two neighboring points
                    if abs(array[idx] - target_value) < abs(array[idx-1] - target_value):
                        return idx
                    else:
                        return idx - 1
            elif is_decreasing:
                # Use binary search for decreasing array (search on reversed)
                reversed_array = array[::-1]
                reversed_idx = np.searchsorted(reversed_array, target_value)
                if reversed_idx == 0:
                    return len(array) - 1
                elif reversed_idx == len(reversed_array):
                    return 0
                else:
                    # Convert back to original index and choose closer point
                    idx1 = len(array) - 1 - reversed_idx
                    idx2 = len(array) - reversed_idx
                    if abs(array[idx1] - target_value) < abs(array[idx2] - target_value):
                        return idx1
                    else:
                        return idx2
        
        # Fall back to linear search for non-monotonic arrays
        return np.argmin(np.abs(array - target_value))
    
    def evaluate_all_constraints(self, constraints: List[AnchorConstraint]) -> float:
        """
        Evaluate all anchor constraints and return total weighted error.
        
        Args:
            constraints: List of anchor constraints
            
        Returns:
            Total weighted squared error
        """
        total_error = 0.0
        for i, constraint in enumerate(constraints):
            try:
                error = self.evaluate_anchor_constraint(constraint)
                weighted_error = constraint.weight * (error ** 2)
                total_error += weighted_error
                logger.debug(f"Constraint {i+1}: error={error:.6f}, weighted_error={weighted_error:.6f}")
            except Exception as e:
                if should_reraise(e):
                    raise
                logger.warning(f"Error evaluating constraint {i+1}: {e}")
                # Return high penalty for constraint evaluation failures
                return cfg.OBJECTIVE_FUNCTION_CONFIG['high_penalty']

        return total_error
    
    
    def get_progress_at_time(self, time: float) -> float:
        """
        Get progress value at a specific time by interpolating computed results.
        
        Args:
            time: Time in decimal years
            
        Returns:
            Interpolated progress value
            
        Raises:
            ValueError: If no results are available or time is outside computed range
        """
        if 'times' not in self.results or 'progress' not in self.results:
            raise ValueError("No results available. Run compute_progress_trajectory first.")
        
        times = self.results['times']
        progress_values = self.results['progress']
        
        # Check if time is within the computed range
        if time < times[0] or time > times[-1]:
            raise ValueError(f"Time {time} is outside computed range [{times[0]:.3f}, {times[-1]:.3f}]")
        
        # Use numpy interpolation
        interpolated_progress = np.interp(time, times, progress_values)
        
        logger.debug(f"Interpolated progress at time {time}: {interpolated_progress:.6f}")
        
        return float(interpolated_progress)
    
    def compute_aggregate_taste_with_sd_schedule(self, progress: float, sd_per_progress_unit: float = 0.5) -> float:
        """
        Compute aggregate research taste with AI research taste specified in standard deviations.
        
        This is a convenience method that implements the pattern:
        ai_research_taste = taste_dist.get_taste_at_sd(sd_per_progress_unit * progress)
        aggregate_taste = taste_dist.get_mean_with_floor(ai_research_taste)
        
        Args:
            progress: Current progress level
            sd_per_progress_unit: How many standard deviations AI research taste grows per progress unit
            
        Returns:
            Aggregate research taste with the AI floor applied
            
        Example:
            # AI research taste grows at 0.5 SD per progress unit
            model = ProgressModel(params, data) 
            progress = 10.0
            aggregate_taste = model.compute_aggregate_taste_with_sd_schedule(progress, 0.5)
        """
        ai_research_taste = self.taste_distribution.get_taste_at_sd(sd_per_progress_unit * progress)
        return self.taste_distribution.get_mean_with_floor(ai_research_taste)


# BlacksiteProgressModel moved to blacksite.py


def load_time_series_data(filename: str) -> TimeSeriesData:
    """Load time series data from CSV file"""
    import csv

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    time = np.array([float(row['time']) for row in data])
    L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
    inference_compute = np.array([float(row['inference_compute']) for row in data])
    experiment_compute = np.array([float(row['experiment_compute']) for row in data])
    training_compute = np.array([float(row['training_compute']) for row in data])

    return TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute)
