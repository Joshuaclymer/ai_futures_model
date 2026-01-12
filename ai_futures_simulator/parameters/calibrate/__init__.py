"""
Calibration module for AI Futures Simulator.

This module calibrates model parameters using trajectory computation.
Parameters like r_software, experiment capacity CES params, and automation
anchors are computed from anchor parameters.

Results are cached based on the input parameters to avoid recomputation
when running multiple simulations with the same settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import hashlib
import json
import logging

import numpy as np
import pandas as pd

from world_updaters.software_r_and_d.data_types import TimeSeriesData
from world_updaters.software_r_and_d.automation_model import AutomationModel
from world_updaters.software_r_and_d.taste_distribution import TasteDistribution

from .trajectory import compute_calibration_trajectory, CalibrationParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationInputs:
    """
    Immutable container for parameters that affect calibration.

    These are the "anchor" parameters that determine the calibrated values.
    The frozen=True makes this hashable for caching.
    """
    # Software progress calibration
    software_progress_rate_at_reference_year: float

    # Automation calibration
    swe_multiplier_at_present_day: float
    present_day: float
    progress_at_aa: float

    # Experiment capacity calibration
    inf_labor_asymptote: float
    inf_compute_asymptote: float
    labor_anchor_exp_cap: float
    inv_compute_anchor_exp_cap: float

    # Other params that affect calibration
    parallel_penalty: float
    rho_coding_labor: float
    coding_labor_normalization: float

    # Horizon trajectory parameters
    present_doubling_time: Optional[float] = None
    doubling_difficulty_growth_factor: Optional[float] = None
    ac_time_horizon_minutes: Optional[float] = None
    present_horizon: Optional[float] = None
    horizon_extrapolation_type: Optional[str] = None

    # Slope parameters (these get converted from per-progress-year to per-progress-unit)
    ai_research_taste_slope: Optional[float] = None
    coding_automation_efficiency_slope: Optional[float] = None

    # Simulation start year (affects r_software calibration)
    start_year: float = 2024.0

    def to_cache_key(self) -> str:
        """Generate a cache key from the inputs."""
        data = {
            'sw_rate': self.software_progress_rate_at_reference_year,
            'swe_mult': self.swe_multiplier_at_present_day,
            'present_day': self.present_day,
            'progress_at_aa': self.progress_at_aa,
            'inf_labor': self.inf_labor_asymptote,
            'inf_compute': self.inf_compute_asymptote,
            'labor_anchor': self.labor_anchor_exp_cap,
            'inv_compute_anchor': self.inv_compute_anchor_exp_cap,
            'parallel_penalty': self.parallel_penalty,
            'rho_coding_labor': self.rho_coding_labor,
            'coding_labor_norm': self.coding_labor_normalization,
            'start_year': self.start_year,
            # Horizon trajectory parameters
            'present_doubling_time': self.present_doubling_time,
            'doubling_difficulty_growth_factor': self.doubling_difficulty_growth_factor,
            'ac_time_horizon_minutes': self.ac_time_horizon_minutes,
            'present_horizon': self.present_horizon,
            'horizon_extrapolation_type': self.horizon_extrapolation_type,
            # Slope parameters
            'ai_research_taste_slope': self.ai_research_taste_slope,
            'coding_automation_efficiency_slope': self.coding_automation_efficiency_slope,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()


@dataclass
class CalibratedParameters:
    """Container for calibrated parameter values."""
    # Software progress
    r_software: float

    # Experiment capacity CES
    rho_experiment_capacity: float
    alpha_experiment_capacity: float
    experiment_compute_exponent: float

    # Automation anchors
    automation_anchors: Dict[float, float]

    # Converted slopes (from "per progress-year" to "per progress-unit")
    ai_research_taste_slope: float
    coding_automation_efficiency_slope: float

    # Calculated progress_at_aa (anchor point for ai_research_taste formula)
    # This is computed from the horizon trajectory, not from config
    progress_at_aa: float

    # Initial state at simulation start (2026)
    initial_progress: float
    initial_research_stock: float

    # Present-day baseline values for ai_sw_progress_mult calculation
    # These are interpolated from time series at present_day during calibration
    present_day_human_labor: float = 0.0
    present_day_inference_compute: float = 0.0
    present_day_experiment_compute: float = 0.0

    # Horizon trajectory function: progress -> horizon_length (minutes)
    # This is a callable that maps progress to horizon length
    horizon_trajectory: Optional[callable] = None

    # Year-based trajectories from calibration (for interpolation)
    # These allow the ODE simulator to use calibrated values by year
    trajectory_years: Optional[np.ndarray] = None
    trajectory_progress: Optional[np.ndarray] = None
    trajectory_horizon_lengths: Optional[np.ndarray] = None
    trajectory_automation_fractions: Optional[np.ndarray] = None
    trajectory_ai_sw_progress_mult: Optional[np.ndarray] = None


# Cache for calibration results
_calibration_cache: Dict[str, CalibratedParameters] = {}


def _load_historical_time_series() -> TimeSeriesData:
    """Load the time series data used for calibration.

    Uses input_data.csv which contains both historical data and projections
    out to 2100, matching the original ai-futures-calculator.
    """
    csv_path = Path(__file__).parent.parent / "input_data.csv"
    df = pd.read_csv(csv_path)
    return TimeSeriesData(
        time=df['time'].values,
        L_HUMAN=df['L_HUMAN'].values,
        inference_compute=df['inference_compute'].values,
        experiment_compute=df['experiment_compute'].values,
        training_compute=df['training_compute'].values
    )


def calibrate(inputs: CalibrationInputs) -> CalibratedParameters:
    """
    Calibrate model parameters.

    This runs the calibration process to compute:
    - r_software (scaled to match software_progress_rate_at_reference_year)
    - experiment capacity CES params (rho, alpha, exponent)
    - automation anchors (from swe_multiplier_at_present_day)
    - initial progress and research_stock at start year

    Results are cached based on the input parameters.

    Args:
        inputs: CalibrationInputs containing the anchor parameters

    Returns:
        CalibratedParameters with the computed values
    """
    # Check cache
    cache_key = inputs.to_cache_key()
    if cache_key in _calibration_cache:
        return _calibration_cache[cache_key]

    # Load historical data
    time_series = _load_historical_time_series()

    # Create CalibrationParams with anchor values
    progress_at_aa_value = inputs.progress_at_aa if inputs.progress_at_aa is not None else 100.0

    # Build kwargs for CalibrationParams
    calib_kwargs = {
        'software_progress_rate_at_reference_year': inputs.software_progress_rate_at_reference_year,
        'swe_multiplier_at_present_day': inputs.swe_multiplier_at_present_day,
        'present_day': inputs.present_day,
        'progress_at_aa': progress_at_aa_value,
        'inf_labor_asymptote': inputs.inf_labor_asymptote,
        'inf_compute_asymptote': inputs.inf_compute_asymptote,
        'labor_anchor_exp_cap': inputs.labor_anchor_exp_cap,
        'inv_compute_anchor_exp_cap': inputs.inv_compute_anchor_exp_cap,
        'parallel_penalty': inputs.parallel_penalty,
        'rho_coding_labor': inputs.rho_coding_labor,
        'coding_labor_normalization': inputs.coding_labor_normalization,
    }

    # Add horizon trajectory parameters if provided
    if inputs.present_doubling_time is not None:
        calib_kwargs['present_doubling_time'] = inputs.present_doubling_time
    if inputs.doubling_difficulty_growth_factor is not None:
        calib_kwargs['doubling_difficulty_growth_factor'] = inputs.doubling_difficulty_growth_factor
    if inputs.ac_time_horizon_minutes is not None:
        calib_kwargs['ac_time_horizon_minutes'] = inputs.ac_time_horizon_minutes
    if inputs.present_horizon is not None:
        calib_kwargs['present_horizon'] = inputs.present_horizon
    if inputs.horizon_extrapolation_type is not None:
        calib_kwargs['horizon_extrapolation_type'] = inputs.horizon_extrapolation_type

    # Add slope parameters if provided
    if inputs.ai_research_taste_slope is not None:
        calib_kwargs['ai_research_taste_slope'] = inputs.ai_research_taste_slope
    if inputs.coding_automation_efficiency_slope is not None:
        calib_kwargs['coding_automation_efficiency_slope'] = inputs.coding_automation_efficiency_slope

    params = CalibrationParams(**calib_kwargs)

    # Run calibration trajectory
    # Start from first time point in data (2017.0) to match original ProgressModel
    # The integration should not extrapolate before the data starts
    data_start_year = float(time_series.time[0])
    time_range = [data_start_year, 2100.0]

    trajectory_result = compute_calibration_trajectory(
        params, time_series, time_range, initial_progress=0.0
    )

    # Extract calibrated parameters
    calibrated_params = trajectory_result.params
    times_arr = trajectory_result.times
    progress_arr = trajectory_result.progress
    research_stock_arr = trajectory_result.research_stock

    # Get initial state at simulation start via interpolation
    initial_progress = float(np.interp(inputs.start_year, times_arr, progress_arr))
    initial_research_stock = float(np.interp(inputs.start_year, times_arr, research_stock_arr))

    # Compute present_day baseline values from time series for ai_sw_progress_mult calculation
    present_day = inputs.present_day
    if time_series.can_use_log_L_HUMAN:
        present_day_human_labor = float(np.exp(np.interp(present_day, time_series.time, time_series.log_L_HUMAN)))
    else:
        present_day_human_labor = float(np.interp(present_day, time_series.time, time_series.L_HUMAN))

    if time_series.can_use_log_inference_compute:
        present_day_inference_compute = float(np.exp(np.interp(present_day, time_series.time, time_series.log_inference_compute)))
    else:
        present_day_inference_compute = float(np.interp(present_day, time_series.time, time_series.inference_compute))

    if time_series.can_use_log_experiment_compute:
        present_day_experiment_compute = float(np.exp(np.interp(present_day, time_series.time, time_series.log_experiment_compute)))
    else:
        present_day_experiment_compute = float(np.interp(present_day, time_series.time, time_series.experiment_compute))

    result = CalibratedParameters(
        r_software=calibrated_params.r_software,
        rho_experiment_capacity=calibrated_params.rho_experiment_capacity,
        alpha_experiment_capacity=calibrated_params.alpha_experiment_capacity,
        experiment_compute_exponent=calibrated_params.experiment_compute_exponent,
        automation_anchors=dict(calibrated_params.automation_anchors),
        ai_research_taste_slope=calibrated_params.ai_research_taste_slope,
        coding_automation_efficiency_slope=calibrated_params.coding_automation_efficiency_slope,
        progress_at_aa=trajectory_result.progress_at_aa,
        initial_progress=initial_progress,
        initial_research_stock=initial_research_stock,
        present_day_human_labor=present_day_human_labor,
        present_day_inference_compute=present_day_inference_compute,
        present_day_experiment_compute=present_day_experiment_compute,
        horizon_trajectory=trajectory_result.horizon_trajectory,
        trajectory_years=times_arr,
        trajectory_progress=progress_arr,
        trajectory_horizon_lengths=trajectory_result.horizon_lengths,
        trajectory_automation_fractions=trajectory_result.automation_fractions,
        trajectory_ai_sw_progress_mult=trajectory_result.ai_sw_progress_mult,
    )

    # Cache the result
    _calibration_cache[cache_key] = result

    return result


def calibrate_from_params(software_r_and_d, start_year: float = 2024.0) -> CalibratedParameters:
    """
    Convenience function to calibrate from a SoftwareRAndDParameters object.

    Args:
        software_r_and_d: SoftwareRAndDParameters from simulation_parameters.py
        start_year: The simulation start year (affects r_software calibration)

    Returns:
        CalibratedParameters with the computed values
    """
    r = software_r_and_d

    inputs = CalibrationInputs(
        software_progress_rate_at_reference_year=r.software_progress_rate_at_reference_year,
        swe_multiplier_at_present_day=r.swe_multiplier_at_present_day,
        present_day=r.present_day,
        progress_at_aa=r.progress_at_aa if r.progress_at_aa is not None else 100.0,
        inf_labor_asymptote=r.inf_labor_asymptote,
        inf_compute_asymptote=r.inf_compute_asymptote,
        labor_anchor_exp_cap=r.labor_anchor_exp_cap,
        inv_compute_anchor_exp_cap=r.inv_compute_anchor_exp_cap,
        parallel_penalty=r.parallel_penalty,
        rho_coding_labor=r.rho_coding_labor,
        coding_labor_normalization=r.coding_labor_normalization,
        # Horizon trajectory parameters
        present_doubling_time=getattr(r, 'present_doubling_time', None),
        doubling_difficulty_growth_factor=getattr(r, 'doubling_difficulty_growth_factor', None),
        ac_time_horizon_minutes=getattr(r, 'ac_time_horizon_minutes', None),
        present_horizon=getattr(r, 'present_horizon', None),
        horizon_extrapolation_type=getattr(r, 'horizon_extrapolation_type', None),
        # Slope parameters
        ai_research_taste_slope=getattr(r, 'ai_research_taste_slope', None),
        coding_automation_efficiency_slope=getattr(r, 'coding_automation_efficiency_slope', None),
        start_year=start_year,
    )

    return calibrate(inputs)


def clear_cache():
    """Clear the calibration cache."""
    _calibration_cache.clear()


# Re-export for backwards compatibility
__all__ = [
    'CalibrationInputs',
    'CalibratedParameters',
    'calibrate',
    'calibrate_from_params',
    'clear_cache',
]
