"""
Calibration module for AI Futures Simulator.

This module calibrates model parameters using the old ai_takeoff_model's
calibration logic. Parameters like r_software, experiment capacity CES params,
and automation anchors are computed from anchor parameters.

Results are cached based on the input parameters to avoid recomputation
when running multiple simulations with the same settings.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
import hashlib
import json
import os
from contextlib import contextmanager

# Add progress_model package to path (progress_model is in world_updaters/software_r_and_d/)
SOFTWARE_R_AND_D_DIR = Path(__file__).resolve().parent.parent / "world_updaters" / "software_r_and_d"
if str(SOFTWARE_R_AND_D_DIR) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_R_AND_D_DIR))


@contextmanager
def working_directory(path):
    """Context manager to temporarily change working directory."""
    old_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)

import numpy as np
import pandas as pd

from progress_model import (
    ProgressModel,
    Parameters as TakeoffParameters,
    TimeSeriesData,
)


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

    # Horizon trajectory function: progress -> horizon_length (minutes)
    # This is a callable that maps progress to horizon length
    horizon_trajectory: Optional[callable] = None

    # Year-based trajectories from ProgressModel (for interpolation)
    # These allow the ODE simulator to use ProgressModel's computed values by year
    trajectory_years: Optional[np.ndarray] = None
    trajectory_progress: Optional[np.ndarray] = None
    trajectory_horizon_lengths: Optional[np.ndarray] = None
    trajectory_automation_fractions: Optional[np.ndarray] = None
    trajectory_ai_sw_progress_mult: Optional[np.ndarray] = None  # AI software progress multiplier ref present day


# Cache for calibration results
_calibration_cache: Dict[str, CalibratedParameters] = {}


def _load_historical_time_series() -> TimeSeriesData:
    """Load the historical time series data used for calibration."""
    csv_path = Path(__file__).parent / "historical_calibration_data.csv"
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
    Calibrate model parameters using the old ai_takeoff_model's logic.

    This runs the old model's calibration process to compute:
    - r_software (scaled to match software_progress_rate_at_reference_year)
    - experiment capacity CES params (rho, alpha, exponent)
    - automation anchors (from swe_multiplier_at_present_day)
    - initial progress and research_stock at 2026

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

    # Create TakeoffParameters with anchor values
    # Use 100.0 as default for progress_at_aa if None (consistent with app's progress_model.py)
    progress_at_aa_value = inputs.progress_at_aa if inputs.progress_at_aa is not None else 100.0

    # Build kwargs for TakeoffParameters, including optional horizon params
    takeoff_kwargs = {
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
        takeoff_kwargs['present_doubling_time'] = inputs.present_doubling_time
    if inputs.doubling_difficulty_growth_factor is not None:
        takeoff_kwargs['doubling_difficulty_growth_factor'] = inputs.doubling_difficulty_growth_factor
    if inputs.ac_time_horizon_minutes is not None:
        takeoff_kwargs['ac_time_horizon_minutes'] = inputs.ac_time_horizon_minutes
    if inputs.present_horizon is not None:
        takeoff_kwargs['present_horizon'] = inputs.present_horizon
    if inputs.horizon_extrapolation_type is not None:
        takeoff_kwargs['horizon_extrapolation_type'] = inputs.horizon_extrapolation_type

    # Add slope parameters if provided
    if inputs.ai_research_taste_slope is not None:
        takeoff_kwargs['ai_research_taste_slope'] = inputs.ai_research_taste_slope
    if inputs.coding_automation_efficiency_slope is not None:
        takeoff_kwargs['coding_automation_efficiency_slope'] = inputs.coding_automation_efficiency_slope

    params = TakeoffParameters(**takeoff_kwargs)

    # Run the old model's calibration by computing a trajectory
    # IMPORTANT: Always start from 2012 (like the reference app) to ensure we have
    # data before "present_day" (2025.6). This is required for computing metrics like
    # ai_sw_progress_mult_ref_present_day which needs present_day_sw_progress_rate.
    # The initial values at inputs.start_year are extracted later via interpolation.
    #
    # NOTE: We change to the ai-futures-calculator directory during calibration so that
    # benchmark_results.yaml can be found (it's loaded via relative path from cwd).
    calibration_start_year = min(2012.0, inputs.start_year)
    model = ProgressModel(params, time_series)
    time_range = [calibration_start_year, 2050.0]
    with working_directory(SOFTWARE_R_AND_D_DIR):
        times, progress_values, research_stock_values = model.compute_progress_trajectory(
            time_range, initial_progress=0.0
        )

    # Extract calibrated parameters from the model
    calibrated_params = model.params

    # Get initial state at simulation start via interpolation
    # (since we now start from 2012, we need to interpolate to inputs.start_year)
    times_arr = np.array(times)
    progress_arr = np.array(progress_values)
    research_stock_arr = np.array(research_stock_values)
    initial_progress = float(np.interp(inputs.start_year, times_arr, progress_arr))
    initial_research_stock = float(np.interp(inputs.start_year, times_arr, research_stock_arr))

    # Get the horizon_trajectory function from the model (computed during trajectory)
    horizon_trajectory = getattr(model, 'horizon_trajectory', None)

    # Extract year-based trajectories for interpolation
    # This allows the ODE simulator to use ProgressModel's horizon and automation values
    trajectory_years = np.array(times)
    trajectory_progress = np.array(progress_values)

    # Get horizon_lengths, automation_fractions, and ai_sw_progress_mult
    # First try from model.results (if populated), otherwise compute from trajectory functions
    trajectory_horizon_lengths = None
    trajectory_automation_fractions = None
    trajectory_ai_sw_progress_mult = None

    if hasattr(model, 'results') and model.results:
        results = model.results
        if 'horizon_lengths' in results:
            trajectory_horizon_lengths = np.array(results['horizon_lengths'])
        if 'automation_fraction' in results:
            trajectory_automation_fractions = np.array(results['automation_fraction'])
        if 'ai_sw_progress_mult_ref_present_day' in results:
            trajectory_ai_sw_progress_mult = np.array(results['ai_sw_progress_mult_ref_present_day'])

    # If horizon_lengths not in results, compute from horizon_trajectory function
    if trajectory_horizon_lengths is None and horizon_trajectory is not None:
        try:
            trajectory_horizon_lengths = horizon_trajectory(np.array(progress_values))
            # Ensure it's the right shape
            if hasattr(trajectory_horizon_lengths, '__len__'):
                trajectory_horizon_lengths = np.array(trajectory_horizon_lengths)
            else:
                trajectory_horizon_lengths = np.full_like(progress_values, trajectory_horizon_lengths)
        except Exception as e:
            logger.warning(f"Failed to compute horizon lengths from trajectory: {e}")
            trajectory_horizon_lengths = None

    # If automation_fractions not in results, compute from automation_model
    if trajectory_automation_fractions is None and hasattr(calibrated_params, 'automation_model'):
        try:
            auto_model = calibrated_params.automation_model
            if auto_model is not None:
                trajectory_automation_fractions = np.array([
                    auto_model.get_automation_fraction(p) for p in progress_values
                ])
        except Exception as e:
            logger.warning(f"Failed to compute automation fractions: {e}")

    # Get progress_at_aa (calculated from horizon trajectory during compute_progress_trajectory)
    # Use 100.0 as fallback if not set
    progress_at_aa = calibrated_params.progress_at_aa if calibrated_params.progress_at_aa is not None else 100.0

    result = CalibratedParameters(
        r_software=calibrated_params.r_software,
        rho_experiment_capacity=calibrated_params.rho_experiment_capacity,
        alpha_experiment_capacity=calibrated_params.alpha_experiment_capacity,
        experiment_compute_exponent=calibrated_params.experiment_compute_exponent,
        automation_anchors=dict(calibrated_params.automation_anchors),
        # Slopes are converted during compute_progress_trajectory from
        # "per progress-year" to "per progress-unit"
        ai_research_taste_slope=calibrated_params.ai_research_taste_slope,
        coding_automation_efficiency_slope=calibrated_params.coding_automation_efficiency_slope,
        # progress_at_aa is calculated from horizon trajectory (where horizon reaches AC threshold)
        progress_at_aa=progress_at_aa,
        initial_progress=initial_progress,
        initial_research_stock=initial_research_stock,
        horizon_trajectory=horizon_trajectory,
        # Year-based trajectories for interpolation
        trajectory_years=trajectory_years,
        trajectory_progress=trajectory_progress,
        trajectory_horizon_lengths=trajectory_horizon_lengths,
        trajectory_automation_fractions=trajectory_automation_fractions,
        trajectory_ai_sw_progress_mult=trajectory_ai_sw_progress_mult,
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


if __name__ == "__main__":
    # Test the calibration
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from parameters.simulation_parameters import ModelParameters

    config_path = Path(__file__).parent / "modal_parameters.yaml"
    model_params = ModelParameters.from_yaml(config_path)
    params = model_params.sample()

    print("Running calibration...")
    result = calibrate_from_params(params.software_r_and_d)

    print("\nCalibrated Parameters:")
    print(f"  r_software: {result.r_software}")
    print(f"  rho_experiment_capacity: {result.rho_experiment_capacity}")
    print(f"  alpha_experiment_capacity: {result.alpha_experiment_capacity}")
    print(f"  experiment_compute_exponent: {result.experiment_compute_exponent}")
    print(f"  automation_anchors: {result.automation_anchors}")
    print(f"  initial_progress: {result.initial_progress}")
    print(f"  initial_research_stock: {result.initial_research_stock}")

    # Test caching
    print("\nRunning calibration again (should be cached)...")
    result2 = calibrate_from_params(params.software_r_and_d)
    print(f"Cache hit: {result is result2}")
