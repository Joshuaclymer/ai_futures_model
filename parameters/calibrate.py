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
from typing import Dict, Optional, Tuple
from functools import lru_cache
import hashlib
import json

# Add ai_takeoff_model to path
AI_TAKEOFF_MODEL_PATH = Path(__file__).resolve().parents[1] / "old" / "update_model_state" / "update_ai_software_r_and_d" / "ai_takeoff_model"
if str(AI_TAKEOFF_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(AI_TAKEOFF_MODEL_PATH))

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

    # Initial state at simulation start (2026)
    initial_progress: float
    initial_research_stock: float


# Cache for calibration results
_calibration_cache: Dict[str, CalibratedParameters] = {}


def _load_historical_time_series() -> TimeSeriesData:
    """Load the historical time series data used for calibration."""
    csv_path = AI_TAKEOFF_MODEL_PATH / "inputs" / "new_simulator_default.csv"
    df = pd.read_csv(csv_path)
    return TimeSeriesData(
        time=df['time'].values,
        L_HUMAN=df['L_HUMAN'].values,
        inference_compute=df['inference_compute'].values,
        experiment_compute=df['experiment_compute'].values,
        training_compute_growth_rate=df['training_compute_growth_rate'].values
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
    params = TakeoffParameters(
        software_progress_rate_at_reference_year=inputs.software_progress_rate_at_reference_year,
        swe_multiplier_at_present_day=inputs.swe_multiplier_at_present_day,
        present_day=inputs.present_day,
        progress_at_aa=inputs.progress_at_aa,
        inf_labor_asymptote=inputs.inf_labor_asymptote,
        inf_compute_asymptote=inputs.inf_compute_asymptote,
        labor_anchor_exp_cap=inputs.labor_anchor_exp_cap,
        inv_compute_anchor_exp_cap=inputs.inv_compute_anchor_exp_cap,
        parallel_penalty=inputs.parallel_penalty,
        rho_coding_labor=inputs.rho_coding_labor,
        coding_labor_normalization=inputs.coding_labor_normalization,
    )

    # Run the old model's calibration by computing a trajectory
    model = ProgressModel(params, time_series)
    time_range = [2012.0, 2050.0]
    times, progress_values, research_stock_values = model.compute_progress_trajectory(
        time_range, initial_progress=0.0
    )

    # Extract calibrated parameters from the model
    calibrated_params = model.params

    # Get initial state at 2026
    idx_2026 = np.argmin(np.abs(times - 2026.0))
    initial_progress = float(progress_values[idx_2026])
    initial_research_stock = float(research_stock_values[idx_2026])

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
        initial_progress=initial_progress,
        initial_research_stock=initial_research_stock,
    )

    # Cache the result
    _calibration_cache[cache_key] = result

    return result


def calibrate_from_params(software_r_and_d) -> CalibratedParameters:
    """
    Convenience function to calibrate from a SoftwareRAndDParameters object.

    Args:
        software_r_and_d: SoftwareRAndDParameters from simulation_parameters.py

    Returns:
        CalibratedParameters with the computed values
    """
    r = software_r_and_d

    inputs = CalibrationInputs(
        software_progress_rate_at_reference_year=r.software_progress_rate_at_reference_year,
        swe_multiplier_at_present_day=r.swe_multiplier_at_present_day,
        present_day=r.present_day,
        progress_at_aa=r.progress_at_aa if r.progress_at_aa is not None else 10.0,
        inf_labor_asymptote=r.inf_labor_asymptote,
        inf_compute_asymptote=r.inf_compute_asymptote,
        labor_anchor_exp_cap=r.labor_anchor_exp_cap,
        inv_compute_anchor_exp_cap=r.inv_compute_anchor_exp_cap,
        parallel_penalty=r.parallel_penalty,
        rho_coding_labor=r.rho_coding_labor,
        coding_labor_normalization=r.coding_labor_normalization,
    )

    return calibrate(inputs)


def clear_cache():
    """Clear the calibration cache."""
    _calibration_cache.clear()


if __name__ == "__main__":
    # Test the calibration
    from simulation_parameters import ModelParameters

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
