"""
Software R&D world updater.

Updates AI software progress for all AI developers in the world.
Uses the full AI takeoff model logic by importing from the progress_model package.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add progress_model package to path
PROGRESS_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "new_version_of_takeoff_model" / "ai-futures-calculator"
if str(PROGRESS_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(PROGRESS_MODEL_PATH))

import torch
from torch import Tensor

from classes.world.world import World
from classes.world.entities import AISoftwareDeveloper
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.simulation_parameters import SimulationParameters
from parameters.calibrate import calibrate_from_params, CalibratedParameters

# Import key functions from the progress_model package
from progress_model import (
    compute_coding_labor_deprecated as compute_coding_labor,
    compute_research_effort,
    compute_software_progress_rate,
    compute_automation_fraction,
    compute_ai_research_taste,
    compute_aggregate_research_taste,
    Parameters as TakeoffParameters,
    AutomationModel,
    TimeSeriesData,
)
import model_config as cfg

# Load historical time series data for interpolation
# This is the same data used by the reference ProgressModel
_HISTORICAL_CSV_PATH = PROGRESS_MODEL_PATH / "input_data.csv"
_historical_time_series = None

def _load_historical_time_series() -> TimeSeriesData:
    """Load the historical time series data for compute interpolation."""
    global _historical_time_series
    if _historical_time_series is None:
        df = pd.read_csv(_HISTORICAL_CSV_PATH)
        _historical_time_series = TimeSeriesData(
            time=df['time'].values,
            L_HUMAN=df['L_HUMAN'].values,
            inference_compute=df['inference_compute'].values,
            experiment_compute=df['experiment_compute'].values,
            training_compute_growth_rate=df['training_compute_growth_rate'].values,
        )
    return _historical_time_series


class SoftwareRAndD(WorldUpdater):
    """
    Updates AI software progress using the full AI takeoff model.

    This updater imports and uses the core functions from the ai_takeoff_model
    to compute progress rates using:
    - CES production functions for coding labor and experiment capacity
    - Automation fraction based on progress
    - AI research taste dynamics

    Inputs (human_labor, inference_compute, experiment_compute) are obtained
    from the AISoftwareDeveloper's properties:
    - human_labor = human_ai_capability_researchers
    - inference_compute = ai_r_and_d_inference_compute_tpp_h100e
    - experiment_compute = ai_r_and_d_training_compute_tpp_h100e

    Parameters are calibrated at initialization using parameters/calibrate.py.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params
        self.r_and_d = params.software_r_and_d

        # Run calibration to get computed parameter values
        # Pass start_year so r_software is calibrated correctly for the requested time range
        start_year = float(params.settings.simulation_start_year)
        self._calibrated = calibrate_from_params(self.r_and_d, start_year=start_year)

        # Create a TakeoffParameters object with calibrated values
        self._takeoff_params = self._create_takeoff_params()

    def _get_inputs_from_developer(self, dev: AISoftwareDeveloper, current_time: float = None) -> tuple:
        """
        Get R&D inputs from the AISoftwareDeveloper entity.

        Returns (human_labor, inference_compute, experiment_compute, training_compute_growth_rate).

        If current_time is provided:
        - For years in the historical range (up to 2026): interpolate from the historical
          time series CSV (same as reference ProgressModel)
        - For years after 2026: apply growth from 2026 base values using slowdown logic
        """
        import math

        # Load historical time series for interpolation
        ts = _load_historical_time_series()

        # Default values from developer entity (for fallback)
        human_labor_default = float(dev.human_ai_capability_researchers)
        inference_compute_default = float(dev.ai_r_and_d_inference_compute_tpp_h100e)
        experiment_compute_default = float(dev.ai_r_and_d_training_compute_tpp_h100e)
        training_compute_growth_rate_default = float(dev.training_compute_growth_rate)

        if current_time is None:
            return (human_labor_default, inference_compute_default,
                    experiment_compute_default, training_compute_growth_rate_default)

        # Get time bounds from historical data
        time_min = float(ts.time.min())  # ~2012
        time_max = float(ts.time.max())  # ~2026

        # Use log-space interpolation for exponentially growing compute values
        # (matches reference model's approach)
        if ts.can_use_log_inference_compute:
            inference_compute = float(np.exp(np.interp(
                min(current_time, time_max), ts.time, ts.log_inference_compute
            )))
        else:
            inference_compute = float(np.interp(
                min(current_time, time_max), ts.time, ts.inference_compute
            ))

        if ts.can_use_log_experiment_compute:
            experiment_compute = float(np.exp(np.interp(
                min(current_time, time_max), ts.time, ts.log_experiment_compute
            )))
        else:
            experiment_compute = float(np.interp(
                min(current_time, time_max), ts.time, ts.experiment_compute
            ))

        # Interpolate human labor and training_compute_growth_rate as well
        if ts.can_use_log_L_HUMAN:
            human_labor = float(np.exp(np.interp(
                min(current_time, time_max), ts.time, ts.log_L_HUMAN
            )))
        else:
            human_labor = float(np.interp(
                min(current_time, time_max), ts.time, ts.L_HUMAN
            ))

        training_compute_growth_rate = float(np.interp(
            min(current_time, time_max), ts.time, ts.training_compute_growth_rate
        ))

        # For years after the historical data range, apply growth from the end point
        if current_time > time_max:
            elapsed_time = current_time - time_max

            # Get slowdown parameters from compute config
            slowdown_year = 2028.0  # Default
            pre_slowdown_rate = math.log10(4.0)  # ~0.602 OOMs/year (4x/year)
            post_slowdown_rate = 0.25  # From parameters

            # Try to get from parameters
            if self.params.compute is not None:
                us_params = getattr(self.params.compute, 'USComputeParameters', None)
                if us_params is not None:
                    slowdown_year = getattr(us_params, 'slowdown_year', slowdown_year)
                    post_slowdown_rate = getattr(us_params, 'post_slowdown_training_compute_growth_rate', post_slowdown_rate)
                    annual_growth = getattr(us_params, 'us_frontier_project_compute_annual_growth_rate', 4.0)
                    pre_slowdown_rate = math.log10(annual_growth)

            # Compute growth factor with proper slowdown handling
            if current_time < slowdown_year:
                # Before slowdown: simple growth
                growth_factor = 10 ** (pre_slowdown_rate * elapsed_time)
            else:
                # After slowdown: growth at pre_slowdown_rate until slowdown,
                # then at post_slowdown_rate from slowdown to current_time
                pre_slowdown_duration = max(0, slowdown_year - time_max)
                post_slowdown_duration = current_time - max(slowdown_year, time_max)
                growth_factor = (
                    10 ** (pre_slowdown_rate * pre_slowdown_duration) *
                    10 ** (post_slowdown_rate * post_slowdown_duration)
                )

            inference_compute *= growth_factor
            experiment_compute *= growth_factor

        return human_labor, inference_compute, experiment_compute, training_compute_growth_rate

    def _compute_horizon_from_progress(self, progress: float) -> float:
        """
        Compute horizon length from progress using the calibrated horizon_trajectory function.

        The horizon_trajectory function maps progress -> horizon_length (minutes).
        This is the correct way to compute horizon since it depends on progress level,
        not time directly.
        """
        cal = self._calibrated
        if cal.horizon_trajectory is None:
            return float('inf')

        try:
            horizon = float(cal.horizon_trajectory(progress))
            if np.isfinite(horizon) and horizon > 0:
                return horizon
            return float('inf')
        except Exception:
            return float('inf')

    def _interpolate_automation_by_year(self, year: float) -> float:
        """
        Interpolate automation fraction by year using calibration trajectory.
        This uses the ProgressModel's computed automation values, ensuring consistency.
        """
        cal = self._calibrated
        if cal.trajectory_years is None or cal.trajectory_automation_fractions is None:
            return 0.0

        # Clamp year to trajectory range
        min_year = float(cal.trajectory_years[0])
        max_year = float(cal.trajectory_years[-1])
        year_clamped = max(min_year, min(max_year, year))

        # Interpolate automation fraction
        automation = float(np.interp(year_clamped, cal.trajectory_years, cal.trajectory_automation_fractions))
        return automation

    def _interpolate_ai_sw_progress_mult_by_year(self, year: float) -> float:
        """
        Interpolate AI software progress multiplier (ref present day) by year.
        This uses the ProgressModel's computed values, ensuring consistency with the chart.

        Falls back to computing from automation_fraction if calibration data is not available.
        """
        cal = self._calibrated

        # Check if we have valid trajectory data
        has_valid_data = (
            cal.trajectory_years is not None and
            cal.trajectory_ai_sw_progress_mult is not None and
            np.any(cal.trajectory_ai_sw_progress_mult > 0)  # Check for non-zero data
        )

        if has_valid_data:
            # Clamp year to trajectory range
            min_year = float(cal.trajectory_years[0])
            max_year = float(cal.trajectory_years[-1])
            year_clamped = max(min_year, min(max_year, year))

            # Use log-space interpolation since this metric grows exponentially
            log_mult = np.log(np.maximum(cal.trajectory_ai_sw_progress_mult, 1e-10))
            log_interp = float(np.interp(year_clamped, cal.trajectory_years, log_mult))
            mult = float(np.exp(log_interp))
            return max(mult, 1.0)

        # Fallback: compute from automation fraction and research effort ratio
        # This approximates the multiplier based on how much automation boosts R&D
        automation = self._interpolate_automation_by_year(year)
        if automation <= 0:
            return 1.0

        # Simple approximation: multiplier grows with automation fraction
        # At full automation (1.0), assume ~1000x multiplier based on typical model outputs
        # This is a rough heuristic to provide non-flat data when calibration fails
        base_multiplier = 1.0 + automation * 1000.0  # Linear interpolation as fallback
        return base_multiplier

    # Fields that should NOT be passed to TakeoffParameters
    _EXCLUDED_FIELDS = {
        # Mode flags specific to our simulator
        'update_software_progress',
        # Fields computed internally by TakeoffParameters.__post_init__
        'automation_model',
        'taste_distribution',
        'ai_research_taste_at_coding_automation_anchor',
        # Our custom fields for time-varying training compute growth
        'us_frontier_project_compute_growth_rate',
        'slowdown_year',
        'post_slowdown_training_compute_growth_rate',
        # Fields that are calibrated (will be set from _calibrated)
        'r_software',
        'rho_experiment_capacity',
        'alpha_experiment_capacity',
        'experiment_compute_exponent',
        'automation_anchors',
        # Slopes are converted during calibration (from "per progress-year" to "per progress-unit")
        'ai_research_taste_slope',
        'coding_automation_efficiency_slope',
        # progress_at_aa is calculated from horizon trajectory during calibration
        'progress_at_aa',
    }

    @property
    def calibrated(self) -> CalibratedParameters:
        """Get the calibrated parameters."""
        return self._calibrated

    def _create_takeoff_params(self) -> TakeoffParameters:
        """Convert SoftwareRAndDParameters to the original TakeoffParameters format."""
        from dataclasses import fields as dataclass_fields

        r = self.r_and_d
        cal = self._calibrated

        # Map all fields except excluded ones
        kwargs = {}
        for f in dataclass_fields(r):
            if f.name not in self._EXCLUDED_FIELDS:
                value = getattr(r, f.name)
                if value is not None:
                    kwargs[f.name] = value

        # Add calibrated values
        kwargs['r_software'] = cal.r_software
        kwargs['rho_experiment_capacity'] = cal.rho_experiment_capacity
        kwargs['alpha_experiment_capacity'] = cal.alpha_experiment_capacity
        kwargs['experiment_compute_exponent'] = cal.experiment_compute_exponent
        # Use converted slopes (from "per progress-year" to "per progress-unit")
        kwargs['ai_research_taste_slope'] = cal.ai_research_taste_slope
        kwargs['coding_automation_efficiency_slope'] = cal.coding_automation_efficiency_slope
        # Use calibrated progress_at_aa (calculated from horizon trajectory)
        kwargs['progress_at_aa'] = cal.progress_at_aa

        params = TakeoffParameters(**kwargs)

        # Set up automation_anchors from calibration
        params.automation_anchors = cal.automation_anchors.copy()

        # Create the automation model
        params.automation_model = AutomationModel(params)

        return params

    def _tensor_to_float(self, x) -> float:
        """Convert tensor or float to float."""
        if isinstance(x, Tensor):
            return x.item()
        return float(x)

    def _compute_coding_labor(
        self,
        progress: float,
        human_labor: float,
        inference_compute: float,
    ) -> float:
        """
        Compute coding labor using either simple_ces or optimal_ces mode.

        The coding_labor_mode parameter determines which method is used:
        - 'simple_ces': Uses compute_coding_labor() with automation_fraction
        - 'optimal_ces': Uses automation_model.coding_labor_optimal_ces()

        Returns:
            coding_labor value
        """
        params = self._takeoff_params
        coding_labor_mode = getattr(self.r_and_d, 'coding_labor_mode', 'simple_ces')

        if coding_labor_mode == 'optimal_ces':
            # Map model quantities to H, C, E (E is effective compute)
            H = float(human_labor)
            C = float(inference_compute)
            logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * progress)

            automation_model = params.automation_model
            L_opt = automation_model.coding_labor_optimal_ces(H, C, logE, params)

            if L_opt is None:
                # Fall back to simple_ces if optimal_ces fails
                automation_fraction = compute_automation_fraction(progress, params)
                return compute_coding_labor(
                    automation_fraction,
                    inference_compute,
                    human_labor,
                    params.rho_coding_labor,
                    params.parallel_penalty,
                    params.coding_labor_normalization
                )

            # Match units with compute_coding_labor: apply parallel_penalty and normalization
            coding_labor = float((L_opt ** params.parallel_penalty) * params.coding_labor_normalization)
            return coding_labor
        else:
            # simple_ces mode (default)
            automation_fraction = compute_automation_fraction(progress, params)
            return compute_coding_labor(
                automation_fraction,
                inference_compute,
                human_labor,
                params.rho_coding_labor,
                params.parallel_penalty,
                params.coding_labor_normalization
            )

    def _compute_rates(
        self,
        progress: float,
        research_stock: float,
        human_labor: float,
        inference_compute: float,
        experiment_compute: float,
        training_compute_growth_rate: float = 0.0,
    ) -> tuple:
        """
        Compute instantaneous rates using the full takeoff model.

        Returns:
            (d(progress)/dt, d(research_stock)/dt, software_progress_rate)
        """
        params = self._takeoff_params

        # Compute AI research taste
        ai_research_taste = compute_ai_research_taste(progress, params)
        aggregate_research_taste = compute_aggregate_research_taste(
            ai_research_taste, params.taste_distribution
        )

        # Compute coding labor (handles optimal_ces vs simple_ces mode)
        coding_labor = self._compute_coding_labor(progress, human_labor, inference_compute)

        # Compute research effort (experiment capacity * taste)
        research_effort = compute_research_effort(
            experiment_compute,
            coding_labor,
            params.alpha_experiment_capacity,
            params.rho_experiment_capacity,
            params.experiment_compute_exponent,
            aggregate_research_taste
        )

        # Compute software progress rate
        software_progress_rate = compute_software_progress_rate(
            research_stock, research_effort, params.r_software
        )

        # Progress rate = software progress rate + training compute growth rate
        # Training compute growth adds directly to progress (OOMs/year)
        progress_rate = software_progress_rate + training_compute_growth_rate

        # d(research_stock)/dt = research_effort
        research_stock_rate = research_effort
        return progress_rate, research_stock_rate, software_progress_rate

    def contribute_state_derivatives(
        self,
        t: Tensor,
        world: World
    ) -> StateDerivative:
        """
        Compute contribution to d(state)/dt for all AI developers.

        For each developer:
        - d(progress)/dt = software_progress_rate
        - d(research_stock)/dt = research_effort
        """
        # Skip if software progress updates are disabled
        if not self.r_and_d.update_software_progress:
            return StateDerivative.zeros(world)

        d_world = World.zeros(world)
        d_world.current_time = torch.tensor(1.0)

        current_time = self._tensor_to_float(t)

        for dev_id, dev in world.ai_software_developers.items():
            d_dev = d_world.ai_software_developers[dev_id]

            # Get inputs from developer entity (with time-varying compute scaling)
            human_labor, inference_compute, experiment_compute, training_compute_growth_rate = self._get_inputs_from_developer(dev, current_time)

            progress = self._tensor_to_float(dev.ai_software_progress.progress)
            research_stock = self._tensor_to_float(dev.ai_software_progress.research_stock)

            # Compute rates using full model
            progress_rate, research_stock_rate, _ = self._compute_rates(
                progress, research_stock,
                human_labor, inference_compute, experiment_compute,
                training_compute_growth_rate
            )

            d_dev.ai_software_progress.progress = torch.tensor(progress_rate)
            d_dev.ai_software_progress.research_stock = torch.tensor(research_stock_rate)

        return StateDerivative(d_world)

    def set_metric_attributes(
        self,
        t: Tensor,
        world: World
    ) -> World:
        """
        Compute derived metrics for all AI developers.
        """
        # Skip if software progress updates are disabled
        if not self.r_and_d.update_software_progress:
            return world

        current_time = self._tensor_to_float(t)
        params = self._takeoff_params

        for dev_id, dev in world.ai_software_developers.items():
            # Get inputs from developer entity (with time-varying compute scaling)
            human_labor, inference_compute, experiment_compute, training_compute_growth_rate = self._get_inputs_from_developer(dev, current_time)

            # Update the developer entity with interpolated values so they show in frontend charts
            # Use plain floats (not tensors) since these fields are typed as float in the dataclass
            dev.human_ai_capability_researchers = float(human_labor)
            dev.training_compute_growth_rate = float(training_compute_growth_rate)
            # Update the compute properties directly - these are what the frontend reads
            dev.ai_r_and_d_inference_compute_tpp_h100e = float(inference_compute)
            dev.ai_r_and_d_training_compute_tpp_h100e = float(experiment_compute)
            # Also update operating_compute for consistency
            if dev.operating_compute:
                total_compute = inference_compute + experiment_compute
                dev.operating_compute[0].functional_tpp_h100e = torch.tensor(total_compute)
                dev.operating_compute[0].tpp_h100e_including_attrition = torch.tensor(total_compute)

            progress = self._tensor_to_float(dev.ai_software_progress.progress)
            research_stock = self._tensor_to_float(dev.ai_software_progress.research_stock)

            # Compute automation fraction using year-based interpolation
            # This ensures consistency with ProgressModel's values
            automation_fraction = self._interpolate_automation_by_year(current_time)
            dev.ai_software_progress.automation_fraction = torch.tensor(automation_fraction)

            # Compute AI research taste
            ai_research_taste = compute_ai_research_taste(progress, params)
            aggregate_research_taste = compute_aggregate_research_taste(
                ai_research_taste, params.taste_distribution
            )
            dev.ai_software_progress.ai_research_taste = torch.tensor(ai_research_taste)
            dev.ai_software_progress.aggregate_research_taste = torch.tensor(aggregate_research_taste)

            # Compute serial_coding_labor using _compute_coding_labor which handles optimal_ces mode
            serial_coding_labor = self._compute_coding_labor(progress, human_labor, inference_compute)
            dev.ai_software_progress.serial_coding_labor = torch.tensor(serial_coding_labor)

            # Compute coding_labor by reversing the transformation
            # serial_coding_labor = (coding_labor ** parallel_penalty) * normalization
            # => coding_labor = (serial_coding_labor / normalization) ** (1/parallel_penalty)
            if params.parallel_penalty != 0 and params.coding_labor_normalization > 0:
                coding_labor = (serial_coding_labor / params.coding_labor_normalization) ** (1.0 / params.parallel_penalty)
            else:
                coding_labor = serial_coding_labor / params.coding_labor_normalization if params.coding_labor_normalization > 0 else serial_coding_labor
            dev.ai_software_progress.coding_labor = torch.tensor(coding_labor)

            # Compute human-only serial coding labor for multiplier calculation
            human_only_serial_coding_labor = (human_labor ** params.parallel_penalty) * params.coding_labor_normalization
            serial_coding_labor_multiplier = serial_coding_labor / human_only_serial_coding_labor if human_only_serial_coding_labor > 0 else 1.0
            dev.ai_software_progress.serial_coding_labor_multiplier = torch.tensor(serial_coding_labor_multiplier)

            # Compute research effort using serial_coding_labor
            research_effort = compute_research_effort(
                experiment_compute,
                serial_coding_labor,
                params.alpha_experiment_capacity,
                params.rho_experiment_capacity,
                params.experiment_compute_exponent,
                aggregate_research_taste
            )
            dev.ai_software_progress.research_effort = torch.tensor(research_effort)

            # Compute experiment capacity: research_effort / aggregate_research_taste
            experiment_capacity = research_effort / aggregate_research_taste if aggregate_research_taste > 0 else 0.0
            dev.ai_software_progress.experiment_capacity = torch.tensor(experiment_capacity)

            # Software progress rate
            sw_rate = compute_software_progress_rate(
                research_stock, research_effort, params.r_software
            )
            dev.ai_software_progress.software_progress_rate = torch.tensor(sw_rate)

            # Progress rate = software progress rate + training compute growth rate
            overall_progress_rate = sw_rate + training_compute_growth_rate
            dev.ai_software_progress.progress_rate = torch.tensor(overall_progress_rate)

            # AI coding labor multiplier: coding_labor / human_labor
            ai_mult = coding_labor / human_labor if human_labor > 0 and coding_labor > human_labor else 1.0
            dev.ai_software_progress.ai_coding_labor_multiplier = torch.tensor(ai_mult)

            # Horizon length computed from progress (not year interpolation)
            # Horizon is a function of progress level, not time directly
            horizon_length = self._compute_horizon_from_progress(progress)
            if np.isfinite(horizon_length) and horizon_length > 0:
                dev.ai_software_progress.horizon_length = torch.tensor(horizon_length)

            # AI software progress multiplier (ref present day) using year-based interpolation
            ai_sw_mult = self._interpolate_ai_sw_progress_mult_by_year(current_time)
            dev.ai_software_progress.ai_sw_progress_mult_ref_present_day = torch.tensor(ai_sw_mult)

            # Compute software_efficiency = progress - initial_progress
            # This represents the cumulative contribution from software improvements (in OOMs)
            initial_progress = self._tensor_to_float(dev.ai_software_progress.initial_progress) if dev.ai_software_progress.initial_progress is not None else 0.0
            software_efficiency = progress - initial_progress
            dev.ai_software_progress.software_efficiency = torch.tensor(software_efficiency)

            # Compute ai_research_taste_sd from ai_research_taste
            ai_research_taste_sd = params.taste_distribution.get_sd_of_taste(ai_research_taste)
            if not np.isfinite(ai_research_taste_sd):
                ai_research_taste_sd = 0.0
            dev.ai_software_progress.ai_research_taste_sd = torch.tensor(ai_research_taste_sd)

        return world
