"""
Software R&D world updater.

Updates AI software progress for all AI developers in the world.
Uses the full AI takeoff model logic by importing from ai_takeoff_model.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add new version of takeoff model to path (contains the progress_model package)
# Path: ai_futures_simulator/ai_futures_simulator/world_updaters/software_r_and_d.py
# Need to go up 3 levels to reach ai_futures_simulator/ then into new_version_of_takeoff_model/
AI_FUTURES_CALCULATOR_PATH = Path(__file__).resolve().parent.parent.parent / "new_version_of_takeoff_model" / "ai-futures-calculator"
if str(AI_FUTURES_CALCULATOR_PATH) not in sys.path:
    sys.path.insert(0, str(AI_FUTURES_CALCULATOR_PATH))

import torch
from torch import Tensor

from classes.world.world import World
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.simulation_parameters import SimulationParameters
from parameters.calibrate import calibrate_from_params, CalibratedParameters

# Import key functions from the new progress_model package
from progress_model import (
    compute_coding_labor_deprecated as compute_coding_labor,
    compute_research_effort,
    compute_software_progress_rate,
    compute_overall_progress_rate,
    compute_automation_fraction,
    compute_ai_research_taste,
    compute_aggregate_research_taste,
    Parameters as TakeoffParameters,
    AutomationModel,
    TimeSeriesData,
)
import model_config as cfg


def _load_time_series_data() -> TimeSeriesData:
    """Load time-varying inputs from the same CSV used by ai-futures-calculator."""
    csv_path = AI_FUTURES_CALCULATOR_PATH / "input_data.csv"
    df = pd.read_csv(csv_path)
    return TimeSeriesData(
        time=df['time'].values,
        L_HUMAN=df['L_HUMAN'].values,
        inference_compute=df['inference_compute'].values,
        experiment_compute=df['experiment_compute'].values,
        training_compute_growth_rate=df['training_compute_growth_rate'].values
    )


class SoftwareRAndD(WorldUpdater):
    """
    Updates AI software progress using the full AI takeoff model.

    This updater imports and uses the core functions from the ai_takeoff_model
    to compute progress rates using:
    - CES production functions for coding labor and experiment capacity
    - Automation fraction based on progress
    - AI research taste dynamics
    - Training compute growth with slowdown

    Parameters are calibrated at initialization using parameters/calibrate.py.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params
        self.r_and_d = params.software_r_and_d

        # Load time-varying inputs (same as old backend)
        self._time_series = _load_time_series_data()

        # Run calibration to get computed parameter values
        # Pass start_year so r_software is calibrated correctly for the requested time range
        start_year = float(params.settings.simulation_start_year)
        self._calibrated = calibrate_from_params(self.r_and_d, start_year=start_year)

        # Create a TakeoffParameters object with calibrated values
        self._takeoff_params = self._create_takeoff_params()

    def _interpolate_inputs(self, t: float) -> tuple:
        """
        Interpolate time-varying inputs at time t.
        Uses log-space interpolation for exponentially growing quantities
        to match the reference model's behavior and prevent scalloping.

        Returns (human_labor, inference_compute, experiment_compute, training_compute_growth_rate).
        """
        ts = self._time_series

        # Use log-space interpolation for exponentially growing quantities
        # This prevents scalloping on log plots and handles exponential growth better
        if ts.can_use_log_L_HUMAN:
            human_labor = float(np.exp(np.interp(t, ts.time, ts.log_L_HUMAN)))
        else:
            human_labor = float(np.interp(t, ts.time, ts.L_HUMAN))

        if ts.can_use_log_inference_compute:
            inference_compute = float(np.exp(np.interp(t, ts.time, ts.log_inference_compute)))
        else:
            inference_compute = float(np.interp(t, ts.time, ts.inference_compute))

        if ts.can_use_log_experiment_compute:
            experiment_compute = float(np.exp(np.interp(t, ts.time, ts.log_experiment_compute)))
        else:
            experiment_compute = float(np.interp(t, ts.time, ts.experiment_compute))

        # Training compute growth rate is not exponential, use linear interpolation
        training_compute_growth_rate = float(np.interp(t, ts.time, ts.training_compute_growth_rate))

        return human_labor, inference_compute, experiment_compute, training_compute_growth_rate

    def _interpolate_horizon_by_year(self, year: float) -> float:
        """
        Interpolate horizon length by year using calibration trajectory.
        This uses the ProgressModel's computed horizon values, ensuring consistency.
        """
        cal = self._calibrated
        if cal.trajectory_years is None or cal.trajectory_horizon_lengths is None:
            return float('inf')

        # Clamp year to trajectory range
        min_year = float(cal.trajectory_years[0])
        max_year = float(cal.trajectory_years[-1])
        year_clamped = max(min_year, min(max_year, year))

        # Interpolate horizon length
        horizon = float(np.interp(year_clamped, cal.trajectory_years, cal.trajectory_horizon_lengths))
        return horizon

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
        training_compute_growth_rate: float,
    ) -> tuple:
        """
        Compute instantaneous rates using the full takeoff model.

        Returns:
            (d(progress)/dt, d(research_stock)/dt)
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

        # Compute overall progress rate (software + training compute)
        progress_rate = compute_overall_progress_rate(
            software_progress_rate, training_compute_growth_rate
        )

        # d(research_stock)/dt = research_effort
        research_stock_rate = research_effort
        return progress_rate, research_stock_rate

    def contribute_state_derivatives(
        self,
        t: Tensor,
        world: World
    ) -> StateDerivative:
        """
        Compute contribution to d(state)/dt for all AI developers.

        For each developer:
        - d(progress)/dt = overall_progress_rate (training + software)
        - d(research_stock)/dt = research_effort
        - d(log_compute)/dt = log_compute_growth_rate
        """
        d_world = World.zeros(world)
        d_world.current_time = torch.tensor(1.0)

        current_time = self._tensor_to_float(t)

        # Get time-varying inputs from interpolation (same as old backend)
        human_labor, inference_compute, experiment_compute, training_compute_growth_rate = \
            self._interpolate_inputs(current_time)

        for dev_id, dev in world.ai_software_developers.items():
            d_dev = d_world.ai_software_developers[dev_id]

            progress = self._tensor_to_float(dev.ai_software_progress.progress)
            research_stock = self._tensor_to_float(dev.ai_software_progress.research_stock)

            # Compute rates using full model
            progress_rate, research_stock_rate = self._compute_rates(
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
        current_time = self._tensor_to_float(t)
        params = self._takeoff_params

        # Get time-varying inputs from interpolation (same as old backend)
        human_labor, inference_compute, experiment_compute, training_compute_growth_rate = \
            self._interpolate_inputs(current_time)

        for dev_id, dev in world.ai_software_developers.items():
            progress = self._tensor_to_float(dev.ai_software_progress.progress)
            research_stock = self._tensor_to_float(dev.ai_software_progress.research_stock)

            # Store input time series values in the world state
            dev.ai_software_progress.human_labor = torch.tensor(human_labor)
            dev.ai_software_progress.inference_compute = torch.tensor(inference_compute)
            dev.ai_software_progress.experiment_compute = torch.tensor(experiment_compute)

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
            # This matches how the reference model computes serial_coding_labor in metrics_computation.py:
            # - optimal_ces mode: L_opt = automation_model.coding_labor_optimal_ces(H, C, logE, params)
            #                     serial_coding_labor = (L_opt ** parallel_penalty) * normalization
            # - simple_ces mode: uses compute_coding_labor() directly
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
            # In reference: human_only_serial_coding_labor = L_HUMAN ** parallel_penalty
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

            # Overall progress rate
            overall_rate = compute_overall_progress_rate(sw_rate, training_compute_growth_rate)
            dev.ai_software_progress.progress_rate = torch.tensor(overall_rate)

            # AI coding labor multiplier: coding_labor / human_labor
            # This measures how much AI amplifies the effective coding labor vs human-only
            ai_mult = coding_labor / human_labor if human_labor > 0 and coding_labor > human_labor else 1.0
            dev.ai_software_progress.ai_coding_labor_multiplier = torch.tensor(ai_mult)

            # Horizon length using year-based interpolation
            # This ensures consistency with ProgressModel's values
            horizon_length = self._interpolate_horizon_by_year(current_time)
            if np.isfinite(horizon_length) and horizon_length > 0:
                dev.ai_software_progress.horizon_length = torch.tensor(horizon_length)

            # AI software progress multiplier (ref present day) using year-based interpolation
            # This is the metric shown on the "Takeoff Period: AI Software R&D Uplift" chart
            ai_sw_mult = self._interpolate_ai_sw_progress_mult_by_year(current_time)
            dev.ai_software_progress.ai_sw_progress_mult_ref_present_day = torch.tensor(ai_sw_mult)

            # Store training_compute_growth_rate for use in computing training_compute at API layer
            dev.ai_software_progress.training_compute_growth_rate = torch.tensor(training_compute_growth_rate)

            # Compute ai_research_taste_sd from ai_research_taste
            # This is needed for computing milestones (SAR, SIAR, TED-AI, ASI)
            ai_research_taste_sd = params.taste_distribution.get_sd_of_taste(ai_research_taste)
            if not np.isfinite(ai_research_taste_sd):
                ai_research_taste_sd = 0.0
            dev.ai_software_progress.ai_research_taste_sd = torch.tensor(ai_research_taste_sd)

        return world
