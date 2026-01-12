"""
Software R&D world updater.

Updates AI software progress for all AI developers in the world.
"""

import numpy as np

import torch
from torch import Tensor

from typing import TYPE_CHECKING

from classes.world.world import World
from classes.world.entities import AISoftwareDeveloper
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.classes import SimulationParameters

if TYPE_CHECKING:
    from parameters.calibrate import CalibratedParameters

# Import from local modules (same directory)
from .ces_functions import compute_coding_labor_deprecated as compute_coding_labor
from .progress_rate import (
    compute_research_effort,
    compute_software_progress_rate,
    compute_automation_fraction,
    compute_ai_research_taste,
    compute_aggregate_research_taste,
)
from .automation_model import AutomationModel
from .taste_distribution import TasteDistribution
from . import model_config as cfg


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

    Requires pre-calibrated parameters (software_r_and_d_calibrated) which are
    computed by ModelParameters.sample() before simulation begins.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params
        self.r_and_d = params.software_r_and_d

        # Use pre-calibrated parameters from SimulationParameters
        # Calibration is performed in ModelParameters.sample() before simulation
        if params.software_r_and_d_calibrated is None:
            raise ValueError(
                "SoftwareRAndD requires calibrated parameters. "
                "Ensure ModelParameters.sample() or sample_modal() was used to create params."
            )
        self._calibrated = params.software_r_and_d_calibrated

        # Create AutomationModel directly with individual params
        self._automation_model = AutomationModel(
            automation_interp_type=self.r_and_d.automation_interp_type,
            automation_anchors=self._calibrated.automation_anchors,
            automation_logistic_asymptote=self.r_and_d.automation_logistic_asymptote or 1.0,
        )

        # Create TasteDistribution directly with individual params
        self._taste_distribution = TasteDistribution(
            top_percentile=self.r_and_d.top_percentile or 0.99,
            median_to_top_gap=self.r_and_d.median_to_top_taste_multiplier or 10.0,
            taste_limit_m2b=self.r_and_d.taste_limit or 8.0,
            taste_limit_smoothing=self.r_and_d.taste_limit_smoothing or 0.51,
        )

        # Store present_day values for computing ai_sw_progress_mult_ref_present_day
        # These are fixed at present_day and used to isolate the effect of automation improvements
        self._init_present_day_values()

    def _init_present_day_values(self):
        """
        Initialize present_day values used for computing ai_sw_progress_mult_ref_present_day.

        These values are fixed at present_day and represent the baseline resources.
        The multiplier isolates the effect of automation/taste improvements by using
        present_day resources with the current progress level.

        Values are read from calibrated parameters (computed during calibration from time series).
        """
        r = self.r_and_d
        cal = self._calibrated
        present_day = r.present_day

        # Read present_day baseline values from calibrated parameters
        # These were computed during calibration by interpolating from time series
        self._present_day_human_labor = cal.present_day_human_labor
        self._present_day_inference_compute = cal.present_day_inference_compute
        self._present_day_experiment_compute = cal.present_day_experiment_compute

        # Compute present_day serial_coding_labor (human-only, no AI automation)
        # At present_day, we use the base coding labor without AI amplification
        self._present_day_serial_coding_labor = (
            (self._present_day_human_labor ** r.parallel_penalty) * r.coding_labor_normalization
        )

        # Get present_day research_stock from calibration (this was computed during trajectory calibration)
        # We need to interpolate from the trajectory if available
        if cal.trajectory_years is not None and cal.trajectory_progress is not None:
            # Get present_day progress from trajectory
            present_day_progress = float(np.interp(present_day, cal.trajectory_years, cal.trajectory_progress))
        else:
            present_day_progress = cal.initial_progress

        # For research_stock, we use the calibrated initial_research_stock scaled by progress
        # Actually, we need to compute present_day sw_rate to get the baseline
        # Use a simple approximation: present_day research_stock ≈ initial_research_stock
        self._present_day_research_stock = cal.initial_research_stock

        # Compute present_day software progress rate (human-only baseline)
        # This is the baseline that the multiplier is relative to
        present_day_ai_taste = compute_ai_research_taste(
            present_day_progress,
            self._taste_distribution,
            cal.ai_research_taste_slope,
            cal.progress_at_aa,
            r.ai_research_taste_at_coding_automation_anchor_sd,
        )
        present_day_agg_taste = compute_aggregate_research_taste(present_day_ai_taste, self._taste_distribution)

        present_day_research_effort = compute_research_effort(
            self._present_day_experiment_compute,
            self._present_day_serial_coding_labor,
            cal.alpha_experiment_capacity,
            cal.rho_experiment_capacity,
            cal.experiment_compute_exponent,
            present_day_agg_taste
        )

        self._present_day_sw_progress_rate = compute_software_progress_rate(
            self._present_day_research_stock, present_day_research_effort, cal.r_software
        )

    def _compute_ai_sw_progress_mult_ref_present_day(self, progress: float) -> float:
        """
        Compute ai_sw_progress_mult_ref_present_day dynamically based on current progress.

        This metric shows how much faster software progress would be at the current
        automation/taste level compared to present_day, holding resources constant.

        The computation uses:
        - Present_day resources (human_labor, inference_compute, experiment_compute, research_stock)
        - Current progress level (which determines automation fraction and taste)

        Returns:
            sw_rate_with_current_automation / present_day_sw_rate
        """
        r = self.r_and_d
        cal = self._calibrated

        # Compute current automation fraction and taste at current progress
        ai_research_taste = compute_ai_research_taste(
            progress,
            self._taste_distribution,
            cal.ai_research_taste_slope,
            cal.progress_at_aa,
            r.ai_research_taste_at_coding_automation_anchor_sd,
        )
        aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste, self._taste_distribution)

        # Compute serial_coding_labor with present_day resources but current automation level
        serial_coding_labor_with_present_resources = self._compute_coding_labor(
            progress,
            self._present_day_human_labor,
            self._present_day_inference_compute
        )

        # Compute research effort with present_day experiment_compute and current coding labor
        research_effort_present_resources = compute_research_effort(
            self._present_day_experiment_compute,
            serial_coding_labor_with_present_resources,
            cal.alpha_experiment_capacity,
            cal.rho_experiment_capacity,
            cal.experiment_compute_exponent,
            aggregate_research_taste
        )

        # Compute software progress rate with present_day research_stock
        sw_rate_present_resources = compute_software_progress_rate(
            self._present_day_research_stock, research_effort_present_resources, cal.r_software
        )

        # Return the multiplier relative to present_day
        if self._present_day_sw_progress_rate > 0:
            return sw_rate_present_resources / self._present_day_sw_progress_rate
        return 1.0

    def _get_inputs_from_developer(self, dev: AISoftwareDeveloper, current_time: float = None) -> tuple:
        """
        Get R&D inputs from the developer entity's properties.

        Returns (human_labor, inference_compute, experiment_compute, training_compute_growth_rate).

        Uses values that were set by the compute and researcher updaters:
        - human_labor from human_ai_capability_researchers (set by AISoftwareDeveloperResearcherUpdater)
        - inference_compute from ai_r_and_d_inference_compute_tpp_h100e (set by AISoftwareDeveloperComputeUpdater)
        - experiment_compute from ai_r_and_d_training_compute_tpp_h100e (set by AISoftwareDeveloperComputeUpdater)
        - training_compute_growth_rate from training_compute_growth_rate (set by AISoftwareDeveloperComputeUpdater)
        """
        # Get values from developer entity (set by other updaters)
        human_labor = float(dev.human_ai_capability_researchers)
        inference_compute = float(dev.ai_r_and_d_inference_compute_tpp_h100e)
        experiment_compute = float(dev.ai_r_and_d_training_compute_tpp_h100e)
        training_compute_growth_rate = float(dev.training_compute_growth_rate)

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

    @property
    def calibrated(self) -> "CalibratedParameters":
        """Get the calibrated parameters."""
        return self._calibrated

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
        r = self.r_and_d
        cal = self._calibrated
        coding_labor_mode = getattr(r, 'coding_labor_mode', 'simple_ces')

        if coding_labor_mode == 'optimal_ces':
            # Map model quantities to H, C, E (E is effective compute)
            H = float(human_labor)
            C = float(inference_compute)
            logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * progress)

            # Create a simple namespace with params needed by coding_labor_optimal_ces
            class OptimalCESParams:
                pass
            frontier_params = OptimalCESParams()
            frontier_params.rho_coding_labor = r.rho_coding_labor
            frontier_params.coding_automation_efficiency_slope = cal.coding_automation_efficiency_slope
            frontier_params.optimal_ces_eta_init = r.optimal_ces_eta_init or 1.0
            frontier_params.optimal_ces_grid_size = r.optimal_ces_grid_size or 512
            frontier_params.optimal_ces_frontier_tail_eps = r.optimal_ces_frontier_tail_eps
            frontier_params.optimal_ces_frontier_cap = r.optimal_ces_frontier_cap
            frontier_params.max_serial_coding_labor_multiplier = r.max_serial_coding_labor_multiplier

            L_opt = self._automation_model.coding_labor_optimal_ces(H, C, logE, frontier_params)

            if L_opt is None:
                # Fall back to simple_ces if optimal_ces fails
                automation_fraction = compute_automation_fraction(progress, self._automation_model)
                return compute_coding_labor(
                    automation_fraction,
                    inference_compute,
                    human_labor,
                    r.rho_coding_labor,
                    r.parallel_penalty,
                    r.coding_labor_normalization
                )

            # Match units with compute_coding_labor: apply parallel_penalty and normalization
            coding_labor = float((L_opt ** r.parallel_penalty) * r.coding_labor_normalization)
            return coding_labor
        else:
            # simple_ces mode (default)
            automation_fraction = compute_automation_fraction(progress, self._automation_model)
            return compute_coding_labor(
                automation_fraction,
                inference_compute,
                human_labor,
                r.rho_coding_labor,
                r.parallel_penalty,
                r.coding_labor_normalization
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
        r = self.r_and_d
        cal = self._calibrated

        # Compute AI research taste
        ai_research_taste = compute_ai_research_taste(
            progress,
            self._taste_distribution,
            cal.ai_research_taste_slope,
            cal.progress_at_aa,
            r.ai_research_taste_at_coding_automation_anchor_sd,
        )
        aggregate_research_taste = compute_aggregate_research_taste(
            ai_research_taste, self._taste_distribution
        )

        # Compute coding labor (handles optimal_ces vs simple_ces mode)
        coding_labor = self._compute_coding_labor(progress, human_labor, inference_compute)

        # Compute research effort (experiment capacity * taste)
        research_effort = compute_research_effort(
            experiment_compute,
            coding_labor,
            cal.alpha_experiment_capacity,
            cal.rho_experiment_capacity,
            cal.experiment_compute_exponent,
            aggregate_research_taste
        )

        # Compute software progress rate
        software_progress_rate = compute_software_progress_rate(
            research_stock, research_effort, cal.r_software
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
        r = self.r_and_d
        cal = self._calibrated

        for dev_id, dev in world.ai_software_developers.items():
            # Get inputs from developer entity (values are set by compute and researcher updaters)
            human_labor, inference_compute, experiment_compute, training_compute_growth_rate = self._get_inputs_from_developer(dev, current_time)

            progress = self._tensor_to_float(dev.ai_software_progress.progress)
            research_stock = self._tensor_to_float(dev.ai_software_progress.research_stock)

            # Compute automation fraction based on progress (not year)
            # This matches how the reference model computes automation fraction
            automation_fraction = compute_automation_fraction(progress, self._automation_model)
            dev.ai_software_progress.automation_fraction = torch.tensor(automation_fraction)

            # Compute AI research taste
            ai_research_taste = compute_ai_research_taste(
                progress,
                self._taste_distribution,
                cal.ai_research_taste_slope,
                cal.progress_at_aa,
                r.ai_research_taste_at_coding_automation_anchor_sd,
            )
            aggregate_research_taste = compute_aggregate_research_taste(
                ai_research_taste, self._taste_distribution
            )
            dev.ai_software_progress.ai_research_taste = torch.tensor(ai_research_taste)
            dev.ai_software_progress.aggregate_research_taste = torch.tensor(aggregate_research_taste)

            # Compute serial_coding_labor using _compute_coding_labor which handles optimal_ces mode
            serial_coding_labor = self._compute_coding_labor(progress, human_labor, inference_compute)
            dev.ai_software_progress.serial_coding_labor = torch.tensor(serial_coding_labor)

            # Compute coding_labor by reversing the transformation
            # serial_coding_labor = (coding_labor ** parallel_penalty) * normalization
            # => coding_labor = (serial_coding_labor / normalization) ** (1/parallel_penalty)
            if r.parallel_penalty != 0 and r.coding_labor_normalization > 0:
                coding_labor = (serial_coding_labor / r.coding_labor_normalization) ** (1.0 / r.parallel_penalty)
            else:
                coding_labor = serial_coding_labor / r.coding_labor_normalization if r.coding_labor_normalization > 0 else serial_coding_labor
            dev.ai_software_progress.coding_labor = torch.tensor(coding_labor)

            # Compute human-only serial coding labor for multiplier calculation
            human_only_serial_coding_labor = (human_labor ** r.parallel_penalty) * r.coding_labor_normalization
            serial_coding_labor_multiplier = serial_coding_labor / human_only_serial_coding_labor if human_only_serial_coding_labor > 0 else 1.0
            dev.ai_software_progress.serial_coding_labor_multiplier = torch.tensor(serial_coding_labor_multiplier)

            # Compute research effort using serial_coding_labor
            research_effort = compute_research_effort(
                experiment_compute,
                serial_coding_labor,
                cal.alpha_experiment_capacity,
                cal.rho_experiment_capacity,
                cal.experiment_compute_exponent,
                aggregate_research_taste
            )
            dev.ai_software_progress.research_effort = torch.tensor(research_effort)

            # Compute experiment capacity: research_effort / aggregate_research_taste
            experiment_capacity = research_effort / aggregate_research_taste if aggregate_research_taste > 0 else 0.0
            dev.ai_software_progress.experiment_capacity = torch.tensor(experiment_capacity)

            # Software progress rate
            sw_rate = compute_software_progress_rate(
                research_stock, research_effort, cal.r_software
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

            # AI software progress multiplier (ref present day) - computed dynamically
            # This shows how much faster sw progress would be with current automation/taste
            # but using present_day resources (isolates the effect of AI improvements)
            ai_sw_mult = self._compute_ai_sw_progress_mult_ref_present_day(progress)
            dev.ai_software_progress.ai_sw_progress_mult_ref_present_day = torch.tensor(ai_sw_mult)

            # Compute software_efficiency = progress - initial_progress
            # This represents the cumulative contribution from software improvements (in OOMs)
            initial_progress = self._tensor_to_float(dev.ai_software_progress.initial_progress) if dev.ai_software_progress.initial_progress is not None else 0.0
            software_efficiency = progress - initial_progress
            dev.ai_software_progress.software_efficiency = torch.tensor(software_efficiency)

            # Compute ai_research_taste_sd from ai_research_taste
            ai_research_taste_sd = self._taste_distribution.get_sd_of_taste(ai_research_taste)
            if not np.isfinite(ai_research_taste_sd):
                ai_research_taste_sd = 0.0
            dev.ai_software_progress.ai_research_taste_sd = torch.tensor(ai_research_taste_sd)

        return world
