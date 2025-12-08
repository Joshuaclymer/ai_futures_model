#!/usr/bin/env python3
"""
Incremental Progress Model

A streamlined version of ProgressModel that supports incremental updates.
Instead of running compute_progress_trajectory all in one go, you can call
increment() to step forward in time, then get_metrics() to compute metrics.

This is useful for scenarios where inputs (human labor, inference compute,
experiment compute) are determined dynamically rather than from a fixed time series.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging
import model_config as cfg

# Import key components from progress_model
from progress_model import (
    Parameters,
    TimeSeriesData,
    TasteDistribution,
    AutomationModel,
    get_or_create_taste_distribution,
    compute_coding_labor,
    compute_research_effort,
    compute_software_progress_rate,
    compute_overall_progress_rate,
    compute_automation_fraction,
    compute_ai_research_taste,
    compute_aggregate_research_taste,
    compute_exp_capacity_params_from_anchors,
    compute_initial_conditions,
    solve_lower_anchor_via_automation_model,
    _log_interp,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IncrementalState:
    """Current state of the incremental model"""
    time: float
    progress: float
    research_stock: float
    training_compute_ooms: float  # Cumulative training compute in OOMs


@dataclass
class IncrementalMetrics:
    """Comprehensive metrics at the current state"""

    progress: float
    research_stock: float

    # Rates
    progress_rate: float
    software_progress_rate: float
    research_effort: float

    # Automation
    automation_fraction: float
    coding_labor: float
    serial_coding_labor: float

    # Taste
    ai_research_taste: float
    ai_research_taste_sd: float
    aggregate_research_taste: float

    # Compute decomposition
    training_compute_ooms: float
    software_efficiency_ooms: float
    effective_compute_ooms: float

    # Experiment capacity
    experiment_capacity: float

    # Time horizon (if horizon trajectory is set)
    horizon_length: Optional[float] = None

    # Human-only counterfactuals
    human_only_coding_labor: float = 0.0
    human_only_research_effort: float = 0.0
    human_only_software_progress_rate: float = 0.0
    human_only_progress_rate: float = 0.0

    # AI multipliers
    ai_coding_labor_multiplier: float = 1.0
    ai_sw_progress_mult_ref_present_day: float = 1.0  # Software progress multiplier ref to present_day resources


class ProgressModelIncremental:
    """
    Incremental version of ProgressModel that supports step-by-step updates.

    Usage:
        # Initialize with parameters (uses default time series for parameter calibration)
        model = ProgressModelIncremental(params, initial_time_series)

        # Step forward in time
        model.increment(
            next_time=2025.5,
            human_labor=1000,
            inference_compute=1e18,
            experiment_compute=1e20,
            training_compute_growth_rate=0.5  # OOMs/year
        )

        # Get comprehensive metrics
        metrics = model.get_metrics()

    Alternative usage (from calibrated ProgressModel):
        # First create and run a ProgressModel to get calibrated parameters
        full_model = ProgressModel(params, time_series)
        full_model.compute_progress_trajectory(time_range, initial_progress=0.0)

        # Then create incremental model from calibrated params
        incr_model = ProgressModelIncremental.from_progress_model(full_model, time_range)
    """

    def __init__(
        self,
        params: Parameters,
        initial_time_series: TimeSeriesData,
        initial_progress: float = 0.0,
        time_range: Optional[List[float]] = None,
        skip_calibration: bool = False
    ):
        """
        Initialize the incremental progress model.

        Args:
            params: Model parameters
            initial_time_series: Time series data for calibration (human-only trajectory fitting, etc.)
            initial_progress: Starting progress value (default 0.0)
            time_range: Time range for calibration [start, end]. If None, uses time series bounds.
            skip_calibration: If True, skip calibration (use when creating from ProgressModel)
        """
        self.params = params
        self.initial_time_series = initial_time_series
        self.initial_progress = initial_progress

        # Determine time range
        if time_range is None:
            time_range = [initial_time_series.time.min(), initial_time_series.time.max()]
        self.time_range = time_range

        # Initialize taste distribution
        self.taste_distribution = params.taste_distribution

        # Calibration results
        self._calibrated = False
        self._anchor_stats = None
        self.horizon_trajectory = None
        self._initial_research_stock = None
        self._initial_research_effort = None
        # Present day baseline values for computing multipliers
        self._present_day_sw_progress_rate = None
        self._present_day_research_stock = None
        self._present_day_human_labor = None
        self._present_day_experiment_compute = None

        # Run calibration unless skipped
        if not skip_calibration:
            self._calibrate()

        # Initialize state at time_range[0]
        # If calibration was run, use the progress at the start time from the human-only trajectory
        # Otherwise use the provided initial_progress
        initial_progress_value = getattr(self, '_initial_progress_at_start', initial_progress)
        self._state = IncrementalState(
            time=time_range[0],
            progress=initial_progress_value,
            research_stock=self._initial_research_stock,
            training_compute_ooms=0.0
        )

        # History tracking (optional, for debugging/analysis)
        self._history: List[IncrementalState] = [IncrementalState(
            time=self._state.time,
            progress=self._state.progress,
            research_stock=self._state.research_stock,
            training_compute_ooms=self._state.training_compute_ooms
        )]

        # Input history for interpolation during RK4
        # Each entry is (time, human_labor, inference_compute, experiment_compute, training_compute)
        self._input_history: List[tuple] = []

    def _calibrate(self):
        """
        Run calibration by creating a full ProgressModel and copying its calibrated state.

        This ensures the incremental model matches the full model exactly.
        """
        from progress_model import ProgressModel

        logger.info("Calibrating incremental model via full ProgressModel...")

        # Run full model to get calibrated params and trajectory
        full_model = ProgressModel(self.params, self.initial_time_series)
        time_range = [self.initial_time_series.time.min(), self.initial_time_series.time.max()]
        times, progress, rs = full_model.compute_progress_trajectory(time_range, initial_progress=self.initial_progress)

        # Copy calibrated params from full model
        self.params = full_model.params

        # Copy anchor stats from full model
        self._anchor_stats = full_model.human_only_results.get('anchor_stats', {})

        # Store present_day baseline values for computing multipliers later
        self._present_day_sw_progress_rate = self._anchor_stats.get('sw_progress_rate', 1.0)
        self._present_day_research_stock = self._anchor_stats.get('research_stock', 1.0)
        self._present_day_human_labor = self._anchor_stats.get('human_labor', 1.0)
        self._present_day_experiment_compute = self._anchor_stats.get('experiment_compute', 1.0)

        # Copy horizon trajectory
        self.horizon_trajectory = full_model.horizon_trajectory

        # Get initial state at time_range[0] from the full trajectory
        start_time = self.time_range[0]
        self._initial_progress_at_start = float(np.interp(start_time, times, progress))
        self._initial_research_stock = float(np.interp(start_time, times, rs))

        logger.info(f"Initial state at {start_time}: progress={self._initial_progress_at_start:.4f}, rs={self._initial_research_stock:.6e}")

        self._calibrated = True
        logger.info("Calibration complete")

    def _estimate_progress_at_aa(self, present_day_progress: float, anchor_progress_rate: float):
        """
        Estimate progress_at_aa from horizon trajectory parameters.
        This matches the logic in ProgressModel.estimate_horizon_trajectory().

        For exponential horizon trajectory:
            log(horizon) = slope * progress + intercept
            progress_at_aa = (log(target_horizon) - intercept) / slope

        For decaying doubling time, the calculation is more complex but follows similar logic.
        """
        # Get parameters
        present_horizon = self.params.present_horizon
        present_doubling_time = self.params.present_doubling_time
        target_horizon = self.params.ac_time_horizon_minutes

        # Check if we have the parameters needed
        if present_horizon is None or target_horizon is None or target_horizon <= 0:
            # Fall back to default offset
            self.params.progress_at_aa = float(present_day_progress + 8.0)
            logger.warning(f"Missing horizon params, using default progress_at_aa: {self.params.progress_at_aa:.4f}")
            return

        if self.params.horizon_extrapolation_type == "exponential":
            # For exponential: log(horizon) = slope * progress + intercept
            # At present_day: log(present_horizon) = slope * present_day_progress + intercept
            # At AA: log(target_horizon) = slope * progress_at_aa + intercept

            if present_doubling_time is not None and anchor_progress_rate > 0:
                # Convert doubling time from time units to progress units
                doubling_time_in_progress_units = present_doubling_time * anchor_progress_rate
                # slope = log(2) / doubling_time (in progress units)
                slope = np.log(2) / doubling_time_in_progress_units
            else:
                # Use a default slope if not specified (typical value from METR data)
                slope = 0.5  # Default slope
                logger.warning(f"No present_doubling_time specified, using default slope: {slope}")

            # Calculate intercept from anchor point
            intercept = np.log(present_horizon) - slope * present_day_progress

            # Calculate progress_at_aa
            calculated_progress_at_aa = (np.log(target_horizon) - intercept) / slope

            # Handle gap if included
            _include_gap_flag = False
            try:
                _inc = getattr(self.params, 'include_gap', 'no gap')
                if isinstance(_inc, str):
                    _include_gap_flag = _inc.strip().lower() == 'gap'
                else:
                    _include_gap_flag = bool(_inc)
            except Exception:
                _include_gap_flag = False

            if _include_gap_flag:
                try:
                    gap_anchor_years = float(self.params.gap_years)
                    gap_progress_units = float(anchor_progress_rate) * gap_anchor_years
                    calculated_progress_at_aa = calculated_progress_at_aa + gap_progress_units
                except Exception:
                    pass

            self.params.progress_at_aa = float(calculated_progress_at_aa)
            logger.info(f"Estimated progress_at_aa from horizon trajectory: {self.params.progress_at_aa:.4f}")
            logger.info(f"  (slope={slope:.6f}, intercept={intercept:.6f}, target_horizon={target_horizon})")

            # Also create the horizon trajectory function for later use
            def horizon_trajectory(progress):
                return np.exp(slope * progress + intercept)
            self.horizon_trajectory = horizon_trajectory

        elif self.params.horizon_extrapolation_type == "decaying doubling time":
            # For decaying doubling time, we need H_0, T_0, A_0
            H_0 = present_horizon

            if present_doubling_time is not None and anchor_progress_rate > 0:
                T_0 = present_doubling_time * anchor_progress_rate
            else:
                T_0 = 0.8  # Default

            if self.params.doubling_difficulty_growth_factor is not None:
                A_0 = 1.0 - self.params.doubling_difficulty_growth_factor
            else:
                A_0 = 0.08  # Default

            # For decaying doubling time: H(t) = H_0 * (1 - A_0 * t / T_0)^(log(2)/log(1-A_0))
            # where t is progress relative to anchor (t = progress - present_day_progress)
            # Solve for t where H(t) = target_horizon

            if abs(A_0) > 1e-6 and A_0 < 1:
                exponent = np.log(2) / np.log(1 - A_0)
                # H_0 * (1 - A_0 * t / T_0)^exponent = target_horizon
                # (1 - A_0 * t / T_0)^exponent = target_horizon / H_0
                # 1 - A_0 * t / T_0 = (target_horizon / H_0)^(1/exponent)
                # t = T_0 / A_0 * (1 - (target_horizon / H_0)^(1/exponent))
                ratio = target_horizon / H_0
                if ratio > 0:
                    t_at_target = T_0 / A_0 * (1 - ratio ** (1 / exponent))
                    calculated_progress_at_aa = present_day_progress + t_at_target
                    self.params.progress_at_aa = float(calculated_progress_at_aa)
                    logger.info(f"Estimated progress_at_aa (decaying DT): {self.params.progress_at_aa:.4f}")
                else:
                    self.params.progress_at_aa = float(present_day_progress + 8.0)
            else:
                self.params.progress_at_aa = float(present_day_progress + 8.0)
        else:
            # Unknown extrapolation type, use default
            self.params.progress_at_aa = float(present_day_progress + 8.0)
            logger.warning(f"Unknown horizon_extrapolation_type, using default progress_at_aa: {self.params.progress_at_aa:.4f}")

    def _run_human_only_calibration(self):
        """Run a simplified human-only trajectory to get anchor stats.

        Also stores the solution for later use in setting initial state at time_range[0].
        """
        from scipy import integrate

        # Temporarily set human_only mode
        original_human_only = self.params.human_only
        self.params.human_only = True

        # Get initial conditions for human-only
        initial_conditions = compute_initial_conditions(
            self.initial_time_series, self.params, self.initial_progress
        )
        initial_research_stock = initial_conditions.research_stock

        def ode_func(t, y):
            progress, research_stock = y
            return self._compute_rates_at_time(
                t, progress, research_stock,
                self.initial_time_series, human_only=True
            )

        # Integrate over the FULL time series range (not self.time_range)
        # This is needed because we need stats at present_day and reference_year
        # which may be before the user's requested start time
        t_start = self.initial_time_series.time.min()
        t_end = self.initial_time_series.time.max()
        sol = integrate.solve_ivp(
            ode_func,
            [t_start, t_end],
            [self.initial_progress, initial_research_stock],
            method='RK45',
            dense_output=True,
            rtol=1e-4, atol=1e-6
        )

        # Store solution for later use (to get state at time_range[0])
        self._calibration_solution = sol if sol.success else None

        if sol.success:
            # Get anchor stats at present_day
            present_day = self.params.present_day
            state_at_present = sol.sol(present_day)
            progress_at_present = state_at_present[0]
            rs_at_present = state_at_present[1]

            # Compute rates at present day
            L_HUMAN = _log_interp(present_day, self.initial_time_series.time, self.initial_time_series.L_HUMAN)
            inference_compute = _log_interp(present_day, self.initial_time_series.time, self.initial_time_series.inference_compute)
            experiment_compute = _log_interp(present_day, self.initial_time_series.time, self.initial_time_series.experiment_compute)
            training_compute_growth_rate = _log_interp(present_day, self.initial_time_series.time, self.initial_time_series.training_compute_growth_rate)

            # Human-only coding labor
            coding_labor = L_HUMAN ** self.params.parallel_penalty * self.params.coding_labor_normalization

            # Human-only research effort
            aggregate_taste = compute_aggregate_research_taste(0, self.params.taste_distribution)
            research_effort = compute_research_effort(
                experiment_compute, coding_labor,
                self.params.alpha_experiment_capacity,
                self.params.rho_experiment_capacity,
                self.params.experiment_compute_exponent,
                aggregate_taste
            )

            # Software progress rate at present_day
            sw_rate = compute_software_progress_rate(rs_at_present, research_effort, self.params.r_software)

            # Overall rate at present_day
            overall_rate = compute_overall_progress_rate(sw_rate, training_compute_growth_rate)

            # IMPORTANT: Also compute reference_sw_progress_rate at SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR
            # This is used for scaling r_software, must match full model behavior
            reference_year = cfg.SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR
            state_at_reference = sol.sol(reference_year)
            rs_at_reference = state_at_reference[1]

            # Get inputs at reference year
            ref_experiment_compute = _log_interp(reference_year, self.initial_time_series.time, self.initial_time_series.experiment_compute)
            ref_L_HUMAN = _log_interp(reference_year, self.initial_time_series.time, self.initial_time_series.L_HUMAN)
            ref_coding_labor = ref_L_HUMAN ** self.params.parallel_penalty * self.params.coding_labor_normalization
            ref_research_effort = compute_research_effort(
                ref_experiment_compute, ref_coding_labor,
                self.params.alpha_experiment_capacity,
                self.params.rho_experiment_capacity,
                self.params.experiment_compute_exponent,
                aggregate_taste
            )
            reference_sw_progress_rate = compute_software_progress_rate(rs_at_reference, ref_research_effort, self.params.r_software)

            self._anchor_stats = {
                'time': present_day,
                'progress': progress_at_present,
                'research_stock': rs_at_present,
                'progress_rate': overall_rate,
                'sw_progress_rate': sw_rate,
                'reference_sw_progress_rate': reference_sw_progress_rate,  # For r_software scaling
                'research_effort': research_effort,
                'human_labor': L_HUMAN,
                'inference_compute': inference_compute,
                'experiment_compute': experiment_compute,
            }
            logger.info(f"Anchor stats at present_day={present_day}: progress={progress_at_present:.4f}, rate={overall_rate:.4f}")
            logger.info(f"Reference sw_progress_rate at {reference_year}: {reference_sw_progress_rate:.6f}")
        else:
            logger.warning("Human-only calibration integration failed")
            self._anchor_stats = {'progress_rate': 1.0, 'sw_progress_rate': 1.0, 'reference_sw_progress_rate': 1.0}

        # Restore original mode
        self.params.human_only = original_human_only

    def _compute_rates_at_time(
        self,
        t: float,
        progress: float,
        research_stock: float,
        time_series: TimeSeriesData,
        human_only: bool = False
    ) -> List[float]:
        """
        Compute instantaneous rates [d(progress)/dt, d(research_stock)/dt].

        This is the RHS of the coupled ODE system.
        """
        # Interpolate time series
        L_HUMAN = _log_interp(t, time_series.time, time_series.L_HUMAN)
        inference_compute = _log_interp(t, time_series.time, time_series.inference_compute)
        experiment_compute = _log_interp(t, time_series.time, time_series.experiment_compute)
        training_compute_growth_rate = _log_interp(t, time_series.time, time_series.training_compute_growth_rate)

        return self._compute_rates(
            progress, research_stock,
            L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate,
            human_only=human_only
        )

    def _compute_rates(
        self,
        progress: float,
        research_stock: float,
        human_labor: float,
        inference_compute: float,
        experiment_compute: float,
        training_compute_growth_rate: float,
        human_only: bool = False
    ) -> List[float]:
        """
        Compute instantaneous rates given current state and inputs.

        Args:
            progress: Current cumulative progress
            research_stock: Current research stock
            human_labor: Human labor supply
            inference_compute: AI inference compute
            experiment_compute: Experiment compute budget
            training_compute_growth_rate: Training compute growth rate (OOMs/year)
            human_only: If True, ignore automation effects

        Returns:
            [d(progress)/dt, d(research_stock)/dt]
        """
        if human_only:
            automation_fraction = 0.0
            aggregate_research_taste = compute_aggregate_research_taste(0, self.params.taste_distribution)
            coding_labor = (human_labor ** self.params.parallel_penalty) * self.params.coding_labor_normalization
        else:
            # Compute automation fraction
            automation_fraction = compute_automation_fraction(progress, self.params)

            # Compute AI research taste
            ai_research_taste = compute_ai_research_taste(progress, self.params)
            aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste, self.params.taste_distribution)

            # Compute coding labor
            if getattr(self.params, 'coding_labor_mode', 'simple_ces') == 'optimal_ces':
                H = float(human_labor)
                C = float(inference_compute)
                logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * progress)
                automation_model = self.params.automation_model
                L_opt = automation_model.coding_labor_optimal_ces(H, C, logE, self.params)
                coding_labor = float((L_opt ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
            else:
                coding_labor = compute_coding_labor(
                    automation_fraction, inference_compute, human_labor,
                    self.params.rho_coding_labor, self.params.parallel_penalty,
                    self.params.coding_labor_normalization
                )

        # Compute research effort
        research_effort = compute_research_effort(
            experiment_compute, coding_labor,
            self.params.alpha_experiment_capacity,
            self.params.rho_experiment_capacity,
            self.params.experiment_compute_exponent,
            aggregate_research_taste
        )

        # Compute software progress rate
        software_progress_rate = compute_software_progress_rate(
            research_stock, research_effort, self.params.r_software
        )

        # Compute overall progress rate
        overall_rate = compute_overall_progress_rate(software_progress_rate, training_compute_growth_rate)

        return [overall_rate, research_effort]

    def _interpolate_inputs(self, t: float) -> tuple:
        """
        Interpolate inputs at time t from stored input history.
        Uses log-space interpolation for exponentially growing variables.

        Args:
            t: Time to interpolate at

        Returns:
            (human_labor, inference_compute, experiment_compute, training_compute)
        """
        if len(self._input_history) == 0:
            raise RuntimeError("No input history available for interpolation")

        if len(self._input_history) == 1:
            # Only one point, return it
            return self._input_history[0][1:]

        # Extract arrays for interpolation
        times = np.array([entry[0] for entry in self._input_history])
        human_labor = np.array([entry[1] for entry in self._input_history])
        inference_compute = np.array([entry[2] for entry in self._input_history])
        experiment_compute = np.array([entry[3] for entry in self._input_history])
        training_compute = np.array([entry[4] for entry in self._input_history])

        # Log-space interpolation for exponentially growing variables
        hl = _log_interp(t, times, human_labor)
        ic = _log_interp(t, times, inference_compute)
        ec = _log_interp(t, times, experiment_compute)
        tc = _log_interp(t, times, training_compute)

        return (hl, ic, ec, tc)

    def _compute_training_compute_growth_rate(self, t: float, training_compute: float) -> float:
        """
        Compute training compute growth rate (OOMs/year) from stored history.

        Uses the derivative of log(training_compute) with respect to time.

        Args:
            t: Current time
            training_compute: Training compute at time t

        Returns:
            Training compute growth rate in OOMs/year
        """
        if len(self._input_history) < 1:
            # No history yet, assume 0 growth rate
            return 0.0

        # Get previous point from history
        prev_t, _, _, _, prev_tc = self._input_history[-1]
        dt = t - prev_t

        if dt <= 0 or prev_tc <= 0 or training_compute <= 0:
            return 0.0

        # Growth rate in OOMs/year = d(log10(tc))/dt
        growth_rate = (np.log10(training_compute) - np.log10(prev_tc)) / dt
        return growth_rate

    def _rk4_substep(
        self,
        dt: float,
        progress: float,
        research_stock: float,
        hl_0: float, ic_0: float, ec_0: float, tc_0: float,
        hl_end: float, ic_end: float, ec_end: float, tc_end: float
    ) -> tuple:
        """
        Perform a single RK4 substep.

        Returns:
            (new_progress, new_research_stock)
        """
        # Inputs at midpoint - geometric mean for exponential quantities
        hl_mid = np.sqrt(hl_0 * hl_end)
        ic_mid = np.sqrt(ic_0 * ic_end)
        ec_mid = np.sqrt(ec_0 * ec_end)

        # Compute training compute growth rate from absolute values
        if tc_0 > 0 and tc_end > 0 and dt > 0:
            tr_avg = (np.log10(tc_end) - np.log10(tc_0)) / dt
        else:
            tr_avg = 0.0

        # k1 at start
        rates_1 = self._compute_rates(progress, research_stock, hl_0, ic_0, ec_0, tr_avg)
        k1_progress, k1_rs = rates_1[0], rates_1[1]

        # k2 at midpoint
        rates_2 = self._compute_rates(
            progress + 0.5 * dt * k1_progress,
            research_stock + 0.5 * dt * k1_rs,
            hl_mid, ic_mid, ec_mid, tr_avg
        )
        k2_progress, k2_rs = rates_2[0], rates_2[1]

        # k3 at midpoint
        rates_3 = self._compute_rates(
            progress + 0.5 * dt * k2_progress,
            research_stock + 0.5 * dt * k2_rs,
            hl_mid, ic_mid, ec_mid, tr_avg
        )
        k3_progress, k3_rs = rates_3[0], rates_3[1]

        # k4 at end
        rates_4 = self._compute_rates(
            progress + dt * k3_progress,
            research_stock + dt * k3_rs,
            hl_end, ic_end, ec_end, tr_avg
        )
        k4_progress, k4_rs = rates_4[0], rates_4[1]

        # RK4 update
        new_progress = progress + (dt / 6.0) * (k1_progress + 2*k2_progress + 2*k3_progress + k4_progress)
        new_rs = research_stock + (dt / 6.0) * (k1_rs + 2*k2_rs + 2*k3_rs + k4_rs)

        return new_progress, max(new_rs, 1e-10)

    def increment(
        self,
        next_time: float,
        human_labor: float,
        inference_compute: float,
        experiment_compute: float,
        training_compute: float
    ):
        """
        Advance the model state to the next time point.

        Uses RK4 integration with adaptive internal substeps for accuracy.
        For large time steps, the interval is subdivided to handle nonlinear dynamics.

        Args:
            next_time: Target time to advance to
            human_labor: Human labor supply at next_time
            inference_compute: AI inference compute at next_time
            experiment_compute: Experiment compute at next_time
            training_compute: Training compute at next_time (absolute value, not growth rate)
        """
        if not self._calibrated:
            raise RuntimeError("Model must be calibrated before incrementing")

        if next_time <= self._state.time:
            raise ValueError(f"next_time ({next_time}) must be > current time ({self._state.time})")

        total_dt = next_time - self._state.time
        t0 = self._state.time

        # Get inputs at t0 (start) - from history
        if len(self._input_history) >= 1:
            hl_0, ic_0, ec_0, tc_0 = self._interpolate_inputs(t0)
        else:
            # First step: use the provided inputs for the whole step
            hl_0, ic_0, ec_0, tc_0 = human_labor, inference_compute, experiment_compute, training_compute

        # Inputs at t_end
        hl_end, ic_end, ec_end, tc_end = human_labor, inference_compute, experiment_compute, training_compute

        # Store the new inputs
        self._input_history.append((next_time, human_labor, inference_compute, experiment_compute, training_compute))

        # Determine number of substeps based on step size and expected rate of change
        # Use smaller substeps for larger time intervals to improve accuracy
        max_substep_dt = 0.01  # Maximum substep size (in years)
        n_substeps = max(1, int(np.ceil(total_dt / max_substep_dt)))
        substep_dt = total_dt / n_substeps

        # Current state
        progress = self._state.progress
        research_stock = self._state.research_stock

        # Perform substeps
        for i in range(n_substeps):
            # Fraction through the interval
            frac_start = i / n_substeps
            frac_end = (i + 1) / n_substeps

            # Interpolate inputs for this substep (log-space for exponential quantities)
            hl_sub_0 = hl_0 * (hl_end / hl_0) ** frac_start if hl_0 > 0 else hl_end
            hl_sub_end = hl_0 * (hl_end / hl_0) ** frac_end if hl_0 > 0 else hl_end
            ic_sub_0 = ic_0 * (ic_end / ic_0) ** frac_start if ic_0 > 0 else ic_end
            ic_sub_end = ic_0 * (ic_end / ic_0) ** frac_end if ic_0 > 0 else ic_end
            ec_sub_0 = ec_0 * (ec_end / ec_0) ** frac_start if ec_0 > 0 else ec_end
            ec_sub_end = ec_0 * (ec_end / ec_0) ** frac_end if ec_0 > 0 else ec_end
            tc_sub_0 = tc_0 * (tc_end / tc_0) ** frac_start if tc_0 > 0 else tc_end
            tc_sub_end = tc_0 * (tc_end / tc_0) ** frac_end if tc_0 > 0 else tc_end

            # Perform RK4 substep
            progress, research_stock = self._rk4_substep(
                substep_dt, progress, research_stock,
                hl_sub_0, ic_sub_0, ec_sub_0, tc_sub_0,
                hl_sub_end, ic_sub_end, ec_sub_end, tc_sub_end
            )

        # Update training compute OOMs (track absolute value in log space)
        new_training_compute_ooms = np.log10(training_compute) if training_compute > 0 else self._state.training_compute_ooms

        self._state = IncrementalState(
            time=next_time,
            progress=progress,
            research_stock=research_stock,
            training_compute_ooms=new_training_compute_ooms
        )

        # Add to history
        self._history.append(IncrementalState(
            time=self._state.time,
            progress=self._state.progress,
            research_stock=self._state.research_stock,
            training_compute_ooms=self._state.training_compute_ooms
        ))

    def get_metrics(
        self,
        human_labor: float,
        inference_compute: float,
        experiment_compute: float,
        training_compute_growth_rate: float = 0.0
    ) -> IncrementalMetrics:
        """
        Compute comprehensive metrics at the current state.

        Args:
            human_labor: Current human labor supply
            inference_compute: Current AI inference compute
            experiment_compute: Current experiment compute
            training_compute_growth_rate: Current training compute growth rate (OOMs/year)

        Returns:
            IncrementalMetrics with all computed values
        """
        progress = self._state.progress
        research_stock = self._state.research_stock

        # Compute automation fraction
        automation_fraction = compute_automation_fraction(progress, self.params)

        # Compute AI research taste
        ai_research_taste = compute_ai_research_taste(progress, self.params)
        ai_research_taste_sd = self.taste_distribution.get_sd_of_taste(ai_research_taste)
        aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste, self.params.taste_distribution)

        # Compute coding labor
        if getattr(self.params, 'coding_labor_mode', 'simple_ces') == 'optimal_ces':
            H = float(human_labor)
            C = float(inference_compute)
            logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * progress)
            automation_model = self.params.automation_model
            L_opt = automation_model.coding_labor_optimal_ces(H, C, logE, self.params)
            coding_labor = L_opt
            serial_coding_labor = float((L_opt ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
        else:
            serial_coding_labor = compute_coding_labor(
                automation_fraction, inference_compute, human_labor,
                self.params.rho_coding_labor, self.params.parallel_penalty,
                self.params.coding_labor_normalization
            )
            coding_labor = serial_coding_labor

        # Compute research effort
        research_effort = compute_research_effort(
            experiment_compute, serial_coding_labor,
            self.params.alpha_experiment_capacity,
            self.params.rho_experiment_capacity,
            self.params.experiment_compute_exponent,
            aggregate_research_taste
        )

        # Experiment capacity
        experiment_capacity = research_effort / aggregate_research_taste if aggregate_research_taste > 0 else 0.0

        # Software progress rate
        software_progress_rate = compute_software_progress_rate(
            research_stock, research_effort, self.params.r_software
        )

        # Overall progress rate
        progress_rate = compute_overall_progress_rate(software_progress_rate, training_compute_growth_rate)

        # Compute decomposition
        training_compute_ooms = self._state.training_compute_ooms
        software_efficiency_ooms = progress - self.initial_progress - training_compute_ooms
        effective_compute_ooms = training_compute_ooms + software_efficiency_ooms

        # Horizon length
        horizon_length = None
        if self.horizon_trajectory is not None:
            try:
                horizon_length = self.horizon_trajectory(progress)
            except:
                pass

        # Human-only counterfactuals
        human_only_coding_labor = human_labor
        human_only_serial_coding_labor = (human_labor ** self.params.parallel_penalty) * self.params.coding_labor_normalization
        human_only_aggregate_taste = compute_aggregate_research_taste(0, self.params.taste_distribution)
        human_only_research_effort = compute_research_effort(
            experiment_compute, human_only_serial_coding_labor,
            self.params.alpha_experiment_capacity,
            self.params.rho_experiment_capacity,
            self.params.experiment_compute_exponent,
            human_only_aggregate_taste
        )
        human_only_sw_rate = compute_software_progress_rate(
            research_stock, human_only_research_effort, self.params.r_software
        )
        human_only_progress_rate = compute_overall_progress_rate(human_only_sw_rate, training_compute_growth_rate)

        # AI coding labor multiplier
        ai_coding_labor_multiplier = coding_labor / human_labor if human_labor > 0 else 1.0

        # AI software progress multiplier ref present_day
        # This is software_rate_with_present_resources / present_day_sw_progress_rate
        # We need to compute what software rate would be with current AI capability but present_day resources
        if self._present_day_sw_progress_rate is not None and self._present_day_sw_progress_rate > 0:
            # Compute coding labor with present_day resources but current AI capability
            if getattr(self.params, 'coding_labor_mode', 'simple_ces') == 'optimal_ces':
                H_present = float(self._present_day_human_labor)
                C_present = float(self._anchor_stats.get('inference_compute', 1.0))  # Use present_day inference compute
                logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * progress)
                automation_model = self.params.automation_model
                L_opt_present = automation_model.coding_labor_optimal_ces(H_present, C_present, logE, self.params)
                serial_coding_labor_present = float((L_opt_present ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
            else:
                serial_coding_labor_present = compute_coding_labor(
                    automation_fraction, self._anchor_stats.get('inference_compute', 1.0), self._present_day_human_labor,
                    self.params.rho_coding_labor, self.params.parallel_penalty,
                    self.params.coding_labor_normalization
                )

            # Research effort with present_day resources but current AI taste
            research_effort_present = compute_research_effort(
                self._present_day_experiment_compute, serial_coding_labor_present,
                self.params.alpha_experiment_capacity,
                self.params.rho_experiment_capacity,
                self.params.experiment_compute_exponent,
                aggregate_research_taste
            )

            # Software progress rate with present_day research_stock but current AI research_effort
            software_rate_present_resources = compute_software_progress_rate(
                self._present_day_research_stock, research_effort_present, self.params.r_software
            )

            ai_sw_progress_mult_ref_present_day = software_rate_present_resources / self._present_day_sw_progress_rate
        else:
            ai_sw_progress_mult_ref_present_day = 1.0

        return IncrementalMetrics(
            time=self._state.time,
            progress=progress,
            research_stock=research_stock,
            progress_rate=progress_rate,
            software_progress_rate=software_progress_rate,
            research_effort=research_effort,
            automation_fraction=automation_fraction,
            coding_labor=coding_labor,
            serial_coding_labor=serial_coding_labor,
            ai_research_taste=ai_research_taste,
            ai_research_taste_sd=ai_research_taste_sd if np.isfinite(ai_research_taste_sd) else 0.0,
            aggregate_research_taste=aggregate_research_taste,
            training_compute_ooms=training_compute_ooms,
            software_efficiency_ooms=software_efficiency_ooms,
            effective_compute_ooms=effective_compute_ooms,
            experiment_capacity=experiment_capacity,
            horizon_length=horizon_length,
            human_only_coding_labor=human_only_coding_labor,
            human_only_research_effort=human_only_research_effort,
            human_only_software_progress_rate=human_only_sw_rate,
            human_only_progress_rate=human_only_progress_rate,
            ai_coding_labor_multiplier=ai_coding_labor_multiplier,
            ai_sw_progress_mult_ref_present_day=ai_sw_progress_mult_ref_present_day,
        )

    @property
    def state(self) -> IncrementalState:
        """Get the current state."""
        return self._state

    @property
    def time(self) -> float:
        """Get the current time."""
        return self._state.time

    @property
    def progress(self) -> float:
        """Get the current progress."""
        return self._state.progress

    @property
    def research_stock(self) -> float:
        """Get the current research stock."""
        return self._state.research_stock

    @property
    def history(self) -> List[IncrementalState]:
        """Get the history of states."""
        return self._history

    def reset(self):
        """Reset the model to initial state."""
        self._state = IncrementalState(
            time=self.time_range[0],
            progress=self.initial_progress,
            research_stock=self._initial_research_stock,
            training_compute_ooms=0.0
        )
        self._history = [IncrementalState(
            time=self._state.time,
            progress=self._state.progress,
            research_stock=self._state.research_stock,
            training_compute_ooms=self._state.training_compute_ooms
        )]
        self._input_history = []

    def set_horizon_trajectory(self, horizon_fn):
        """
        Set a horizon trajectory function.

        Args:
            horizon_fn: Function that maps progress to horizon length in minutes
        """
        self.horizon_trajectory = horizon_fn

    @classmethod
    def from_progress_model(
        cls,
        progress_model,
        time_range: List[float],
        initial_progress: float = 0.0
    ) -> 'ProgressModelIncremental':
        """
        Create an incremental model from a calibrated ProgressModel.

        This ensures the incremental model uses the exact same calibrated parameters
        as the full ProgressModel, which is important for consistency.

        Args:
            progress_model: A ProgressModel instance that has been calibrated
                           (i.e., compute_progress_trajectory has been called)
            time_range: Time range [start, end] for the incremental model
            initial_progress: Starting progress value

        Returns:
            ProgressModelIncremental with matching calibration
        """
        # Get the calibrated parameters from the progress model
        params = progress_model.params
        time_series = progress_model.data

        # Create incremental model without running its own calibration
        incr_model = cls(
            params=params,
            initial_time_series=time_series,
            initial_progress=initial_progress,
            time_range=time_range,
            skip_calibration=True
        )

        # Set calibration results from the progress model
        incr_model._calibrated = True
        incr_model.taste_distribution = params.taste_distribution

        # Compute initial conditions using the calibrated params
        initial_conditions = compute_initial_conditions(
            time_series, params, initial_progress
        )
        incr_model._initial_research_stock = initial_conditions.research_stock
        incr_model._initial_research_effort = initial_conditions.research_effort

        # Re-initialize state with correct values
        incr_model._state = IncrementalState(
            time=time_range[0],
            progress=initial_progress,
            research_stock=incr_model._initial_research_stock,
            training_compute_ooms=0.0
        )
        incr_model._history = [IncrementalState(
            time=incr_model._state.time,
            progress=incr_model._state.progress,
            research_stock=incr_model._state.research_stock,
            training_compute_ooms=incr_model._state.training_compute_ooms
        )]

        # Copy horizon trajectory if available
        if hasattr(progress_model, 'horizon_trajectory') and progress_model.horizon_trajectory is not None:
            incr_model.horizon_trajectory = progress_model.horizon_trajectory

        logger.info(f"Created incremental model from ProgressModel with initial RS={incr_model._initial_research_stock:.6f}")

        return incr_model


def create_incremental_model_from_defaults(
    param_overrides: Optional[Dict[str, Any]] = None,
    input_csv_path: str = "input_data.csv"
) -> ProgressModelIncremental:
    """
    Create an incremental model with default parameters.

    Args:
        param_overrides: Optional dict of parameter values to override defaults
        input_csv_path: Path to input CSV with time series data

    Returns:
        Configured ProgressModelIncremental instance
    """
    import pandas as pd

    # Load time series data
    df = pd.read_csv(input_csv_path)
    time_series = TimeSeriesData(
        time=df['time'].values,
        L_HUMAN=df['L_HUMAN'].values,
        inference_compute=df['inference_compute'].values,
        experiment_compute=df['experiment_compute'].values,
        training_compute_growth_rate=df['training_compute_growth_rate'].values
    )

    # Create parameters
    param_dict = dict(cfg.DEFAULT_PARAMETERS)
    if param_overrides:
        param_dict.update(param_overrides)

    params = Parameters(**{k: v for k, v in param_dict.items() if k in Parameters.__dataclass_fields__})

    # Create and return model
    return ProgressModelIncremental(params, time_series)
