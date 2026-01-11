#!/usr/bin/env python3
"""
Blacksite Progress Model

This module contains the BlacksiteProgressModel class, which extends the base
ProgressModel to simulate a "blacksite" scenario - a separate AI development
effort that starts behind the main model but may catch up.
"""

import numpy as np
from typing import List, Tuple
import logging
import time

import model_config as cfg

from .types import TimeSeriesData
from .parameters import Parameters
from .utils import _log_interp, should_reraise
from .ces_functions import compute_coding_labor_deprecated
from .progress_rate import (
    compute_research_effort,
    compute_software_progress_rate,
    compute_overall_progress_rate,
    compute_automation_fraction,
    compute_ai_research_taste,
    compute_aggregate_research_taste,
    progress_rate_at_time,
)
from .integration import (
    _find_exponential_crossing_time,
    integrate_progress,
)

# Import ProgressModel - need to import from _impl to avoid circular import
# This will be imported at runtime to avoid circular dependency
ProgressModel = None

logger = logging.getLogger(__name__)


def _get_progress_model():
    """Lazy import of ProgressModel to avoid circular dependency."""
    global ProgressModel
    if ProgressModel is None:
        from ._impl import ProgressModel as PM
        ProgressModel = PM
    return ProgressModel


class BlacksiteProgressModel:
    """
    Progress model for blacksite mode.

    A blacksite is a separate AI development effort that:
    - Starts at a specified time (blacksite_start_time)
    - Begins with models that are some years behind the main trajectory
    - May have reduced resources (compute, labor) compared to the main site
    - Can optionally receive software efficiency improvements from the main site

    The blacksite inherits human-only results and horizon trajectory from the
    main model, but computes its own progress trajectory based on its constrained
    resources.
    """

    def __init__(self, params: Parameters, data: TimeSeriesData, main_model):
        """
        Initialize a blacksite progress model.

        Args:
            params: Model parameters including blacksite-specific settings
            data: Input time series data
            main_model: The main ProgressModel that this blacksite branches from
        """
        # Get ProgressModel class and call its __init__
        PM = _get_progress_model()
        PM.__init__(self, params, data)
        self.main_model = main_model



        # set up input time series
        self.data = self.create_blacksite_input_data(data)


        # set up initial conditions
        logger.info(f"BLACKSITE::: params.blacksite_start_time: {params.blacksite_start_time}")
        logger.info(f"BLACKSITE::: data.time[0]: {data.time[0]}")
        start_time = self.data.time[0]
        assert start_time == params.blacksite_start_time
        initial_years_behind = params.blacksite_initial_years_behind


        main_proj_training_compute_at_start = np.interp(start_time, self.main_model.results['times'], self.main_model.results['training_compute'])
        main_proj_sw_eff_at_start = np.interp(start_time, self.main_model.results['times'], self.main_model.results['software_efficiency'])
        main_proj_progress_at_start = np.interp(start_time, self.main_model.results['times'], self.main_model.results['progress'])
        main_proj_effective_compute_at_start = np.interp(start_time, self.main_model.results['times'], self.main_model.results['effective_compute'])
        self.eOOMs_minus_progress = main_proj_effective_compute_at_start - main_proj_progress_at_start
        logger.info(f"BLACKSITE::: eOOMs_minus_progress: {self.eOOMs_minus_progress}")

        self.initial_model_progress = self.main_model.get_progress_at_time(start_time - initial_years_behind)
        self.params.blacksite_initial_model_progress = self.initial_model_progress
        self.initial_research_stock = _log_interp(start_time - initial_years_behind, self.main_model.results['times'], self.main_model.results['research_stock'])
        self.initial_software_efficiency = np.interp(start_time - initial_years_behind, self.main_model.results['times'], self.main_model.results['software_efficiency'])
        self.initial_training_compute = main_proj_training_compute_at_start - self.params.blacksite_training_compute_penalty_ooms
        # Account for both training compute penalty AND software efficiency deficit from being years behind
        software_efficiency_deficit = main_proj_sw_eff_at_start - self.initial_software_efficiency
        self.initial_progress = (main_proj_progress_at_start
                                 - self.params.blacksite_training_compute_penalty_ooms
                                 - software_efficiency_deficit)
        eOOMs_minus_progress_for_blacksite = (self.initial_training_compute + self.initial_software_efficiency) - self.initial_progress

        self.params.main_site = self.main_model
        # take human-only results from main model
        self.human_only_results = self.main_model.human_only_results

        # take horizon trajectory from main model
        self.horizon_trajectory = self.main_model.horizon_trajectory

    def create_blacksite_input_data(self, data: TimeSeriesData) -> TimeSeriesData:
        """
        Create blacksite input data by modifying main model's data.
        """
        assert self.params.blacksite_start_time in data.time
        start_time_idx = np.argmin(np.abs(data.time - self.params.blacksite_start_time))

        # TODO: maybe make this constant after start time
        time = data.time[start_time_idx:]
        L_HUMAN = np.full(len(data.L_HUMAN[start_time_idx:]), data.L_HUMAN[start_time_idx]/(10**self.params.blacksite_human_labor_penalty_ooms))
        inference_compute = np.full(len(data.inference_compute[start_time_idx:]), data.inference_compute[start_time_idx]/(10**self.params.blacksite_inference_compute_penalty_ooms))
        experiment_compute = np.full(len(data.experiment_compute[start_time_idx:]), data.experiment_compute[start_time_idx]/(10**self.params.blacksite_experiment_compute_penalty_ooms))

        # Compute training_compute values that produce the desired growth rate behavior
        # Start with the blacksite's penalized initial training compute
        initial_tc = data.training_compute[start_time_idx] - self.params.blacksite_training_compute_penalty_ooms
        training_compute = np.zeros(len(time))
        training_compute[0] = initial_tc

        if self.params.blacksite_can_stack_training_compute:
            # Constant growth rate from the start
            for i in range(1, len(time)):
                dt = time[i] - time[i-1]
                training_compute[i] = training_compute[i-1] + self.params.blacksite_training_compute_growth_rate * dt
        else:
            if self.params.blacksite_training_compute_growth_rate > 0:
                time_until_training_compute_parity = self.params.blacksite_training_compute_penalty_ooms / self.params.blacksite_training_compute_growth_rate
                parity_time = time[0] + time_until_training_compute_parity
                for i in range(1, len(time)):
                    dt = time[i] - time[i-1]
                    if time[i] <= parity_time:
                        # Zero growth rate until parity
                        training_compute[i] = training_compute[i-1]
                    else:
                        # Use blacksite growth rate after parity
                        training_compute[i] = training_compute[i-1] + self.params.blacksite_training_compute_growth_rate * dt
            else:
                # Zero growth rate throughout
                for i in range(1, len(time)):
                    training_compute[i] = training_compute[i-1]

        return TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute)

    def compute_progress_trajectory(self, time_range: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute progress over specified time range with comprehensive metrics

        Args:
            time_range: [start_time, end_time]

        Returns:
            Tuple of (times, cumulative_progress_values, research_stock_values)
        """
        initial_progress = self.initial_progress
        initial_research_stock_val = self.initial_research_stock

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



        # Below gives you at each time what is the effectige compute value and what is the research stock value. It runs the whole model.
        # With just time -> effective compute and research stock, you can compute all the other metrics.
        _t_integrate_start = time.perf_counter()
        times, progress_values, research_stock_values = integrate_progress(time_range, initial_progress, initial_research_stock_val, self.data, self.params)

        # SIE MODE
        takeoff_start_human_only_stats = None
        # assert self.params.sos_mode is False

        # Calculate all metrics in a single pass to avoid redundancy
        progress_rates = []
        research_efforts = []
        automation_fractions = []
        ai_research_tastes = []
        ai_research_taste_sds = []
        ai_research_taste_quantiles = []
        aggregate_research_tastes = []
        coding_labors = []
        coding_labors_with_present_resources = []
        serial_coding_labors = []
        software_progress_rates = []
        software_progress_rates_present_resources = []
        software_efficiency = []  # Integral of software_progress_rate
        human_only_research_efforts = []
        human_only_software_progress_rates = []
        human_only_progress_rates = []
        ai_labor_contributions = []
        human_labor_contributions = []
        ai_coding_labor_multipliers = []
        ai_coding_labor_mult_ref_present_day = []
        serial_coding_labor_multipliers = []
        ai_sw_progress_mult_ref_present_day = []
        takeoff_progress_multipliers = []
        discounted_exp_compute = []
        horizon_lengths = []
        effective_compute = []
        training_compute = []
        experiment_capacity = []
        exp_cap_mult_with_infinite_labor = []
        exp_cap_mult_with_infinite_compute = []

        # logger.info(f"Computing comprehensive metrics for {len(times)} time points")

        _t_metrics_loop_start = time.perf_counter()
        for i, (t, progress, rs) in enumerate(zip(times, progress_values, research_stock_values)):
            try:
                state = [progress, rs]
                rates = progress_rate_at_time(t, state, self.data, self.params)
                progress_rates.append(rates[0])
                research_efforts.append(rates[1])
                model_progress = max(progress, self.initial_model_progress)

                # INPUT TIME SERIES
                L_HUMAN = _log_interp(t, self.data.time, self.data.L_HUMAN)
                inference_compute = _log_interp(t, self.data.time, self.data.inference_compute)
                experiment_compute = _log_interp(t, self.data.time, self.data.experiment_compute)
                training_compute_growth_rate = self.data.get_training_compute_growth_rate(t)
                # Compute discounted experiment compute
                discounted_exp_compute_val = experiment_compute ** self.params.experiment_compute_exponent
                discounted_exp_compute.append(discounted_exp_compute_val if np.isfinite(discounted_exp_compute_val) else 0.0)


                # AUTOMATION FRACTION
                automation_fraction = compute_automation_fraction(model_progress, self.params)
                automation_fractions.append(automation_fraction)

                # RESEARCH TASTE
                ai_research_taste = compute_ai_research_taste(model_progress, self.params)
                ai_research_taste_sd = self.taste_distribution.get_sd_of_taste(ai_research_taste)
                ai_research_taste_quantile = self.taste_distribution.get_quantile_of_taste(ai_research_taste)
                aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste, self.params.taste_distribution)
                ai_research_tastes.append(ai_research_taste)
                ai_research_taste_sds.append(ai_research_taste_sd if np.isfinite(ai_research_taste_sd) else 0.0)
                ai_research_taste_quantiles.append(ai_research_taste_quantile if np.isfinite(ai_research_taste_quantile) else 0.0)
                aggregate_research_tastes.append(aggregate_research_taste)

                # CODING LABOR
                if getattr(self.params, 'coding_labor_mode', 'simple_ces') == 'optimal_ces':
                    H = float(L_HUMAN)
                    C = float(inference_compute)
                    logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * model_progress)
                    try:
                        automation_model = self.params.automation_model
                        L_opt = automation_model.coding_labor_optimal_ces(H, C, logE, self.params)
                        L_opt_present_resources = automation_model.coding_labor_optimal_ces(present_day_human_labor, present_day_inference_compute, logE, self.params)
                        if takeoff_start_human_only_stats is not None:
                            L_opt_takeoff_start = automation_model.coding_labor_optimal_ces(takeoff_start_human_only_stats['human_labor'], takeoff_start_human_only_stats['inference_compute'], logE, self.params)
                        if L_opt is None or not np.isfinite(L_opt):
                            assert False, "L_opt is None or not np.isfinite(L_opt)"
                        else:
                            # TODO: why is this being converted to serial-equivalent?
                            coding_labor = L_opt
                            coding_labor_with_present_resources = L_opt_present_resources
                            serial_coding_labor = float((L_opt ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
                            serial_coding_labor_with_present_resources = float((L_opt_present_resources ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
                            if takeoff_start_human_only_stats is not None:
                                serial_coding_labor_takeoff_start = float((L_opt_takeoff_start ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
                    except Exception as e:
                        if should_reraise(e):
                            raise
                        raise AssertionError(f"Falling back to simple CES in metrics due to optimal_ces error: {e}")
                else:
                    serial_coding_labor = compute_coding_labor_deprecated(
                        automation_fraction, inference_compute, L_HUMAN,
                        self.params.rho_coding_labor, self.params.parallel_penalty, self.params.coding_labor_normalization
                    )
                coding_labors.append(coding_labor if np.isfinite(coding_labor) else 0.0)
                coding_labors_with_present_resources.append(coding_labor_with_present_resources if np.isfinite(coding_labor_with_present_resources) else 0.0)
                serial_coding_labors.append(serial_coding_labor if np.isfinite(serial_coding_labor) else 0.0)

                # EXPERIMENT CAPACITY
                current_research_effort = research_efforts[i]
                exp_capacity = current_research_effort / aggregate_research_taste if aggregate_research_taste > 0 else 0.0
                experiment_capacity.append(exp_capacity if np.isfinite(exp_capacity) else 0.0)

                if self.params.rho_experiment_capacity < 0:
                    exp_cap_with_infinite_compute = (1-self.params.alpha_experiment_capacity) ** (1/self.params.rho_experiment_capacity) * serial_coding_labor
                    exp_cap_with_infinite_labor = self.params.alpha_experiment_capacity ** (1/self.params.rho_experiment_capacity) * discounted_exp_compute_val
                    exp_cap_mult_with_infinite_labor.append(exp_cap_with_infinite_labor / exp_capacity if exp_capacity > 0 else 0.0)
                    exp_cap_mult_with_infinite_compute.append(exp_cap_with_infinite_compute / exp_capacity if exp_capacity > 0 else 0.0)

                # RESEARCH EFFORT
                research_effort_present_resources = compute_research_effort(
                    present_day_experiment_compute, serial_coding_labor_with_present_resources,
                    self.params.alpha_experiment_capacity, self.params.rho_experiment_capacity, self.params.experiment_compute_exponent, aggregate_research_taste
                )
                if takeoff_start_human_only_stats is not None:
                    research_effort_takeoff_start_resources = compute_research_effort(
                        takeoff_start_human_only_stats['experiment_compute'], serial_coding_labor_takeoff_start,
                        self.params.alpha_experiment_capacity, self.params.rho_experiment_capacity, self.params.experiment_compute_exponent, aggregate_research_taste
                    )

                # SOFTWARE PROGRESS RATE
                # assert current_research_effort == compute_research_effort(
                #     experiment_compute, serial_coding_labor,
                #     self.params.alpha_experiment_capacity, self.params.rho_experiment_capacity, self.params.experiment_compute_exponent, aggregate_research_taste
                # )
                current_research_effort = research_efforts[i]
                software_rate = compute_software_progress_rate(
                    rs, current_research_effort,
                    self.params.r_software
                )
                if self.params.sw_leaks_to_blacksite:
                    software_rate = max(software_rate, np.interp(t, self.params.main_site.results['times'], self.params.main_site.results['software_progress_rates']))
                software_progress_rates.append(software_rate if np.isfinite(software_rate) else 0.0)
                software_rate_present_resources = compute_software_progress_rate(
                    present_day_research_stock, research_effort_present_resources,
                    self.params.r_software
                )
                software_progress_rates_present_resources.append(software_rate_present_resources if np.isfinite(software_rate_present_resources) else 0.0)
                # SOFTWARE EFFICIENCY (OOMS)
                if i == 0:
                    # Initialize software efficiency at 0
                    software_efficiency_val = self.initial_software_efficiency
                else:
                    # Trapezoidal integration: add area of trapezoid from previous time step
                    dt = times[i] - times[i-1]
                    avg_rate = (software_progress_rates[i] + software_progress_rates[i-1]) / 2.0
                    software_efficiency_val = software_efficiency[i-1] + avg_rate * dt
                software_efficiency.append(software_efficiency_val if np.isfinite(software_efficiency_val) else 0.0)

                # TRAINING COMPUTE (OOMS)
                if i == 0:
                    # Initialize training compute at 0
                    training_compute_val = self.initial_training_compute
                    logger.info(f"BLACKSITE::: initial_training_compute: {self.initial_training_compute}")
                else:
                    # Trapezoidal integration: add area of trapezoid from previous time step
                    dt = times[i] - times[i-1]
                    # Get the previous training_compute_growth_rate for trapezoidal rule
                    prev_training_compute_growth_rate = self.data.get_training_compute_growth_rate(times[i-1])
                    avg_growth_rate = (training_compute_growth_rate + prev_training_compute_growth_rate) / 2.0
                    training_compute_val = training_compute[i-1] + avg_growth_rate * dt

                training_compute.append(training_compute_val if np.isfinite(training_compute_val) else 0.0)

                # EFFECTIVE COMPUTE (OOMS)
                effective_compute_val = training_compute_val + software_efficiency_val
                effective_compute.append(effective_compute_val)

                # TIME HORIZON
                horizon_length = 0.0  # Default fallback
                if self.horizon_trajectory is not None:
                    try:
                        horizon_length = self.horizon_trajectory(model_progress)
                        if not np.isfinite(horizon_length) or horizon_length < 0:
                            horizon_length = 0.0
                    except Exception as horizon_e:
                        if should_reraise(horizon_e):
                            raise
                        logger.warning(f"Error computing horizon at progress {progress}: {horizon_e}")
                        horizon_length = 0.0
                horizon_lengths.append(horizon_length)

                # INSTANTANEOUS HUMAN-ONLY METRICS
                human_only_coding_labor = L_HUMAN
                human_only_serial_coding_labor = L_HUMAN**self.params.parallel_penalty
                human_only_aggregate_research_taste = compute_aggregate_research_taste(0, self.params.taste_distribution) # No AI research taste
                human_only_research_effort = compute_research_effort(
                    experiment_compute, human_only_serial_coding_labor,
                    self.params.alpha_experiment_capacity, self.params.rho_experiment_capacity, self.params.experiment_compute_exponent, human_only_aggregate_research_taste
                )
                human_only_research_efforts.append(human_only_research_effort if np.isfinite(human_only_research_effort) else 0.0)
                human_only_software_rate = compute_software_progress_rate(
                    rs, human_only_research_effort,
                    self.params.r_software
                )
                human_only_software_progress_rates.append(human_only_software_rate if np.isfinite(human_only_software_rate) else 0.0)
                human_only_overall_rate = compute_overall_progress_rate(
                    human_only_software_rate, training_compute_growth_rate
                )
                human_only_progress_rates.append(
                    human_only_overall_rate if np.isfinite(human_only_overall_rate) else 0.0
                )

                # Calculate labor contributions to cognitive output
                human_contrib = L_HUMAN
                ai_contrib = max(0.0, coding_labor - human_contrib)  # Ensure non-negative

                human_labor_contributions.append(human_contrib)
                ai_labor_contributions.append(ai_contrib)

                # AUTOMATION MULTIPLIERS
                if self.params.parallel_penalty and self.params.parallel_penalty != 0:
                    ai_coding_labor_multipliers.append(coding_labor / human_contrib if ai_contrib > 0 else 1.0)
                    ai_coding_labor_mult_ref_present_day.append(coding_labor_with_present_resources / present_day_human_labor if ai_contrib > 0 else 1.0)
                    # Serial coding labor multiplier: ratio of serial coding labor to human-only serial coding labor
                    serial_coding_labor_multipliers.append(serial_coding_labor / human_only_serial_coding_labor if human_only_serial_coding_labor > 0 else 1.0)
                else:
                    ai_coding_labor_multipliers.append(0.0)
                    serial_coding_labor_multipliers.append(1.0)
                ai_sw_progress_mult_ref_present_day.append(software_rate_present_resources / present_day_sw_progress_rate if present_day_sw_progress_rate > 0 else 0.0)

            except Exception as e:
                if should_reraise(e):
                    raise
                logger.warning(f"Error calculating metrics at t={t}: {e}")
                # Use safe fallback values
                if len(progress_rates) <= i:
                    progress_rates.append(0.0)
                if len(research_efforts) <= i:
                    research_efforts.append(0.0)
                automation_fractions.append(0.0)
                ai_research_tastes.append(0.0)
                ai_research_taste_sds.append(0.0)
                ai_research_taste_quantiles.append(0.0)
                aggregate_research_tastes.append(cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)  # Default to no enhancement
                coding_labors.append(0.0)
                software_progress_rates.append(0.0)
                software_efficiency.append(0.0)
                human_only_progress_rates.append(0.0)
                human_only_research_efforts.append(0.0)
                human_only_software_progress_rates.append(0.0)
                human_labor_contributions.append(0.0)
                ai_labor_contributions.append(0.0)
                ai_coding_labor_multipliers.append(0.0)
                serial_coding_labor_multipliers.append(1.0)
                ai_research_stock_multipliers.append(0.0)
                ai_software_progress_multipliers.append(0.0)
                ai_overall_progress_multipliers.append(0.0)
                discounted_exp_compute.append(0.0)
                horizon_lengths.append(0.0)
                effective_compute.append(0.0)
                training_compute.append(0.0)
                experiment_capacity.append(0.0)
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
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed computing top taste metrics: {e}")

        logger.info(f"BLACKSITE::: training_compute: {software_efficiency}")
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
            'training_compute': np.array(training_compute),
            'experiment_capacity': experiment_capacity,
            'aa_time': aa_time,  # Time when superhuman coder level is reached
            'sc_progress_level': self.params.progress_at_aa,  # Progress level for SC
            'sc_sw_multiplier': self.sc_sw_multiplier if hasattr(self, 'sc_sw_multiplier') else None,  # Software progress multiplier at SC
            'ai2027_sc_time': ai2027_sc_time,  # Time when @AI2027 SC condition is met
            # 'present_day': present_day,  # Anchor time for manual horizon fitting
            # 'anchor_progress_rate': anchor_progress_rate,  # Progress rate at anchor time
            # 'instantaneous_anchor_doubling_time_years': instantaneous_anchor_doubling_time_years,  # Instantaneous doubling time of horizon at anchor (years)
            # 'ai_research_taste_slope_per_anchor_progress_year': ai_taste_slope_per_anchor_progress_year,  # SD per anchor-progress-year
            # 'ai_research_taste_slope_per_effective_oom': ai_taste_slope_per_effective_oom,  # SD per effective OOM
            # 'automation_efficiency_slope_per_anchor_progress_year': automation_efficiency_slope_per_anchor_progress_year,  # OOMs per anchor-progress-year
            # 'automation_efficiency_slope_per_effective_oom': automation_efficiency_slope_per_effective_oom,  # OOMs per effective OOM
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
            'exp_cap_mult_with_infinite_labor': exp_cap_mult_with_infinite_labor,
            'exp_cap_mult_with_infinite_compute': exp_cap_mult_with_infinite_compute,
        }
        self.results['milestones'] = self.compute_milestones()

        return times, progress_values, research_stock_values
