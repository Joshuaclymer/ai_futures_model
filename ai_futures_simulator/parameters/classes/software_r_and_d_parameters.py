"""
AI Software R&D parameters for modeling AI research and development dynamics.

Contains parameters for:
- CES production functions (coding labor, experiment capacity)
- Software progress rates
- Automation schedules
- AI research taste dynamics
- Horizon/milestone parameters
"""

from dataclasses import dataclass

from parameters.classes.base_spec import BaseSpec
from parameters.distribution_spec import ParamValue


@dataclass
class SoftwareRAndDParameters(BaseSpec):
    """
    All parameters for AI Software R&D / takeoff model.

    Parameters are organized into logical groups matching the model structure.
    """

    # Mode flags
    update_software_progress: ParamValue = None

    # Coding labor CES
    rho_coding_labor: ParamValue = None
    coding_labor_normalization: ParamValue = None

    # Experiment capacity CES
    direct_input_exp_cap_ces_params: ParamValue = None
    rho_experiment_capacity: ParamValue = None
    alpha_experiment_capacity: ParamValue = None
    experiment_compute_exponent: ParamValue = None

    # Experiment capacity asymptotes
    inf_labor_asymptote: ParamValue = None
    inf_compute_asymptote: ParamValue = None
    labor_anchor_exp_cap: ParamValue = None
    compute_anchor_exp_cap: ParamValue = None
    inv_compute_anchor_exp_cap: ParamValue = None

    # Parallel penalty
    parallel_penalty: ParamValue = None

    # Software progress parameters
    r_software: ParamValue = None
    software_progress_rate_at_reference_year: ParamValue = None

    # Automation schedule parameters
    automation_fraction_at_coding_automation_anchor: ParamValue = None
    automation_anchors: ParamValue = None
    automation_interp_type: ParamValue = None
    automation_logistic_asymptote: ParamValue = None
    swe_multiplier_at_present_day: ParamValue = None

    # Coding labor mode
    coding_labor_mode: ParamValue = None
    coding_automation_efficiency_slope: ParamValue = None
    optimal_ces_eta_init: ParamValue = None
    optimal_ces_grid_size: ParamValue = None
    optimal_ces_frontier_tail_eps: ParamValue = None
    optimal_ces_frontier_cap: ParamValue = None
    max_serial_coding_labor_multiplier: ParamValue = None

    # AI research taste parameters
    ai_research_taste_at_coding_automation_anchor_sd: ParamValue = None
    ai_research_taste_slope: ParamValue = None
    taste_schedule_type: ParamValue = None
    median_to_top_taste_multiplier: ParamValue = None
    top_percentile: ParamValue = None
    taste_limit: ParamValue = None
    taste_limit_smoothing: ParamValue = None

    # Horizon / milestone parameters
    progress_at_aa: ParamValue = None
    ac_time_horizon_minutes: ParamValue = None
    pre_gap_ac_time_horizon: ParamValue = None
    horizon_extrapolation_type: ParamValue = None

    # Manual horizon fitting parameters
    present_day: ParamValue = None
    present_horizon: ParamValue = None
    present_doubling_time: ParamValue = None
    doubling_difficulty_growth_factor: ParamValue = None

    # Milestone multipliers
    strat_ai_m2b: ParamValue = None
    ted_ai_m2b: ParamValue = None

    # Gap mode parameters
    include_gap: ParamValue = None
    gap_years: ParamValue = None
