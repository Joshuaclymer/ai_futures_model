"""
AI Software R&D parameters for modeling AI research and development dynamics.

Contains parameters for:
- CES production functions (coding labor, experiment capacity)
- Software progress rates
- Automation schedules
- AI research taste dynamics
- Horizon/milestone parameters
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Union


@dataclass
class SoftwareRAndDParameters:
    """
    All parameters for AI Software R&D / takeoff model.

    Parameters are organized into logical groups matching the model structure.
    """

    # =========================================================================
    # MODE FLAGS
    # =========================================================================
    human_only: bool

    # =========================================================================
    # PRODUCTION FUNCTION PARAMETERS (CES)
    # =========================================================================

    # Coding labor CES
    rho_coding_labor: float  # Elasticity of substitution parameter
    coding_labor_normalization: float  # Normalization constant

    # Experiment capacity CES
    direct_input_exp_cap_ces_params: bool  # If True, use direct CES params; else derive from asymptotes
    rho_experiment_capacity: float  # Elasticity of substitution
    alpha_experiment_capacity: float  # Labor share parameter
    experiment_compute_exponent: float  # Exponent on compute in production

    # Experiment capacity asymptotes (used when direct_input_exp_cap_ces_params=False)
    inf_labor_asymptote: float  # Asymptote as labor -> infinity
    inf_compute_asymptote: float  # Asymptote as compute -> infinity
    labor_anchor_exp_cap: float  # Labor anchor point for calibration
    compute_anchor_exp_cap: Optional[float]  # Compute anchor (if None, derived from inverse)
    inv_compute_anchor_exp_cap: float  # Inverse of compute anchor (1/compute)

    # Parallel penalty
    parallel_penalty: float  # Penalty for parallel research efforts

    # =========================================================================
    # SOFTWARE PROGRESS PARAMETERS
    # =========================================================================

    r_software: float  # Returns to software R&D (exponent in progress function)
    software_progress_rate_at_reference_year: float  # Calibration point

    # =========================================================================
    # AUTOMATION SCHEDULE PARAMETERS
    # =========================================================================

    # Automation fraction dynamics
    automation_fraction_at_coding_automation_anchor: float  # Automation at anchor progress
    automation_anchors: Optional[Dict[float, float]]  # Progress -> automation fraction map
    automation_interp_type: str  # Interpolation type: "linear", "logistic", etc.
    automation_logistic_asymptote: float  # Max automation fraction (for logistic)
    swe_multiplier_at_present_day: float  # SWE productivity multiplier at present

    # Coding labor mode
    coding_labor_mode: str  # "simple" or "optimal_ces"
    coding_automation_efficiency_slope: float  # How quickly automation improves with progress
    optimal_ces_eta_init: float  # Initial eta for optimal CES
    optimal_ces_grid_size: int  # Grid size for optimal CES computation
    optimal_ces_frontier_tail_eps: float  # Epsilon for frontier tail
    optimal_ces_frontier_cap: float  # Cap on frontier
    max_serial_coding_labor_multiplier: float  # Max multiplier on serial coding labor

    # =========================================================================
    # AI RESEARCH TASTE PARAMETERS
    # =========================================================================

    ai_research_taste_at_coding_automation_anchor_sd: float  # AI research taste at anchor (SD units)
    ai_research_taste_slope: float  # Slope of taste schedule
    taste_schedule_type: str  # "linear", "logistic", etc.
    median_to_top_taste_multiplier: float  # Ratio of top to median taste
    top_percentile: float  # Percentile considered "top"
    taste_limit: float  # Upper limit on taste
    taste_limit_smoothing: float  # Smoothing parameter for taste limit

    # =========================================================================
    # HORIZON / MILESTONE PARAMETERS
    # =========================================================================

    progress_at_aa: Optional[float]  # Progress at advanced automation milestone
    ac_time_horizon_minutes: float  # Autonomy time horizon in minutes
    pre_gap_ac_time_horizon: float  # Horizon before capability gap
    horizon_extrapolation_type: str  # "linear", "exponential", etc.

    # Manual horizon fitting parameters
    present_day: float  # Reference year for calibration
    present_horizon: float  # Horizon at present day
    present_doubling_time: float  # Time for horizon to double
    doubling_difficulty_growth_factor: float  # Growth factor for doubling difficulty

    # Milestone multipliers (years to reach milestone = base * multiplier)
    strat_ai_m2b: float  # Strategic AI milestone multiplier
    ted_ai_m2b: float  # Ted AI milestone multiplier

    # =========================================================================
    # GAP MODE PARAMETERS
    # =========================================================================

    include_gap: Union[str, bool]  # Whether to include capability gap
    gap_years: float  # Duration of capability gap in years


# =============================================================================
# DEFAULT PARAMETER VALUES
# =============================================================================

def get_default_software_r_and_d_parameters() -> SoftwareRAndDParameters:
    """
    Get default software R&D parameters.

    These are reasonable defaults based on the ai-futures-calculator model.
    """
    return SoftwareRAndDParameters(
        # Mode flags
        human_only=False,

        # Production function (CES)
        rho_coding_labor=-2.0,
        coding_labor_normalization=1.0,
        direct_input_exp_cap_ces_params=False,
        rho_experiment_capacity=-1.0,
        alpha_experiment_capacity=0.5,
        experiment_compute_exponent=0.5,

        # Experiment capacity asymptotes
        inf_labor_asymptote=10.0,
        inf_compute_asymptote=10.0,
        labor_anchor_exp_cap=1000.0,
        compute_anchor_exp_cap=None,
        inv_compute_anchor_exp_cap=1e-6,

        # Parallel penalty
        parallel_penalty=0.5,

        # Software progress
        r_software=2.4,
        software_progress_rate_at_reference_year=0.1,

        # Automation schedule
        automation_fraction_at_coding_automation_anchor=0.5,
        automation_anchors=None,
        automation_interp_type="logistic",
        automation_logistic_asymptote=0.95,
        swe_multiplier_at_present_day=1.0,

        # Coding labor mode
        coding_labor_mode="simple",
        coding_automation_efficiency_slope=1.0,
        optimal_ces_eta_init=0.5,
        optimal_ces_grid_size=100,
        optimal_ces_frontier_tail_eps=0.01,
        optimal_ces_frontier_cap=10.0,
        max_serial_coding_labor_multiplier=10.0,

        # AI research taste
        ai_research_taste_at_coding_automation_anchor_sd=0.0,
        ai_research_taste_slope=1.0,
        taste_schedule_type="linear",
        median_to_top_taste_multiplier=2.0,
        top_percentile=0.1,
        taste_limit=10.0,
        taste_limit_smoothing=0.1,

        # Horizon / milestones
        progress_at_aa=None,
        ac_time_horizon_minutes=60.0,
        pre_gap_ac_time_horizon=30.0,
        horizon_extrapolation_type="exponential",
        present_day=2024.0,
        present_horizon=1.0,
        present_doubling_time=2.0,
        doubling_difficulty_growth_factor=1.5,
        strat_ai_m2b=1.0,
        ted_ai_m2b=1.0,

        # Gap mode
        include_gap=False,
        gap_years=0.0,
    )
