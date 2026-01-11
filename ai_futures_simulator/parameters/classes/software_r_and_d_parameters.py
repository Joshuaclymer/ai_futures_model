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
    update_software_progress: bool

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
