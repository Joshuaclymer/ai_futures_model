"""
Configuration file for the progress model.
Contains hardcoded values for numerical stability and internal implementation constants.

NOTE: User-configurable parameters should be in default_parameters.yaml, not here.
This file should only contain implementation details that users shouldn't need to change.
"""

# =============================================================================
# NUMERICAL STABILITY & PRECISION
# =============================================================================
RHO_COBB_DOUGLAS_THRESHOLD = 1e-9
RHO_LEONTIEF_THRESHOLD = -50.0
SIGMOID_EXPONENT_CLAMP = 100.0
AUTOMATION_FRACTION_CLIP_MIN = 1e-9

# =============================================================================
# PARAMETER CLIPPING (in Parameters.__post_init__)
# =============================================================================
RHO_CLIP_MIN = -50.0
PARAM_CLIP_MIN = 1e-6
AUTOMATION_SLOPE_CLIP_MIN = 0.1
AUTOMATION_SLOPE_CLIP_MAX = 10.0
RESEARCH_STOCK_START_MIN = 1e-10
NORMALIZATION_MIN = 1e-10
experiment_compute_exponent_CLIP_MIN = 0.001
experiment_compute_exponent_CLIP_MAX = 10.0
PARALLEL_PENALTY_MIN = 0.0
PARALLEL_PENALTY_MAX = 1.0

# =============================================================================
# INTERNAL CONSTANTS (not user-configurable)
# =============================================================================
PARALLEL_LABOR_MULT_BETWEEN_AVERAGE_AND_TOP_FOR_AI2027_SC = 25
AGGREGATE_RESEARCH_TASTE_BASELINE = 1.0
AGGREGATE_RESEARCH_TASTE_FALLBACK = 1.0

# Standard deviations for milestone definitions (computed from scipy.stats.norm.ppf)
TOP_RESEARCHER_SD = 3.090232306167813  # SD corresponding to 99.9th percentile

# Taste floor interpolation grid configuration
TASTE_FLOOR_GRID_LOW_PERCENTILE = 0.001
TASTE_FLOOR_GRID_HIGH_PERCENTILE = 0.9999
TASTE_FLOOR_GRID_POINTS = 200

# Research Taste Schedule Configuration
TASTE_SCHEDULE_TYPES = ["SDs per effective OOM", "SDs per progress-year"]

# Horizon Extrapolation Configuration
HORIZON_EXTRAPOLATION_TYPES = ["exponential", "decaying doubling time"]

# SOS (Software-Only Singularity) Mode Configuration
SOS_START_MILESTONES = ["AC", "AI2027-SC", "AIR-5x", "AIR-25x", "AIR-250x", "AIR-2000x", "AIR-10000x"]

# AI Research Taste clipping bounds
AI_RESEARCH_TASTE_MIN = 0.0
AI_RESEARCH_TASTE_MAX = 1e30
AI_RESEARCH_TASTE_MAX_SD = 10e30

BASE_FOR_SOFTWARE_LOM = 10.0

# Reference year constants for anchoring calculations
REFERENCE_YEAR = 2024.8
REFERENCE_LABOR_CHANGE = 30.0
REFERENCE_COMPUTE_CHANGE = 0.1
SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR = 2024.0

# Training compute reference for normalizing software efficiency
TRAINING_COMPUTE_REFERENCE_YEAR = 2025.13
TRAINING_COMPUTE_REFERENCE_OOMS = 26.54

# Capability reference for ECI calculations
CAPABILITY_REFERENCE_SCORE = 142
CAPABILITY_POINTS_PER_OOM = 9.6

# =============================================================================
# MODEL RATE & VALUE CAPS
# =============================================================================
MAX_RESEARCH_EFFORT = 1e30
MAX_NORMALIZED_PROGRESS_RATE = 1e30
TIME_EXTRAPOLATION_WINDOW = 10.0

# =============================================================================
# ODE INTEGRATION
# =============================================================================
PROGRESS_ODE_CLAMP_MAX = 1e30
RESEARCH_STOCK_ODE_CLAMP_MAX = 1e30
ODE_MAX_STEP = 1.0
EULER_FALLBACK_MIN_STEPS = 100
EULER_FALLBACK_STEPS_PER_YEAR = 10
DENSE_OUTPUT_STEP_SIZE = 0.1

# ODE step size logging configuration
ODE_STEP_SIZE_LOGGING = False
ODE_SMALL_STEP_THRESHOLD = 1e-6
ODE_STEP_VARIATION_THRESHOLD = 100.0

# =============================================================================
# PARAMETER ESTIMATION
# =============================================================================
RELATIVE_ERROR_CLIP = 100.0

PARAM_VALIDATION_THRESHOLDS = {
    'automation_fraction_superhuman_coder_min': 0.05,
    'automation_fraction_superhuman_coder_max': 1,
    'rho_extreme_abs': 0.8,
    'rho_product_max': 0.5,
    'coding_labor_normalization_max': 10
}

FEASIBILITY_CHECK_THRESHOLDS = {
    'progress_rate_target_max': 1000.0,
}

OBJECTIVE_FUNCTION_CONFIG = {
    'high_penalty': 1e6,
    'elasticity_regularization_weight': 0.001,
    'boundary_avoidance_regularization_weight': 0.001,
    'boundary_avoidance_threshold': 0.35,
}

OPTIMIZATION_CONFIG = {
    'early_termination_fun_threshold_excellent': 1e-6,
    'early_termination_fun_threshold_good': 1e-3,
}

STRATEGIC_STARTING_POINTS_CONFIG = {
    'extreme_factor_min': 0.1,
    'extreme_factor_max': 0.9,
    'lhs_points': 5,
    'high_progress_rate_threshold': 2.0,
    'rho_adjustment_factor': 0.8,
    'high_automation_threshold': 0.5,
    'progress_at_half_automation_adjustment_factor': 0.7,
    'perturbed_points': 3,
    'critical_param_perturbation_factor': 0.1,
    'other_param_perturbation_factor': 0.2
}

# =============================================================================
# DEFAULT PARAMETERS (fallbacks when not specified in default_parameters.yaml)
# These should match the values in default_parameters.yaml where applicable.
# =============================================================================
DEFAULT_PARAMETERS = {
    # CES parameters
    'rho_coding_labor': -2,
    'direct_input_exp_cap_ces_params': False,
    'rho_experiment_capacity': -0.155,
    'alpha_experiment_capacity': 0.809,
    'r_software': 2.40,
    'software_progress_rate_at_reference_year': 1,
    'experiment_compute_exponent': 0.655,
    # Automation parameters
    'automation_fraction_at_coding_automation_anchor': 1,
    'automation_anchors': None,
    'automation_interp_type': "linear",
    'automation_logistic_asymptote': 1.05,
    'swe_multiplier_at_present_day': 1.6,
    'coding_labor_normalization': 1,
    # AI Research Taste parameters
    'ai_research_taste_at_coding_automation_anchor_sd': 0.5,
    'ai_research_taste_at_coding_automation_anchor_fallback': 1,
    'ai_research_taste_slope': 2.2,
    'taste_schedule_type': "SDs per progress-year",
    # Horizon parameters
    'progress_at_aa': None,
    'ac_time_horizon_minutes': 1.5e7,
    'horizon_extrapolation_type': "decaying doubling time",
    'present_day': 2025.6,
    'present_horizon': 26.0,
    'present_doubling_time': 0.458,
    'doubling_difficulty_growth_factor': 0.92,
    # Experiment capacity parameters
    'inf_labor_asymptote': 15.0,
    'inf_compute_asymptote': 1000,
    'labor_anchor_exp_cap': 1.6,
    'compute_anchor_exp_cap': None,
    'inv_compute_anchor_exp_cap': 2.8,
    'parallel_penalty': 0.5,
    # Gap mode
    'include_gap': 'no gap',
    'gap_years': 0.7,
    # Research taste distribution
    'median_to_top_taste_multiplier': 3.7,
    'top_percentile': 0.999,
    'taste_limit': 8,
    'taste_limit_smoothing': 0.5,
    # Milestone parameters
    'strat_ai_m2b': 2.0,
    'ted_ai_m2b': 1.5,
    # Coding labor mode
    'coding_automation_efficiency_slope': 3.0,
    'coding_labor_mode': 'optimal_ces',
    'optimal_ces_eta_init': 0.05,
    'optimal_ces_grid_size': 4096,
    'optimal_ces_frontier_tail_eps': 1e-6,
    'optimal_ces_frontier_cap': 1e12,
    'max_serial_coding_labor_multiplier': 1e8,
    # SOS mode
    'sos_mode': False,
    'sos_start_milestone': "AC",
    # Blacksite parameters
    'plan_a_mode': False,
    'is_blacksite': False,
    'show_blacksite': False,
    'sw_leaks_to_blacksite': True,
    'plan_a_start_time': 2030.0,
    'main_project_training_compute_growth_rate': 0.8,
    'main_project_software_progress_rate': 0.5,
    'blacksite_initial_years_behind': 0.0,
    'blacksite_training_compute_penalty_ooms': 2,
    'blacksite_training_compute_growth_rate': 0,
    'blacksite_human_labor_penalty_ooms': 2,
    'blacksite_experiment_compute_penalty_ooms': 2,
    'blacksite_inference_compute_penalty_ooms': 2,
    'blacksite_human_taste_penalty': 2,
    'blacksite_can_stack_training_compute': False,
    'blacksite_can_stack_software_progress': False,
    'blacksite_start_time': 2030.0,
    # Training compute growth
    'constant_training_compute_growth_rate': 0.6,
    'slowdown_year': 2028.0,
    'post_slowdown_training_compute_growth_rate': 0.25,
}

# Convenience accessors for commonly used defaults
TOP_PERCENTILE = DEFAULT_PARAMETERS['top_percentile']
MEDIAN_TO_TOP_TASTE_MULTIPLIER = DEFAULT_PARAMETERS['median_to_top_taste_multiplier']
DEFAULT_TASTE_SCHEDULE_TYPE = DEFAULT_PARAMETERS['taste_schedule_type']
DEFAULT_HORIZON_EXTRAPOLATION_TYPE = DEFAULT_PARAMETERS['horizon_extrapolation_type']
DEFAULT_SOS_START_MILESTONE = DEFAULT_PARAMETERS['sos_start_milestone']
DEFAULT_present_day = DEFAULT_PARAMETERS['present_day']
DEFAULT_present_horizon = DEFAULT_PARAMETERS['present_horizon']
DEFAULT_present_doubling_time = DEFAULT_PARAMETERS['present_doubling_time']
DEFAULT_DOUBLING_DIFFICULTY_GROWTH_FACTOR = DEFAULT_PARAMETERS['doubling_difficulty_growth_factor']

# Taste slope defaults by schedule type
TASTE_SLOPE_DEFAULTS = {
    "SDs per effective OOM": 1.47,
    "SDs per progress-year": 2.2
}
