"""
Configuration file for the progress model.
Contains hardcoded values for numerical stability and internal implementation constants.

NOTE: User-configurable parameters should be in default_parameters.yaml, not here.
This file only contains implementation details that users shouldn't need to change.
"""

# =============================================================================
# NUMERICAL STABILITY & PRECISION
# =============================================================================
RHO_COBB_DOUGLAS_THRESHOLD = 1e-9
RHO_LEONTIEF_THRESHOLD = -50.0
SIGMOID_EXPONENT_CLAMP = 100.0
AUTOMATION_FRACTION_CLIP_MIN = 1e-9

# =============================================================================
# PARAMETER CLIPPING
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
# INTERNAL CONSTANTS
# =============================================================================
PARALLEL_LABOR_MULT_BETWEEN_AVERAGE_AND_TOP_FOR_AI2027_SC = 25
AGGREGATE_RESEARCH_TASTE_BASELINE = 1.0
AGGREGATE_RESEARCH_TASTE_FALLBACK = 1.0

# Standard deviations for milestone definitions
TOP_RESEARCHER_SD = 3.090232306167813  # SD corresponding to 99.9th percentile

# Taste floor interpolation grid configuration
TASTE_FLOOR_GRID_LOW_PERCENTILE = 0.001
TASTE_FLOOR_GRID_HIGH_PERCENTILE = 0.9999
TASTE_FLOOR_GRID_POINTS = 200

# Valid options for categorical parameters
TASTE_SCHEDULE_TYPES = ["SDs per effective OOM", "SDs per progress-year"]
HORIZON_EXTRAPOLATION_TYPES = ["exponential", "decaying doubling time"]
SOS_START_MILESTONES = ["AC", "AI2027-SC", "AIR-5x", "AIR-25x", "AIR-250x", "AIR-2000x", "AIR-10000x"]

# AI Research Taste clipping bounds
AI_RESEARCH_TASTE_MIN = 0.0
AI_RESEARCH_TASTE_MAX = 1e30
AI_RESEARCH_TASTE_MAX_SD = 10e30

BASE_FOR_SOFTWARE_LOM = 10.0

# Reference year constants
REFERENCE_YEAR = 2024.8
REFERENCE_LABOR_CHANGE = 30.0
REFERENCE_COMPUTE_CHANGE = 0.1
SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR = 2024.0

# Training compute reference
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
