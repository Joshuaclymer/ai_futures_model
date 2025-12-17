import pythonParameterConfig from '../config/python-parameter-config.json' assert { type: 'json' };

export type ParameterPrimitive = number | string | boolean | null;

export interface ParametersType {
    [key: string]: ParameterPrimitive;
    taste_schedule_type: string;
    present_doubling_time: number;
    ac_time_horizon_minutes: number;
    doubling_difficulty_growth_factor: number;
    rho_coding_labor: number;
    rho_experiment_capacity: number;
    alpha_experiment_capacity: number;
    direct_input_exp_cap_ces_params: boolean;
    r_software: number;
    software_progress_rate_at_reference_year: number;
    coding_labor_normalization: number;
    experiment_compute_exponent: number;
    coding_labor_exponent: number;
    automation_fraction_at_coding_automation_anchor: number;
    automation_interp_type: string;
    automation_logistic_asymptote: number;
    swe_multiplier_at_present_day: number;
    automation_anchors: ParameterPrimitive;
    ai_research_taste_at_coding_automation_anchor_sd: number;
    ai_research_taste_slope: number;
    progress_at_aa: number;
    saturation_horizon_minutes: number;
    present_day: number;
    present_horizon: number;
    horizon_extrapolation_type: string;
    inf_labor_asymptote: number;
    inf_compute_asymptote: number;
    labor_anchor_exp_cap: number;
    compute_anchor_exp_cap: ParameterPrimitive;
    inv_compute_anchor_exp_cap: number;
    benchmarks_and_gaps_mode: boolean;
    gap_years: number;
    coding_automation_efficiency_slope: number;
    max_serial_coding_labor_multiplier: number;
    median_to_top_taste_multiplier: number;
    top_percentile: number;
    taste_limit: number;
    taste_limit_smoothing: number;
    strat_ai_m2b: number;
    ted_ai_m2b: number;
    optimal_ces_eta_init: number;
    constant_training_compute_growth_rate: number;
    slowdown_year: number;
    post_slowdown_training_compute_growth_rate: number;
}

const uiDefaults = pythonParameterConfig.ui_defaults as ParametersType;

export const DEFAULT_PARAMETERS: ParametersType = {
    ...uiDefaults,
};

export const PYTHON_RAW_DEFAULTS = pythonParameterConfig.raw_defaults;
export const PYTHON_PARAMETER_BOUNDS = pythonParameterConfig.parameter_bounds;

// Model reference constants synced from Python model_config.py
export const MODEL_CONSTANTS = pythonParameterConfig.model_constants as {
    training_compute_reference_year: number;
    training_compute_reference_ooms: number;
    software_progress_scale_reference_year: number;
    base_for_software_lom: number;
};

// Convenience exports for commonly used constants
export const TRAINING_COMPUTE_REFERENCE_OOMS = MODEL_CONSTANTS.training_compute_reference_ooms;
export const TRAINING_COMPUTE_REFERENCE_YEAR = MODEL_CONSTANTS.training_compute_reference_year;

export function areParametersAtDefaults(parameters: ParametersType): boolean {
    return Object.keys(DEFAULT_PARAMETERS).every(key => {
        const paramKey = key as keyof ParametersType;
        const defaultValue = DEFAULT_PARAMETERS[paramKey];
        const currentValue = parameters[paramKey];
        return currentValue === defaultValue;
    });
}
