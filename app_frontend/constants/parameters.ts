import pythonParameterConfig from '../config/python-parameter-config.json' assert { type: 'json' };

export type ParameterPrimitive = number | string | boolean | null;

// Parameters now use full dot-notation paths matching the backend structure
// e.g., 'software_r_and_d.rho_coding_labor' instead of 'rho_coding_labor'
export type ParametersType = Record<string, ParameterPrimitive>;

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

// Offset to convert H100e units (tpp_h100e) to FLOP OOMs
// Derived from: TRAINING_COMPUTE_REFERENCE_OOMS - log10(experimentCompute at reference year)
// At 2025.13, experimentCompute ≈ 140000, so offset ≈ 26.54 - 5.15 ≈ 21.4
export const H100E_TPP_TO_FLOP_OOM_OFFSET = 21.4;

export function areParametersAtDefaults(parameters: ParametersType): boolean {
    return Object.keys(DEFAULT_PARAMETERS).every(key => {
        const defaultValue = DEFAULT_PARAMETERS[key];
        const currentValue = parameters[key];
        return currentValue === defaultValue;
    });
}
