export type ParameterPrimitive = number | string | boolean | null;

// Parameters now use full dot-notation paths matching the backend structure
// e.g., 'software_r_and_d.rho_coding_labor' instead of 'rho_coding_labor'
export type ParametersType = Record<string, ParameterPrimitive>;

// Model reference constants - these rarely change so kept as hardcoded values
// Synced from default_parameters.yaml software_r_and_d section
export const MODEL_CONSTANTS = {
    training_compute_reference_year: 2025.13,
    training_compute_reference_ooms: 26.54,
    software_progress_scale_reference_year: 2024.0,
    base_for_software_lom: 10.0,
};

// Convenience exports for commonly used constants
export const TRAINING_COMPUTE_REFERENCE_OOMS = MODEL_CONSTANTS.training_compute_reference_ooms;
export const TRAINING_COMPUTE_REFERENCE_YEAR = MODEL_CONSTANTS.training_compute_reference_year;

// Offset to convert H100e units (tpp_h100e) to FLOP OOMs
// Derived from: TRAINING_COMPUTE_REFERENCE_OOMS - log10(experimentCompute at reference year)
// At 2025.13, experimentCompute ≈ 140000, so offset ≈ 26.54 - 5.15 ≈ 21.4
export const H100E_TPP_TO_FLOP_OOM_OFFSET = 21.4;
