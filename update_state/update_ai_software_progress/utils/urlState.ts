"use client";

import { DEFAULT_PARAMETERS, ParametersType } from '@/constants/parameters';

type CheckboxStateKey =
  | 'enableCodingAutomation'
  | 'enableExperimentAutomation'
  | 'useExperimentThroughputCES'
  | 'enableSoftwareProgress'
  | 'useComputeLaborGrowthSlowdown'
  | 'useVariableHorizonDifficulty';

export interface FullUIState {
  parameters: ParametersType;
  enableCodingAutomation: boolean;
  enableExperimentAutomation: boolean;
  useExperimentThroughputCES: boolean;
  enableSoftwareProgress: boolean;
  useComputeLaborGrowthSlowdown: boolean;
  useVariableHorizonDifficulty: boolean;
}

export const DEFAULT_CHECKBOX_STATES: Record<CheckboxStateKey, boolean> = {
  enableCodingAutomation: true,
  enableExperimentAutomation: true,
  useExperimentThroughputCES: true,
  enableSoftwareProgress: true,
  useComputeLaborGrowthSlowdown: true,
  useVariableHorizonDifficulty: true,
};

const PARAM_ABBREVIATIONS: Record<string, keyof ParametersType> = {
  'tst': 'taste_schedule_type',
  'pdt': 'present_doubling_time',
  'acth': 'ac_time_horizon_minutes',
  'ddgf': 'doubling_difficulty_growth_factor',
  'rcl': 'rho_coding_labor',
  'rec': 'rho_experiment_capacity',
  'aec': 'alpha_experiment_capacity',
  'diec': 'direct_input_exp_cap_ces_params',
  'rsw': 'r_software',
  'spry': 'software_progress_rate_at_reference_year',
  'cln': 'coding_labor_normalization',
  'ece': 'experiment_compute_exponent',
  'cle': 'coding_labor_exponent',
  'afca': 'automation_fraction_at_coding_automation_anchor',
  'ait': 'automation_interp_type',
  'swm': 'swe_multiplier_at_present_day',
  'aa': 'automation_anchors',
  'artc': 'ai_research_taste_at_coding_automation_anchor_sd',
  'arts': 'ai_research_taste_slope',
  'paa': 'progress_at_aa',
  'shm': 'saturation_horizon_minutes',
  'pd': 'present_day',
  'ph': 'present_horizon',
  'het': 'horizon_extrapolation_type',
  'ila': 'inf_labor_asymptote',
  'ica': 'inf_compute_asymptote',
  'laec': 'labor_anchor_exp_cap',
  'caec': 'compute_anchor_exp_cap',
  'icaec': 'inv_compute_anchor_exp_cap',
  'bgm': 'benchmarks_and_gaps_mode',
  'gy': 'gap_years',
  'caes': 'coding_automation_efficiency_slope',
  'mscl': 'max_serial_coding_labor_multiplier',
  'mttm': 'median_to_top_taste_multiplier',
  'tp': 'top_percentile',
  'sam': 'strat_ai_m2b',
  'tam': 'ted_ai_m2b',
  'ocei': 'optimal_ces_eta_init',
};

const PARAM_ABBREVIATIONS_REVERSE: Record<string, string> = Object.fromEntries(
  Object.entries(PARAM_ABBREVIATIONS).map(([short, long]) => [long, short])
);

const CHECKBOX_ABBREVIATIONS: Record<string, string> = {
  'eca': 'enableCodingAutomation',
  'eea': 'enableExperimentAutomation',
  'uetc': 'useExperimentThroughputCES',
  'esp': 'enableSoftwareProgress',
  'uclgs': 'useComputeLaborGrowthSlowdown',
  'uvhd': 'useVariableHorizonDifficulty',
};

const CHECKBOX_ABBREVIATIONS_REVERSE: Record<string, string> = Object.fromEntries(
  Object.entries(CHECKBOX_ABBREVIATIONS).map(([short, long]) => [long, short])
);

const sanitizeParameterValue = (
  key: keyof ParametersType,
  value: string | null
): ParametersType[typeof key] | undefined => {
  if (value === null) {
    return undefined;
  }

  const defaultValue = DEFAULT_PARAMETERS[key];

  if (typeof defaultValue === 'number') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed as ParametersType[typeof key];
    }
    return undefined;
  }

  if (typeof defaultValue === 'string') {
    return value as ParametersType[typeof key];
  }

  if (typeof defaultValue === 'boolean') {
    if (value === 'true') {
      return true as ParametersType[typeof key];
    }
    if (value === 'false') {
      return false as ParametersType[typeof key];
    }
    return undefined;
  }

  if (defaultValue === null) {
    if (value === 'null') {
      return null as ParametersType[typeof key];
    }
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed as ParametersType[typeof key];
    }
    return value as ParametersType[typeof key];
  }

  return undefined;
};

export const encodeFullStateToParams = (state: FullUIState): URLSearchParams => {
  const params = new URLSearchParams();

  // Add all parameters that differ from defaults (using short names)
  (Object.keys(DEFAULT_PARAMETERS) as Array<keyof ParametersType>).forEach((paramKey) => {
    const value = state.parameters[paramKey];
    const defaultValue = DEFAULT_PARAMETERS[paramKey];

    if (value !== defaultValue) {
      const shortName = PARAM_ABBREVIATIONS_REVERSE[paramKey];
      if (shortName) {
        params.set(shortName, String(value));
      }
    }
  });

  // Add checkbox states that differ from defaults (using short names)
  if (state.enableCodingAutomation !== DEFAULT_CHECKBOX_STATES.enableCodingAutomation) {
    params.set(CHECKBOX_ABBREVIATIONS_REVERSE.enableCodingAutomation, String(state.enableCodingAutomation));
  }
  if (state.enableExperimentAutomation !== DEFAULT_CHECKBOX_STATES.enableExperimentAutomation) {
    params.set(CHECKBOX_ABBREVIATIONS_REVERSE.enableExperimentAutomation, String(state.enableExperimentAutomation));
  }
  if (state.useExperimentThroughputCES !== DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES) {
    params.set(CHECKBOX_ABBREVIATIONS_REVERSE.useExperimentThroughputCES, String(state.useExperimentThroughputCES));
  }
  if (state.enableSoftwareProgress !== DEFAULT_CHECKBOX_STATES.enableSoftwareProgress) {
    params.set(CHECKBOX_ABBREVIATIONS_REVERSE.enableSoftwareProgress, String(state.enableSoftwareProgress));
  }
  if (state.useComputeLaborGrowthSlowdown !== DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown) {
    params.set(CHECKBOX_ABBREVIATIONS_REVERSE.useComputeLaborGrowthSlowdown, String(state.useComputeLaborGrowthSlowdown));
  }
  if (state.useVariableHorizonDifficulty !== DEFAULT_CHECKBOX_STATES.useVariableHorizonDifficulty) {
    params.set(CHECKBOX_ABBREVIATIONS_REVERSE.useVariableHorizonDifficulty, String(state.useVariableHorizonDifficulty));
  }

  return params;
};

export const decodeFullStateFromParams = (searchParams: URLSearchParams): FullUIState => {
  const parameters: ParametersType = { ...DEFAULT_PARAMETERS };

  Object.entries(PARAM_ABBREVIATIONS).forEach(([shortName, paramKey]) => {
    const value = searchParams.get(shortName);

    if (value !== null) {
      const sanitized = sanitizeParameterValue(paramKey, value);
      if (sanitized !== undefined) {
        parameters[paramKey] = sanitized;
      }
    }
  });

  const enableCodingAutomation = searchParams.has(CHECKBOX_ABBREVIATIONS_REVERSE.enableCodingAutomation)
    ? searchParams.get(CHECKBOX_ABBREVIATIONS_REVERSE.enableCodingAutomation) === 'true'
    : DEFAULT_CHECKBOX_STATES.enableCodingAutomation;

  const enableExperimentAutomation = searchParams.has(CHECKBOX_ABBREVIATIONS_REVERSE.enableExperimentAutomation)
    ? searchParams.get(CHECKBOX_ABBREVIATIONS_REVERSE.enableExperimentAutomation) === 'true'
    : DEFAULT_CHECKBOX_STATES.enableExperimentAutomation;

  const useExperimentThroughputCES = searchParams.has(CHECKBOX_ABBREVIATIONS_REVERSE.useExperimentThroughputCES)
    ? searchParams.get(CHECKBOX_ABBREVIATIONS_REVERSE.useExperimentThroughputCES) === 'true'
    : DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES;

  const enableSoftwareProgress = searchParams.has(CHECKBOX_ABBREVIATIONS_REVERSE.enableSoftwareProgress)
    ? searchParams.get(CHECKBOX_ABBREVIATIONS_REVERSE.enableSoftwareProgress) === 'true'
    : DEFAULT_CHECKBOX_STATES.enableSoftwareProgress;

  const useComputeLaborGrowthSlowdown = searchParams.has(CHECKBOX_ABBREVIATIONS_REVERSE.useComputeLaborGrowthSlowdown)
    ? searchParams.get(CHECKBOX_ABBREVIATIONS_REVERSE.useComputeLaborGrowthSlowdown) === 'true'
    : DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown;

  const useVariableHorizonDifficulty = searchParams.has(CHECKBOX_ABBREVIATIONS_REVERSE.useVariableHorizonDifficulty)
    ? searchParams.get(CHECKBOX_ABBREVIATIONS_REVERSE.useVariableHorizonDifficulty) === 'true'
    : DEFAULT_CHECKBOX_STATES.useVariableHorizonDifficulty;

  return {
    parameters,
    enableCodingAutomation,
    enableExperimentAutomation,
    useExperimentThroughputCES,
    enableSoftwareProgress,
    useComputeLaborGrowthSlowdown,
    useVariableHorizonDifficulty,
  };
};

