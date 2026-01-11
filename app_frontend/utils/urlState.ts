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

// Maps short URL codes to full parameter paths (matching backend structure)
const PARAM_ABBREVIATIONS: Record<string, string> = {
  'tst': 'software_r_and_d.taste_schedule_type',
  'pdt': 'software_r_and_d.present_doubling_time',
  'acth': 'software_r_and_d.ac_time_horizon_minutes',
  'ddgf': 'software_r_and_d.doubling_difficulty_growth_factor',
  'rcl': 'software_r_and_d.rho_coding_labor',
  'rec': 'software_r_and_d.rho_experiment_capacity',
  'aec': 'software_r_and_d.alpha_experiment_capacity',
  'diec': 'software_r_and_d.direct_input_exp_cap_ces_params',
  'rsw': 'software_r_and_d.r_software',
  'spry': 'software_r_and_d.software_progress_rate_at_reference_year',
  'cln': 'software_r_and_d.coding_labor_normalization',
  'ece': 'software_r_and_d.experiment_compute_exponent',
  'cle': 'software_r_and_d.coding_labor_exponent',
  'afca': 'software_r_and_d.automation_fraction_at_coding_automation_anchor',
  'ait': 'software_r_and_d.automation_interp_type',
  'swm': 'software_r_and_d.swe_multiplier_at_present_day',
  'aa': 'software_r_and_d.automation_anchors',
  'artc': 'software_r_and_d.ai_research_taste_at_coding_automation_anchor_sd',
  'arts': 'software_r_and_d.ai_research_taste_slope',
  'paa': 'software_r_and_d.progress_at_aa',
  'shm': 'software_r_and_d.saturation_horizon_minutes',
  'pd': 'software_r_and_d.present_day',
  'ph': 'software_r_and_d.present_horizon',
  'het': 'software_r_and_d.horizon_extrapolation_type',
  'ila': 'software_r_and_d.inf_labor_asymptote',
  'ica': 'software_r_and_d.inf_compute_asymptote',
  'laec': 'software_r_and_d.labor_anchor_exp_cap',
  'caec': 'software_r_and_d.compute_anchor_exp_cap',
  'icaec': 'software_r_and_d.inv_compute_anchor_exp_cap',
  'bgm': 'software_r_and_d.benchmarks_and_gaps_mode',
  'gy': 'software_r_and_d.gap_years',
  'caes': 'software_r_and_d.coding_automation_efficiency_slope',
  'mscl': 'software_r_and_d.max_serial_coding_labor_multiplier',
  'mttm': 'software_r_and_d.median_to_top_taste_multiplier',
  'tp': 'software_r_and_d.top_percentile',
  'sam': 'software_r_and_d.strat_ai_m2b',
  'tam': 'software_r_and_d.ted_ai_m2b',
  'ocei': 'software_r_and_d.optimal_ces_eta_init',
  'ala': 'software_r_and_d.automation_logistic_asymptote',
  'tl': 'software_r_and_d.taste_limit',
  'tls': 'software_r_and_d.taste_limit_smoothing',
  'uscgr': 'compute.USComputeParameters.total_us_compute_annual_growth_rate',
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
  key: string,
  value: string | null
): string | number | boolean | null | undefined => {
  if (value === null) {
    return undefined;
  }

  const defaultValue = DEFAULT_PARAMETERS[key];

  if (typeof defaultValue === 'number') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
    return undefined;
  }

  if (typeof defaultValue === 'string') {
    return value;
  }

  if (typeof defaultValue === 'boolean') {
    if (value === 'true') {
      return true;
    }
    if (value === 'false') {
      return false;
    }
    return undefined;
  }

  if (defaultValue === null) {
    if (value === 'null') {
      return null;
    }
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
    return value;
  }

  return undefined;
};

export const encodeFullStateToParams = (state: FullUIState): URLSearchParams => {
  const params = new URLSearchParams();

  // Add all parameters that differ from defaults (using short names)
  Object.keys(DEFAULT_PARAMETERS).forEach((paramKey) => {
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
