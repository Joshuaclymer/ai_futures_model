/**
 * Parameter bounds overrides
 *
 * These bounds take the highest priority when determining slider min/max values.
 * Priority order: overrides > customMin/customMax props > API bounds > fallbackMin/fallbackMax
 *
 * Set min/max to override the bounds for a parameter, or leave undefined to use other sources.
 *
 * Example usage:
 * ```typescript
 * export const PARAMETER_BOUNDS_OVERRIDES: ParameterBoundsOverrides = {
 *   'software_r_and_d.present_doubling_time': { min: 0.05, max: 1.5 },  // Override both bounds
 *   'software_r_and_d.ai_research_taste_slope': { min: 0.5 },          // Override only min
 *   'software_r_and_d.median_to_top_taste_multiplier': undefined,       // No override
 * };
 * ```
 */

export interface BoundsOverride {
  min?: number;
  max?: number;
}

export type ParameterBoundsOverrides = Record<string, BoundsOverride | undefined>;

export const PARAMETER_BOUNDS_OVERRIDES: ParameterBoundsOverrides = {
  // Main UI Parameters (ProgressChartClient.tsx)
  'software_r_and_d.present_doubling_time': undefined,
  'software_r_and_d.ac_time_horizon_minutes': undefined,
  'software_r_and_d.doubling_difficulty_growth_factor': undefined,
  'software_r_and_d.ai_research_taste_slope': undefined,
  'software_r_and_d.median_to_top_taste_multiplier': undefined,

  // Advanced Parameters (AdvancedSections.tsx)

  // Time Horizon & Progress
  'software_r_and_d.saturation_horizon_minutes': { max: 50 * 2000 * 60 },  // 50 work years in minutes
  'software_r_and_d.gap_years': { max: 50 },

  // Coding Automation
  'software_r_and_d.swe_multiplier_at_present_day': { max: 100 },
  'software_r_and_d.coding_automation_efficiency_slope': undefined,
  'software_r_and_d.rho_coding_labor': undefined,
  'software_r_and_d.max_serial_coding_labor_multiplier': undefined,

  // Experiment Throughput Production
  'software_r_and_d.rho_experiment_capacity': undefined,
  'software_r_and_d.alpha_experiment_capacity': undefined,
  'software_r_and_d.experiment_compute_exponent': undefined,
  'software_r_and_d.coding_labor_exponent': undefined,
  'software_r_and_d.inf_labor_asymptote': undefined,
  'software_r_and_d.inf_compute_asymptote': undefined,
  'software_r_and_d.inv_compute_anchor_exp_cap': undefined,

  // Experiment Selection Automation
  'software_r_and_d.ai_research_taste_at_coding_automation_anchor_sd': undefined,
  'software_r_and_d.taste_limit': undefined,
  'software_r_and_d.taste_limit_smoothing': undefined,

  // General Capabilities
  'software_r_and_d.ted_ai_m2b': undefined,

  // Effective Compute
  'software_r_and_d.software_progress_rate_at_reference_year': undefined,

  // Training Compute Growth
  'compute.USComputeParameters.total_us_compute_annual_growth_rate': { min: 0, max: 10 },

  // Extra Parameters
  'software_r_and_d.present_day': undefined,
  'software_r_and_d.present_horizon': { max: 2000 * 60 },  // 1 work year in minutes
  'software_r_and_d.automation_fraction_at_coding_automation_anchor': undefined,
  'software_r_and_d.optimal_ces_eta_init': undefined,
  'software_r_and_d.top_percentile': undefined,
};
