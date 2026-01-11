/**
 * TypeScript types for Black Project API response and components.
 *
 * These types match the structure returned by format_data_for_black_project_plots.py
 */

// Percentile data for time series
export interface TimeSeriesPercentiles {
  p25: number[];
  median: number[];
  p75: number[];
  individual?: number[][];
}

// CCDF data point
export interface CCDFPoint {
  x: number;
  y: number;
}

// CCDF data for multiple thresholds
export type CCDFByThreshold = Record<number, CCDFPoint[]>;

// Black project model main data
export interface BlackProjectModel {
  years: number[];

  // Dashboard values (individual simulation data at detection threshold)
  project_80th_h100_years: number[];
  project_80th_h100e: number[];
  project_80th_time: number[];

  // CCDF plots
  h100_years_ccdf: CCDFByThreshold;
  average_covert_compute_ccdf: CCDFByThreshold;
  time_to_detection_ccdf: CCDFByThreshold;
  ai_rd_reduction_ccdf: {
    global: CCDFByThreshold;
    prc: CCDFByThreshold;
  };
  chip_production_reduction_ccdf: CCDFByThreshold;
  likelihood_ratios: number[];

  // Time series data
  covert_computation_h100_years: TimeSeriesPercentiles;
  cumulative_lr: TimeSeriesPercentiles;
  initial_black_project: TimeSeriesPercentiles;
  black_fab_flow: TimeSeriesPercentiles;
  black_fab_flow_all_sims: TimeSeriesPercentiles;
  black_fab_monthly_flow_all_sims: TimeSeriesPercentiles;
  survival_rate: TimeSeriesPercentiles;
  total_black_project: TimeSeriesPercentiles;
  datacenter_capacity: TimeSeriesPercentiles;
  black_project_energy: number[][];
  energy_source_labels: string[];
  operational_compute: TimeSeriesPercentiles;

  // LR components
  lr_initial_stock: TimeSeriesPercentiles;
  lr_diverted_sme: TimeSeriesPercentiles;
  lr_other_intel: TimeSeriesPercentiles;
  posterior_prob_project: TimeSeriesPercentiles;
  lr_prc_accounting: TimeSeriesPercentiles;
  lr_sme_inventory: TimeSeriesPercentiles;
  lr_satellite_datacenter: { individual: number[] };
  lr_reported_energy: TimeSeriesPercentiles;
  lr_combined_reported_assets: TimeSeriesPercentiles;

  // Individual simulation data for dashboard
  individual_project_h100e_before_detection: number[];
  individual_project_energy_before_detection: number[];
  individual_project_time_before_detection: number[];
  individual_project_h100_years_before_detection: number[];
}

// Initial stock data
export interface InitialBlackProject {
  years: number[];
  h100e: TimeSeriesPercentiles;
  survival_rate: TimeSeriesPercentiles;
  black_project: TimeSeriesPercentiles;
}

// Initial stock data
export interface InitialStock {
  initial_prc_stock_samples: number[];
  initial_compute_stock_samples: number[];
  diversion_proportion: number;
  lr_prc_accounting_samples: number[];
  initial_black_project_detection_probs: Record<string, number>;
}

// Black datacenters data
export interface BlackDatacenters {
  years: number[];
  datacenter_capacity: TimeSeriesPercentiles;
  energy_by_source: number[][];
  source_labels: string[];
  operational_compute: TimeSeriesPercentiles;
  lr_datacenters: TimeSeriesPercentiles;
  datacenter_detection_prob: number[];
  capacity_ccdfs: CCDFByThreshold;
  likelihood_ratios: number[];
  individual_capacity_before_detection: number[];
  individual_time_before_detection: number[];

  // PRC capacity trajectory
  prc_capacity_years: number[];
  prc_capacity_gw: {
    p25: number[];
    median: number[];
    p75: number[];
  };
  prc_capacity_at_ai_slowdown_start_year_gw: number;
  prc_capacity_at_ai_slowdown_start_year_samples: number[];
  fraction_diverted: number;

  // Min[] formula parameters
  total_prc_energy_gw?: number;
  max_proportion_energy?: number;
  construction_workers?: number;
  mw_per_worker_per_year?: number;
  datacenter_start_year?: number;
}

// Black fab data
export interface BlackFab {
  years: number[];

  // Detection data
  compute_ccdf: CCDFPoint[];
  compute_ccdfs: CCDFByThreshold;
  op_time_ccdf: CCDFPoint[];
  op_time_ccdfs: CCDFByThreshold;
  likelihood_ratios: number[];

  // LR components
  lr_inventory: TimeSeriesPercentiles;
  lr_procurement: TimeSeriesPercentiles;
  lr_other: TimeSeriesPercentiles;
  lr_combined: TimeSeriesPercentiles;

  // Production parameters
  is_operational: {
    proportion: number[];
    individual: boolean[][];
  };
  wafer_starts: TimeSeriesPercentiles;
  chips_per_wafer: TimeSeriesPercentiles;
  architecture_efficiency: TimeSeriesPercentiles;
  compute_per_wafer_2022_arch: TimeSeriesPercentiles;
  transistor_density: TimeSeriesPercentiles;
  watts_per_tpp: TimeSeriesPercentiles;
  process_node_by_sim: string[];
  architecture_efficiency_at_agreement: number;
  watts_per_tpp_curve: { x: number; y: number }[];

  // Individual simulation data
  individual_h100e_before_detection: number[];
  individual_time_before_detection: number[];
  individual_process_node: string[];
  individual_energy_before_detection: number[];
}

// Complete API response
export interface BlackProjectData {
  num_simulations: number;
  prob_fab_built: number;
  p_project_exists: number;
  researcher_headcount: number;

  black_project_model: BlackProjectModel;
  initial_black_project: InitialBlackProject;
  initial_stock?: InitialStock;
  black_datacenters: BlackDatacenters;
  black_fab: BlackFab;
}

// Parameter types for the sidebar
export interface BlackProjectParameters {
  // Simulation settings
  agreementYear: number;
  timeStepYears: number;
  numSimulations: number;

  // Initial compute
  proportionOfInitialChipStockToDivert: number;

  // Datacenters
  datacenterConstructionLabor: number;
  yearsBeforeAgreementYearPRCStartsBuildingBlackDatacenters: number;
  maxProportionOfPRCEnergyConsumption: number;
  fractionOfDatacenterCapacityDiverted: number;

  // Fab
  buildCovertFab: boolean;
  operatingLabor: number;
  constructionLabor: number;
  processNode: string;
  scannerProportion: number;

  // AI Research
  researcherHeadcount: number;

  // Detection
  pProjectExists: number;
  meanDetectionTime100: number;
  meanDetectionTime1000: number;
  varianceDetectionTime: number;
  detectionThreshold: number;
}

// Color palette
export const COLOR_PALETTE = {
  chip_stock: '#5E6FB8',
  fab: '#E9A842',
  datacenters_and_energy: '#4AA896',
  detection: '#7BA3C4',
  survival_rate: '#E05A4F',
  gray: '#7F8C8D',
} as const;

export type ColorName = keyof typeof COLOR_PALETTE;

// Helper function to convert hex to rgba
export function hexToRgba(colorName: ColorName, alpha: number): string {
  const hex = COLOR_PALETTE[colorName] || '#000000';
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
