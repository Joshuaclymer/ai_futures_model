/**
 * Types for the Black Project page components.
 */

// CCDF point type
export interface CCDFPoint {
  x: number;
  y: number;
}

// Percentile data for time series
export interface TimeSeriesPercentiles {
  p25: number[];
  median: number[];
  p75: number[];
  individual?: number[][];
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
  lr_compute_accounting: TimeSeriesPercentiles;
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
  lr_compute_accounting_samples: number[];
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

  // PRC datacenter capacity trajectory (energy in GW)
  prc_datacenter_capacity_years: number[];
  prc_datacenter_capacity_gw: {
    p25: number[];
    median: number[];
    p75: number[];
  };
  prc_datacenter_capacity_at_black_project_start_year_gw: number;
  prc_datacenter_capacity_at_black_project_start_year_samples: number[];
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

// Parameter types for the simulation
export interface Parameters {
  // Key Parameters
  numSimulations: number;
  agreementYear: number;
  blackProjectStartYear: number;
  workersInCovertProject: number;
  meanDetectionTime100: number;
  meanDetectionTime1000: number;
  varianceDetectionTime: number;
  proportionOfInitialChipStockToDivert: number;
  intelligenceMedianError: number;

  // Black Project Properties (from BlackProjectProperties class)
  totalLabor: number;
  fractionOfLaborDevotedToDatacenterConstruction: number;
  fractionOfLaborDevotedToBlackFabConstruction: number;
  fractionOfLaborDevotedToBlackFabOperation: number;
  fractionOfLaborDevotedToAiResearch: number;
  fractionOfDatacenterCapacityToDivert: number;
  fractionOfLithographyScannersToDivert: number;
  maxFractionOfTotalNationalEnergyConsumption: number;
  buildCovertFab: boolean;
  blackFabMaxProcessNode: string;

  // Detection Parameters (from BlackProjectPerceptionsParameters class)
  intelligenceMedianErrorInEstimateOfFabStock: number;
  intelligenceMedianErrorInEnergyConsumptionEstimate: number;
  intelligenceMedianErrorInSatelliteEstimate: number;
  detectionThreshold: number;
  priorOddsOfCovertProject: number;

  // PRC Compute (from PRCComputeParameters class)
  totalPrcComputeTppH100eIn2025: number;
  annualGrowthRateOfPrcComputeStock: number;
  prcArchitectureEfficiencyRelativeToStateOfTheArt: number;
  proportionOfPrcChipStockProducedDomestically2026: number;
  proportionOfPrcChipStockProducedDomestically2030: number;
  prcLithographyScannersProducedInFirstYear: number;
  prcAdditionalLithographyScannersProducedPerYear: number;
  pLocalization28nm2030: number;
  pLocalization14nm2030: number;
  pLocalization7nm2030: number;
  h100SizedChipsPerWafer: number;
  wafersPerMonthPerLithographyScanner: number;
  constructionTimeFor5kWafersPerMonth: number;
  constructionTimeFor100kWafersPerMonth: number;
  fabWafersPerMonthPerOperatingWorker: number;
  fabWafersPerMonthPerConstructionWorker: number;

  // PRC Data Centers and Energy (from PRCDataCenterAndEnergyParameters class)
  energyEfficiencyOfComputeStockRelativeToStateOfTheArt: number;
  totalPrcEnergyConsumptionGw: number;
  dataCenterMwPerYearPerConstructionWorker: number;
  dataCenterMwPerOperatingWorker: number;

  // US Compute (from USComputeParameters class)
  usFrontierProjectComputeTppH100eIn2025: number;
  usFrontierProjectComputeAnnualGrowthRate: number;

  // Compute Survival (from SurvivalRateParameters class)
  initialAnnualHazardRate: number;
  annualHazardRateIncreasePerYear: number;

  // Exogenous Compute Trends (from ExogenousComputeTrends class)
  transistorDensityScalingExponent: number;
  stateOfTheArtArchitectureEfficiencyImprovementPerYear: number;
  transistorDensityAtEndOfDennardScaling: number;
  wattsTppDensityExponentBeforeDennard: number;
  wattsTppDensityExponentAfterDennard: number;
  stateOfTheArtEnergyEfficiencyImprovementPerYear: number;
}

// Default parameters
export const defaultParameters: Parameters = {
  // Key Parameters
  numSimulations: 100,
  agreementYear: 2030,
  blackProjectStartYear: 2029,
  workersInCovertProject: 11300,
  meanDetectionTime100: 6.95,
  meanDetectionTime1000: 3.42,
  varianceDetectionTime: 3.88,
  proportionOfInitialChipStockToDivert: 0.05,
  intelligenceMedianError: 0.07,

  // Black Project Properties
  totalLabor: 11300,
  fractionOfLaborDevotedToDatacenterConstruction: 0.885,
  fractionOfLaborDevotedToBlackFabConstruction: 0.022,
  fractionOfLaborDevotedToBlackFabOperation: 0.049,
  fractionOfLaborDevotedToAiResearch: 0.044,
  fractionOfDatacenterCapacityToDivert: 0.5,
  fractionOfLithographyScannersToDivert: 0.10,
  maxFractionOfTotalNationalEnergyConsumption: 0.05,
  buildCovertFab: true,
  blackFabMaxProcessNode: '28',

  // Detection Parameters
  intelligenceMedianErrorInEstimateOfFabStock: 0.07,
  intelligenceMedianErrorInEnergyConsumptionEstimate: 0.07,
  intelligenceMedianErrorInSatelliteEstimate: 0.01,
  detectionThreshold: 100.0,
  priorOddsOfCovertProject: 0.3,

  // PRC Compute
  totalPrcComputeTppH100eIn2025: 100000,
  annualGrowthRateOfPrcComputeStock: 2.2,
  prcArchitectureEfficiencyRelativeToStateOfTheArt: 1.0,
  proportionOfPrcChipStockProducedDomestically2026: 0.10,
  proportionOfPrcChipStockProducedDomestically2030: 0.40,
  prcLithographyScannersProducedInFirstYear: 20,
  prcAdditionalLithographyScannersProducedPerYear: 16,
  pLocalization28nm2030: 0.25,
  pLocalization14nm2030: 0.10,
  pLocalization7nm2030: 0.06,
  h100SizedChipsPerWafer: 28,
  wafersPerMonthPerLithographyScanner: 1000,
  constructionTimeFor5kWafersPerMonth: 1.4,
  constructionTimeFor100kWafersPerMonth: 2.41,
  fabWafersPerMonthPerOperatingWorker: 24.64,
  fabWafersPerMonthPerConstructionWorker: 14.1,

  // PRC Data Centers and Energy
  energyEfficiencyOfComputeStockRelativeToStateOfTheArt: 0.20,
  totalPrcEnergyConsumptionGw: 1100,
  dataCenterMwPerYearPerConstructionWorker: 1.0,
  dataCenterMwPerOperatingWorker: 10.0,

  // US Compute
  usFrontierProjectComputeTppH100eIn2025: 120325,
  usFrontierProjectComputeAnnualGrowthRate: 4.0,

  // Compute Survival
  initialAnnualHazardRate: 0.05,
  annualHazardRateIncreasePerYear: 0.02,

  // Exogenous Compute Trends
  transistorDensityScalingExponent: 1.49,
  stateOfTheArtArchitectureEfficiencyImprovementPerYear: 1.23,
  transistorDensityAtEndOfDennardScaling: 1.98,
  wattsTppDensityExponentBeforeDennard: -2.0,
  wattsTppDensityExponentAfterDennard: -0.91,
  stateOfTheArtEnergyEfficiencyImprovementPerYear: 1.26,
};

// Multi-threshold CCDF data
export type MultiThresholdCCDF = Record<string | number, CCDFPoint[]>;

// Time series data structure
export interface TimeSeriesData {
  years: number[];
  median: number[];
  p25: number[];
  p75: number[];
}

// Rate of computation section data (HowWeEstimate)
export interface RateOfComputationData {
  years: number[];
  initial_chip_stock_samples: number[];
  acquired_hardware: TimeSeriesData;
  surviving_fraction: TimeSeriesData;
  covert_chip_stock: TimeSeriesData;
  datacenter_capacity: TimeSeriesData;
  energy_usage: TimeSeriesData;
  operating_chips: TimeSeriesData;
  covert_computation: TimeSeriesData;
  energy_stacked_data?: number[][];
  energy_source_labels?: string[];
}

// Covert fab section data
export interface CovertFabData {
  dashboard?: {
    production: string;
    energy: string;
    probFabBuilt: string;
    yearsOperational: string;
    processNode: string;
  };
  compute_ccdf?: Record<number | string, CCDFPoint[]>;
  time_series_data?: {
    years: number[];
    lr_combined: TimeSeriesData;
    h100e_flow: TimeSeriesData;
  };
  is_operational?: TimeSeriesData;
  wafer_starts_samples?: number[];
  chips_per_wafer?: number;
  architecture_efficiency?: number;
  h100_power?: number;
  transistor_density?: Array<{ node: string; density: number; wattsPerTpp?: number }>;
  compute_per_month?: TimeSeriesData;
  watts_per_tpp_curve?: {
    densityRelative: number[];
    wattsPerTppRelative: number[];
  };
  energy_per_month?: TimeSeriesData;
}

// Detection likelihood section data
export interface DetectionLikelihoodData {
  years: number[];
  chip_evidence_samples: number[];
  sme_evidence_samples: number[];
  dc_evidence_samples: number[];
  energy_evidence: TimeSeriesData;
  combined_evidence: TimeSeriesData;
  direct_evidence: TimeSeriesData;
  posterior_prob: TimeSeriesData;
}

// Simulation data types
export interface SimulationData {
  num_simulations?: number;
  black_project_model?: {
    years?: number[];
    total_dark_compute?: {
      median: number[];
      p25: number[];
      p75: number[];
      individual?: number[][];
    };
    h100_years_before_detection?: {
      median: number;
      p25: number;
      p75: number;
      individual?: number[];
    };
    h100_years_ccdf?: MultiThresholdCCDF;
    average_covert_compute_ccdf?: MultiThresholdCCDF;
    time_to_detection?: { median: number };
    time_to_detection_ccdf?: MultiThresholdCCDF;
    ai_rd_reduction?: { median: number };
    ai_rd_reduction_ccdf?: MultiThresholdCCDF;
    ai_rd_reduction_ccdf_flat?: Record<string, CCDFPoint[]>;
    chip_production_reduction_ccdf?: MultiThresholdCCDF;
    chip_production_reduction_ccdf_flat?: Record<string, CCDFPoint[]>;
    chips_produced?: { median: number };
    posterior_prob_project?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    individual_project_h100_years_before_detection?: number[];
    individual_project_time_before_detection?: number[];
    individual_project_h100e_before_detection?: number[];
    ai_rd_reduction_median?: number;
    // Additional time series data
    black_fab_flow_all_sims?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    survival_rate?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    covert_chip_stock?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    datacenter_capacity?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    energy_usage?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    operating_chips?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    covert_computation?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    chip_evidence?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    sme_evidence?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    dc_evidence?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    energy_evidence?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    combined_evidence?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    direct_evidence?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    chip_evidence_samples?: number[];
    sme_evidence_samples?: number[];
    dc_evidence_samples?: number[];
    // Cumulative H100-years computation
    h100_years?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    // Likelihood ratio time series
    lr_reported_energy?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    lr_combined_reported_assets?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    lr_other_intel?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    // Monthly fab production flow
    black_fab_monthly_flow_all_sims?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    // Cumulative fab production flow
    black_fab_flow?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    // Energy data
    black_project_energy?: number[][];
    energy_source_labels?: string[];
  };
  black_fab?: {
    years: number[];
    is_operational?: { proportion: number[] };
    wafer_starts?: {
      median: number[];
      individual?: number[][];
    };
    individual_process_node?: string[];
    process_node_by_sim?: string[];
    transistor_density?: {
      median?: number[];
      individual?: number[][];
    };
    watts_per_tpp?: {
      median?: number[];
      individual?: number[][];
    };
    dashboard?: {
      production: string;
      energy: string;
      probFabBuilt: string;
      yearsOperational: string;
      processNode: string;
    };
    compute_ccdfs?: Record<string | number, CCDFPoint[]>;
    lr_combined?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
    chips_per_wafer?: {
      median?: number[];
    };
    architecture_efficiency_at_agreement?: number;
    watts_per_tpp_curve?: {
      density_relative?: number[];
      watts_per_tpp_relative?: number[];
    };
  };
  black_datacenters?: {
    years?: number[];
    capacity_ccdfs?: Record<string, CCDFPoint[]>;
    individual_capacity_before_detection?: number[];
    individual_time_before_detection?: number[];
    datacenter_capacity?: { median: number[]; p25: number[]; p75: number[] };
    lr_datacenters?: { median: number[]; p25: number[]; p75: number[] };
    operational_compute?: { median: number[]; p25: number[]; p75: number[] };
    prc_datacenter_capacity_years?: number[];
    prc_datacenter_capacity_gw?: { median: number[]; p25: number[]; p75: number[] };
    prc_datacenter_capacity_at_black_project_start_year_gw?: number;
    prc_datacenter_capacity_at_black_project_start_year_samples?: number[];
    fraction_diverted?: number;
    energy_by_source?: number[][];
    source_labels?: string[];
    datacenter_detection_prob?: number[];
    likelihood_ratios?: number[];
    construction_workers?: number;
    mw_per_worker_per_year?: number;
    datacenter_start_year?: number;
    total_prc_energy_gw?: number;
    max_proportion_energy?: number;
  };
  initial_stock?: {
    lr_compute_accounting_samples?: number[];
    lr_sme_inventory_samples?: number[];
    lr_satellite_datacenter_samples?: number[];
    initial_prc_stock_samples?: number[];
    initial_compute_stock_samples?: number[];
    initial_energy_samples?: number[];
    diversion_proportion?: number;
    initial_black_project_detection_probs?: Record<string, number>;
    prc_compute_years?: number[];
    prc_compute_over_time?: { p25: number[]; median: number[]; p75: number[] };
    prc_domestic_compute_over_time?: { median: number[] };
    proportion_domestic_by_year?: number[];
    largest_company_compute_over_time?: number[];
    state_of_the_art_energy_efficiency_relative_to_h100?: number;
  };
  // New section-specific data from API
  rate_of_computation?: RateOfComputationData;
  covert_fab?: CovertFabData;
  detection_likelihood?: DetectionLikelihoodData;
  [key: string]: unknown;
}
