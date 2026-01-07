/**
 * Types for the Black Project page components.
 */

// Re-export common types from the shared types file
export type { BlackProjectData } from '@/types/blackProject';

// Parameter types for the simulation
export interface Parameters {
  // Key Parameters
  numYearsToSimulate: number;
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
  numYearsToSimulate: 10,
  numSimulations: 100,
  agreementYear: 2027,
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

// CCDF point type (for local use)
export interface CCDFPoint {
  x: number;
  y: number;
}

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
  energy_stacked_data?: [number, number][];
  energy_source_labels?: [string, string];
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
  };
  black_fab?: {
    years: number[];
    is_operational?: { proportion: number[] };
    wafer_starts?: {
      median: number[];
      individual?: number[][];
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
    prc_capacity_years?: number[];
    prc_capacity_gw?: { median: number[]; p25: number[]; p75: number[] };
    prc_capacity_at_agreement_year_gw?: number;
    prc_capacity_at_agreement_year_samples?: number[];
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
    lr_prc_accounting_samples?: number[];
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
