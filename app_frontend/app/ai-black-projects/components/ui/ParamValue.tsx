'use client';

import { scrollToParameter } from './ParamLink';
import { Parameters } from '../../types';

// Formatters for different parameter types
const formatters: Partial<Record<keyof Parameters, (v: number | string | boolean) => string>> = {
  totalPrcComputeTppH100eIn2025: (v) => `${((v as number) / 1000000).toFixed(1)}M`,
  annualGrowthRateOfPrcComputeStock: (v) => `${(v as number).toFixed(1)}x`,
  proportionOfInitialChipStockToDivert: (v) => `${((v as number) * 100).toFixed(0)}%`,
  fractionOfDatacenterCapacityToDivert: (v) => `${((v as number) * 100).toFixed(0)}%`,
  fractionOfLithographyScannersToDivert: (v) => `${((v as number) * 100).toFixed(0)}%`,
  maxFractionOfTotalNationalEnergyConsumption: (v) => `${((v as number) * 100).toFixed(0)}%`,
  priorOddsOfCovertProject: (v) => `${((v as number) * 100).toFixed(0)}%`,
  fractionOfLaborDevotedToDatacenterConstruction: (v) => `${((v as number) * 100).toFixed(0)}%`,
  workersInCovertProject: (v) => (v as number).toLocaleString(),
  totalLabor: (v) => (v as number).toLocaleString(),
  agreementYear: (v) => String(v),
  blackProjectStartYear: (v) => String(v),
  totalPrcEnergyConsumptionGw: (v) => `${v} GW`,
  energyEfficiencyOfComputeStockRelativeToStateOfTheArt: (v) => (v as number).toFixed(2),
  transistorDensityScalingExponent: (v) => `${(v as number).toPrecision(2)}x`,
  stateOfTheArtArchitectureEfficiencyImprovementPerYear: (v) => `${(v as number).toPrecision(2)}x`,
  wattsTppDensityExponentBeforeDennard: (v) => (v as number).toPrecision(2),
  wattsTppDensityExponentAfterDennard: (v) => (v as number).toPrecision(2),
  fabWafersPerMonthPerOperatingWorker: (v) => (v as number).toPrecision(2),
  wafersPerMonthPerLithographyScanner: (v) => (v as number).toLocaleString(),
};

// Map parameter keys to their sidebar input IDs
// These IDs must match the `id` props on Slider components in ParameterSidebar
const paramIdMap: Partial<Record<keyof Parameters, string>> = {
  workersInCovertProject: 'param-workers-in-project',
  fractionOfLaborDevotedToDatacenterConstruction: 'param-fraction-labor-datacenter',
  proportionOfInitialChipStockToDivert: 'param-fraction-compute-divert',
  fractionOfLithographyScannersToDivert: 'param-fraction-scanners-divert',
  fractionOfDatacenterCapacityToDivert: 'param-fraction-datacenter-divert',
  h100SizedChipsPerWafer: 'param-chips-per-wafer',
  wafersPerMonthPerLithographyScanner: 'param-wafers-per-scanner',
  constructionTimeFor5kWafersPerMonth: 'param-construction-time-5k',
  constructionTimeFor100kWafersPerMonth: 'param-construction-time-100k',
  fabWafersPerMonthPerOperatingWorker: 'param-wafers-per-operating-worker',
  initialAnnualHazardRate: 'param-initial-hazard-rate',
  annualHazardRateIncreasePerYear: 'param-hazard-rate-increase',
  transistorDensityScalingExponent: 'param-transistor-density-scaling',
  stateOfTheArtArchitectureEfficiencyImprovementPerYear: 'param-architecture-improvement',
  wattsTppDensityExponentBeforeDennard: 'param-watts-tpp-before-dennard',
  wattsTppDensityExponentAfterDennard: 'param-watts-tpp-after-dennard',
  intelligenceMedianErrorInEstimateOfFabStock: 'param-median-error-chip-stock',
  intelligenceMedianErrorInEnergyConsumptionEstimate: 'param-median-error-energy',
  intelligenceMedianErrorInSatelliteEstimate: 'param-median-error-satellite',
  detectionThreshold: 'param-detection-threshold',
  priorOddsOfCovertProject: 'param-prior-odds',
};

interface ParamValueProps {
  paramKey: keyof Parameters;
  parameters: Parameters;
  format?: (value: number | string | boolean) => string;
}

/**
 * Displays a parameter value as a clickable link that scrolls to the parameter in the sidebar.
 */
export function ParamValue({ paramKey, parameters, format }: ParamValueProps) {
  const value = parameters[paramKey];
  const paramId = paramIdMap[paramKey];

  // Use custom format, or default formatter, or just convert to string
  const formatter = format || formatters[paramKey] || ((v: number | string | boolean) => String(v));
  const displayValue = formatter(value);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (paramId) {
      scrollToParameter(paramId);
    }
  };

  return (
    <span
      className="param-link"
      onClick={handleClick}
      style={{
        cursor: paramId ? 'pointer' : 'default',
      }}
    >
      {displayValue}
    </span>
  );
}

export default ParamValue;
