"use client";

import { Parameters, defaultParameters } from '@/app/ai-black-projects/types';

/**
 * URL state utilities for the AI Black Projects page.
 * Follows the same pattern as urlState.ts for ai-timelines-and-takeoff.
 */

// Parameter abbreviations to keep URLs short
// Maps short URL param names to full parameter keys
const BP_PARAM_ABBREVIATIONS: Record<string, keyof Parameters> = {
  // Key Parameters
  'ns': 'numSimulations',
  'ay': 'agreementYear',
  'bpsy': 'blackProjectStartYear',
  'wcp': 'workersInCovertProject',
  'mdt100': 'meanDetectionTime100',
  'mdt1k': 'meanDetectionTime1000',
  'vdt': 'varianceDetectionTime',
  'picsd': 'proportionOfInitialChipStockToDivert',
  'ime': 'intelligenceMedianError',

  // Black Project Properties
  'tl': 'totalLabor',
  'fldcc': 'fractionOfLaborDevotedToDatacenterConstruction',
  'fldfbc': 'fractionOfLaborDevotedToBlackFabConstruction',
  'fldfbo': 'fractionOfLaborDevotedToBlackFabOperation',
  'fldair': 'fractionOfLaborDevotedToAiResearch',
  'fdctd': 'fractionOfDatacenterCapacityToDivert',
  'flstd': 'fractionOfLithographyScannersToDivert',
  'mftnec': 'maxFractionOfTotalNationalEnergyConsumption',
  'bcf': 'buildCovertFab',
  'bfmpn': 'blackFabMaxProcessNode',

  // Detection Parameters
  'imeifs': 'intelligenceMedianErrorInEstimateOfFabStock',
  'imeece': 'intelligenceMedianErrorInEnergyConsumptionEstimate',
  'imeise': 'intelligenceMedianErrorInSatelliteEstimate',
  'dt': 'detectionThreshold',
  'pocp': 'priorOddsOfCovertProject',

  // PRC Compute
  'tpct25': 'totalPrcComputeTppH100eIn2025',
  'agrpcs': 'annualGrowthRateOfPrcComputeStock',
  'paer': 'prcArchitectureEfficiencyRelativeToStateOfTheArt',
  'ppd26': 'proportionOfPrcChipStockProducedDomestically2026',
  'ppd30': 'proportionOfPrcChipStockProducedDomestically2030',
  'plsfy': 'prcLithographyScannersProducedInFirstYear',
  'palspy': 'prcAdditionalLithographyScannersProducedPerYear',
  'pl28': 'pLocalization28nm2030',
  'pl14': 'pLocalization14nm2030',
  'pl7': 'pLocalization7nm2030',
  'hcpw': 'h100SizedChipsPerWafer',
  'wpmls': 'wafersPerMonthPerLithographyScanner',
  'ct5k': 'constructionTimeFor5kWafersPerMonth',
  'ct100k': 'constructionTimeFor100kWafersPerMonth',
  'fwpow': 'fabWafersPerMonthPerOperatingWorker',
  'fwpcw': 'fabWafersPerMonthPerConstructionWorker',

  // PRC Data Centers and Energy
  'eecs': 'energyEfficiencyOfComputeStockRelativeToStateOfTheArt',
  'tpecgw': 'totalPrcEnergyConsumptionGw',
  'dcmwcw': 'dataCenterMwPerYearPerConstructionWorker',
  'dcmwow': 'dataCenterMwPerOperatingWorker',

  // US Compute
  'usfpct': 'usFrontierProjectComputeTppH100eIn2025',
  'usfpcgr': 'usFrontierProjectComputeAnnualGrowthRate',

  // Compute Survival
  'iahr': 'initialAnnualHazardRate',
  'ahripy': 'annualHazardRateIncreasePerYear',

  // Exogenous Compute Trends
  'tdse': 'transistorDensityScalingExponent',
  'saeipy': 'stateOfTheArtArchitectureEfficiencyImprovementPerYear',
  'tdeds': 'transistorDensityAtEndOfDennardScaling',
  'wtdebd': 'wattsTppDensityExponentBeforeDennard',
  'wtdead': 'wattsTppDensityExponentAfterDennard',
  'saeeip': 'stateOfTheArtEnergyEfficiencyImprovementPerYear',
};

// Reverse mapping: full param key -> short URL param name
const BP_PARAM_ABBREVIATIONS_REVERSE: Record<string, string> = Object.fromEntries(
  Object.entries(BP_PARAM_ABBREVIATIONS).map(([short, long]) => [long, short])
);

/**
 * Sanitize a parameter value from URL string to the correct type.
 */
const sanitizeParameterValue = (
  key: keyof Parameters,
  value: string | null
): Parameters[typeof key] | undefined => {
  if (value === null) {
    return undefined;
  }

  const defaultValue = defaultParameters[key];

  if (typeof defaultValue === 'number') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed as Parameters[typeof key];
    }
    return undefined;
  }

  if (typeof defaultValue === 'string') {
    return value as Parameters[typeof key];
  }

  if (typeof defaultValue === 'boolean') {
    if (value === 'true') {
      return true as Parameters[typeof key];
    }
    if (value === 'false') {
      return false as Parameters[typeof key];
    }
    return undefined;
  }

  return undefined;
};

/**
 * Encode black project parameters to URL search params.
 * Only includes parameters that differ from defaults to keep URLs short.
 */
export const encodeBlackProjectParamsToUrl = (parameters: Parameters): URLSearchParams => {
  const urlParams = new URLSearchParams();

  (Object.keys(defaultParameters) as Array<keyof Parameters>).forEach((paramKey) => {
    const value = parameters[paramKey];
    const defaultValue = defaultParameters[paramKey];

    if (value !== defaultValue) {
      const shortName = BP_PARAM_ABBREVIATIONS_REVERSE[paramKey];
      if (shortName) {
        urlParams.set(shortName, String(value));
      }
    }
  });

  return urlParams;
};

/**
 * Decode black project parameters from URL search params.
 * Returns full parameters object with defaults for any missing values.
 */
export const decodeBlackProjectParamsFromUrl = (searchParams: URLSearchParams): Parameters => {
  const parameters: Parameters = { ...defaultParameters };

  Object.entries(BP_PARAM_ABBREVIATIONS).forEach(([shortName, paramKey]) => {
    const value = searchParams.get(shortName);

    if (value !== null) {
      const sanitized = sanitizeParameterValue(paramKey, value);
      if (sanitized !== undefined) {
        (parameters as unknown as Record<string, unknown>)[paramKey] = sanitized;
      }
    }
  });

  return parameters;
};

/**
 * Check if URL has any black project parameters set.
 */
export const hasBlackProjectParamsInUrl = (searchParams: URLSearchParams): boolean => {
  return Object.keys(BP_PARAM_ABBREVIATIONS).some((shortName) => searchParams.has(shortName));
};
