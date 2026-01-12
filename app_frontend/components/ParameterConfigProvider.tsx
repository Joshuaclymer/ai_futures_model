'use client';

import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';

export type ParameterPrimitive = number | string | boolean | null;
export type ParametersType = Record<string, ParameterPrimitive>;

interface ParameterBounds {
  min?: number;
  max?: number;
}

interface ModelConstants {
  training_compute_reference_year: number;
  training_compute_reference_ooms: number;
  software_progress_scale_reference_year: number;
  base_for_software_lom: number;
}

export interface BlackProjectDefaults {
  numSimulations: number;
  agreementYear: number;
  blackProjectStartYear: number;
  workersInCovertProject: number;
  fractionOfLaborDevotedToDatacenterConstruction: number;
  fractionOfLaborDevotedToBlackFabConstruction: number;
  fractionOfLaborDevotedToBlackFabOperation: number;
  fractionOfLaborDevotedToAiResearch: number;
  proportionOfInitialChipStockToDivert: number;
  fractionOfDatacenterCapacityToDivert: number;
  fractionOfLithographyScannersToDivert: number;
  maxFractionOfTotalNationalEnergyConsumption: number;
  buildCovertFab: boolean;
  blackFabMaxProcessNode: string;
  priorOddsOfCovertProject: number;
  intelligenceMedianError: number;
  meanDetectionTime100: number;
  meanDetectionTime1000: number;
  varianceDetectionTime: number;
  detectionThreshold: number;
  totalPrcComputeTppH100eIn2025: number;
  annualGrowthRateOfPrcComputeStock: number;
  h100SizedChipsPerWafer: number;
  wafersPerMonthPerLithographyScanner: number;
  energyEfficiencyOfComputeStockRelativeToStateOfTheArt: number;
  totalPrcEnergyConsumptionGw: number;
  dataCenterMwPerYearPerConstructionWorker: number;
  initialAnnualHazardRate: number;
  annualHazardRateIncreasePerYear: number;
  transistorDensityScalingExponent: number;
  stateOfTheArtArchitectureEfficiencyImprovementPerYear: number;
  transistorDensityAtEndOfDennardScaling: number;
  wattsTppDensityExponentBeforeDennard: number;
  wattsTppDensityExponentAfterDennard: number;
  stateOfTheArtEnergyEfficiencyImprovementPerYear: number;
}

interface ParameterConfigContextValue {
  defaults: ParametersType;
  bounds: Record<string, ParameterBounds>;
  modelConstants: ModelConstants | null;
  config: Record<string, unknown> | null;
  blackProjectDefaults: BlackProjectDefaults | null;
  isLoading: boolean;
  error: string | null;
}

const defaultModelConstants: ModelConstants = {
  training_compute_reference_year: 2025.13,
  training_compute_reference_ooms: 26.54,
  software_progress_scale_reference_year: 2024.0,
  base_for_software_lom: 10.0,
};

const ParameterConfigContext = createContext<ParameterConfigContextValue | undefined>(undefined);

export function useParameterConfig() {
  const context = useContext(ParameterConfigContext);
  if (!context) {
    throw new Error('useParameterConfig must be used within a ParameterConfigProvider');
  }
  return context;
}

export function ParameterConfigProvider({ children }: { children: ReactNode }) {
  const [defaults, setDefaults] = useState<ParametersType>({});
  const [bounds, setBounds] = useState<Record<string, ParameterBounds>>({});
  const [modelConstants, setModelConstants] = useState<ModelConstants | null>(defaultModelConstants);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [blackProjectDefaults, setBlackProjectDefaults] = useState<BlackProjectDefaults | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isCancelled = false;

    const loadConfig = async () => {
      try {
        const response = await fetch('/api/parameter-config');
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();

        if (!isCancelled && data.success) {
          setDefaults(data.defaults || {});
          setBounds(data.bounds || {});
          if (data.model_constants) {
            setModelConstants(data.model_constants);
          }
          if (data.config) {
            setConfig(data.config);
          }
          if (data.black_project_defaults) {
            setBlackProjectDefaults(data.black_project_defaults);
          }
        }
      } catch (err) {
        if (!isCancelled) {
          console.error('Failed to load parameter config:', err);
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
      } finally {
        if (!isCancelled) {
          setIsLoading(false);
        }
      }
    };

    loadConfig();

    return () => {
      isCancelled = true;
    };
  }, []);

  return (
    <ParameterConfigContext.Provider value={{ defaults, bounds, modelConstants, config, blackProjectDefaults, isLoading, error }}>
      {children}
    </ParameterConfigContext.Provider>
  );
}
