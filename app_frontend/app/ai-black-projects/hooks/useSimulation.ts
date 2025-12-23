'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { Parameters, SimulationData, defaultParameters } from '../types';

// Flask backend URL
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5329';

// Configuration for API behavior
const USE_REAL_BACKEND = true;
const FALLBACK_TO_DUMMY = false; // DISABLED - only use real backend data

interface UseSimulationResult {
  data: SimulationData | null;
  isLoading: boolean;
  error: string | null;
  parameters: Parameters;
  updateParameter: <K extends keyof Parameters>(key: K, value: Parameters[K]) => void;
  runSimulation: () => Promise<void>;
  usingRealBackend: boolean;
  defaultsLoaded: boolean;
}

// Map backend parameter names to frontend parameter names
function mapBackendDefaults(backendDefaults: Record<string, unknown>): Partial<Parameters> {
  return {
    numYearsToSimulate: backendDefaults.numYearsToSimulate as number,
    numSimulations: backendDefaults.numSimulations as number,
    agreementYear: backendDefaults.agreementYear as number,
    blackProjectStartYear: backendDefaults.blackProjectStartYear as number,
    workersInCovertProject: backendDefaults.workersInCovertProject as number,
    fractionOfLaborDevotedToDatacenterConstruction: backendDefaults.fractionOfLaborDevotedToDatacenterConstruction as number,
    fractionOfLaborDevotedToBlackFabConstruction: backendDefaults.fractionOfLaborDevotedToBlackFabConstruction as number,
    fractionOfLaborDevotedToBlackFabOperation: backendDefaults.fractionOfLaborDevotedToBlackFabOperation as number,
    fractionOfLaborDevotedToAiResearch: backendDefaults.fractionOfLaborDevotedToAiResearch as number,
    proportionOfInitialChipStockToDivert: backendDefaults.proportionOfInitialChipStockToDivert as number,
    fractionOfDatacenterCapacityToDivert: backendDefaults.fractionOfDatacenterCapacityToDivert as number,
    fractionOfLithographyScannersToDivert: backendDefaults.fractionOfLithographyScannersToDivert as number,
    maxFractionOfTotalNationalEnergyConsumption: backendDefaults.maxFractionOfTotalNationalEnergyConsumption as number,
    buildCovertFab: backendDefaults.buildCovertFab as boolean,
    blackFabMaxProcessNode: backendDefaults.blackFabMaxProcessNode as string,
    priorOddsOfCovertProject: backendDefaults.priorOddsOfCovertProject as number,
    intelligenceMedianError: backendDefaults.intelligenceMedianError as number,
    meanDetectionTime100: backendDefaults.meanDetectionTime100 as number,
    meanDetectionTime1000: backendDefaults.meanDetectionTime1000 as number,
    varianceDetectionTime: backendDefaults.varianceDetectionTime as number,
    detectionThreshold: backendDefaults.detectionThreshold as number,
    totalPrcComputeTppH100eIn2025: backendDefaults.totalPrcComputeTppH100eIn2025 as number,
    annualGrowthRateOfPrcComputeStock: backendDefaults.annualGrowthRateOfPrcComputeStock as number,
    h100SizedChipsPerWafer: backendDefaults.h100SizedChipsPerWafer as number,
    wafersPerMonthPerLithographyScanner: backendDefaults.wafersPerMonthPerLithographyScanner as number,
    energyEfficiencyOfComputeStockRelativeToStateOfTheArt: backendDefaults.energyEfficiencyOfComputeStockRelativeToStateOfTheArt as number,
    totalPrcEnergyConsumptionGw: backendDefaults.totalPrcEnergyConsumptionGw as number,
    dataCenterMwPerYearPerConstructionWorker: backendDefaults.dataCenterMwPerYearPerConstructionWorker as number,
    initialAnnualHazardRate: backendDefaults.initialAnnualHazardRate as number,
    annualHazardRateIncreasePerYear: backendDefaults.annualHazardRateIncreasePerYear as number,
  };
}

export function useSimulation(initialData: SimulationData | null): UseSimulationResult {
  const [data, setData] = useState<SimulationData | null>(initialData);
  const [isLoading, setIsLoading] = useState(!initialData);
  const [error, setError] = useState<string | null>(null);
  const [parameters, setParameters] = useState<Parameters>(defaultParameters);
  const [usingRealBackend, setUsingRealBackend] = useState(USE_REAL_BACKEND);
  const [defaultsLoaded, setDefaultsLoaded] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const initialLoadDone = useRef(false);

  // Fetch from real backend
  const fetchFromRealBackend = async (params: Parameters, signal: AbortSignal) => {
    const response = await fetch(`${BACKEND_URL}/api/run-black-project-simulation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        parameters: params,
        num_simulations: params.numSimulations || 100,
        time_range: [params.agreementYear || 2027, (params.agreementYear || 2027) + (params.numYearsToSimulate || 10)],
      }),
      signal,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Backend error (${response.status}): ${text}`);
    }

    const result = await response.json();
    if (!result.success) {
      throw new Error(result.error || 'Backend simulation failed');
    }

    return result;
  };

  // Load defaults from YAML and run initial simulation
  useEffect(() => {
    if (initialData || initialLoadDone.current) return;
    initialLoadDone.current = true;

    const loadDefaultsAndSimulation = async () => {
      const controller = new AbortController();

      try {
        // Step 1: Fetch defaults from backend YAML
        console.log('[useSimulation] Fetching defaults from YAML...');
        const defaultsResponse = await fetch(`${BACKEND_URL}/api/black-project-defaults`, {
          signal: controller.signal,
        });

        let yamlDefaults = defaultParameters;

        if (defaultsResponse.ok) {
          const defaultsResult = await defaultsResponse.json();
          if (defaultsResult.success && defaultsResult.defaults) {
            const mapped = mapBackendDefaults(defaultsResult.defaults);
            yamlDefaults = { ...defaultParameters, ...mapped };
            setParameters(yamlDefaults);
            setDefaultsLoaded(true);
            console.log('[useSimulation] YAML defaults loaded:', yamlDefaults);
          }
        } else {
          console.warn('[useSimulation] Failed to fetch YAML defaults, using hardcoded defaults');
        }

        // Step 2: Run simulation with defaults
        if (USE_REAL_BACKEND) {
          console.log('[useSimulation] Running initial simulation...');
          const result = await fetchFromRealBackend(yamlDefaults, controller.signal);
          setData(result);
          setUsingRealBackend(true);
          console.log('[useSimulation] Initial simulation complete');
          console.log('[useSimulation] Full backend response:', result);
          console.log('[useSimulation] Available top-level keys:', Object.keys(result));
          if (result._debug_raw_simulations) {
            console.log('[useSimulation] === RAW SIMULATION ROLLOUTS ===');
            console.log('[useSimulation] Central simulation:', result._debug_raw_simulations.central);
            console.log('[useSimulation] MC results:', result._debug_raw_simulations.mc_results);
          }
        }

      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') return;
        console.error('[useSimulation] Error during initialization:', err);
        setError(err instanceof Error ? err.message : 'Failed to initialize');
      } finally {
        setIsLoading(false);
      }
    };

    loadDefaultsAndSimulation();
  }, [initialData]);

  const runSimulation = useCallback(async () => {
    // Cancel any previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setError(null);

    try {
      if (USE_REAL_BACKEND) {
        console.log('[useSimulation] Running simulation...');
        const result = await fetchFromRealBackend(parameters, controller.signal);
        setData(result);
        setUsingRealBackend(true);
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return;
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
    } finally {
      setIsLoading(false);
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  }, [parameters]);

  // Auto-run simulation when parameters change (with debounce)
  useEffect(() => {
    // Don't auto-run until initial load is complete
    if (!defaultsLoaded || (isLoading && !data)) return;

    const debounceTime = 1000;

    const timeoutId = setTimeout(() => {
      runSimulation();
    }, debounceTime);

    return () => clearTimeout(timeoutId);
  }, [parameters, defaultsLoaded]); // eslint-disable-line react-hooks/exhaustive-deps

  const updateParameter = useCallback(<K extends keyof Parameters>(key: K, value: Parameters[K]) => {
    setParameters(prev => ({ ...prev, [key]: value }));
  }, []);

  return {
    data,
    isLoading,
    error,
    parameters,
    updateParameter,
    runSimulation,
    usingRealBackend,
    defaultsLoaded,
  };
}
