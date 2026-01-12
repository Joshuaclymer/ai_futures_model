'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { Parameters, SimulationData, defaultParameters } from '../types';
import {
  encodeBlackProjectParamsToUrl,
  decodeBlackProjectParamsFromUrl,
  hasBlackProjectParamsInUrl,
} from '../utils/blackProjectUrlState';
import { useParameterConfig } from '@/components/ParameterConfigProvider';

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
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const { blackProjectDefaults, isLoading: configLoading } = useParameterConfig();

  const [data, setData] = useState<SimulationData | null>(initialData);
  const [isLoading, setIsLoading] = useState(!initialData);
  const [error, setError] = useState<string | null>(null);
  const [parameters, setParameters] = useState<Parameters>(defaultParameters);
  const [usingRealBackend, setUsingRealBackend] = useState(USE_REAL_BACKEND);
  const [defaultsLoaded, setDefaultsLoaded] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const initialLoadDone = useRef(false);
  const isHydratingFromUrlRef = useRef(false);
  const isUrlSyncReady = useRef(false);

  // Fetch from real backend
  const fetchFromRealBackend = async (params: Parameters, signal: AbortSignal) => {
    const response = await fetch(`${BACKEND_URL}/api/get-data-for-ai-black-projects-page`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        parameters: params,
        num_simulations: params.numSimulations || 100,
        ai_slowdown_start_year: params.agreementYear || 2030,
        end_year: 2037,
      }),
      signal,
      cache: 'no-store',
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Backend error (${response.status}): ${text}`);
    }

    const result = await response.json();
    // Backend returns data directly without success wrapper, but check for error response
    if (result.success === false) {
      throw new Error(result.error || 'Backend simulation failed');
    }

    return result;
  };

  // Load defaults from context, check URL params, and run initial simulation
  useEffect(() => {
    // Wait for config to load before initializing
    if (configLoading || initialData || initialLoadDone.current) return;
    initialLoadDone.current = true;

    const loadDefaultsAndSimulation = async () => {
      const controller = new AbortController();
      isHydratingFromUrlRef.current = true;

      try {
        // Step 1: Check if URL has parameters
        const hasUrlParams = hasBlackProjectParamsInUrl(searchParams);
        let initialParams = defaultParameters;

        if (hasUrlParams) {
          // Use URL parameters if present
          console.log('[useSimulation] Loading parameters from URL...');
          initialParams = decodeBlackProjectParamsFromUrl(searchParams);
          setParameters(initialParams);
          console.log('[useSimulation] URL parameters loaded:', initialParams);
        } else if (blackProjectDefaults) {
          // Step 2: Use defaults from context (loaded from /api/parameter-config)
          console.log('[useSimulation] Using defaults from context...');
          const mapped = mapBackendDefaults(blackProjectDefaults as unknown as Record<string, unknown>);
          initialParams = { ...defaultParameters, ...mapped };
          setParameters(initialParams);
          console.log('[useSimulation] Context defaults loaded:', initialParams);
        } else {
          console.warn('[useSimulation] No defaults available, using hardcoded defaults');
        }

        // Always mark defaults as loaded after the attempt, so parameter changes trigger simulations
        setDefaultsLoaded(true);
        isUrlSyncReady.current = true;

        // Step 3: Run simulation with initial params
        if (USE_REAL_BACKEND) {
          console.log('[useSimulation] Running initial simulation...');
          const result = await fetchFromRealBackend(initialParams, controller.signal);
          setData(result);
          setUsingRealBackend(true);
          console.log('[useSimulation] Initial simulation complete');
        }

      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') return;
        console.error('[useSimulation] Error during initialization:', err);
        setError(err instanceof Error ? err.message : 'Failed to initialize');
      } finally {
        setIsLoading(false);
        isHydratingFromUrlRef.current = false;
      }
    };

    loadDefaultsAndSimulation();
  }, [initialData, searchParams, configLoading, blackProjectDefaults]);

  // Sync parameters to URL when they change
  useEffect(() => {
    // Don't sync until initial hydration is complete
    if (!isUrlSyncReady.current || isHydratingFromUrlRef.current) return;

    const newParams = encodeBlackProjectParamsToUrl(parameters);
    const newUrl = newParams.toString()
      ? `${pathname}?${newParams.toString()}`
      : pathname;

    // Only update if URL actually changed
    const currentUrl = window.location.pathname + window.location.search;
    const targetUrl = newParams.toString() ? `${pathname}?${newParams.toString()}` : pathname;

    if (currentUrl !== targetUrl) {
      router.replace(newUrl, { scroll: false });
    }
  }, [parameters, pathname, router]);

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
  }, [parameters, defaultsLoaded, runSimulation]);

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
