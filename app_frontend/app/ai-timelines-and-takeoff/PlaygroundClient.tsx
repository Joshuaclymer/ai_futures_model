'use client';

import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';

// Local page-specific components
import CustomHorizonChart from './components/charts/CustomHorizonChart';
import { CombinedComputeChart } from './components/charts/CombinedComputeChart';
import { CustomMetricChart } from './components/charts/CustomMetricChart';
import { AIRnDProgressMultiplierChart } from './components/charts/AIRnDProgressMultiplierChart';
import { SmallChartMetricTooltip } from './components/charts/SmallChartMetricTooltip';
import { ChartLoadingOverlay } from './components/charts/ChartLoadingOverlay';
import { AdvancedSections } from './components/sections/AdvancedSections';
import { ParameterSlider } from './components/ui/ParameterSlider';

// Shared components
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './components/charts/chartTooltipStyle';
import { ChartDataPoint, BenchmarkPoint } from '@/app/types';
import type { DataPoint } from './components/charts/CustomLineChart';
import { ChartSyncProvider } from './components/charts/ChartSyncContext';
import { ParameterHoverProvider } from '@/components/ParameterHoverContext';

// Utils and constants
import { formatTo3SigFigs, formatWorkTimeDuration, formatUplift, formatCompactNumberNode, formatAsPowerOfTenText, formatSCHorizon, formatTimeDuration, yearsToMinutes } from './utils/formatting';
import { ParametersType, H100E_TPP_TO_FLOP_OOM_OFFSET } from './constants/parameters';
import { useParameterConfig } from '@/components/ParameterConfigProvider';
import { CHART_LAYOUT } from './constants/chartLayout';
import { convertParametersToAPIFormat, convertSampledParametersToAPIFormat, ParameterRecord } from './utils/monteCarlo';
import { encodeFullStateToParams, decodeFullStateFromParams, DEFAULT_CHECKBOX_STATES } from './utils/urlState';
import type { MilestoneMap } from './types/milestones';
import { SamplingConfig, generateParameterSampleWithUserValues, initializeCorrelationSampling, extractSamplingConfigBounds, flattenMonteCarloConfig } from './utils/sampling';
import type { ComputeApiResponse } from '@/app/types';

interface SampleTrajectoryWithParams {
  trajectory: ChartDataPoint[];
  params: Record<string, number | string | boolean>;
}

interface PlaygroundClientProps {
  benchmarkData: BenchmarkPoint[];
  initialComputeData: ComputeApiResponse;
  initialParameters: ParametersType;
  initialSampleTrajectories?: { trajectory: { year: number; horizonLength: number; effectiveCompute: number; automationFraction?: number; trainingCompute?: number; frontierTrainingCompute?: number; aiSwProgressMultRefPresentDay?: number }[]; params: Record<string, number | string | boolean> }[];
  initialSeed?: number;
  hideHeader?: boolean;
}

interface SmallChartDef {
  key: string;
  title: string;
  tooltip: (p: DataPoint) => React.ReactNode;
  yFormatter?: (v: number) => string | React.ReactNode;
  yScale?: 'linear' | 'log';
  scaleType: 'log' | 'linear';
  logSuffix?: string;
  isDataInLogForm?: boolean;
}

const SIMULATION_START_YEAR = 2026;
const SIMULATION_END_YEAR = 2045;
// Use relative URL to go through Next.js proxy (configured in next.config.ts)
// This avoids CORS issues and works in both dev and production
const API_BASE_URL = '';

function processInitialData(data: ComputeApiResponse): ChartDataPoint[] {
  if (!data?.time_series) return [];
  return data.time_series.map((point) => {
    // Helper to safely convert API values to number | null
    const toNumberOrNull = (val: unknown): number | null =>
      typeof val === 'number' ? val : null;

    return {
      year: point.year,
      horizonLength: point.horizonLength,
      horizonFormatted: formatWorkTimeDuration(point.horizonLength),
      effectiveCompute: point.effectiveCompute,
      // Use frontierTrainingCompute (actual training compute in H100e units) converted to FLOP OOMs
      trainingCompute: typeof point.frontierTrainingCompute === 'number' && point.frontierTrainingCompute > 0
        ? Math.log10(point.frontierTrainingCompute) + H100E_TPP_TO_FLOP_OOM_OFFSET
        : null,
      automationFraction: point.automationFraction ?? null,
      aiSoftwareProgressMultiplier: toNumberOrNull(point.aiSoftwareProgressMultiplier),
      aiSwProgressMultRefPresentDay: toNumberOrNull(point.aiSwProgressMultRefPresentDay),
      aiResearchTaste: toNumberOrNull(point.aiResearchTaste),
      experimentCapacity: toNumberOrNull(point.experimentCapacity),
      inferenceCompute: toNumberOrNull(point.inferenceCompute),
      experimentCompute: toNumberOrNull(point.experimentCompute),
      researchEffort: toNumberOrNull(point.researchEffort),
      researchStock: toNumberOrNull(point.researchStock),
      softwareProgressRate: toNumberOrNull(point.softwareProgressRate),
      softwareEfficiency: toNumberOrNull(point.softwareEfficiency),
      aiCodingLaborMultiplier: toNumberOrNull(point.aiCodingLaborMultiplier),
      serialCodingLaborMultiplier: toNumberOrNull(point.serialCodingLaborMultiplier),
      humanLabor: toNumberOrNull(point.humanLabor),
    };
  });
}

// Simple slider component for playground
function SimpleSlider({ 
  label, 
  value, 
  onChange, 
  onAfterChange,
  min, 
  max, 
  step,
  formatValue,
  disabled = false,
}: { 
  label: string;
  value: number;
  onChange: (v: number) => void;
  onAfterChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  formatValue?: (v: number) => string;
  disabled?: boolean;
}) {
  return (
    <div className={`space-y-1 ${disabled ? 'opacity-50' : ''}`}>
      <div className="flex justify-between text-xs">
        <span className="text-gray-600">{label}</span>
        <span className="text-gray-800 font-mono">{formatValue ? formatValue(value) : value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        onMouseUp={(e) => onAfterChange(parseFloat((e.target as HTMLInputElement).value))}
        onTouchEnd={(e) => onAfterChange(parseFloat((e.target as HTMLInputElement).value))}
        disabled={disabled}
        className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-600 disabled:cursor-not-allowed"
      />
    </div>
  );
}

// Simple checkbox component
function SimpleCheckbox({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="w-3.5 h-3.5 rounded border-gray-300 text-emerald-600 focus:ring-emerald-500"
      />
      <span className="text-gray-700">{label}</span>
    </label>
  );
}

export default function PlaygroundClient({
  benchmarkData = [],
  initialComputeData,
  initialParameters,
  initialSampleTrajectories = [],
  hideHeader = false,
}: PlaygroundClientProps) {
  const { defaults: contextDefaults, config: contextConfig, isLoading: configLoading } = useParameterConfig();
  const initialChartData = useMemo(() => processInitialData(initialComputeData), [initialComputeData]);
  const resolvedInitialParameters = useMemo(() => initialParameters ?? { ...contextDefaults }, [initialParameters, contextDefaults]);
  
  const convertedInitialSamples = useMemo(() => {
    return initialSampleTrajectories.map(sample => ({
      trajectory: sample.trajectory.map(point => ({
        year: point.year,
        horizonLength: point.horizonLength,
        horizonFormatted: formatWorkTimeDuration(point.horizonLength),
        effectiveCompute: point.effectiveCompute,
        automationFraction: point.automationFraction,
        // Use frontierTrainingCompute (actual training compute in H100e units) converted to FLOP OOMs
        trainingCompute: point.frontierTrainingCompute && point.frontierTrainingCompute > 0
          ? Math.log10(point.frontierTrainingCompute) + H100E_TPP_TO_FLOP_OOM_OFFSET
          : null,
        aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
      })) as ChartDataPoint[],
      params: sample.params
    }));
  }, [initialSampleTrajectories]);
  
  const [chartData, setChartData] = useState<ChartDataPoint[]>(initialChartData);
  const [scHorizonMinutes, setScHorizonMinutes] = useState<number>(
    Math.pow(10, contextDefaults['software_r_and_d.ac_time_horizon_minutes'] as number || 7.08)
  );
  const [parameters, setParameters] = useState<ParametersType>(resolvedInitialParameters);
  const [uiParameters, setUiParameters] = useState<ParametersType>(resolvedInitialParameters);
  const uiParametersRef = useRef<ParametersType>(resolvedInitialParameters);
  const [milestones, setMilestones] = useState<MilestoneMap | null>(
    initialComputeData?.milestones && typeof initialComputeData.milestones === 'object'
      ? (initialComputeData.milestones as MilestoneMap)
      : null
  );
  const [mainLoading, setMainLoading] = useState(false);
  const [mainlineLoaded, setMainlineLoaded] = useState(false);
  const [sampleTrajectories, setSampleTrajectories] = useState<SampleTrajectoryWithParams[]>(convertedInitialSamples);
  const [samplingConfig, setSamplingConfig] = useState<SamplingConfig | null>(null);
  const [enabledSamplingParams, setEnabledSamplingParams] = useState<Set<string>>(new Set());
  const [resampleTrigger, setResampleTrigger] = useState(0);
  const [numSamples, setNumSamples] = useState(10);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [allParameters] = useState<{ defaults?: Record<string, unknown>; bounds?: Record<string, [number, number]>; metadata?: Record<string, unknown> } | null>(null);
  
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  
  // Model simplification checkboxes
  const [enableCodingAutomation, setEnableCodingAutomation] = useState(DEFAULT_CHECKBOX_STATES.enableCodingAutomation);
  const [enableExperimentAutomation, setEnableExperimentAutomation] = useState(DEFAULT_CHECKBOX_STATES.enableExperimentAutomation);
  const [enableSoftwareProgress, setEnableSoftwareProgress] = useState(DEFAULT_CHECKBOX_STATES.enableSoftwareProgress);
  const [useExperimentThroughputCES, setUseExperimentThroughputCES] = useState(DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES);
  const [useComputeLaborGrowthSlowdown, setUseComputeLaborGrowthSlowdown] = useState(DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown);
  const [useVariableHorizonDifficulty, setUseVariableHorizonDifficulty] = useState(DEFAULT_CHECKBOX_STATES.useVariableHorizonDifficulty);
  
  const currentRequestRef = useRef<AbortController | null>(null);
  const sampleTrajectoriesAbortRef = useRef<AbortController | null>(null);
  const hasUsedInitialSamplesRef = useRef(convertedInitialSamples.length > 0);

  const displayEndYear = 2060;

  // Sampling config bounds for slider ranges
  const samplingConfigBounds = useMemo(() => {
    if (!samplingConfig) return {};
    return extractSamplingConfigBounds(samplingConfig);
  }, [samplingConfig]);

  // Locked parameters based on checkbox state
  const lockedParameters = useMemo(() => {
    const locked = new Set<string>();
    if (!enableCodingAutomation) {
      locked.add('optimal_ces_eta_init');
      locked.add('coding_automation_efficiency_slope');
      locked.add('swe_multiplier_at_present_day');
    }
    if (!enableExperimentAutomation) {
      locked.add('median_to_top_taste_multiplier');
    }
    if (!useExperimentThroughputCES) {
      locked.add('direct_input_exp_cap_ces_params');
      locked.add('inf_labor_asymptote');
      locked.add('inf_compute_asymptote');
      locked.add('inv_compute_anchor_exp_cap');
      locked.add('rho_experiment_capacity');
    }
    if (!enableSoftwareProgress) {
      locked.add('software_progress_rate_at_reference_year');
    }
    if (!useVariableHorizonDifficulty) {
      locked.add('doubling_difficulty_growth_factor');
    }
    return locked;
  }, [enableCodingAutomation, enableExperimentAutomation, useExperimentThroughputCES, enableSoftwareProgress, useVariableHorizonDifficulty]);

  // Bounds for specific parameter groups
  const scHorizonLogBounds = useMemo(() => {
    const samplingBounds = samplingConfigBounds?.ac_time_horizon_minutes;
    if (samplingBounds?.min !== undefined && samplingBounds?.max !== undefined) {
      return { min: samplingBounds.min, max: samplingBounds.max };
    }
    return { min: 3, max: 11 };
  }, [samplingConfigBounds]);

  const preGapHorizonBounds = useMemo(() => {
    const samplingBounds = samplingConfigBounds?.pre_gap_ac_time_horizon;
    if (samplingBounds?.min !== undefined && samplingBounds?.max !== undefined) {
      return { min: samplingBounds.min, max: samplingBounds.max };
    }
    return { min: 1000, max: 100000000000 };
  }, [samplingConfigBounds]);

  const parallelPenaltyBounds = useMemo(() => {
    const samplingBounds = samplingConfigBounds?.parallel_penalty;
    if (samplingBounds?.min !== undefined && samplingBounds?.max !== undefined) {
      return { min: samplingBounds.min, max: samplingBounds.max };
    }
    return { min: 0, max: 1 };
  }, [samplingConfigBounds]);

  // Keep uiParametersRef in sync with uiParameters state
  useEffect(() => {
    uiParametersRef.current = uiParameters;
  }, [uiParameters]);

  // Fetch compute data - defined before commitParameters since it depends on this
  const fetchComputeData = useCallback(async (params: ParametersType) => {
    if (currentRequestRef.current) {
      currentRequestRef.current.abort();
    }

    const controller = new AbortController();
    currentRequestRef.current = controller;
    setMainLoading(true);
    setMainlineLoaded(false);

    try {
      const apiParameters = convertParametersToAPIFormat(params as unknown as ParameterRecord);
      const response = await fetch(`${API_BASE_URL}/api/get-data-for-ai-timelines-and-takeoff-page`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parameters: apiParameters,
          time_range: [SIMULATION_START_YEAR, SIMULATION_END_YEAR],
        }),
        signal: controller.signal,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (data.success && data.time_series) {
        console.log('[Mainline] Loaded - showing chart data now, MC samples will start next');
        setChartData(processInitialData(data));
        setMainlineLoaded(true);
        if (data.milestones) {
          setMilestones(data.milestones as MilestoneMap);
        }
      }
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') return;
      console.error('Fetch error:', e);
    } finally {
      setMainLoading(false);
    }
  }, []);

  // Commit UI parameters to actual parameters when dragging ends
  // Uses ref instead of closure to always get the latest uiParameters value
  const commitParameters = useCallback((nextParameters?: ParametersType) => {
    const newParams = nextParameters ?? uiParametersRef.current;
    setParameters(newParams);
    setIsDragging(false);
    fetchComputeData(newParams);
    setResampleTrigger(prev => prev + 1);
  }, [fetchComputeData]);

  // Initialize sampling config from context
  useEffect(() => {
    if (!contextConfig) return;

    // Flatten nested default_parameters.yaml structure into flat SamplingConfig
    const flatConfig = flattenMonteCarloConfig(contextConfig as Record<string, unknown>);
    initializeCorrelationSampling(flatConfig.correlation_matrix);
    setSamplingConfig(flatConfig);

    const allParams = new Set<string>();
    for (const paramName of Object.keys(flatConfig.parameters)) {
      allParams.add(paramName);
    }
    if (flatConfig.time_series_parameters) {
      for (const paramName of Object.keys(flatConfig.time_series_parameters)) {
        allParams.add(paramName);
      }
    }
    setEnabledSamplingParams(allParams);
  }, [contextConfig]);

  // Update SC horizon line immediately while slider moves
  useEffect(() => {
    const instantHorizon = Math.pow(10, uiParameters['software_r_and_d.ac_time_horizon_minutes'] as number);
    setScHorizonMinutes(instantHorizon);
  }, [uiParameters['software_r_and_d.ac_time_horizon_minutes'] as number]);

  // Fetch sample trajectory
  const fetchSampleTrajectory = useCallback(async (params: Record<string, number | string | boolean>, signal?: AbortSignal) => {
    try {
      const apiParams = convertSampledParametersToAPIFormat(params as ParameterRecord);
      const response = await fetch(`${API_BASE_URL}/api/get-data-for-ai-timelines-and-takeoff-page`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parameters: apiParams,
          time_range: [SIMULATION_START_YEAR, SIMULATION_END_YEAR],
        }),
        signal,
      });

      if (!response.ok) return null;
      const data = await response.json();
      return data.success ? data.time_series : null;
    } catch {
      return null;
    }
  }, []);

  // Convert parameters for sampling (handles UI format differences)
  const convertParametersForSampling = useCallback((params: ParametersType): Record<string, number | string | boolean> => {
    const rawParams = params as unknown as Record<string, number | string | boolean>;
    const userParams: Record<string, number | string | boolean> = { ...rawParams };

    // Convert log-scale ac_time_horizon_minutes back to linear
    const acTimeHorizonKey = 'software_r_and_d.ac_time_horizon_minutes';
    if (typeof rawParams[acTimeHorizonKey] === 'number') {
      userParams[acTimeHorizonKey] = Math.pow(10, rawParams[acTimeHorizonKey] as number);
    }

    // Map saturation_horizon_minutes to pre_gap_ac_time_horizon
    const satHorizonKey = 'software_r_and_d.saturation_horizon_minutes';
    const preGapKey = 'software_r_and_d.pre_gap_ac_time_horizon';
    if (satHorizonKey in rawParams) {
      userParams[preGapKey] = rawParams[satHorizonKey];
    }

    // Map benchmarks_and_gaps_mode to include_gap
    const bgmKey = 'software_r_and_d.benchmarks_and_gaps_mode';
    const includeGapKey = 'software_r_and_d.include_gap';
    if (bgmKey in rawParams) {
      const includeGap = rawParams[bgmKey] === true || rawParams[bgmKey] === 'gap';
      userParams[includeGapKey] = includeGap ? 'gap' : 'no gap';
    }

    // Map coding_labor_exponent to parallel_penalty
    const cleKey = 'software_r_and_d.coding_labor_exponent';
    const ppKey = 'software_r_and_d.parallel_penalty';
    if (cleKey in rawParams && typeof rawParams[cleKey] === 'number') {
      userParams[ppKey] = rawParams[cleKey];
    }

    return userParams;
  }, []);

  // Fetch sample trajectories when parameters change
  // Samples are fetched SEQUENTIALLY to avoid overwhelming the backend with parallel calibrations
  // Wait for mainline to load first so user sees mainline data before MC samples start
  useEffect(() => {
    if (!samplingConfig || !mainlineLoaded) return;

    if (hasUsedInitialSamplesRef.current && sampleTrajectories.length > 0) {
      hasUsedInitialSamplesRef.current = false;
      return;
    }

    if (sampleTrajectoriesAbortRef.current) {
      sampleTrajectoriesAbortRef.current.abort();
    }

    const abortController = new AbortController();
    sampleTrajectoriesAbortRef.current = abortController;

    const fetchSamplesSequentially = async () => {
      console.log(`[MC Samples] Starting to fetch ${numSamples} samples sequentially...`);
      const userParams = convertParametersForSampling(parameters);

      // Fetch samples one at a time to avoid overwhelming backend with parallel calibrations
      // Each sample appears on the chart as soon as it's ready
      for (let i = 0; i < numSamples; i++) {
        if (abortController.signal.aborted) break;

        const sampleParams = generateParameterSampleWithUserValues(samplingConfig, userParams, enabledSamplingParams);

        try {
          const trajectory = await fetchSampleTrajectory(sampleParams, abortController.signal);

          if (abortController.signal.aborted) break;

          if (trajectory) {
            console.log(`[MC Sample ${i + 1}/${numSamples}] Loaded and added to chart`);
            setSampleTrajectories(prev => [...prev, {
              trajectory: trajectory.map((p: { year: number; horizonLength: number; effectiveCompute: number; automationFraction?: number; trainingCompute?: number; aiSwProgressMultRefPresentDay?: number }) => ({
                ...p,
                horizonFormatted: formatWorkTimeDuration(p.horizonLength),
              })) as ChartDataPoint[],
              params: sampleParams as Record<string, number | string | boolean>
            }]);
          }
        } catch {
          // Continue to next sample on error
        }
      }
    };

    setSampleTrajectories([]);
    fetchSamplesSequentially();

    return () => {
      abortController.abort();
    };
  }, [samplingConfig, mainlineLoaded, fetchSampleTrajectory, resampleTrigger, enabledSamplingParams, parameters, convertParametersForSampling, numSamples]);

  // Handle parameter change
  const handleParameterChange = useCallback((param: keyof ParametersType, value: number) => {
    setUiParameters(prev => ({ ...prev, [param]: value }));
  }, []);

  const handleParameterCommit = useCallback((param: keyof ParametersType, value: number) => {
    setParameters(prev => {
      const newParams = { ...prev, [param]: value };
      fetchComputeData(newParams);
      return newParams;
    });
    setResampleTrigger(prev => prev + 1);
  }, [fetchComputeData]);

  // Handle checkbox changes
  const handleCheckboxChange = useCallback((
    setter: React.Dispatch<React.SetStateAction<boolean>>,
    value: boolean
  ) => {
    setter(value);
    fetchComputeData(parameters);
    setResampleTrigger(prev => prev + 1);
  }, [fetchComputeData, parameters]);

  // URL sync and initial fetch
  useEffect(() => {
    if (Object.keys(contextDefaults).length === 0) return; // Wait for defaults to load
    const urlState = decodeFullStateFromParams(searchParams, contextDefaults);
    if (urlState) {
      setParameters(urlState.parameters);
      setUiParameters(urlState.parameters);
      setEnableCodingAutomation(urlState.enableCodingAutomation);
      setEnableExperimentAutomation(urlState.enableExperimentAutomation);
      setEnableSoftwareProgress(urlState.enableSoftwareProgress);
      fetchComputeData(urlState.parameters);
    } else if (chartData.length === 0) {
      // No URL state and no initial data - fetch with default parameters
      fetchComputeData(parameters);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [contextDefaults]);

  useEffect(() => {
    if (Object.keys(contextDefaults).length === 0) return; // Wait for defaults to load
    const newParams = encodeFullStateToParams({
      parameters,
      enableCodingAutomation,
      enableExperimentAutomation,
      useExperimentThroughputCES: DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES,
      enableSoftwareProgress,
      useComputeLaborGrowthSlowdown: DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown,
      useVariableHorizonDifficulty: DEFAULT_CHECKBOX_STATES.useVariableHorizonDifficulty,
    }, contextDefaults);
    const newUrl = `${pathname}?${newParams.toString()}`;
    router.replace(newUrl, { scroll: false });
  }, [parameters, enableCodingAutomation, enableExperimentAutomation, enableSoftwareProgress, pathname, router, contextDefaults]);

  // Render data
  const renderData = useMemo(() => chartData, [chartData]);
  const hasData = renderData.length > 0;

  // Tooltips
  const AIProgressMultiplierTooltip = useCallback((point: DataPoint) => {
    const y = point.aiSwProgressMultRefPresentDay;
    if (typeof y !== 'number' || !Number.isFinite(y)) return null;
    return (
      <div style={tooltipBoxStyle}>
        <div style={tooltipHeaderStyle}>{Math.round(point.x)}</div>
        <div style={tooltipValueStyle}>{formatUplift(y)}</div>
      </div>
    );
  }, []);

  const HorizonTooltip = useCallback((point: DataPoint) => {
    const y = point.horizonLength;
    if (typeof y !== 'number' || !Number.isFinite(y)) return null;
    return (
      <div style={tooltipBoxStyle}>
        <div style={tooltipHeaderStyle}>{Math.round(point.x)}</div>
        <div style={tooltipValueStyle}>{formatWorkTimeDuration(y)}</div>
      </div>
    );
  }, []);

  // Reference line for automation threshold
  const automationReferenceLineYear = milestones?.['100%-automation-fraction']?.year;
  const automationReferenceLine = typeof automationReferenceLineYear === 'number' 
    ? { x: automationReferenceLineYear, stroke: '#888', strokeDasharray: '4 4', strokeWidth: 1 }
    : undefined;

  // All small chart definitions (matching ProgressChartClient)
  const smallCharts: SmallChartDef[] = useMemo(() => ([
    {
      key: 'automationFraction',
      title: 'Coding Automation Fraction',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="automationFraction" formatter={(v) => `${(v * 100).toFixed(1)}%`} />,
      yFormatter: (v) => `${(v * 100).toFixed(1)}%`,
      scaleType: 'linear'
    },
    {
      key: 'aiCodingLaborMultiplier',
      title: 'AI Parallel Coding Labor Multiplier',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="aiCodingLaborMultiplier" formatter={(v) => formatCompactNumberNode(v, { suffix: ' x', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      logSuffix: ' x',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: ' x' })
    },
    {
      key: 'serialCodingLaborMultiplier',
      title: 'AI Serial Coding Labor Multiplier',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="serialCodingLaborMultiplier" formatter={(v) => formatCompactNumberNode(v, { suffix: ' x', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      logSuffix: ' x',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: ' x' })
    },
    {
      key: 'aiResearchTaste',
      title: 'AI Experiment Selection Skill',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="aiResearchTaste" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'researchEffort',
      title: 'Software Research Effort',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="researchEffort" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'researchStock',
      title: 'Cumulative Research Effort',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="researchStock" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'softwareProgressRate',
      title: 'Software Efficiency Growth Rate',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="softwareProgressRate" formatter={formatTo3SigFigs} />,
      scaleType: 'linear'
    },
    {
      key: 'softwareEfficiency',
      title: 'Software Efficiency',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="softwareEfficiency" formatter={(v) => formatCompactNumberNode(Math.pow(10, v), { renderMode: 'html' })} />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(Math.pow(10, v)),
      isDataInLogForm: true
    },
    {
      key: 'experimentCapacity',
      title: 'Experiment Throughput',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="experimentCapacity" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'inferenceCompute',
      title: 'Inference Compute for Coding',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="inferenceCompute" formatter={(v) => formatCompactNumberNode(v, { suffix: ' H100e', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: ' H100e' }),
      logSuffix: ' H100e'
    },
    {
      key: 'experimentCompute',
      title: 'Experiment Compute',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="experimentCompute" formatter={(v) => formatCompactNumberNode(v, { suffix: ' H100e', renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v, { suffix: ' H100e' }),
      logSuffix: ' H100e'
    },
    {
      key: 'humanLabor',
      title: 'Human Coding Labor',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="humanLabor" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
  ]), []);

  const mainChartHeight = 350;
  const smallChartHeight = 100;

  return (
    <ParameterHoverProvider>
      {/* Fixed Header - only shown when not embedded */}
      {!hideHeader && (
        <header className="fixed top-0 left-0 right-0 z-50 flex flex-row items-center justify-between px-6 py-0 border-b border-gray-200 bg-white">
          <nav className="flex flex-row">
            <Link
              href="/ai-timelines-and-takeoff"
              className="px-4 py-4 text-sm font-medium text-gray-900 border-b-2 border-[#5E6FB8]"
            >
              AI Timelines and Takeoff
            </Link>
            <Link
              href="/ai-black-projects"
              className="px-4 py-4 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50 border-b-2 border-transparent"
            >
              Black Projects
            </Link>
          </nav>
        </header>
      )}

      {/* Desktop-only warning */}
      <div className={`lg:hidden h-screen flex items-center justify-center bg-vivid-background p-8 ${hideHeader ? '' : 'pt-20'}`}>
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Playground is Desktop Only</h1>
          <p className="text-gray-600 mb-6">The playground requires a larger screen for the best experience.</p>
          <Link href="/" className="text-blue-600 hover:underline">← Back to main site</Link>
        </div>
      </div>

      {/* Desktop layout using CSS Grid */}
      {/*
        Grid structure:
        +------------------+---------------------------+
        |   Sidebar with   | Chart | Chart | Chart     |
        |   Model Diagram  |   1   |   2   |   3       |
        |   & Parameters   +---------------------------+
        |   (row-span-2)   |    All Small Charts       |
        +------------------+---------------------------+
      */}
      <div
        className="hidden lg:grid bg-vivid-background overflow-hidden"
        style={{
          gridTemplateColumns: '260px 1fr',
          gridTemplateRows: 'minmax(300px, 1fr) 1fr',
          height: hideHeader ? '100vh' : 'calc(100vh - 57px)',
          marginTop: hideHeader ? '0' : '57px',
        }}
      >
        {/* Sidebar - spans all rows */}
        <aside className="row-span-2 border-r border-gray-200/60 bg-vivid-background flex flex-col overflow-hidden">
          <div className="px-3 py-2 border-b border-gray-200/60 shrink-0">
            <h1 className="text-sm font-bold text-primary font-et-book">Parameters</h1>
          </div>
          
          {/* Model Diagram */}
          <div className="px-2 py-2 border-b border-gray-200/60 shrink-0">
            <div className="flex gap-1 items-center mb-1">
              <span className="leading-tight text-[10px] font-semibold text-primary text-left font-system-mono">Model</span>
              <div className="flex-1 border-t border-gray-500/30" />
            </div>
            <div className="opacity-80 hover:opacity-100 transition-opacity">
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto px-2 py-2">
            {/* Monte Carlo Samples */}
            <div className="mb-2">
              <div className="flex gap-1 items-center mb-1.5">
                <span className="leading-tight text-[10px] font-semibold text-primary text-left font-system-mono">Monte Carlo</span>
                <div className="flex-1 border-t border-gray-500/30" />
              </div>
              <div className="space-y-0.5">
                <label className="block text-[11px] font-medium text-foreground font-sans leading-tight">
                  Sample Trajectories: {numSamples}
                </label>
                <input
                  type="range"
                  min={0}
                  max={50}
                  step={1}
                  value={numSamples}
                  onChange={(e) => setNumSamples(parseInt(e.target.value))}
                  className="w-full rounded-lg appearance-none cursor-pointer slider"
                />
              </div>
            </div>

            {/* Key Parameters */}
            <div className="mb-2">
              <div className="flex gap-1 items-center mb-1.5">
                <span className="leading-tight text-[10px] font-semibold text-primary text-left font-system-mono">Key Parameters</span>
                <div className="flex-1 border-t border-gray-500/30" />
              </div>
              <div className="space-y-2">
                <ParameterSlider
                  paramName="ac_time_horizon_minutes"
                  label="AC Time Horizon Requirement"
                  customMin={scHorizonLogBounds.min}
                  customMax={scHorizonLogBounds.max}
                  step={0.1}
                  customFormatValue={formatSCHorizon}
                  value={uiParameters['software_r_and_d.ac_time_horizon_minutes'] as number}
                  uiParameters={uiParameters}
                  setUiParameters={setUiParameters}
                  allParameters={allParameters}
                  isDragging={isDragging}
                  setIsDragging={setIsDragging}
                  commitParameters={commitParameters}
                  useLogScale={true}
                />
                <ParameterSlider
                  paramName="present_doubling_time"
                  label="Present Doubling Time"
                  customMin={samplingConfigBounds.present_doubling_time?.min}
                  fallbackMin={0.01}
                  fallbackMax={2.0}
                  step={0.01}
                  decimalPlaces={2}
                  customFormatValue={(years) => formatTimeDuration(yearsToMinutes(years))}
                  value={uiParameters['software_r_and_d.present_doubling_time'] as number}
                  uiParameters={uiParameters}
                  setUiParameters={setUiParameters}
                  allParameters={allParameters}
                  isDragging={isDragging}
                  setIsDragging={setIsDragging}
                  commitParameters={commitParameters}
                  useLogScale={true}
                />
                <ParameterSlider
                  paramName="doubling_difficulty_growth_factor"
                  label="Doubling Difficulty Growth Factor"
                  customMin={samplingConfigBounds.doubling_difficulty_growth_factor?.min}
                  customMax={samplingConfigBounds.doubling_difficulty_growth_factor?.max}
                  fallbackMin={0.5}
                  fallbackMax={1.5}
                  step={0.01}
                  decimalPlaces={3}
                  value={uiParameters['software_r_and_d.doubling_difficulty_growth_factor'] as number}
                  uiParameters={uiParameters}
                  setUiParameters={setUiParameters}
                  allParameters={allParameters}
                  isDragging={isDragging}
                  setIsDragging={setIsDragging}
                  commitParameters={commitParameters}
                  disabled={lockedParameters.has('doubling_difficulty_growth_factor')}
                />
                <ParameterSlider
                  paramName="ai_research_taste_slope"
                  label="Experiment Selection Skill Slope"
                  step={0.1}
                  customMin={samplingConfigBounds.ai_research_taste_slope?.min}
                  customMax={samplingConfigBounds.ai_research_taste_slope?.max}
                  fallbackMin={0.1}
                  fallbackMax={10.0}
                  decimalPlaces={1}
                  value={uiParameters['software_r_and_d.ai_research_taste_slope'] as number}
                  uiParameters={uiParameters}
                  setUiParameters={setUiParameters}
                  allParameters={allParameters}
                  isDragging={isDragging}
                  setIsDragging={setIsDragging}
                  commitParameters={commitParameters}
                  useLogScale={true}
                />
                <ParameterSlider
                  paramName="median_to_top_taste_multiplier"
                  label="Experiment Selection Skill Median to Top Multiplier"
                  customMin={samplingConfigBounds.median_to_top_taste_multiplier?.min}
                  customMax={samplingConfigBounds.median_to_top_taste_multiplier?.max}
                  fallbackMin={1.1}
                  fallbackMax={20.0}
                  step={0.1}
                  decimalPlaces={2}
                  value={uiParameters['software_r_and_d.median_to_top_taste_multiplier'] as number}
                  uiParameters={uiParameters}
                  setUiParameters={setUiParameters}
                  allParameters={allParameters}
                  isDragging={isDragging}
                  setIsDragging={setIsDragging}
                  commitParameters={commitParameters}
                  disabled={lockedParameters.has('median_to_top_taste_multiplier')}
                  useLogScale={true}
                />
              </div>
            </div>


            <AdvancedSections
              uiParameters={uiParameters}
              setUiParameters={setUiParameters}
              allParameters={allParameters}
              isDragging={isDragging}
              setIsDragging={setIsDragging}
              commitParameters={commitParameters}
              scHorizonLogBounds={scHorizonLogBounds}
              preGapHorizonBounds={preGapHorizonBounds}
              parallelPenaltyBounds={parallelPenaltyBounds}
              lockedParameters={lockedParameters}
              samplingConfigBounds={samplingConfigBounds}
              simplificationCheckboxes={{
                enableCodingAutomation,
                setEnableCodingAutomation,
                enableExperimentAutomation,
                setEnableExperimentAutomation,
                useExperimentThroughputCES,
                setUseExperimentThroughputCES,
                enableSoftwareProgress,
                setEnableSoftwareProgress,
                useComputeLaborGrowthSlowdown,
                setUseComputeLaborGrowthSlowdown,
                useVariableHorizonDifficulty,
                setUseVariableHorizonDifficulty,
              }}
            />
            
            <div className="mt-2 pt-2 border-t border-gray-200/60">
              <button
                onClick={() => {
                  setParameters({ ...contextDefaults });
                  setUiParameters({ ...contextDefaults });
                  setEnableCodingAutomation(DEFAULT_CHECKBOX_STATES.enableCodingAutomation);
                  setEnableExperimentAutomation(DEFAULT_CHECKBOX_STATES.enableExperimentAutomation);
                  setEnableSoftwareProgress(DEFAULT_CHECKBOX_STATES.enableSoftwareProgress);
                  setUseExperimentThroughputCES(DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES);
                  setUseComputeLaborGrowthSlowdown(DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown);
                  setUseVariableHorizonDifficulty(DEFAULT_CHECKBOX_STATES.useVariableHorizonDifficulty);
                  fetchComputeData({ ...contextDefaults });
                  setResampleTrigger(prev => prev + 1);
                }}
                className="text-[10px] text-gray-500 hover:text-gray-700"
              >
                Reset to defaults
              </button>
            </div>
          </div>
        </aside>

        <ChartSyncProvider>
          {/* Row 1: Main Charts */}
          <div className="border-b border-gray-200/60 grid grid-cols-3 gap-0 min-h-0">
            {/* Main Chart 1 - AI R&D Progress */}
            <div className="p-3 min-w-0 overflow-hidden border-r border-gray-200/60">
              <ChartLoadingOverlay isLoading={mainLoading} blocking={!hasData} className="h-full">
                <AIRnDProgressMultiplierChart
                  chartData={renderData}
                  tooltip={AIProgressMultiplierTooltip}
                  className="h-full"
                  milestones={milestones}
                  displayEndYear={displayEndYear}
                  verticalReferenceLine={automationReferenceLine}
                  height={mainChartHeight}
                  sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                />
              </ChartLoadingOverlay>
            </div>

            {/* Main Chart 2 - Coding Time Horizon */}
            <div className="p-3 min-w-0 overflow-hidden border-r border-gray-200/60">
              <ChartLoadingOverlay isLoading={mainLoading} blocking={!hasData} className="h-full">
                <CustomHorizonChart
                  chartData={renderData}
                  scHorizonMinutes={scHorizonMinutes}
                  tooltip={HorizonTooltip}
                  formatTimeDuration={formatWorkTimeDuration}
                  benchmarkData={benchmarkData}
                  className="h-full"
                  displayEndYear={displayEndYear}
                  height={mainChartHeight}
                  sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                />
              </ChartLoadingOverlay>
            </div>

            {/* Main Chart 3 - Effective Compute */}
            <div className="p-3 min-w-0 overflow-hidden">
              <ChartLoadingOverlay isLoading={mainLoading} blocking={!hasData} className="h-full">
                <CombinedComputeChart
                  chartData={renderData}
                  title={enableSoftwareProgress ? 'Effective & Training Compute' : 'Effective Compute'}
                  verticalReferenceLine={automationReferenceLine}
                  className="h-full"
                  displayEndYear={displayEndYear}
                  height={mainChartHeight}
                  showTrainingSeries={enableSoftwareProgress}
                  sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                />
              </ChartLoadingOverlay>
            </div>
          </div>

          {/* Row 2: All Small Charts in Uniform Grid */}
          <div className="p-3 overflow-auto">
            <div 
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gridAutoRows: `${smallChartHeight}px`,
                gap: '12px',
              }}
            >
              {smallCharts.map((chart) => (
                <ChartLoadingOverlay key={chart.key} isLoading={mainLoading} blocking={!hasData} className="h-full w-full">
                  <CustomMetricChart
                    chartData={renderData}
                    dataKey={chart.key}
                    title={chart.title}
                    tooltip={chart.tooltip}
                    yScale={chart.yScale}
                    yFormatter={chart.yFormatter}
                    displayEndYear={displayEndYear}
                    height={smallChartHeight}
                    logSuffix={chart.logSuffix}
                    isDataInLogForm={chart.isDataInLogForm}
                    sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
                  />
                </ChartLoadingOverlay>
              ))}
            </div>
          </div>
        </ChartSyncProvider>
      </div>
    </ParameterHoverProvider>
  );
}
