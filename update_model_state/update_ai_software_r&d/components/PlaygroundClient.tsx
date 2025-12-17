'use client';

import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import CustomHorizonChart from './CustomHorizonChart';
import CombinedComputeChart from './CombinedComputeChart';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { ChartDataPoint, BenchmarkPoint } from '@/app/types';
import { CustomMetricChart } from './CustomMetricChart';
import type { DataPoint } from './CustomLineChart';

import { formatTo3SigFigs, formatWorkTimeDuration, formatUplift, formatCompactNumberNode, formatAsPowerOfTenText, formatSCHorizon, formatTimeDuration, yearsToMinutes } from '@/utils/formatting';
import { DEFAULT_PARAMETERS, ParametersType } from '@/constants/parameters';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import { convertParametersToAPIFormat, ParameterRecord } from '@/utils/monteCarlo';
import { ChartSyncProvider } from './ChartSyncContext';
import { ChartLoadingOverlay } from './ChartLoadingOverlay';
import { encodeFullStateToParams, decodeFullStateFromParams, DEFAULT_CHECKBOX_STATES } from '@/utils/urlState';
import { AIRnDProgressMultiplierChart } from './AIRnDProgressMultiplierChart';
import type { MilestoneMap } from '@/types/milestones';
import { SmallChartMetricTooltip } from './SmallChartMetricTooltip';
import { ParameterHoverProvider } from './ParameterHoverContext';
import ModelDiagram from '@/svgs/model-diagram-semantic.svg';
import { SamplingConfig, generateParameterSampleWithUserValues, initializeCorrelationSampling, extractSamplingConfigBounds } from '@/utils/sampling';
import type { ComputeApiResponse } from '@/lib/serverApi';
import Link from 'next/link';
import { AdvancedSections } from './AdvancedSections';
import { ParameterSlider } from './ParameterSlider';

interface SampleTrajectoryWithParams {
  trajectory: ChartDataPoint[];
  params: Record<string, number | string | boolean>;
}

interface PlaygroundClientProps {
  benchmarkData: BenchmarkPoint[];
  initialComputeData: ComputeApiResponse;
  initialParameters: ParametersType;
  initialSampleTrajectories?: { trajectory: { year: number; horizonLength: number; effectiveCompute: number; automationFraction?: number; trainingCompute?: number; aiSwProgressMultRefPresentDay?: number }[]; params: Record<string, number | string | boolean> }[];
  initialSeed?: number;
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

const SIMULATION_END_YEAR = 2045;
// Use relative URL to go through Next.js proxy (configured in next.config.ts)
// This avoids CORS issues and works in both dev and production
const API_BASE_URL = '';

function processInitialData(data: ComputeApiResponse): ChartDataPoint[] {
  if (!data?.time_series) return [];
  return data.time_series.map((point) => ({
    year: point.year,
    horizonLength: point.horizonLength,
    horizonFormatted: formatWorkTimeDuration(point.horizonLength),
    effectiveCompute: point.effectiveCompute,
    trainingCompute: point.trainingCompute ?? null,
    automationFraction: point.automationFraction ?? null,
    aiSoftwareProgressMultiplier: point.aiSoftwareProgressMultiplier,
    aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
    aiResearchTaste: point.aiResearchTaste ?? null,
    experimentCapacity: point.experimentCapacity ?? null,
    inferenceCompute: point.inferenceCompute ?? null,
    experimentCompute: point.experimentCompute ?? null,
    researchEffort: point.researchEffort ?? null,
    researchStock: point.researchStock ?? null,
    softwareProgressRate: point.softwareProgressRate ?? null,
    softwareEfficiency: point.softwareEfficiency ?? null,
    aiCodingLaborMultiplier: point.aiCodingLaborMultiplier ?? null,
    humanLabor: point.humanLabor ?? null,
  }));
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
}: PlaygroundClientProps) {
  const initialChartData = useMemo(() => processInitialData(initialComputeData), [initialComputeData]);
  const resolvedInitialParameters = useMemo(() => initialParameters ?? { ...DEFAULT_PARAMETERS }, [initialParameters]);
  
  const convertedInitialSamples = useMemo(() => {
    return initialSampleTrajectories.map(sample => ({
      trajectory: sample.trajectory.map(point => ({
        year: point.year,
        horizonLength: point.horizonLength,
        horizonFormatted: formatWorkTimeDuration(point.horizonLength),
        effectiveCompute: point.effectiveCompute,
        automationFraction: point.automationFraction,
        trainingCompute: point.trainingCompute ?? null,
        aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
      })) as ChartDataPoint[],
      params: sample.params
    }));
  }, [initialSampleTrajectories]);
  
  const [chartData, setChartData] = useState<ChartDataPoint[]>(initialChartData);
  const [scHorizonMinutes, setScHorizonMinutes] = useState<number>(
    Math.pow(10, DEFAULT_PARAMETERS.ac_time_horizon_minutes)
  );
  const [parameters, setParameters] = useState<ParametersType>(resolvedInitialParameters);
  const [uiParameters, setUiParameters] = useState<ParametersType>(resolvedInitialParameters);
  const [milestones, setMilestones] = useState<MilestoneMap | null>(
    initialComputeData?.milestones && typeof initialComputeData.milestones === 'object'
      ? (initialComputeData.milestones as MilestoneMap)
      : null
  );
  const [mainLoading, setMainLoading] = useState(false);
  const [sampleTrajectories, setSampleTrajectories] = useState<SampleTrajectoryWithParams[]>(convertedInitialSamples);
  const [samplingConfig, setSamplingConfig] = useState<SamplingConfig | null>(null);
  const [enabledSamplingParams, setEnabledSamplingParams] = useState<Set<string>>(new Set());
  const [resampleTrigger, setResampleTrigger] = useState(0);
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

  // Commit UI parameters to actual parameters when dragging ends
  const commitParameters = useCallback((nextParameters?: ParametersType) => {
    const newParams = nextParameters ?? uiParameters;
    setParameters(newParams);
    setIsDragging(false);
    fetchComputeData(newParams);
    setResampleTrigger(prev => prev + 1);
  }, [uiParameters]);

  // Load sampling config
  useEffect(() => {
    let isCancelled = false;

    const loadSamplingConfig = async () => {
      try {
        const response = await fetch('/api/sampling-config');
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!isCancelled && data.success) {
          initializeCorrelationSampling(data.config.correlation_matrix);
          setSamplingConfig(data.config);
          
          const allParams = new Set<string>();
          for (const paramName of Object.keys(data.config.parameters)) {
            allParams.add(paramName);
          }
          if (data.config.time_series_parameters) {
            for (const paramName of Object.keys(data.config.time_series_parameters)) {
              allParams.add(paramName);
            }
          }
          setEnabledSamplingParams(allParams);
        }
      } catch (error) {
        console.error('Failed to load sampling configuration:', error);
      }
    };

    loadSamplingConfig();

    return () => {
      isCancelled = true;
    };
  }, []);

  // Update SC horizon line immediately while slider moves
  useEffect(() => {
    const instantHorizon = Math.pow(10, uiParameters.ac_time_horizon_minutes);
    setScHorizonMinutes(instantHorizon);
  }, [uiParameters.ac_time_horizon_minutes]);

  // Fetch compute data
  const fetchComputeData = useCallback(async (params: ParametersType) => {
    if (currentRequestRef.current) {
      currentRequestRef.current.abort();
    }
    
    const controller = new AbortController();
    currentRequestRef.current = controller;
    setMainLoading(true);
    
    try {
      const apiParameters = convertParametersToAPIFormat(params as unknown as ParameterRecord);
      const response = await fetch(`${API_BASE_URL}/api/compute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parameters: apiParameters,
          time_range: [2012, SIMULATION_END_YEAR],
          initial_progress: 0.0
        }),
        signal: controller.signal,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      
      if (data.success && data.time_series) {
        setChartData(processInitialData(data));
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

  // Fetch sample trajectory
  const fetchSampleTrajectory = useCallback(async (params: Record<string, number | string | boolean>, signal?: AbortSignal) => {
    try {
      const apiParams = convertParametersToAPIFormat(params as ParameterRecord);
      const response = await fetch(`${API_BASE_URL}/api/compute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parameters: apiParams,
          time_range: [2012, SIMULATION_END_YEAR],
          initial_progress: 0.0
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
    
    if (typeof rawParams.ac_time_horizon_minutes === 'number') {
      userParams.ac_time_horizon_minutes = Math.pow(10, rawParams.ac_time_horizon_minutes);
    }
    
    if ('saturation_horizon_minutes' in rawParams) {
      userParams.pre_gap_ac_time_horizon = rawParams.saturation_horizon_minutes;
    }
    if ('benchmarks_and_gaps_mode' in rawParams) {
      const includeGap = rawParams.benchmarks_and_gaps_mode === true || rawParams.benchmarks_and_gaps_mode === 'gap';
      userParams.include_gap = includeGap ? 'gap' : 'no gap';
    }
    if ('coding_labor_exponent' in rawParams && typeof rawParams.coding_labor_exponent === 'number') {
      userParams.parallel_penalty = rawParams.coding_labor_exponent;
    }
    
    return userParams;
  }, []);

  // Fetch sample trajectories when parameters change
  useEffect(() => {
    if (!samplingConfig) return;

    if (hasUsedInitialSamplesRef.current && sampleTrajectories.length > 0) {
      hasUsedInitialSamplesRef.current = false;
      return;
    }

    if (sampleTrajectoriesAbortRef.current) {
      sampleTrajectoriesAbortRef.current.abort();
    }

    const abortController = new AbortController();
    sampleTrajectoriesAbortRef.current = abortController;

    const fetchAllSamples = async () => {
      const userParams = convertParametersForSampling(parameters);

      const NUM_SAMPLES = 10;
      const samplePromises = Array.from({ length: NUM_SAMPLES }, async () => {
        const sampleParams = generateParameterSampleWithUserValues(samplingConfig, userParams, enabledSamplingParams);
        
        try {
          const trajectory = await fetchSampleTrajectory(sampleParams, abortController.signal);
          
          if (abortController.signal.aborted) return null;
          
          if (trajectory) {
            setSampleTrajectories(prev => [...prev, { 
              trajectory: trajectory.map((p: { year: number; horizonLength: number; effectiveCompute: number; automationFraction?: number; trainingCompute?: number; aiSwProgressMultRefPresentDay?: number }) => ({
                ...p,
                horizonFormatted: formatWorkTimeDuration(p.horizonLength),
              })) as ChartDataPoint[], 
              params: sampleParams as Record<string, number | string | boolean>
            }]);
          }
          return trajectory;
        } catch {
          return null;
        }
      });

      await Promise.allSettled(samplePromises);
    };

    setSampleTrajectories([]);
    fetchAllSamples();

    return () => {
      abortController.abort();
    };
  }, [samplingConfig, fetchSampleTrajectory, resampleTrigger, enabledSamplingParams, parameters, convertParametersForSampling]);

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

  // URL sync
  useEffect(() => {
    const urlState = decodeFullStateFromParams(searchParams);
    if (urlState) {
      setParameters(urlState.parameters);
      setUiParameters(urlState.parameters);
      setEnableCodingAutomation(urlState.enableCodingAutomation);
      setEnableExperimentAutomation(urlState.enableExperimentAutomation);
      setEnableSoftwareProgress(urlState.enableSoftwareProgress);
      fetchComputeData(urlState.parameters);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const newParams = encodeFullStateToParams({
      parameters,
      enableCodingAutomation,
      enableExperimentAutomation,
      useExperimentThroughputCES: DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES,
      enableSoftwareProgress,
      useComputeLaborGrowthSlowdown: DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown,
      useVariableHorizonDifficulty: DEFAULT_CHECKBOX_STATES.useVariableHorizonDifficulty,
    });
    const newUrl = `${pathname}?${newParams.toString()}`;
    router.replace(newUrl, { scroll: false });
  }, [parameters, enableCodingAutomation, enableExperimentAutomation, enableSoftwareProgress, pathname, router]);

  // Render data
  const renderData = useMemo(() => chartData, [chartData]);

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
      {/* Desktop-only warning */}
      <div className="lg:hidden h-screen flex items-center justify-center bg-vivid-background p-8">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Playground is Desktop Only</h1>
          <p className="text-gray-600 mb-6">The playground requires a larger screen for the best experience.</p>
          <Link href="/" className="text-blue-600 hover:underline">← Back to main site</Link>
        </div>
      </div>

      {/* Desktop layout using CSS Grid */}
      {/* 
        Grid structure:
        +------------------+-------+-------+-------+
        |                  | Chart | Chart | Chart |
        |     Sidebar      |   1   |   2   |   3   |
        |   (row-span-3)   +-------+-------+-------+
        |                  |  Log  |  Log  |  Log  |
        |                  | Charts| Charts| Charts|
        |                  +-------+-------+-------+
        |                  | Model | Model |Linear |
        |                  | Diag  | Diag  |Charts |
        +------------------+-------+-------+-------+
      */}
      <div 
        className="hidden lg:grid h-screen bg-vivid-background overflow-hidden"
        style={{
          gridTemplateColumns: '280px 1fr 1fr 1fr',
          gridTemplateRows: `${mainChartHeight + 30}px 1fr minmax(250px, auto)`,
        }}
      >
        {/* Sidebar - spans all rows */}
        <aside className="row-span-3 border-r border-gray-200 bg-white flex flex-col overflow-hidden">
          <div className="p-4 border-b border-gray-100 shrink-0">
            <Link href="/" className="text-sm text-gray-500 hover:text-gray-700 mb-2 block">← Back</Link>
            <h1 className="text-lg font-bold text-primary font-et-book">Playground</h1>
          </div>
          
          <div className="flex-1 overflow-y-auto px-4 py-4">
            {/* Key Parameters */}
            <div className="mb-6">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Key Parameters</h2>
              <div className="space-y-4">
                <ParameterSlider
                  paramName="ac_time_horizon_minutes"
                  label="AC Time Horizon Requirement"
                  customMin={scHorizonLogBounds.min}
                  customMax={scHorizonLogBounds.max}
                  step={0.1}
                  customFormatValue={formatSCHorizon}
                  value={uiParameters.ac_time_horizon_minutes}
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
                  value={uiParameters.present_doubling_time}
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
                  value={uiParameters.doubling_difficulty_growth_factor}
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
                  value={uiParameters.ai_research_taste_slope}
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
                  value={uiParameters.median_to_top_taste_multiplier}
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

            <div className="border-t border-gray-100 pt-4 mb-4" />

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
            
            <div className="mt-6 pt-4 border-t border-gray-100">
              <button
                onClick={() => {
                  setParameters({ ...DEFAULT_PARAMETERS });
                  setUiParameters({ ...DEFAULT_PARAMETERS });
                  setEnableCodingAutomation(DEFAULT_CHECKBOX_STATES.enableCodingAutomation);
                  setEnableExperimentAutomation(DEFAULT_CHECKBOX_STATES.enableExperimentAutomation);
                  setEnableSoftwareProgress(DEFAULT_CHECKBOX_STATES.enableSoftwareProgress);
                  setUseExperimentThroughputCES(DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES);
                  setUseComputeLaborGrowthSlowdown(DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown);
                  setUseVariableHorizonDifficulty(DEFAULT_CHECKBOX_STATES.useVariableHorizonDifficulty);
                  fetchComputeData({ ...DEFAULT_PARAMETERS });
                  setResampleTrigger(prev => prev + 1);
                }}
                className="text-xs text-gray-500 hover:text-gray-700"
              >
                Reset to defaults
              </button>
            </div>
          </div>
        </aside>

        <ChartSyncProvider>
          {/* Main Chart 1 - AI R&D Progress */}
          <div className="p-3 border-b border-gray-100 min-w-0 overflow-hidden">
            <ChartLoadingOverlay isLoading={mainLoading} className="h-full">
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
          <div className="p-3 border-b border-gray-100 min-w-0 overflow-hidden">
            <ChartLoadingOverlay isLoading={mainLoading} className="h-full">
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
          <div className="p-3 border-b border-gray-100 min-w-0 overflow-hidden">
            <ChartLoadingOverlay isLoading={mainLoading} className="h-full">
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

          {/* Log-scale Small Charts - spans 3 columns, exactly 2 rows */}
          <div 
            className="col-span-3 p-3 overflow-hidden"
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(5, 1fr)',
              gridTemplateRows: 'repeat(2, 1fr)',
              gap: '12px',
            }}
          >
            {smallCharts.filter(c => c.scaleType === 'log').map((chart) => (
              <ChartLoadingOverlay key={chart.key} isLoading={mainLoading} className="h-full w-full">
                <CustomMetricChart
                  chartData={renderData}
                  dataKey={chart.key}
                  title={chart.title}
                  tooltip={chart.tooltip}
                  yScale={chart.yScale}
                  yFormatter={chart.yFormatter}
                  displayEndYear={displayEndYear}
                  logSuffix={chart.logSuffix}
                  isDataInLogForm={chart.isDataInLogForm}
                />
              </ChartLoadingOverlay>
            ))}
          </div>

          {/* Bottom row: Model Diagram + Linear Charts - spans 3 columns, uses 6-column inner grid */}
          <div 
            className="col-span-3 border-t border-gray-100 grid"
            style={{
              gridTemplateColumns: 'repeat(6, 1fr)',
            }}
          >
            {/* Model Diagram - 3 columns */}
            <div className="col-span-3 p-4 bg-white/50 flex flex-col">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 shrink-0">Model Diagram</h2>
              <div className="flex-1 min-h-0 flex items-center justify-center opacity-60 hover:opacity-100 transition-opacity overflow-hidden">
                <ModelDiagram 
                  style={{ 
                    maxWidth: 'calc(100% - 8px)', 
                    maxHeight: '100%',
                    width: 'auto',
                    height: 'auto',
                  }} 
                />
              </div>
            </div>

            {/* Linear-scale Small Charts - 3 columns */}
            <div 
              className="col-span-3 border-l border-gray-100 p-3 overflow-y-auto"
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
                gridAutoRows: `${smallChartHeight}px`,
                gap: '12px',
                alignContent: 'start',
              }}
            >
              {smallCharts.filter(c => c.scaleType === 'linear').map((chart) => (
                <ChartLoadingOverlay key={chart.key} isLoading={mainLoading} className="h-full">
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
