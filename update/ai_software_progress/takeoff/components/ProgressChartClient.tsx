'use client';

import { useState, useCallback, useEffect, useRef, useMemo, ChangeEvent } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import CustomHorizonChart from './CustomHorizonChart';
import CombinedComputeChart from './CombinedComputeChart';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { ChartDataPoint, BenchmarkPoint } from '@/app/types';
import { CustomMetricChart } from './CustomMetricChart';
import type { DataPoint } from './CustomLineChart';

import { formatTo3SigFigs, formatTimeDuration, formatSCHorizon, yearsToMinutes, formatYearMonth, formatWorkTimeDuration, formatWorkTimeDurationDetailed, formatUplift, formatCompactNumberNode, formatAsPowerOfTenText } from '@/utils/formatting';
import { DEFAULT_PARAMETERS, ParametersType, ParameterPrimitive } from '@/constants/parameters';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import { convertParametersToAPIFormat, convertSampledParametersToAPIFormat, ParameterRecord } from '@/utils/monteCarlo';
import { ParameterSlider } from './ParameterSlider';
import { AdvancedSections } from './AdvancedSections';
import { ChartType } from '@/types/chartConfig';
import { CHART_CONFIGS } from '@/utils/chartConfigs';
import { ChartSyncProvider } from './ChartSyncContext';
import { ChartLoadingOverlay } from './ChartLoadingOverlay';
import { encodeFullStateToParams, decodeFullStateFromParams, DEFAULT_CHECKBOX_STATES } from '@/utils/urlState';
import { AIRnDProgressMultiplierChart } from './AIRnDProgressMultiplierChart';
import type { MilestoneMap } from '@/types/milestones';
import { SmallChartMetricTooltip } from './SmallChartMetricTooltip';
import { MILESTONE_EXPLANATIONS, MILESTONE_FULL_NAMES } from '@/constants/chartExplanations';
import { ParameterHoverProvider } from './ParameterHoverContext';
import { HeaderContent } from './HeaderContent';
import { WithChartTooltip } from './ChartTitleTooltip';
import ModelArchitecture from './ModelArchitecture';
import { SamplingConfig, generateParameterSample, generateParameterSampleWithFixedParams, generateParameterSampleWithUserValues, getDistributionMedian, initializeCorrelationSampling, extractSamplingConfigBounds } from '@/utils/sampling';
import type { ComputeApiResponse } from '@/lib/serverApi';

interface ModelDefaults {
  optimal_ces_eta_init?: number;
  automation_interp_type?: string;
  ai_research_taste_slope?: number;
  anchor_progress_at_strong_cognitive_horizon?: number;
  present_year?: number;
  present_progress?: number;
  progress_at_aa?: number;
  [key: string]: number | string | undefined;
}

interface ParameterConfig {
  defaults?: ModelDefaults;
  bounds?: Record<string, [number, number]>;
  metadata?: Record<string, unknown>;
}

interface SampleTrajectoryWithParams {
  trajectory: ChartDataPoint[];
  params: Record<string, number | string | boolean>;
}

interface TimeSeriesPoint {
  year: number;
  progress: number;
  effectiveCompute: number;
  horizonLength: number;
  researchStock: number;
  automationFraction?: number;
  trainingCompute?: number;
  experimentCapacity?: number;
  aiResearchTaste?: number;
  aiSoftwareProgressMultiplier?: number;
  aiSwProgressMultRefPresentDay?: number;
  serialCodingLaborMultiplier?: number;
  // Newly exposed metrics from API
  humanLabor?: number;
  inferenceCompute?: number;
  experimentCompute?: number;
  researchEffort?: number;
  softwareProgressRate?: number;
  softwareEfficiency?: number;
  aiCodingLaborMultiplier?: number;
  aiCodingLaborMultRefPresentDay?: number;
}

const CHART_ANIMATION_DURATION_MS = 450;

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const interpolateNumericValue = (
  startValue: unknown,
  endValue: unknown,
  progress: number
): number | null => {
  const startNumeric = isFiniteNumber(startValue)
    ? startValue
    : isFiniteNumber(endValue)
      ? endValue
      : null;
  const endNumeric = isFiniteNumber(endValue) ? endValue : startNumeric;

  if (startNumeric == null || endNumeric == null) {
    return null;
  }

  if (progress <= 0) {
    return startNumeric;
  }

  if (progress >= 1) {
    return endNumeric;
  }

  return startNumeric + (endNumeric - startNumeric) * progress;
};

const interpolateChartDataPoints = (
  fromData: ChartDataPoint[],
  toData: ChartDataPoint[],
  progress: number
): ChartDataPoint[] => {
  if (progress >= 1) {
    return toData.map(point => ({ ...point }));
  }

  if (fromData.length === 0) {
    return toData.map(point => ({ ...point }));
  }

  const fromMap = new Map<number, ChartDataPoint>();
  const toMap = new Map<number, ChartDataPoint>();

  fromData.forEach(point => {
    fromMap.set(point.year, point);
  });
  toData.forEach(point => {
    toMap.set(point.year, point);
  });

  const years = Array.from(new Set([...fromMap.keys(), ...toMap.keys()])).sort(
    (a, b) => a - b
  );

  return years.map(year => {
    const startPoint = fromMap.get(year);
    const endPoint = toMap.get(year);

    if (!startPoint && !endPoint) {
      return {
        year,
        horizonLength: 0,
        horizonFormatted: '',
      };
    }

    const resolvedStart = startPoint ?? endPoint!;
    const resolvedEnd = endPoint ?? startPoint!;
    const mergedKeys = new Set([
      ...Object.keys(resolvedStart),
      ...Object.keys(resolvedEnd),
    ]);

    const interpolatedHorizon =
      interpolateNumericValue(
        resolvedStart.horizonLength,
        resolvedEnd.horizonLength,
        progress
      ) ??
      (typeof resolvedEnd.horizonLength === 'number'
        ? resolvedEnd.horizonLength
        : typeof resolvedStart.horizonLength === 'number'
          ? resolvedStart.horizonLength
          : 0);

    const interpolatedPoint: ChartDataPoint = {
      year,
      horizonLength: interpolatedHorizon,
      horizonFormatted:
        progress >= 1
          ? resolvedEnd.horizonFormatted ?? resolvedStart.horizonFormatted ?? ''
          : resolvedStart.horizonFormatted ?? resolvedEnd.horizonFormatted ?? '',
    };

    mergedKeys.forEach(key => {
      if (key === 'year' || key === 'horizonLength' || key === 'horizonFormatted') {
        return;
      }

      const startValue = resolvedStart[key];
      const endValue = resolvedEnd[key];

      if (isFiniteNumber(startValue) || isFiniteNumber(endValue)) {
        interpolatedPoint[key] = interpolateNumericValue(startValue, endValue, progress);
      } else {
        interpolatedPoint[key] = progress >= 1
          ? endValue ?? startValue ?? null
          : startValue ?? endValue ?? null;
      }
    });

    return interpolatedPoint;
  });
};

const VISIBLE_CHART_START_YEAR = 2019;
const SIMULATION_END_YEAR = 2045; // Simulation end year (matches sampling_config.yaml)
const CODING_AUTOMATION_MARKER_COLOR = '#6b7280';
// Benchmark data will be loaded dynamically from YAML file

type MilestoneDisplay = {
  month: string;
  monthNumber: number;
  year: string;
};

const getMilestoneDisplay = (decimalYear: number | null): MilestoneDisplay | null => {
  if (decimalYear == null || !Number.isFinite(decimalYear)) {
    return null;
  }

  const year = Math.floor(decimalYear);
  const fraction = decimalYear - year;
  const monthIndexRaw = Math.round(fraction * 12);

  const monthIndex = monthIndexRaw % 12;
  const adjustedYear = monthIndexRaw === 12 ? year + 1 : year;

  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  return {
    month: months[monthIndex],
    monthNumber: monthIndex + 1,
    year: String(adjustedYear),
  };
};

interface KeyMilestoneValueProps {
  label: string;
  milestone: MilestoneDisplay | null;
  fallback: string;
  isLoading: boolean;
  explanation?: string;
}

const KeyMilestoneValue = ({
  label,
  milestone,
  fallback,
  isLoading,
  explanation,
}: KeyMilestoneValueProps) => (
  <div className="mt-3">
    <div className="flex items-center gap-2">
      <p className="leading-tight text-[12px] font-semibold text-primary text-left font-system-mono">{label}</p>
      <div className="flex-1 border-t border-gray-500/30" />
    </div>
    <div className="mt-1">
      {isLoading ? (
        <div className="h-12 w-2/3 animate-pulse rounded bg-slate-200/70" />
      ) : milestone ? (
        <div className="flex items-center gap-2">
          <div className="text-4xl font-et-book font-bold text-primary mb-0 leading-none flex justify-between flex-1">
            {`${String(milestone.monthNumber).padStart(2, '0')}/${milestone.year}`.split('').map((char, idx) => (
              <span key={idx}>{char}</span>
            ))}
          </div>
          {explanation && (
            <WithChartTooltip explanation={explanation} tooltipPlacement="left">
              <span className="sr-only">Info</span>
            </WithChartTooltip>
          )}
        </div>
      ) : (
        <div className="text-xl font-semibold text-slate-500">{fallback}</div>
      )}
    </div>
  </div>
);

interface KeyMilestonePanelProps {
  automationYear: number | null;
  asiYear: number | null;
  isLoading: boolean;
}

const KeyMilestonePanel = ({
  automationYear,
  asiYear,
  isLoading,
}: KeyMilestonePanelProps) => {
  const automationMilestone = getMilestoneDisplay(automationYear);
  const asiMilestone = getMilestoneDisplay(asiYear);

  return (
    <section
      className="flex h-full flex-col gap-1"
      style={{
        minWidth: CHART_LAYOUT.keyStats.width,
        maxWidth: 350,
        minHeight: CHART_LAYOUT.primary.height,
      }}
    >
      <div className="flex flex-col gap-0 justify-between h-full">
        <div>
        <div className="flex gap-2 items-center my-1">
          <span className="leading-tight text-[12px] font-semibold text-primary text-left font-system-mono">Explanation</span>
          <div className="flex-1 border-t border-gray-500/30" />
        </div>
        <p className="prose text-[15px] lg:text-[12px] mb-2">
          {`This website presents AI Futures Project's latest AI capabilities model, \
following up on the `}
          <a href="https://ai-2027.com/research/timelines-forecast" className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">timelines</a>
          {` and `}
          <a href="https://ai-2027.com/research/takeoff-forecast" className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">takeoff</a>
          {` models we published alongside `}
          <a href="https://ai-2027.com/" className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">AI 2027</a>
          {`.`}
          {` Predicting superhuman AI capabilities inherently requires much intuition and guesswork, but we've nonetheless found quantitative modeling to be useful. `}
        </p>
        <p className="prose text-[15px] lg:text-[12px] mb-2">
          {`To the right you can see the output of our model with each parameter \
set to Eli's median estimate (Eli is a co-author). `}
          <em>{`The default displayed milestone dates differ from \
the median of our Monte Carlo simulations; you can see the results of those \
on the `}
          <a href="https://ai-rates-calculator.vercel.app/forecast" className="text-blue-600 underline" target="_blank" rel="noopener noreferrer">Forecast page</a>
          {`.`}</em>
        </p>
        {/* <p className="prose text-[13px]">
          {`Beginning at the bottom of the page is an explanation of our model, \
including motivation and limitations. If you scroll down to the core model description, \
you'll see a diagram of how our model works, and you can click on variables in the \
diagram to navigate to the corresponding part of the description.`}
        </p> */}
        </div>
        <div>
          <KeyMilestoneValue
            label="Date of Automated Coder (AC)"
            milestone={automationMilestone}
            fallback="After 2045"
            isLoading={isLoading}
            explanation="An AC can (if dropped into present day) autonomously replace the AGI project's coding staff without human help, given 5% of the project's compute."
          />
          <KeyMilestoneValue
            label="Date of Superintelligence (ASI)"
            milestone={asiMilestone}
            fallback="After 2045"
            isLoading={isLoading}
            explanation="An Artificial Superintelligence (ASI) is 2 times as far above the top human as the top human is above the median. (That is, 2x more SDs within the human distribution, not necessarily a full 2x larger difference in absolute performance metrics.)"
          />
        </div>
        
      </div>
    </section>
  );
};

const PARAMETER_PRESETS = [
  { id: 'baseline', label: 'Baseline [not yet functional]' },
  { id: 'automation-forward', label: 'Alternative [not yet functional]' },
//  { id: 'compute-heavy', label: 'Compute-Heavy' },
];

// Custom hook to sync checkbox state with parameter values
interface CheckboxParameterSyncConfig {
  checkboxValue: boolean;
  checkboxKey: keyof typeof DEFAULT_CHECKBOX_STATES;
  nonDefaultParameterValues: Partial<ParametersType>;
  setParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
  setUiParameters: React.Dispatch<React.SetStateAction<ParametersType>>;
  isHydratingFromUrlRef: React.MutableRefObject<boolean>;
}

function useCheckboxParameterSync({
  checkboxValue,
  checkboxKey,
  nonDefaultParameterValues,
  setParameters,
  setUiParameters,
  isHydratingFromUrlRef,
}: CheckboxParameterSyncConfig) {
  useEffect(() => {
    if (isHydratingFromUrlRef.current) {
      return;
    }

    const isDefaultState = checkboxValue === DEFAULT_CHECKBOX_STATES[checkboxKey];
    const defaultCheckboxValue = DEFAULT_CHECKBOX_STATES[checkboxKey];
    const isInNonDefaultState = checkboxValue !== defaultCheckboxValue;

    const nonDefaultUpdaterFn = (prev: ParametersType) => ({
      ...prev,
      ...nonDefaultParameterValues,
      // We need the cast because in theory we could be getting an object with explicit undefined values for some keys
      // Hopefully nobody does that...
    } as ParametersType);

    // When in NON-DEFAULT state (unchecked), apply the special parameter values
    // When in DEFAULT state (checked), apply the default parameter values
    if (isInNonDefaultState) {
      // Apply non-default parameter values when checkbox is unchecked
      setUiParameters(nonDefaultUpdaterFn);
      setParameters(nonDefaultUpdaterFn);
    } else if (isDefaultState) {
      const parameterKeys = Object.keys(nonDefaultParameterValues) as Array<keyof ParametersType>;
      const defaultValues: Partial<ParametersType> = {};
      parameterKeys.forEach(key => {
        defaultValues[key] = DEFAULT_PARAMETERS[key];
      });

      const defaultUpdaterFn = (prev: ParametersType) => ({
        ...prev,
        ...defaultValues,
      } as ParametersType);

      setUiParameters(defaultUpdaterFn);
      setParameters(defaultUpdaterFn);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [checkboxValue]);
}

// Types
interface InitialSampleTrajectory {
  trajectory: Array<{
    year: number;
    horizonLength: number;
    effectiveCompute: number;
    automationFraction: number;
    trainingCompute?: number;
    aiSwProgressMultRefPresentDay?: number;
  }>;
  params: Record<string, number | string | boolean>;
}

interface ProgressChartProps {
  benchmarkData?: BenchmarkPoint[];
  modelDescriptionGDocPortionMarkdown?: React.ReactNode;
  initialComputeData?: ComputeApiResponse;
  initialParameters?: ParametersType;
  initialSampleTrajectories?: InitialSampleTrajectory[];
  initialSeed?: number;
}

// Process initial compute data into ChartDataPoint format
function processInitialData(data: ComputeApiResponse | undefined): ChartDataPoint[] {
  if (!data?.success || !data.time_series?.length) return [];
  
  return data.time_series
    .filter((point) => point.year >= VISIBLE_CHART_START_YEAR)
    .sort((a, b) => a.year - b.year)
    .map((point) => ({
      year: point.year,
      horizonLength: point.horizonLength,
      horizonFormatted: formatWorkTimeDuration(point.horizonLength),
      effectiveCompute: point.effectiveCompute,
      automationFraction: point.automationFraction,
      trainingCompute: point.trainingCompute ?? null,
      experimentCapacity: point.experimentCapacity,
      aiResearchTaste: point.aiResearchTaste,
      aiSoftwareProgressMultiplier: point.aiSoftwareProgressMultiplier,
      aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
      serialCodingLaborMultiplier: point.serialCodingLaborMultiplier,
      humanLabor: point.humanLabor ?? null,
      inferenceCompute: point.inferenceCompute ?? null,
      experimentCompute: point.experimentCompute ?? null,
      researchEffort: point.researchEffort ?? null,
      researchStock: point.researchStock ?? null,
      softwareProgressRate: point.softwareProgressRate ?? null,
      softwareEfficiency: point.softwareEfficiency ?? null,
      aiCodingLaborMultiplier: point.aiCodingLaborMultiplier ?? null,
      aiCodingLaborMultRefPresentDay: point.aiCodingLaborMultRefPresentDay ?? null,
    }));
}

export default function ProgressChart({ 
  benchmarkData = [], 
  modelDescriptionGDocPortionMarkdown,
  initialComputeData, 
  initialParameters,
  initialSampleTrajectories = [],
  initialSeed = 12345
}: ProgressChartProps) {
  // Initialize state directly from server-provided data
  const initialChartData = useMemo(() => processInitialData(initialComputeData), [initialComputeData]);
  const resolvedInitialParameters = useMemo(() => initialParameters ?? { ...DEFAULT_PARAMETERS }, [initialParameters]);
  
  // Convert server sample trajectories to client format
  const convertedInitialSamples = useMemo(() => {
    return initialSampleTrajectories.map(sample => ({
      trajectory: sample.trajectory.map(point => ({
        year: point.year,
        horizonLength: point.horizonLength,
        effectiveCompute: point.effectiveCompute,
        automationFraction: point.automationFraction,
        trainingCompute: point.trainingCompute ?? null,
        aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
      })) as ChartDataPoint[],
      params: sample.params
    }));
  }, [initialSampleTrajectories]);
  
  const [chartData, setChartData] = useState<ChartDataPoint[]>(initialChartData);
  const [animatedChartData, setAnimatedChartData] = useState<ChartDataPoint[]>(initialChartData);
  const [scHorizonMinutes, setScHorizonMinutes] = useState<number>(
    Math.pow(10, DEFAULT_PARAMETERS.ac_time_horizon_minutes)
  );
  const [activePopover, setActivePopover] = useState<ChartType | null>(null);
  const [parameters, setParameters] = useState<ParametersType>(resolvedInitialParameters);
  // Separate UI parameters for immediate slider updates vs committed parameters for API calls
  const [uiParameters, setUiParameters] = useState<ParametersType>(resolvedInitialParameters);
  const [isDragging, setIsDragging] = useState(false);
  const [allParameters, setAllParameters] = useState<ParameterConfig | null>(null);
  const [trajectoryData, setTrajectoryData] = useState<ChartDataPoint[]>([]);
  const [milestones, setMilestones] = useState<MilestoneMap | null>(
    initialComputeData?.milestones && typeof initialComputeData.milestones === 'object'
      ? (initialComputeData.milestones as MilestoneMap)
      : null
  );
  const [mainLoading, setMainLoading] = useState(false);
  const [isParameterConfigReady, setIsParameterConfigReady] = useState(false);
  const [isAdvancedPanelOpen, setIsAdvancedPanelOpen] = useState(false);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('');
  const [horizonParams, setHorizonParams] = useState<{ uses_shifted_form: boolean; anchor_progress: number | null } | null>(
    initialComputeData?.horizon_params ?? null
  );
  const [summary, setSummary] = useState<{ beta_software?: number; r_software?: number;[key: string]: unknown } | null>(
    initialComputeData?.summary as { beta_software?: number; r_software?: number;[key: string]: unknown } | null ?? null
  );
  const [hiddenCharts, setHiddenCharts] = useState<Set<string>>(new Set());
  
  // Sample trajectories state
  const [samplingConfig, setSamplingConfig] = useState<SamplingConfig | null>(null);
  const [sampleTrajectories, setSampleTrajectories] = useState<SampleTrajectoryWithParams[]>(convertedInitialSamples);
  const [isSamplingEnabled, setIsSamplingEnabled] = useState(true);
  const NUM_SAMPLES = 10; // Number of sample trajectories to show
  
  // Debug state for sample trajectories
  const [showSamplingDebug, setShowSamplingDebug] = useState(false);
  const [expandedTrajectoryIndex, setExpandedTrajectoryIndex] = useState<number | null>(null);
  const [showParamToggles, setShowParamToggles] = useState(false);
  const [enabledSamplingParams, setEnabledSamplingParams] = useState<Set<string>>(new Set());
  const [resampleTrigger, setResampleTrigger] = useState(0);
  const [samplingDebugLog, setSamplingDebugLog] = useState<Array<{
    id: number;
    timestamp: Date;
    type: 'request' | 'response' | 'error';
    params?: Record<string, unknown>;
    dataPoints?: number;
    error?: string;
  }>>([]);

  // Profiling state
  const [profilingEnabled, setProfilingEnabled] = useState(false);
  const [profilingData, setProfilingData] = useState<{
    totalTimeSeconds: number;
    timingBreakdown: Record<string, number>;
    stats: string;
    timestamp: Date;
  } | null>(null);

  // Initialize enabled params when sampling config loads
  useEffect(() => {
    if (samplingConfig) {
      const allParams = new Set<string>();
      for (const paramName of Object.keys(samplingConfig.parameters)) {
        allParams.add(paramName);
      }
      if (samplingConfig.time_series_parameters) {
        for (const paramName of Object.keys(samplingConfig.time_series_parameters)) {
          allParams.add(paramName);
        }
      }
      setEnabledSamplingParams(allParams);
    }
  }, [samplingConfig]);

  // Extract bounds from sampling config for slider ranges
  // These ensure sliders stay within the valid sampling ranges
  const samplingConfigBounds = useMemo(() => {
    if (!samplingConfig) return {};
    return extractSamplingConfigBounds(samplingConfig);
  }, [samplingConfig]);
  
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [isUrlSyncReady, setIsUrlSyncReady] = useState(false);

  const [enableCodingAutomation, setEnableCodingAutomation] = useState(DEFAULT_CHECKBOX_STATES.enableCodingAutomation);
  const [enableExperimentAutomation, setEnableExperimentAutomation] = useState(DEFAULT_CHECKBOX_STATES.enableExperimentAutomation);
  const [useExperimentThroughputCES, setUseExperimentThroughputCES] = useState(DEFAULT_CHECKBOX_STATES.useExperimentThroughputCES);
  const [enableSoftwareProgress, setEnableSoftwareProgress] = useState(DEFAULT_CHECKBOX_STATES.enableSoftwareProgress);
  const [useComputeLaborGrowthSlowdown, setUseComputeLaborGrowthSlowdown] = useState(DEFAULT_CHECKBOX_STATES.useComputeLaborGrowthSlowdown);
  const [useVariableHorizonDifficulty, setUseVariableHorizonDifficulty] = useState(DEFAULT_CHECKBOX_STATES.useVariableHorizonDifficulty);

  // Flag to prevent checkbox effects during initial URL hydration
  const isHydratingFromUrlRef = useRef(false);

  const lockedParameters = useMemo(() => {
    const locked = new Set<string>();
    // Lock when unchecked (disabled state)
    if (!enableCodingAutomation) {
      locked.add('optimal_ces_eta_init');
      locked.add('coding_automation_efficiency_slope');
      locked.add('swe_multiplier_at_present_day');
    }
    if (!enableExperimentAutomation) {
      locked.add('median_to_top_taste_multiplier');
    }
    // Lock when unchecked (uses Cobb-Douglas with high asymptotes)
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
    // Lock when unchecked (sets to constant=1)
    if (!useVariableHorizonDifficulty) {
      locked.add('doubling_difficulty_growth_factor');
    }
    return locked;
  }, [enableCodingAutomation, enableExperimentAutomation, useExperimentThroughputCES, enableSoftwareProgress, useVariableHorizonDifficulty]);

  const scHorizonLogBounds = useMemo(() => {
    // Priority: sampling config bounds > API bounds > fallback
    // Note: samplingConfigBounds already has ac_time_horizon_minutes converted to log10
    const samplingBounds = samplingConfigBounds?.ac_time_horizon_minutes;
    if (samplingBounds?.min !== undefined && samplingBounds?.max !== undefined) {
      return {
        min: samplingBounds.min,
        max: samplingBounds.max
      };
    }
    const rawBounds = allParameters?.bounds?.ac_time_horizon_minutes;
    if (rawBounds && rawBounds[0] > 0 && rawBounds[1] > 0) {
      return {
        min: Math.log10(rawBounds[0]),
        max: Math.log10(rawBounds[1])
      };
    }
    return { min: 3, max: 11 };
  }, [allParameters, samplingConfigBounds]);

  const preGapHorizonBounds = useMemo(() => {
    // Priority: sampling config bounds > API bounds > fallback
    const samplingBounds = samplingConfigBounds?.pre_gap_ac_time_horizon;
    if (samplingBounds?.min !== undefined && samplingBounds?.max !== undefined) {
      return {
        min: samplingBounds.min,
        max: samplingBounds.max
      };
    }
    const rawBounds = allParameters?.bounds?.pre_gap_ac_time_horizon;
    if (rawBounds && rawBounds[0] > 0 && rawBounds[1] > 0) {
      return {
        min: rawBounds[0],
        max: rawBounds[1]
      };
    }
    return { min: 1000, max: 100000000000 };
  }, [allParameters, samplingConfigBounds]);

  const parallelPenaltyBounds = useMemo(() => {
    // Priority: sampling config bounds > API bounds > fallback
    const samplingBounds = samplingConfigBounds?.parallel_penalty;
    if (samplingBounds?.min !== undefined && samplingBounds?.max !== undefined) {
      return {
        min: samplingBounds.min,
        max: samplingBounds.max
      };
    }
    const rawBounds = allParameters?.bounds?.parallel_penalty;
    if (rawBounds) {
      return {
        min: rawBounds[0],
        max: rawBounds[1]
      };
    }
    return { min: 0, max: 1 };
  }, [allParameters, samplingConfigBounds]);

  // Ref for debouncing Monte Carlo updates
  const monteCarloTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  // Ref for request deduplication
  const currentMonteCarloRequestRef = useRef<AbortController | null>(null);
  // Ref to cancel in-flight chart computations when parameters change quickly
  const chartRequestControllerRef = useRef<AbortController | null>(null);
  const latestChartRequestIdRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  const animationStartRef = useRef<number>(0);
  const animationFromRef = useRef<ChartDataPoint[]>([]);
  const animationToRef = useRef<ChartDataPoint[]>([]);
  const animatedDataRef = useRef<ChartDataPoint[]>([]);
  const importInputRef = useRef<HTMLInputElement | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);

  const commitAnimatedChartData = useCallback((nextData: ChartDataPoint[]) => {
    animatedDataRef.current = nextData;
    setAnimatedChartData(nextData);
  }, []);

  const cancelChartAnimation = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  const startChartAnimation = useCallback(
    (nextData: ChartDataPoint[], options: { immediate?: boolean } = {}) => {
      const shouldSkipAnimation =
        options.immediate ?? animatedDataRef.current.length === 0;

      if (shouldSkipAnimation) {
        cancelChartAnimation();
        animationFromRef.current = nextData;
        animationToRef.current = nextData;
        animationStartRef.current = 0;
        commitAnimatedChartData(nextData);
        return;
      }

      animationFromRef.current = animatedDataRef.current;
      animationToRef.current = nextData;
      animationStartRef.current = performance.now();
      cancelChartAnimation();

      const step = (timestamp: number) => {
        const elapsed = timestamp - animationStartRef.current;
        const progress = Math.min(1, elapsed / CHART_ANIMATION_DURATION_MS);
        const interpolated = interpolateChartDataPoints(
          animationFromRef.current,
          animationToRef.current,
          progress
        );
        commitAnimatedChartData(interpolated);

        if (progress < 1) {
          animationFrameRef.current = requestAnimationFrame(step);
        } else {
          animationFrameRef.current = null;
        }
      };

      animationFrameRef.current = requestAnimationFrame(step);
    },
    [cancelChartAnimation, commitAnimatedChartData]
  );

  useEffect(() => {
    return () => {
      cancelChartAnimation();
    };
  }, [cancelChartAnimation]);

  // Keyboard shortcut to toggle sampling debug panel (Alt+D)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Use Alt+D to avoid browser shortcut conflicts
      if (e.altKey && (e.key === 'D' || e.key === 'd')) {
        e.preventDefault();
        setShowSamplingDebug(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Memoized popover handlers to prevent chart re-renders
  const handlePopoverOpen = useCallback((type: ChartType) => {
    setActivePopover(type);
  }, []);

  const handlePopoverClose = useCallback(() => {
    setActivePopover(null);
  }, []);

  const handleExportParameters = useCallback(() => {
    const parameterData = JSON.stringify(uiParameters, null, 2);
    const blob = new Blob([parameterData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'parameter-settings.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [uiParameters]);

  const handleImportParameters = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const fileContents = await file.text();
      const parsed = JSON.parse(fileContents);
      if (!parsed || typeof parsed !== 'object') {
        throw new Error('Invalid parameter file');
      }

      const parsedParameters = parsed as ParametersType;
      setUiParameters(parsedParameters);
      setParameters(parsedParameters);
      setIsDragging(false);
    } catch (error) {
      console.error('Failed to import parameters:', error);
    } finally {
      event.target.value = '';
    }
  }, [setUiParameters, setParameters, setIsDragging]);

  const handleImportButtonClick = useCallback(() => {
    importInputRef.current?.click();
  }, []);

  const handlePresetSelect = useCallback((event: ChangeEvent<HTMLSelectElement>) => {
    setSelectedPresetId(event.target.value);
  }, []);

  // Commit UI parameters to actual parameters when dragging ends
  const commitParameters = useCallback((nextParameters?: ParametersType) => {
    setParameters(nextParameters ?? uiParameters);
    setIsDragging(false);
  }, [uiParameters]);

  // Apply a sample trajectory's parameters to the main charts
  const applySampleTrajectory = useCallback((sampleWithParams: SampleTrajectoryWithParams) => {
    const { trajectory, params } = sampleWithParams;
    
    // Convert sampled parameters to UI format
    // Key conversion: ac_time_horizon_minutes needs to go from actual minutes to log10
    const uiFormatParams = { ...DEFAULT_PARAMETERS };
    
    for (const [key, value] of Object.entries(params)) {
      if (key === 'ac_time_horizon_minutes' && typeof value === 'number') {
        // Convert from actual minutes to log10 for UI slider
        uiFormatParams[key] = Math.log10(value);
      } else if (key === 'include_gap') {
        // Convert include_gap back to benchmarks_and_gaps_mode
        uiFormatParams.benchmarks_and_gaps_mode = value === 'gap';
      } else if (key === 'parallel_penalty' && typeof value === 'number') {
        // Convert parallel_penalty back to coding_labor_exponent
        uiFormatParams.coding_labor_exponent = value;
      } else if (key === 'pre_gap_ac_time_horizon' && typeof value === 'number') {
        // Convert pre_gap_ac_time_horizon back to saturation_horizon_minutes
        uiFormatParams.saturation_horizon_minutes = value;
      } else if (key in uiFormatParams) {
        (uiFormatParams as Record<string, ParameterPrimitive>)[key] = value as ParameterPrimitive;
      }
    }
    
    // Set the parameters to update sliders
    setUiParameters(uiFormatParams);
    setParameters(uiFormatParams);
    setIsDragging(false);
    
    // Set the chart data to this trajectory
    setChartData(trajectory);
    startChartAnimation(trajectory);
  }, [startChartAnimation]);

  // Sync UI parameters with committed parameters when they change (unless we're dragging)
  useEffect(() => {
    if (!isDragging) {
      setUiParameters(parameters);
    }
  }, [parameters, isDragging]);

  useEffect(() => {
    if (!searchParams || isUrlSyncReady) {
      return;
    }

    const urlParams = new URLSearchParams(searchParams.toString());
    const state = decodeFullStateFromParams(urlParams);

    isHydratingFromUrlRef.current = true;
    setParameters(state.parameters);
    setUiParameters(state.parameters);
    setEnableCodingAutomation(state.enableCodingAutomation);
    setEnableExperimentAutomation(state.enableExperimentAutomation);
    setUseExperimentThroughputCES(state.useExperimentThroughputCES);
    setEnableSoftwareProgress(state.enableSoftwareProgress);
    setUseComputeLaborGrowthSlowdown(state.useComputeLaborGrowthSlowdown);
    setUseVariableHorizonDifficulty(state.useVariableHorizonDifficulty);

    // Allow checkbox effects to run after hydration
    setTimeout(() => {
      isHydratingFromUrlRef.current = false;
    }, 0);

    setIsUrlSyncReady(true);
  }, [searchParams, isUrlSyncReady]);

  useEffect(() => {
    if (!isUrlSyncReady) {
      return;
    }

    const nextParams = encodeFullStateToParams({
      parameters,
      enableCodingAutomation,
      enableExperimentAutomation,
      useExperimentThroughputCES,
      enableSoftwareProgress,
      useComputeLaborGrowthSlowdown,
      useVariableHorizonDifficulty,
    });

    const nextSearch = nextParams.toString();
    const currentSearch = searchParams?.toString() ?? '';

    // Only update if changed
    if (nextSearch !== currentSearch) {
      router.replace(nextSearch ? `${pathname}?${nextSearch}` : pathname, { scroll: false });
    }
  }, [parameters, enableCodingAutomation, enableExperimentAutomation, useExperimentThroughputCES, enableSoftwareProgress, useComputeLaborGrowthSlowdown, useVariableHorizonDifficulty, isUrlSyncReady, router, pathname, searchParams]);

  // Load full parameter configuration on mount
  useEffect(() => {
    let isCancelled = false;

    const loadParameterConfig = async () => {
      try {
        const response = await fetch('/api/parameter-config');
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!isCancelled) {
          // Unwrap the config from the {success, config} wrapper
          setAllParameters(data.config || data);
        }
      } catch (error) {
        if (!isCancelled) {
          console.error('Failed to load parameter configuration:', error);
        }
      } finally {
        if (!isCancelled) {
          setIsParameterConfigReady(true);
        }
      }
    };

    loadParameterConfig();

    return () => {
      isCancelled = true;
    };
  }, []);

  // Load sampling configuration on mount
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
          // Initialize correlation sampling with the loaded config
          initializeCorrelationSampling(data.config.correlation_matrix);
          setSamplingConfig(data.config);
        }
      } catch (error) {
        if (!isCancelled) {
          console.error('Failed to load sampling configuration:', error);
        }
      }
    };

    loadSamplingConfig();

    return () => {
      isCancelled = true;
    };
  }, []);

  // API now returns values already in log form (OOMs), so no conversion needed
  const toEffectiveComputeOOM = useCallback((value?: number | null): number | null => {
    if (value === null || value === undefined || !Number.isFinite(value)) {
      return null;
    }
    // Values are already in OOM form from the API, just pass through with clamping
    return value
  }, []);

  const normalizeTrajectoryPoint = useCallback((point: ChartDataPoint): ChartDataPoint => {
    const normalized: ChartDataPoint = {
      ...point,
      // Values already in OOM form from API
      effectiveCompute: toEffectiveComputeOOM(point.effectiveCompute),
    };

    Object.keys(point).forEach(key => {
      if (key.startsWith('effective_compute_trajectory_')) {
        const original = point[key] as number | undefined | null;
        // Values already in OOM form from API
        normalized[key] = toEffectiveComputeOOM(original);
      }
    });

    return normalized;
  }, [toEffectiveComputeOOM]);

  const isVisibleYear = useCallback((year: number | null | undefined): boolean => {
    if (year === null || year === undefined) {
      return false;
    }
    return year >= VISIBLE_CHART_START_YEAR;
  }, []);

  const fetchChartData = useCallback(
    async (params: ParametersType, signal: AbortSignal, requestId: number) => {
      try {
        const apiParameters = convertParametersToAPIFormat(params as unknown as ParameterRecord);
        const response = await fetch('/api/compute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            parameters: apiParameters,
            time_range: [2012, SIMULATION_END_YEAR],
            initial_progress: 0.0,
            enable_profiling: profilingEnabled
          }),
          signal,
          priority: 'high'
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        if (signal.aborted) {
          return false;
        }

        if (!data.success) {
          throw new Error(data.error || 'Failed to compute model');
        }

        if (requestId !== latestChartRequestIdRef.current) {
          return false;
        }

        // Store horizon params if available
        if (data.horizon_params) {
          setHorizonParams(data.horizon_params);
        }

        // Store summary data including beta_software
        if (data.summary) {
          setSummary(data.summary);
        }

        // Store profiling data if available
        if (data.profiling?.enabled) {
          setProfilingData({
            totalTimeSeconds: data.profiling.total_time_seconds,
            timingBreakdown: data.profiling.timing_breakdown || {},
            stats: data.profiling.stats,
            timestamp: new Date()
          });
        }

        setMilestones(
          data.milestones && typeof data.milestones === 'object'
            ? (data.milestones as MilestoneMap)
            : null
        );

        if (data.time_series && data.time_series.length > 0) {
          const formattedData: ChartDataPoint[] = data.time_series
            .filter((point: TimeSeriesPoint) => isVisibleYear(point.year))
            .sort((a: TimeSeriesPoint, b: TimeSeriesPoint) => a.year - b.year)
            .map((point: TimeSeriesPoint) => ({
              year: point.year,
              horizonLength: point.horizonLength,
              horizonFormatted: formatWorkTimeDuration(point.horizonLength),
              effectiveCompute: toEffectiveComputeOOM(point.effectiveCompute),
              automationFraction: point.automationFraction,
              trainingCompute: point.trainingCompute ? toEffectiveComputeOOM(point.trainingCompute) : null,
              experimentCapacity: point.experimentCapacity,
              aiResearchTaste: point.aiResearchTaste,
              aiSoftwareProgressMultiplier: point.aiSoftwareProgressMultiplier,
              aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
              serialCodingLaborMultiplier: point.serialCodingLaborMultiplier,
              humanLabor: point.humanLabor ?? null,
              inferenceCompute: point.inferenceCompute ?? null,
              experimentCompute: point.experimentCompute ?? null,
              researchEffort: point.researchEffort ?? null,
              researchStock: point.researchStock ?? null,
              softwareProgressRate: point.softwareProgressRate ?? null,
              softwareEfficiency: point.softwareEfficiency ?? null,
              aiCodingLaborMultiplier: point.aiCodingLaborMultiplier ?? null,
              aiCodingLaborMultRefPresentDay: point.aiCodingLaborMultRefPresentDay ?? null,
            }));

          setChartData(formattedData);
          startChartAnimation(formattedData);
        }

        return true;
      } catch (error) {
        setMilestones(null);
        if (signal.aborted) {
          return false;
        }
        console.error('Error updating chart:', error);
        return false;
      }
    },
    [isVisibleYear, toEffectiveComputeOOM, startChartAnimation, profilingEnabled]
  );

  // Fetch a single sample trajectory (doesn't set main chart state)
  const fetchSampleTrajectory = useCallback(
    async (sampleParams: Record<string, number | string | boolean>, signal: AbortSignal): Promise<ChartDataPoint[] | null> => {
      try {
        // Use convertSampledParametersToAPIFormat because sampled params already have
        // ac_time_horizon_minutes in actual minutes (not log scale like UI params)
        const apiParameters = convertSampledParametersToAPIFormat(sampleParams as unknown as ParameterRecord);
        const response = await fetch('/api/compute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            parameters: apiParameters,
            time_range: [2012, SIMULATION_END_YEAR],
            initial_progress: 0.0
          }),
          signal,
          priority: 'low'
        });

        if (!response.ok || signal.aborted) {
          return null;
        }

        const data = await response.json();
        if (!data.success || !data.time_series) {
          return null;
        }

        const formattedData: ChartDataPoint[] = data.time_series
          .filter((point: TimeSeriesPoint) => isVisibleYear(point.year))
          .sort((a: TimeSeriesPoint, b: TimeSeriesPoint) => a.year - b.year)
          .map((point: TimeSeriesPoint) => ({
            year: point.year,
            horizonLength: point.horizonLength,
            effectiveCompute: toEffectiveComputeOOM(point.effectiveCompute),
            trainingCompute: point.trainingCompute ? toEffectiveComputeOOM(point.trainingCompute) : null,
            aiSwProgressMultRefPresentDay: point.aiSwProgressMultRefPresentDay,
          }));

        return formattedData;
      } catch {
        return null;
      }
    },
    [isVisibleYear, toEffectiveComputeOOM]
  );

  // Fetch sample trajectories when sampling config is loaded
  const sampleTrajectoriesAbortRef = useRef<AbortController | null>(null);
  const sampleIdCounterRef = useRef(0);
  
  // Track if we've already used server-provided samples
  const hasUsedInitialSamplesRef = useRef(convertedInitialSamples.length > 0);

  useEffect(() => {
    if (!samplingConfig || !isSamplingEnabled) {
      return;
    }

    // Skip fetching if we already have server-provided samples (first render only)
    if (hasUsedInitialSamplesRef.current && sampleTrajectories.length > 0) {
      hasUsedInitialSamplesRef.current = false; // Allow future fetches
      return;
    }

    // Cancel any existing sample requests
    if (sampleTrajectoriesAbortRef.current) {
      sampleTrajectoriesAbortRef.current.abort();
    }

    const abortController = new AbortController();
    sampleTrajectoriesAbortRef.current = abortController;

    const fetchAllSamples = async () => {
      // Use current parameters as the base - distributions will be shifted to center on these values
      // Convert UI parameter format to sampling config format:
      // - ac_time_horizon_minutes: UI stores as log10(minutes), sampling expects actual minutes
      // - saturation_horizon_minutes → pre_gap_ac_time_horizon
      // - benchmarks_and_gaps_mode → include_gap  
      // - coding_labor_exponent → parallel_penalty
      const rawParams = parameters as unknown as Record<string, number | string | boolean>;
      const userParams: Record<string, number | string | boolean> = { ...rawParams };
      
      // Convert ac_time_horizon_minutes from log10 to actual minutes
      if (typeof rawParams.ac_time_horizon_minutes === 'number') {
        userParams.ac_time_horizon_minutes = Math.pow(10, rawParams.ac_time_horizon_minutes);
      }
      
      // Map UI parameter names to sampling config names
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

      // Clear debug log when starting fresh
      setSamplingDebugLog([]);

      // Log sampling info for debugging
      if (enabledSamplingParams.size <= 3) {
        console.log('=== SAMPLING DEBUG ===');
        console.log('Enabled params (uncertainty will be sampled):', Array.from(enabledSamplingParams));
        console.log('Distributions are shifted to center on user values:');
        for (const [paramName, distConfig] of Object.entries(samplingConfig.parameters)) {
          if (enabledSamplingParams.has(paramName) && distConfig.dist !== 'fixed') {
            const defaultMedian = getDistributionMedian(distConfig);
            const userValue = userParams[paramName];
            console.log(`  ${paramName}: user=${typeof userValue === 'number' ? userValue.toExponential(4) : userValue}, default=${typeof defaultMedian === 'number' ? defaultMedian.toExponential(4) : defaultMedian} (dist: ${distConfig.dist})`);
          }
        }
        console.log('=== END SAMPLING DEBUG ===');
      }

      // Generate samples and fetch in parallel, but add them as they complete
      // Use generateParameterSampleWithUserValues to shift distributions to user's current values
      // This preserves uncertainty but centers it around the user's choices
      const currentEnabledParams = enabledSamplingParams;
      const samplePromises = Array.from({ length: NUM_SAMPLES }, async (_, index) => {
        const sampleId = ++sampleIdCounterRef.current;
        const sampleParams = generateParameterSampleWithUserValues(samplingConfig, userParams, currentEnabledParams);
        
        // Log the request with full sampled parameters
        setSamplingDebugLog(prev => [...prev, {
          id: sampleId,
          timestamp: new Date(),
          type: 'request',
          params: sampleParams as Record<string, unknown>,
        }]);
        
        try {
          const trajectory = await fetchSampleTrajectory(sampleParams, abortController.signal);
          
          if (abortController.signal.aborted) {
            return null;
          }
          
          if (trajectory) {
            // Log successful response
            setSamplingDebugLog(prev => [...prev, {
              id: sampleId,
              timestamp: new Date(),
              type: 'response',
              dataPoints: trajectory.length,
            }]);
            
            // Add trajectory with full params as it completes
            setSampleTrajectories(prev => [...prev, { 
              trajectory, 
              params: sampleParams as Record<string, number | string | boolean>
            }]);
          } else {
            // Log null response
            setSamplingDebugLog(prev => [...prev, {
              id: sampleId,
              timestamp: new Date(),
              type: 'error',
              error: 'Null trajectory returned',
            }]);
          }
          return trajectory;
        } catch (err) {
          // Log error
          setSamplingDebugLog(prev => [...prev, {
            id: sampleId,
            timestamp: new Date(),
            type: 'error',
            error: String(err),
          }]);
          return null;
        }
      });

      // Wait for all to complete (for cleanup purposes)
      await Promise.allSettled(samplePromises);
    };

    // Clear existing samples and start fresh
    setSampleTrajectories([]);
    fetchAllSamples();

    return () => {
      abortController.abort();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [samplingConfig, isSamplingEnabled, fetchSampleTrajectory, resampleTrigger, enabledSamplingParams, parameters]);

  // Update the SC horizon line immediately while the slider moves
  useEffect(() => {
    const instantHorizon = Math.pow(10, uiParameters.ac_time_horizon_minutes);
    setScHorizonMinutes(instantHorizon);
  }, [uiParameters.ac_time_horizon_minutes]);

  // Load dynamic Monte Carlo trajectories with streaming based on current slider values
  const loadDynamicMonteCarloTrajectories = useCallback(async (allParams: typeof parameters) => {
    // Cancel any existing request
    if (currentMonteCarloRequestRef.current) {
      currentMonteCarloRequestRef.current.abort();
      currentMonteCarloRequestRef.current = null;
    }

    // Create new AbortController for this request
    const abortController = new AbortController();
    currentMonteCarloRequestRef.current = abortController;

    const trajectoryMap: { [year: number]: ChartDataPoint } = {};

    try {
      const response = await fetch('/api/monte-carlo-dynamic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sliderParams: allParams,
          timeRange: [2012, SIMULATION_END_YEAR]
        }),
        signal: abortController.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('Response body is not readable');
      }

      let buffer = '';
      while (true) {
        const { done, value } = await reader!.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        // Keep the last potentially incomplete line in the buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const message = JSON.parse(line);

            if (message.type === 'trajectory' && message.data && Array.isArray(message.data)) {
              // Dynamic endpoint streams individual trajectory arrays - merge by year
              message.data.forEach((point: ChartDataPoint) => {
                if (!isVisibleYear(point.year)) {
                  return;
                }

                const normalizedPoint = normalizeTrajectoryPoint(point);
                const year = normalizedPoint.year;
                if (!trajectoryMap[year]) {
                  trajectoryMap[year] = normalizedPoint;
                }

                Object.keys(normalizedPoint).forEach(key => {
                  if (key.startsWith('trajectory_') || key.startsWith('effective_compute_trajectory_')) {
                    trajectoryMap[year][key] = normalizedPoint[key];
                  }
                });
              });

              // Update trajectories progressively as each one completes
              const mergedTrajectories = Object.values(trajectoryMap)
                .filter((point) => isVisibleYear(point.year))
                .sort((a, b) => a.year - b.year);
              setTrajectoryData(mergedTrajectories);
            } else if (message.type === 'complete') {
              // Final update when all trajectories are complete
              const mergedTrajectories = Object.values(trajectoryMap)
                .filter((point) => isVisibleYear(point.year))
                .sort((a, b) => a.year - b.year);
              setTrajectoryData(mergedTrajectories);
            } else if (message.type === 'trajectory_error') {
              console.warn(`Failed to generate trajectory ${message.index}:`, message.error);
            } else if (message.type === 'error') {
              throw new Error(message.error);
            }
          } catch (parseError) {
            console.warn('Failed to parse streaming line:', line, parseError);
          }
        }
      }

    } catch (error: unknown) {
      // Ignore aborted requests - they were cancelled intentionally
      if (error instanceof Error && error.name === 'AbortError') {
        return;
      }
      console.error('Failed to load dynamic Monte Carlo trajectories:', error);
      setTrajectoryData([]);
    } finally {
      // Clear the request reference when done (whether success or error)
      if (currentMonteCarloRequestRef.current === abortController) {
        currentMonteCarloRequestRef.current = null;
      }
    }
  }, [normalizeTrajectoryPoint, isVisibleYear]);

  const scheduleMonteCarloUpdate = useCallback((allParams: typeof parameters, skipDelay: boolean = false) => {
    if (monteCarloTimeoutRef.current) {
      clearTimeout(monteCarloTimeoutRef.current);
    }

    setTrajectoryData([]);
    if (skipDelay) {
      loadDynamicMonteCarloTrajectories(allParams);
      return;
    }
    monteCarloTimeoutRef.current = setTimeout(() => {
      monteCarloTimeoutRef.current = null;
      loadDynamicMonteCarloTrajectories(allParams);
    }, 500);
  }, [loadDynamicMonteCarloTrajectories]);

  const cancelMonteCarloUpdate = useCallback(() => {
    if (monteCarloTimeoutRef.current) {
      clearTimeout(monteCarloTimeoutRef.current);
      monteCarloTimeoutRef.current = null;
    }

    if (currentMonteCarloRequestRef.current) {
      currentMonteCarloRequestRef.current.abort();
      currentMonteCarloRequestRef.current = null;
    }
  }, []);

  // Track if this is the first parameter change (to skip initial fetch if we have server data)
  const isFirstParameterRenderRef = useRef(true);

  useEffect(() => {
    if (!isParameterConfigReady) {
      return;
    }

    // Skip the initial fetch if we already have server-provided data
    if (isFirstParameterRenderRef.current && initialChartData.length > 0) {
      isFirstParameterRenderRef.current = false;
      return;
    }
    isFirstParameterRenderRef.current = false;

    const paramsSnapshot = { ...parameters };
    const controller = new AbortController();

    if (chartRequestControllerRef.current) {
      chartRequestControllerRef.current.abort();
    }
    chartRequestControllerRef.current = controller;

    const requestId = latestChartRequestIdRef.current + 1;
    latestChartRequestIdRef.current = requestId;

    let isActive = true;
    setMainLoading(true);
    // scheduleMonteCarloUpdate(paramsSnapshot, true);

    (async () => {
      const didUpdate = await fetchChartData(paramsSnapshot, controller.signal, requestId);
      const isCurrentRequest = latestChartRequestIdRef.current === requestId;

      if (!isActive || controller.signal.aborted || !isCurrentRequest) {
        return;
      }

      setMainLoading(false);
      if (!didUpdate) {
        cancelMonteCarloUpdate();
      }
    })();

    return () => {
      isActive = false;
      if (chartRequestControllerRef.current === controller) {
        chartRequestControllerRef.current = null;
      }
      controller.abort();
      cancelMonteCarloUpdate();
      if (latestChartRequestIdRef.current === requestId) {
        setMainLoading(false);
      }
    };
  }, [parameters, fetchChartData, scheduleMonteCarloUpdate, cancelMonteCarloUpdate, isParameterConfigReady]);

  useEffect(() => () => {
    cancelMonteCarloUpdate();
    if (chartRequestControllerRef.current) {
      chartRequestControllerRef.current.abort();
      chartRequestControllerRef.current = null;
    }
  }, [cancelMonteCarloUpdate]);

  // Sync checkbox states with parameter values
  // When checked (default): uses DEFAULT_PARAMETERS
  // When unchecked: applies nonDefaultParameterValues and locks parameters
  useCheckboxParameterSync({
    checkboxValue: enableCodingAutomation,
    checkboxKey: 'enableCodingAutomation',
    nonDefaultParameterValues: {
      optimal_ces_eta_init: 0.0000001,
      coding_automation_efficiency_slope: 0.000000000001,
      swe_multiplier_at_present_day: 1.0000001,
    },
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });

  useCheckboxParameterSync({
    checkboxValue: enableExperimentAutomation,
    checkboxKey: 'enableExperimentAutomation',
    nonDefaultParameterValues: {
      median_to_top_taste_multiplier: 1.000000000001,
    },
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });

  useCheckboxParameterSync({
    checkboxValue: useExperimentThroughputCES,
    checkboxKey: 'useExperimentThroughputCES',
    nonDefaultParameterValues: {
      direct_input_exp_cap_ces_params: true,
      inf_labor_asymptote: 1.26697e30,
      inf_compute_asymptote: 5.19636e44,
      inv_compute_anchor_exp_cap: 2.8,
      parallel_penalty: 0.5,
      rho_experiment_capacity: 0,
      alpha_experiment_capacity: 0.699,
      experiment_compute_exponent: 0.640,
    },
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });

  useCheckboxParameterSync({
    checkboxValue: enableSoftwareProgress,
    checkboxKey: 'enableSoftwareProgress',
    nonDefaultParameterValues: {
      software_progress_rate_at_reference_year: 0,
    },
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });

  useCheckboxParameterSync({
    checkboxValue: useVariableHorizonDifficulty,
    checkboxKey: 'useVariableHorizonDifficulty',
    nonDefaultParameterValues: {
      doubling_difficulty_growth_factor: 1,
    },
    setParameters,
    setUiParameters,
    isHydratingFromUrlRef,
  });



  const resolvedChartData = animatedChartData.length > 0 ? animatedChartData : chartData;

  // Merge chartData with trajectoryData for rendering (computed, not stored)
  const renderData = useMemo(() => {
    if (!Array.isArray(trajectoryData) || trajectoryData.length === 0) {
      return resolvedChartData;
    }

    const trajectoryLookup = new Map<string, ChartDataPoint>();
    trajectoryData.forEach(tp => {
      const key = Number(tp.year).toFixed(6);
      trajectoryLookup.set(key, tp);
    });

    const merged = resolvedChartData.map(point => {
      const lookupKey = Number(point.year).toFixed(6);
      const trajectoryPoint = trajectoryLookup.get(lookupKey);
      if (!trajectoryPoint) {
        return point;
      }

      const trajectoryFields: Record<string, string | number | null | undefined> = {};
      Object.keys(trajectoryPoint).forEach(key => {
        const rawValue = trajectoryPoint[key as keyof ChartDataPoint];

        if (key.startsWith('trajectory_')) {
          const numeric = typeof rawValue === 'number' ? rawValue : Number(rawValue);
          if (!Number.isFinite(numeric) || numeric <= 0) {
            trajectoryFields[key] = null;
          } else {
            trajectoryFields[key] = numeric
          }
        } else if (key.startsWith('effective_compute_trajectory_')) {
          const numeric = typeof rawValue === 'number' ? rawValue : Number(rawValue);
          if (!Number.isFinite(numeric)) {
            trajectoryFields[key] = null;
          } else {
            trajectoryFields[key] = numeric;
          }
        }
      });
      return { ...point, ...trajectoryFields };
    });

    return merged;
  }, [resolvedChartData, trajectoryData]);

  // Calculate the year when coding automation horizon is reached
  const codingAutomationHorizonYear = useMemo(() => {
    // Find the first year where horizonLength >= scHorizonMinutes
    const point = resolvedChartData.find(p =>
      p.horizonLength != null &&
      typeof p.horizonLength === 'number' &&
      p.horizonLength >= scHorizonMinutes
    );
    return point?.year ?? null;
  }, [resolvedChartData, scHorizonMinutes]);

  const asiMilestoneYear = useMemo(() => {
    if (!milestones) {
      return null;
    }
    const raw = milestones['ASI']?.time;
    return typeof raw === 'number' && Number.isFinite(raw) ? raw : null;
  }, [milestones]);

  // Dynamic display end year: show at least to 2030, extend to show automation achievement, max 2045
  const displayEndYear = useMemo(() => {
    const MIN_END_YEAR = 2030;
    const MAX_END_YEAR = SIMULATION_END_YEAR; // 2045

    const targetYear = asiMilestoneYear ?? codingAutomationHorizonYear;
    if (targetYear == null) {
      return MAX_END_YEAR; // Show full simulation if no milestones
    }

    return Math.min(MAX_END_YEAR, Math.max(MIN_END_YEAR, targetYear) + 0.5);
  }, [asiMilestoneYear, codingAutomationHorizonYear]);

  const automationReferenceLine = useMemo(() => {
    if (codingAutomationHorizonYear == null) {
      return undefined;
    }

    return {
      x: codingAutomationHorizonYear,
      stroke: CODING_AUTOMATION_MARKER_COLOR,
      strokeWidth: 1,
      strokeOpacity: 0.8,
    };
  }, [codingAutomationHorizonYear]);

  const isInitialMilestoneLoading = mainLoading && resolvedChartData.length === 0;

  // Small chart configurations and paging (snap per page)
  type SmallChartDef = {
    key: string;
    title: string;
    tooltip: (p: DataPoint) => React.ReactNode;
    yScale?: 'linear' | 'log';
    domain?: [number, number];
    yFormatter?: (v: number) => string | React.ReactNode;
    // Explicit tag to control layout rows irrespective of yScale defaulting
    scaleType: 'linear' | 'log';
    logSuffix?: string;
    /** If true, the data values are already in log form (OOMs) */
    isDataInLogForm?: boolean;
  };

  // Tooltip for Horizon Length chart
  const HorizonTooltip = useCallback((point: ChartDataPoint | { x: number;[key: string]: number | null | undefined | string }) => {
    const year = 'year' in point ? point.year : (point.x as number);
    const horizonLength = 'horizonLength' in point ? point.horizonLength : null;

    return (
      <div style={tooltipBoxStyle}>
        <span style={tooltipHeaderStyle}>{formatYearMonth(year as number)}</span>
        {horizonLength != null && typeof horizonLength === 'number' && !isNaN(horizonLength) && (
          <span style={tooltipValueStyle}>{formatWorkTimeDurationDetailed(horizonLength as number)}</span>
        )}
      </div>
    );
  }, []);

  // Tooltip for AI R&D Progress Multiplier chart
  const AIProgressMultiplierTooltip = useCallback((point: ChartDataPoint | { x: number;[key: string]: number | null | undefined | string }) => {
    const year = 'year' in point ? point.year : (point.x as number);
    const value = 'aiSwProgressMultRefPresentDay' in point ? point.aiSwProgressMultRefPresentDay : null;

    // Check for multiple milestone labels (stored as array)
    const milestoneLabelsArray = Array.isArray((point as Record<string, unknown>)['milestoneLabels'])
      ? (point as Record<string, unknown>)['milestoneLabels'] as string[]
      : undefined;

    // Fallback to single milestone label (for backwards compatibility)
    const singleMilestoneLabel = typeof (point as Record<string, unknown>)['milestoneLabel'] === 'string'
      ? (point as Record<string, unknown>)['milestoneLabel'] as string
      : undefined;

    return (
      <div style={tooltipBoxStyle}>
        <span style={tooltipHeaderStyle}>{formatYearMonth(year as number)}</span>
        {value != null && typeof value === 'number' && !isNaN(value) && value > 0 && (
          <span style={tooltipValueStyle}>
          {formatUplift(value as number)} x
        </span>
        )}
        {milestoneLabelsArray && milestoneLabelsArray.length > 0 ? (
          // Multiple milestones - show each with its explanation
          milestoneLabelsArray.map((label, index) => {
            const explanation = MILESTONE_EXPLANATIONS[label];
            const fullName = MILESTONE_FULL_NAMES[label] || label;
            return explanation ? (
              <div key={label} className={index > 0 ? 'mt-2' : ''}>
                <div className="font-semibold">{fullName}</div>
                <span style={{ color: 'var(--vivid-foreground)' }}>{explanation}</span>
              </div>
            ) : null;
          })
        ) : singleMilestoneLabel && MILESTONE_EXPLANATIONS[singleMilestoneLabel] ? (
          // Single milestone - show as before
          <>
            <div className="font-semibold">{MILESTONE_FULL_NAMES[singleMilestoneLabel] || singleMilestoneLabel}</div>
            <span style={{ color: 'var(--vivid-foreground)' }}>{MILESTONE_EXPLANATIONS[singleMilestoneLabel]}</span>
          </>
        ) : null}
      </div>
    );
  }, []);

  type PrimaryChartConfig = {
    id: string;
    title: string;
    render: () => React.ReactNode;
  };

  const primaryCharts: PrimaryChartConfig[] = useMemo(
    () => [
      {
        id: 'primary-ai-software-rd',
        title: 'AI Software R&D Uplift',
        render: () => (
          <ChartLoadingOverlay isLoading={mainLoading} className="flex flex-col">
            <AIRnDProgressMultiplierChart
              chartData={renderData}
              tooltip={AIProgressMultiplierTooltip}
              milestones={milestones}
              displayEndYear={displayEndYear}
              verticalReferenceLine={automationReferenceLine}
            />
          </ChartLoadingOverlay>
        ),
      },
      {
        id: 'primary-coding-horizon',
        title: 'Coding Time Horizon',
        render: () => (
          <ChartLoadingOverlay isLoading={mainLoading} className="flex flex-col">
            <CustomHorizonChart
              chartData={renderData}
              scHorizonMinutes={scHorizonMinutes}
              tooltip={HorizonTooltip}
              formatTimeDuration={formatWorkTimeDuration}
              benchmarkData={benchmarkData}
              className="flex flex-col"
              displayEndYear={displayEndYear}
              height={CHART_LAYOUT.primary.height}
              sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
            />
          </ChartLoadingOverlay>
        ),
      },
      {
        id: 'primary-effective-compute',
        title: 'Effective & Training Compute',
        render: () => (
          <ChartLoadingOverlay isLoading={mainLoading} className="flex flex-col">
            <CombinedComputeChart
              chartData={renderData}
              title={enableSoftwareProgress ? 'Effective & Training Compute' : 'Effective Compute'}
              verticalReferenceLine={automationReferenceLine}
              className="flex flex-col"
              displayEndYear={displayEndYear}
              height={CHART_LAYOUT.primary.height}
              showTrainingSeries={enableSoftwareProgress}
              sampleTrajectories={sampleTrajectories.map(s => s.trajectory)}
            />
          </ChartLoadingOverlay>
        ),
      },
    ],
    [
      AIProgressMultiplierTooltip,
      HorizonTooltip,
      automationReferenceLine,
      benchmarkData,
      displayEndYear,
      enableSoftwareProgress,
      formatWorkTimeDuration,
      mainLoading,
      milestones,
      renderData,
      sampleTrajectories,
      scHorizonMinutes,
    ],
  );

  const visiblePrimaryCharts = useMemo(
    () => primaryCharts.filter((chart) => !hiddenCharts.has(chart.id)),
    [primaryCharts, hiddenCharts],
  );
  const visiblePrimaryCount = visiblePrimaryCharts.length;
  const primaryGridWidth = visiblePrimaryCount > 0
    ? visiblePrimaryCount * CHART_LAYOUT.primary.width
    : 0;


  const smallCharts: SmallChartDef[] = useMemo(() => ([
    {
      key: 'softwareProgressRate',
      title: 'Software Efficiency Growth Rate',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="softwareProgressRate" formatter={formatTo3SigFigs} />,
      scaleType: 'linear'
    },
    {
      key: 'automationFraction',
      title: 'Coding Automation Fraction',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="automationFraction" formatter={(v) => `${(v * 100).toFixed(1)}%`} />,
      yFormatter: (v) => `${(v * 100).toFixed(1)}%`,
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
      key: 'aiResearchTaste',
      title: 'AI Experiment Selection Skill',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="aiResearchTaste" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
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
      key: 'researchStock',
      title: 'Cumulative Research Effort',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="researchStock" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
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
      key: 'experimentCapacity',
      title: 'Experiment Throughput',
      tooltip: (p) => <SmallChartMetricTooltip point={p} fieldName="experimentCapacity" formatter={(v) => formatCompactNumberNode(v, { renderMode: 'html' })} requirePositive />,
      yScale: 'log',
      scaleType: 'log',
      yFormatter: (v) => formatAsPowerOfTenText(v)
    },
    {
      key: 'inferenceCompute',
      title: 'Inference Compute for Coding Automation',
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

  const visibleSmallCharts = useMemo(
    () => smallCharts.filter((chart) => !hiddenCharts.has(`metric-${chart.key}`)),
    [smallCharts, hiddenCharts],
  );

  type ColumnDef = { log1?: SmallChartDef; log2?: SmallChartDef; log3?: SmallChartDef; linear?: SmallChartDef };
  const columns: ColumnDef[] = useMemo(() => {
    const logs = visibleSmallCharts.filter(c => c.scaleType === 'log');
    const linears = visibleSmallCharts.filter(c => c.scaleType !== 'log');
    const numCols = Math.max(Math.ceil(logs.length / 3), linears.length);
    return Array.from({ length: numCols }, (_, i) => ({
      log1: logs[3 * i],
      log2: logs[3 * i + 1],
      log3: logs[3 * i + 2],
      linear: linears[i],
    })).filter(col => col.log1 || col.log2 || col.log3 || col.linear);
  }, [visibleSmallCharts]);
  const hasLogCharts = useMemo(() => columns.some(col => col.log1 || col.log2 || col.log3), [columns]);
  const hasLinearCharts = useMemo(() => columns.some(col => col.linear), [columns]);
  const hasMetricCharts = hasLogCharts || hasLinearCharts;

  // Mobile layout: separate log and linear charts for 2-screen layout
  const mobileLogCharts = useMemo(() => visibleSmallCharts.filter(c => c.scaleType === 'log'), [visibleSmallCharts]);
  const mobileLinearCharts = useMemo(() => visibleSmallCharts.filter(c => c.scaleType !== 'log'), [visibleSmallCharts]);

  const SOFTWARE_PROGRESS_CHART_ID = 'primary-ai-software-rd';

  useEffect(() => {
    setHiddenCharts((prev) => {
      const next = new Set(prev);
      const shouldHide = !enableSoftwareProgress;
      const currentlyHidden = next.has(SOFTWARE_PROGRESS_CHART_ID);
      if (shouldHide && !currentlyHidden) {
        next.add(SOFTWARE_PROGRESS_CHART_ID);
        return next;
      }
      if (!shouldHide && currentlyHidden) {
        next.delete(SOFTWARE_PROGRESS_CHART_ID);
        return next;
      }
      return prev;
    });
  }, [enableSoftwareProgress]);

  return (
    <ParameterHoverProvider>
      {/* Mobile/Tablet Advanced Parameters Drawer */}
      {isAdvancedPanelOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={() => setIsAdvancedPanelOpen(false)} />
          <div className="absolute inset-y-0 right-0 w-full max-w-sm bg-white shadow-xl flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
              <h2 className="text-sm font-semibold text-primary uppercase tracking-[0.2em] font-system-mono">
                Advanced Parameters
              </h2>
              <button
                type="button"
                onClick={() => setIsAdvancedPanelOpen(false)}
                className="p-2 -mr-2 text-gray-500 hover:text-gray-700"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-4 py-4">
              <div className="space-y-6">
                <div className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={handleExportParameters}
                      className="flex-1 min-w-[120px] px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-white bg-[#2A623D] border border-[#215337] rounded-sm shadow-sm hover:bg-[#245632]"
                    >
                      Export
                    </button>
                    <button
                      type="button"
                      onClick={handleImportButtonClick}
                      className="flex-1 min-w-[120px] px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-[#2A623D] bg-white border border-[#2A623D]/40 rounded-sm shadow-sm hover:bg-[#F4F8F5]"
                    >
                      Import
                    </button>
                  </div>
                  <div className="space-y-2">
                    <label className="block text-xs font-semibold uppercase tracking-[0.15em] text-primary font-system-mono">
                      Presets
                    </label>
                    <select
                      value={selectedPresetId}
                      onChange={handlePresetSelect}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md bg-white text-sm"
                    >
                      <option value="">Select preset</option>
                      {PARAMETER_PRESETS.map((preset) => (
                        <option key={preset.id} value={preset.id}>{preset.label}</option>
                      ))}
                    </select>
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
                  summary={summary}
                  lockedParameters={lockedParameters}
                  onToggleTrajectoryDebugger={() => setShowSamplingDebug(prev => !prev)}
                  isTrajectoryDebuggerOpen={showSamplingDebug}
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
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid h-screen w-full grid-cols-[minmax(0,1fr)_auto] grid-rows-[minmax(0,1fr)] gap-0">
        <div className="relative col-start-1 row-start-1 flex min-h-0 flex-col">
          <div
            ref={scrollContainerRef}
            className="flex min-h-0 flex-col overflow-y-auto pb-10"
          >
            <HeaderContent variant="inline" className="pt-6 pb-4 pl-6 lg:pl-8 pr-6" onOpenAdvancedParams={() => setIsAdvancedPanelOpen(true)} />
            {resolvedChartData.length === 0 && !mainLoading && (
              <div className="text-center py-8 text-red-500 text-lg">No chart data available</div>
            )}

            <main className="w-full px-0 lg:px-8">
              {/* Charts Container */}
              <div className="flex flex-col flex-1 min-h-0">
                <ChartSyncProvider>
                  {/* Mobile/Tablet Layout: Horizontal scroll-snap */}
                  <div className="lg:hidden relative w-full">
                    <div 
                      className="flex gap-x-4 overflow-x-auto overflow-y-hidden snap-x snap-mandatory scrollbar-none"
                      style={{ 
                        scrollBehavior: 'smooth',
                        WebkitOverflowScrolling: 'touch',
                      }}
                    >
                      {/* Column 1: Explanation & Key Dates */}
                      <div className="snap-start flex-shrink-0 pl-6 pr-2">
                        <div className="h-full">
                          <KeyMilestonePanel
                            automationYear={codingAutomationHorizonYear}
                            asiYear={asiMilestoneYear}
                            isLoading={isInitialMilestoneLoading}
                          />
                        </div>
                      </div>

                      {/* Columns 2-4: Big Charts (one per column) */}
                      {visiblePrimaryCharts.map((chart) => (
                        <div key={chart.id} className="snap-center flex-shrink-0" style={{ width: 'calc(100vw - 48px)' }}>
                          <div className="w-full">
                            {chart.render()}
                          </div>
                        </div>
                      ))}

                      {/* Column 5: Small Charts Screen 1 - first 4 log + first 2 linear */}
                      {hasMetricCharts && (
                        <div className="snap-center flex-shrink-0" style={{ width: 'calc(100vw - 48px)' }}>
                          <div className="grid grid-cols-2 grid-rows-[auto_1fr_1fr_auto_1fr] gap-x-2 gap-y-1 h-[420px]">
                            {/* Log Scale Section - uses subgrid to inherit parent row sizing */}
                            <section className="col-span-2 row-span-3 grid grid-cols-subgrid grid-rows-subgrid">
                              <div className="col-span-2 flex items-center gap-2">
                                <span className="font-mono small-caps text-[10px] font-semibold text-gray-500">Log Scale</span>
                                <div className="border-t border-gray-300/50 flex-1" />
                              </div>
                              {mobileLogCharts.slice(0, 4).map((chart) => (
                                <ChartLoadingOverlay key={chart.key} isLoading={mainLoading} className="h-full">
                                  <CustomMetricChart
                                    chartData={renderData}
                                    tooltip={chart.tooltip}
                                    title={chart.title}
                                    dataKey={chart.key}
                                    yFormatter={chart.yFormatter || ((v) => formatTo3SigFigs(v))}
                                    yScale={chart.yScale || 'linear'}
                                    domainOverride={chart.domain}
                                    verticalReferenceLine={automationReferenceLine}
                                    className="h-full"
                                    displayEndYear={displayEndYear}
                                    logSuffix={chart.logSuffix}
                                    isDataInLogForm={chart.isDataInLogForm}
                                  />
                                </ChartLoadingOverlay>
                              ))}
                            </section>
                            {/* Linear Scale Section - uses subgrid to inherit parent row sizing */}
                            <section className="col-span-2 row-span-2 grid grid-cols-subgrid grid-rows-subgrid">
                              <div className="col-span-2 flex items-center gap-2">
                                <span className="font-mono small-caps text-[10px] font-semibold text-gray-500">Linear Scale</span>
                                <div className="border-t border-gray-300/50 flex-1" />
                              </div>
                              {mobileLinearCharts.slice(0, 2).map((chart) => (
                                <ChartLoadingOverlay key={chart.key} isLoading={mainLoading} className="h-full">
                                  <CustomMetricChart
                                    chartData={renderData}
                                    tooltip={chart.tooltip}
                                    title={chart.title}
                                    dataKey={chart.key}
                                    yFormatter={chart.yFormatter || ((v) => formatTo3SigFigs(v))}
                                    yScale={chart.yScale || 'linear'}
                                    domainOverride={chart.domain}
                                    verticalReferenceLine={automationReferenceLine}
                                    className="h-full"
                                    displayEndYear={displayEndYear}
                                    logSuffix={chart.logSuffix}
                                    isDataInLogForm={chart.isDataInLogForm}
                                  />
                                </ChartLoadingOverlay>
                              ))}
                            </section>
                          </div>
                        </div>
                      )}

                      {/* Column 6: Small Charts Screen 2 - remaining log charts + remaining linear charts */}
                      {hasMetricCharts && (mobileLogCharts.length > 4 || mobileLinearCharts.length > 2) && (
                        <div className="snap-center flex-shrink-0" style={{ width: 'calc(100vw - 48px)' }}>
                          <div className={`grid grid-cols-2 gap-x-2 gap-y-1 h-[420px] ${
                            mobileLinearCharts.length > 2 
                              ? 'grid-rows-[auto_1fr_1fr_1fr_auto_1fr]' 
                              : 'grid-rows-[auto_1fr_1fr_1fr]'
                          }`}>
                            {/* Log Scale Section */}
                            <section className="col-span-2 row-span-4 grid grid-cols-subgrid grid-rows-subgrid">
                              <div className="col-span-2 flex items-center gap-2">
                                <span className="font-mono small-caps text-[10px] font-semibold text-gray-500">Log Scale</span>
                                <div className="border-t border-gray-300/50 flex-1" />
                              </div>
                              {mobileLogCharts.slice(4).map((chart) => (
                                <ChartLoadingOverlay key={chart.key} isLoading={mainLoading} className="h-full">
                                  <CustomMetricChart
                                    chartData={renderData}
                                    tooltip={chart.tooltip}
                                    title={chart.title}
                                    dataKey={chart.key}
                                    yFormatter={chart.yFormatter || ((v) => formatTo3SigFigs(v))}
                                    yScale={chart.yScale || 'linear'}
                                    domainOverride={chart.domain}
                                    verticalReferenceLine={automationReferenceLine}
                                    className="h-full"
                                    displayEndYear={displayEndYear}
                                    logSuffix={chart.logSuffix}
                                    isDataInLogForm={chart.isDataInLogForm}
                                  />
                                </ChartLoadingOverlay>
                              ))}
                              {/* Add empty cell if odd number of remaining log charts */}
                              {mobileLogCharts.slice(4).length % 2 === 1 && <div className="h-full" />}
                            </section>
                            {/* Linear Scale Section - only shown if there are remaining linear charts */}
                            {mobileLinearCharts.length > 2 && (
                              <section className="col-span-2 row-span-2 grid grid-cols-subgrid grid-rows-subgrid">
                                <div className="col-span-2 flex items-center gap-2">
                                  <span className="font-mono small-caps text-[10px] font-semibold text-gray-500">Linear Scale</span>
                                  <div className="border-t border-gray-300/50 flex-1" />
                                </div>
                                {mobileLinearCharts.slice(2).map((chart) => (
                                  <ChartLoadingOverlay key={chart.key} isLoading={mainLoading} className="h-full">
                                    <CustomMetricChart
                                      chartData={renderData}
                                      tooltip={chart.tooltip}
                                      title={chart.title}
                                      dataKey={chart.key}
                                      yFormatter={chart.yFormatter || ((v) => formatTo3SigFigs(v))}
                                      yScale={chart.yScale || 'linear'}
                                      domainOverride={chart.domain}
                                      verticalReferenceLine={automationReferenceLine}
                                      className="h-full"
                                      displayEndYear={displayEndYear}
                                      logSuffix={chart.logSuffix}
                                      isDataInLogForm={chart.isDataInLogForm}
                                    />
                                  </ChartLoadingOverlay>
                                ))}
                                {/* Add empty cell if odd number of remaining linear charts */}
                                {mobileLinearCharts.slice(2).length % 2 === 1 && <div className="h-full" />}
                              </section>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Column 7: Parameters */}
                      <div className="snap-center flex-shrink-0 px-6" style={{ width: '100vw' }}>
                        <section className="w-full flex flex-col">
                          <div className="flex items-center gap-2 pb-4">
                            <span className="text-[12px] font-semibold text-primary text-left font-system-mono">Parameters</span>
                            <div className="border-t border-gray-500/30 flex-1" />
                            <button
                              onClick={() => {
                                setParameters({ ...DEFAULT_PARAMETERS });
                                setUiParameters({ ...DEFAULT_PARAMETERS });
                              }}
                              className="text-[10px] text-gray-400 hover:text-gray-600 font-system-mono uppercase tracking-wider"
                            >
                              Reset
                            </button>
                            <div className="w-3 border-t border-gray-500/30" />
                            
                            <button
                              onClick={() => setIsAdvancedPanelOpen(true)}
                              className="text-[10px] text-gray-400 hover:text-gray-600 font-system-mono uppercase tracking-wider"
                            >
                              Advanced
                            </button>
                            {/* <div className="w-3 border-t border-gray-500/30" /> */}
                          </div>
                          <div className="flex flex-col gap-2">
                            {/* Present Doubling Time Slider */}
                            <div className="relative flex w-full flex-col">
                              <div className="flex items-center justify-between mb-2">
                                <label className="block w-full text-xs font-medium text-foreground">
                                  <WithChartTooltip
                                    explanation="How long it takes the coding time horizon to double, beginning at present day but extended as a long-term trend."
                                    fullWidth
                                  >
                                    How long does AIs' coding time horizon currently take to double?
                                  </WithChartTooltip>
                                </label>
                              </div>
                              <ParameterSlider
                                paramName="present_doubling_time"
                                label=""
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
                            </div>
                            {/* SC Time Horizon Slider */}
                            <div className="relative flex w-full flex-col">
                              <div className="flex items-center justify-between mb-2">
                                <label className="block w-full text-xs font-medium text-foreground">
                                  <WithChartTooltip
                                    explanation="If an AI can do 80% of tasks that take a skilled human this long, then an AI can do every type of coding work as well as the average company engineer."
                                    fullWidth
                                  >
                                    Automated Coder Time Horizon Requirement
                                  </WithChartTooltip>
                                </label>
                              </div>
                              <ParameterSlider
                                paramName="ac_time_horizon_minutes"
                                label=""
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
                            </div>
                            {/* Doubling Difficulty Growth Factor Slider */}
                            <div className="relative flex w-full flex-col">
                              <div className="flex items-center justify-between mb-2">
                                <label className="block w-full text-xs font-medium text-foreground">
                                  <WithChartTooltip
                                    explanation='For each successive coding time horizon doubling, by what factor does the required orders of magnitude of effective compute change?'
                                    fullWidth
                                  >
                                    How much easier/harder each coding time horizon doubling gets
                                  </WithChartTooltip>
                                </label>
                              </div>
                              <ParameterSlider
                                paramName="doubling_difficulty_growth_factor"
                                label=""
                                customMin={samplingConfigBounds.doubling_difficulty_growth_factor?.min}
                                customMax={samplingConfigBounds.doubling_difficulty_growth_factor?.max}
                                fallbackMin={0.5}
                                fallbackMax={1.5}
                                step={0.01}
                                customFormatValue={(val) => `${val.toFixed(2)}x/doubling`}
                                value={uiParameters.doubling_difficulty_growth_factor}
                                uiParameters={uiParameters}
                                setUiParameters={setUiParameters}
                                allParameters={allParameters}
                                isDragging={isDragging}
                                setIsDragging={setIsDragging}
                                commitParameters={commitParameters}
                                disabled={lockedParameters.has('doubling_difficulty_growth_factor')}
                              />
                            </div>
                            {/* Median to Top Taste Multiplier Slider */}
                            <div className="relative flex w-full flex-col">
                              <div className="flex items-center justify-between mb-2">
                                <label className="block w-full text-xs font-medium text-foreground">
                                  <WithChartTooltip
                                    explanation="Ratio of the top researcher's experiment selection skill to median researcher's experiment selection skill, i.e. how much more value per experiment the top researcher gets holding capacity to implement experiments fixed."
                                    fullWidth
                                  >
                                  Ratio of top to median researchers&apos; value per selected experiment
                                  </WithChartTooltip>
                                </label>
                              </div>
                              <ParameterSlider
                                paramName="median_to_top_taste_multiplier"
                                label=""
                                customMin={samplingConfigBounds.median_to_top_taste_multiplier?.min}
                                customMax={samplingConfigBounds.median_to_top_taste_multiplier?.max}
                                fallbackMin={1.1}
                                fallbackMax={20.0}
                                step={0.1}
                                customFormatValue={(val) => `${val.toFixed(2)}x`}
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
                            {/* AI Research Taste Slope Slider */}
                            <div className="relative flex w-full flex-col">
                              <div className="flex items-start justify-between mb-2">
                                <label className="flex w-full text-xs font-medium text-foreground">
                                  <WithChartTooltip
                                    explanation="Each time effective compute increases by as much as it does in 2025, by how many standard deviations (SDs) does AI experiment selection skill increase (median to best AGI company researcher requires 3 SDs)?"
                                    fullWidth
                                  >
                                    How quickly AIs improve at experiment selection
                                  </WithChartTooltip>
                                </label>
                              </div>
                              <ParameterSlider
                                paramName="ai_research_taste_slope"
                                label=""
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
                            </div>
                          </div>
                        </section>
                      </div>
                    </div>
                  </div>

                  {/* Desktop Layout: Original grid */}
                  <div className="hidden lg:block relative w-full">
                    <div className="overflow-x-auto overflow-y-hidden scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent">
                      {/* Clean 4-row grid layout */}
                      <div
                        className="grid gap-4 min-w-max"
                        style={{
                          gridTemplateRows: 'repeat(3, 125px) 125px',
                          gridTemplateColumns: `${CHART_LAYOUT.keyStats.width}px ${visiblePrimaryCount > 0 ? `${primaryGridWidth}px` : '0px'} ${hasMetricCharts ? `${columns.length * CHART_LAYOUT.metric.columnMinWidth + 100}px` : '0px'}`,
                        }}
                      >
                        {/* Column 1: Summary + KeyMilestone (spans all 4 rows) */}
                        <div className="row-span-4">
                          <KeyMilestonePanel
                            automationYear={codingAutomationHorizonYear}
                            asiYear={asiMilestoneYear}
                            isLoading={isInitialMilestoneLoading}
                          />
                        </div>

                        {/* Column 2, Rows 1-3: Big Graphs */}
                        {visiblePrimaryCount > 0 && (
                          <div
                            className="col-start-2 row-span-3 grid gap-4"
                            style={{
                              gridTemplateColumns: `repeat(${visiblePrimaryCount}, minmax(0, 1fr))`,
                            }}
                          >
                            {visiblePrimaryCharts.map((chart) => (
                              <div key={chart.id} className="flex flex-col h-full">
                                {chart.render()}
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Column 2, Row 4: Parameters */}
                        {visiblePrimaryCount > 0 && (
                          <section className="col-start-2 row-start-4 w-full px-2 flex flex-col justify-between">
                            <div className="flex items-center gap-2 pb-2" style={{ marginBottom: 10 }}>
                              <span className="text-[12px] font-semibold text-primary text-left font-system-mono">Parameters</span>
                              <div className="border-t border-gray-500/30 flex-1" />
                              <button
                                onClick={() => {
                                  setParameters({ ...DEFAULT_PARAMETERS });
                                  setUiParameters({ ...DEFAULT_PARAMETERS });
                                }}
                                className="text-[10px] text-gray-400 hover:text-gray-600 font-system-mono uppercase tracking-wider"
                              >
                                Reset
                              </button>
                              
                              <div className="w-3 border-t border-gray-500/30" />
                              <button
                                onClick={() => setIsAdvancedPanelOpen(true)}
                                className="text-[10px] text-gray-400 hover:text-gray-600 font-system-mono uppercase tracking-wider"
                              >
                                Advanced
                              </button>
                              <div className="w-3 border-t border-gray-500/30" />
                            </div>
                            <div className="grid grid-cols-5 gap-10">
                              {/* Present Doubling Time Slider */}
                              <div className="relative flex w-full flex-col justify-between">
                                <div className="flex items-center justify-between mb-1">
                                  <label className="block w-full text-xs font-medium text-foreground">
                                    <WithChartTooltip
                                      explanation="How long it takes the coding time horizon to double, beginning at present day but extended as a long-term trend."
                                      fullWidth
                                    >
                                      How long it currently takes AIs' coding time horizon to double
                                    </WithChartTooltip>
                                  </label>
                                </div>
                                <ParameterSlider
                                  paramName="present_doubling_time"
                                  label=""
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
                                  onMouseEnter={() => handlePopoverOpen('doubling_time')}
                                  onMouseLeave={handlePopoverClose}
                                  onFocus={() => handlePopoverOpen('doubling_time')}
                                  onBlur={handlePopoverClose}
                                  popoverConfig={CHART_CONFIGS.doubling_time}
                                  popoverVisible={activePopover === 'doubling_time'}
                                  onPopoverClose={handlePopoverClose}
                                  horizonParams={horizonParams}
                                  useLogScale={true}
                                />
                              </div>
                              {/* SC Time Horizon Slider */}
                              <div className="relative flex w-full flex-col justify-between">
                                <div className="flex items-center justify-between mb-1">
                                  <label className="block w-full text-xs font-medium text-foreground">
                                    <WithChartTooltip
                                      explanation="If an AI can do 80% of tasks that take a coder this long, then an AI can replace the coding staff of an AGI project."
                                      fullWidth
                                    >
                                      Coding time horizon required to achieve Automated Coder (AC)
                                    </WithChartTooltip>
                                  </label>
                                </div>
                                <ParameterSlider
                                  paramName="ac_time_horizon_minutes"
                                  label=""
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
                                  onMouseEnter={() => handlePopoverOpen('sc_horizon')}
                                  onMouseLeave={handlePopoverClose}
                                  onFocus={() => handlePopoverOpen('sc_horizon')}
                                  onBlur={handlePopoverClose}
                                  popoverConfig={CHART_CONFIGS.sc_horizon}
                                  popoverVisible={activePopover === 'sc_horizon'}
                                  onPopoverClose={handlePopoverClose}
                                  horizonParams={horizonParams}
                                  useLogScale={true}
                                />
                              </div>
                              {/* Doubling Difficulty Growth Factor Slider */}
                              <div className="relative flex w-full flex-col justify-between">
                                <div className="flex items-center justify-between mb-1">
                                  <label className="block w-full text-xs font-medium text-foreground">
                                    <WithChartTooltip
                                      explanation='For each successive coding time horizon doubling, by what factor does the required orders of magnitude of effective compute change? Lower values mean doublings get easier over time. This parameter is called doubling difficulty growth factor in the writeup + parameter export/import.'
                                      fullWidth
                                    >
                                      How much easier/harder each coding time horizon doubling gets
                                    </WithChartTooltip>
                                  </label>
                                </div>
                                <ParameterSlider
                                  paramName="doubling_difficulty_growth_factor"
                                  label=""
                                  fallbackMin={0.5}
                                  fallbackMax={1.5}
                                  step={0.01}
                                  customFormatValue={(val) => `${val.toFixed(2)}x/doubling`}
                                  value={uiParameters.doubling_difficulty_growth_factor}
                                  uiParameters={uiParameters}
                                  setUiParameters={setUiParameters}
                                  allParameters={allParameters}
                                  isDragging={isDragging}
                                  setIsDragging={setIsDragging}
                                  commitParameters={commitParameters}
                                  onMouseEnter={() => handlePopoverOpen('difficulty_growth_rate')}
                                  onMouseLeave={handlePopoverClose}
                                  onFocus={() => handlePopoverOpen('difficulty_growth_rate')}
                                  onBlur={handlePopoverClose}
                                  popoverConfig={CHART_CONFIGS.difficulty_growth_rate}
                                  popoverVisible={activePopover === 'difficulty_growth_rate'}
                                  onPopoverClose={handlePopoverClose}
                                  horizonParams={horizonParams}
                                  disabled={lockedParameters.has('doubling_difficulty_growth_factor')}
                                />
                              </div>
                              {/* Median to Top Taste Multiplier Slider */}
                              <div className="relative flex w-full flex-col justify-between">
                                <div className="flex items-center justify-between mb-1">
                                  <label className="block w-full text-xs font-medium text-foreground">
                                    <WithChartTooltip
                                      explanation="Ratio of the top researcher's experiment selection skill to median researcher's experiment selection skill, i.e. how much more value per experiment the top researcher gets holding capacity to implement experiments fixed."
                                      fullWidth
                                    >
                                      Ratio of top to median researchers&apos; value per selected experiment
                                    </WithChartTooltip>
                                  </label>
                                </div>
                                <ParameterSlider
                                  paramName="median_to_top_taste_multiplier"
                                  label=""
                                  customMin={samplingConfigBounds.median_to_top_taste_multiplier?.min}
                                  customMax={samplingConfigBounds.median_to_top_taste_multiplier?.max}
                                  fallbackMin={1.1}
                                  fallbackMax={20.0}
                                  step={0.1}
                                  customFormatValue={(val) => `${val.toFixed(2)}x`}
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
                              {/* AI Research Taste Slope Slider */}
                              <div className="relative flex w-full flex-col justify-between">
                                <div className="flex items-start justify-between mb-1">
                                  <label className="flex w-full text-xs font-medium text-foreground">
                                    <WithChartTooltip
                                      explanation="Each time effective compute increases by as much as in 2025, by how many standard deviations (SDs) does AI experiment selection skill increase (median to best AGI company researcher requires 3 SDs)?"
                                      fullWidth
                                    >
                                      How quickly AIs improve at experiment selection
                                    </WithChartTooltip>
                                  </label>
                                </div>
                                <ParameterSlider
                                  paramName="ai_research_taste_slope"
                                  label=""
                                  step={0.1}
                                  customMin={samplingConfigBounds.ai_research_taste_slope?.min}
                                  customMax={samplingConfigBounds.ai_research_taste_slope?.max}
                                  fallbackMin={0.1}
                                  fallbackMax={10.0}
                                  customFormatValue={(val) => `${val.toFixed(1)} SDs/2025-effective-FLOP-growth`}
                                  value={uiParameters.ai_research_taste_slope}
                                  uiParameters={uiParameters}
                                  setUiParameters={setUiParameters}
                                  allParameters={allParameters}
                                  isDragging={isDragging}
                                  setIsDragging={setIsDragging}
                                  commitParameters={commitParameters}
                                  useLogScale={true}
                                />
                              </div>
                            </div>
                          </section>
                        )}

                        {/* Column 3, Rows 1-3: Small Log Graphs */}
                        {hasMetricCharts && (
                          <div className="col-start-3 row-span-3 flex flex-col">
                            <div className="flex items-center gap-2 pb-2">
                              <span className="font-mono small-caps text-[12px] font-semibold">Log Scale</span>
                              <div className="border-t border-gray-500/30 flex-1" />
                            </div>
                            <div className="flex flex-1 flex-row gap-4 px-1 pr-10 min-h-0">
                              {hasLogCharts ? (
                                columns.map((col, columnIndex) => (
                                  <div
                                    key={`log-col-${columnIndex}`}
                                    className="h-full flex-shrink-0"
                                    style={{ width: CHART_LAYOUT.metric.columnMinWidth }}
                                  >
                                    <div className="grid h-full grid-rows-3 gap-4">
                                      {[col.log1, col.log2, col.log3].map((cfg, rowIndex) => (
                                        cfg ? (
                                          <ChartLoadingOverlay key={`${cfg.key}-${columnIndex}-${rowIndex}`} isLoading={mainLoading} className="h-full">
                                            <CustomMetricChart
                                              chartData={renderData}
                                              tooltip={cfg.tooltip}
                                              title={cfg.title}
                                              dataKey={cfg.key}
                                              yFormatter={cfg.yFormatter || ((v) => formatTo3SigFigs(v))}
                                              yScale={cfg.yScale || 'linear'}
                                              domainOverride={cfg.domain}
                                              verticalReferenceLine={automationReferenceLine}
                                              className="h-full"
                                              displayEndYear={displayEndYear}
                                              logSuffix={cfg.logSuffix}
                                              isDataInLogForm={cfg.isDataInLogForm}
                                            />
                                          </ChartLoadingOverlay>
                                        ) : (
                                          <div
                                            key={`log-placeholder-${columnIndex}-${rowIndex}`}
                                            className="h-full rounded-lg border border-dashed border-slate-200/60"
                                          />
                                        )
                                      ))}
                                    </div>
                                  </div>
                                ))
                              ) : (
                                <div className="flex h-full flex-1 items-center justify-center rounded-lg border border-dashed border-slate-200/60 text-xs text-slate-500">
                                  All log-scale charts are hidden.
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Column 3, Row 4: Small Linear Graphs */}
                        {hasMetricCharts && (
                          <div className="col-start-3 row-start-4 flex flex-col">
                            <div className="flex items-center gap-2 pb-2">
                              <span className="font-mono small-caps text-[12px] font-semibold">Linear Scale</span>
                              <div className="border-t border-gray-500/30 flex-1" />
                            </div>
                            <div className="flex flex-1 flex-row gap-4 px-1 pr-10 min-h-0 h-[100px] max-h-[100px]">
                              {hasLinearCharts ? (
                                columns.map((col, columnIndex) => (
                                  <div
                                    key={`lin-col-${columnIndex}`}
                                    className="h-full flex-shrink-0"
                                    style={{ width: CHART_LAYOUT.metric.columnMinWidth }}
                                  >
                                    {col.linear ? (
                                      <ChartLoadingOverlay key={`${col.linear.key}-${columnIndex}-linear`} isLoading={mainLoading} className="h-full">
                                        <CustomMetricChart
                                          chartData={renderData}
                                          tooltip={col.linear.tooltip}
                                          title={col.linear.title}
                                          dataKey={col.linear.key}
                                          yFormatter={col.linear.yFormatter || ((v) => formatTo3SigFigs(v))}
                                          yScale={col.linear.yScale || 'linear'}
                                          domainOverride={col.linear.domain}
                                          verticalReferenceLine={automationReferenceLine}
                                          className="h-full"
                                          displayEndYear={displayEndYear}
                                          logSuffix={col.linear.logSuffix}
                                          isDataInLogForm={col.linear.isDataInLogForm}
                                        />
                                      </ChartLoadingOverlay>
                                    ) : (
                                      <div className="h-full rounded-lg border border-dashed border-slate-200/60" />
                                    )}
                                  </div>
                                ))
                              ) : (
                                <div className="flex h-full flex-1 items-center justify-center rounded-lg border border-dashed border-slate-200/60 text-xs text-slate-500">
                                  All linear-scale charts are hidden.
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                    <div
                      className="pointer-events-none absolute inset-y-0 right-0 w-32 bg-gradient-to-r from-transparent to-vivid-background"
                    />
                  </div>
                </ChartSyncProvider>
              </div>
            </main>

            {/* Model Architecture with Intro Content */}
            <ModelArchitecture
              modelDescriptionGDocPortionMarkdown={modelDescriptionGDocPortionMarkdown}
              scrollContainerRef={scrollContainerRef}
            />

          </div>
        </div>

        {/* Advanced Parameters Column - hidden on mobile/tablet, opens via Parameters section button */}
        {isAdvancedPanelOpen && (
          <div className="col-start-2 row-start-1 hidden lg:flex h-full overflow-visible shadow-[0_0_1em_0_rgba(0,0,0,0.5)] z-10">
            <aside
              id="advanced-panel"
              className="w-[22rem] bg-white border-l border-gray-200 flex flex-col"
            >
                <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
                  <h2 className="text-sm font-semibold text-primary uppercase tracking-[0.3em] font-system-mono">
                    Advanced Parameters
                  </h2>
                  <button
                    type="button"
                    onClick={() => setIsAdvancedPanelOpen(false)}
                    className="text-xs font-medium text-gray-500 hover:text-gray-700"
                  >
                    Close
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto px-4 py-4">
                  <div className="space-y-6">
                    <div className="space-y-3">
                      <div className="flex flex-wrap gap-2 sm:gap-3">
                        <button
                          type="button"
                          onClick={handleExportParameters}
                          className="flex-1 min-w-[140px] px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-white bg-[#2A623D] border border-[#215337] rounded-sm shadow-[0_1px_2px_rgba(32,83,53,0.25)] hover:bg-[#245632] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#2A623D] font-system-mono"
                        >
                          Export Settings
                        </button>
                        <button
                          type="button"
                          onClick={handleImportButtonClick}
                          className="flex-1 min-w-[140px] px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-[#2A623D] bg-white border border-[#2A623D]/40 rounded-sm shadow-[0_1px_2px_rgba(32,83,53,0.1)] hover:bg-[#F4F8F5] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#2A623D] font-system-mono"
                        >
                          Import Settings
                        </button>
                        <input
                          ref={importInputRef}
                          type="file"
                          accept="application/json"
                          className="hidden"
                          onChange={handleImportParameters}
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="block text-xs font-semibold uppercase tracking-[0.18em] text-primary font-system-mono">
                          Parameter Presets
                        </label>
                        <div className="relative">
                          <span className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-[11px] uppercase tracking-[0.2em] text-gray-400 font-system-mono">
                            Preset
                          </span>
                          <select
                            value={selectedPresetId}
                            onChange={handlePresetSelect}
                            className="w-full pl-20 pr-8 py-3 border border-gray-300 rounded-md shadow-sm bg-white text-sm font-system-mono focus:outline-none focus:ring-2 focus:ring-[#2A623D] appearance-none"
                          >
                            <option value="">Select</option>
                            {PARAMETER_PRESETS.map((preset) => (
                              <option key={preset.id} value={preset.id}>
                                {preset.label}
                              </option>
                            ))}
                          </select>
                          <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-gray-400">
                            <svg
                              className="h-3 w-3"
                              viewBox="0 0 12 8"
                              fill="none"
                              xmlns="http://www.w3.org/2000/svg"
                              aria-hidden="true"
                            >
                              <path d="M1 1.5L6 6.5L11 1.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          </span>
                        </div>
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
                      summary={summary}
                      lockedParameters={lockedParameters}
                      onToggleTrajectoryDebugger={() => setShowSamplingDebug(prev => !prev)}
                      isTrajectoryDebuggerOpen={showSamplingDebug}
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
                  </div>
                </div>
              </aside>
          </div>
        )}
      </div>
      
      {/* Sampling Debug Panel - Toggle with Alt+D */}
      {showSamplingDebug && (
        <div className="fixed bottom-4 right-4 z-[9999] w-[600px] max-h-[500px] bg-gray-900 text-white rounded-lg shadow-2xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
            <h3 className="font-mono text-sm font-semibold">Sampling Debug</h3>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setShowParamToggles(prev => !prev)}
                className={`px-2 py-0.5 text-[10px] font-semibold rounded uppercase tracking-wider ${
                  showParamToggles ? 'bg-yellow-600 hover:bg-yellow-500' : 'bg-gray-600 hover:bg-gray-500'
                }`}
              >
                {showParamToggles ? 'Hide' : 'Show'} Param Toggles
              </button>
              <button
                onClick={() => setProfilingEnabled(prev => !prev)}
                className={`px-2 py-0.5 text-[10px] font-semibold rounded uppercase tracking-wider ${
                  profilingEnabled ? 'bg-purple-600 hover:bg-purple-500' : 'bg-gray-600 hover:bg-gray-500'
                }`}
              >
                Profiling {profilingEnabled ? 'ON' : 'OFF'}
              </button>
              <span className="text-xs text-gray-400">
                {enabledSamplingParams.size}/{samplingConfig ? Object.keys(samplingConfig.parameters).length + (samplingConfig.time_series_parameters ? Object.keys(samplingConfig.time_series_parameters).length : 0) : 0} sampling
              </span>
              <button
                onClick={() => setShowSamplingDebug(false)}
                className="text-gray-400 hover:text-white"
              >
                ✕
              </button>
            </div>
          </div>
          
          {/* Parameter Toggles Section */}
          {showParamToggles && samplingConfig && (
            <div className="p-2 bg-gray-800/50 border-b border-gray-700 max-h-[200px] overflow-auto">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-400 font-semibold">Parameters to Sample (unchecked = fixed to median)</span>
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      const allParams = new Set<string>();
                      for (const paramName of Object.keys(samplingConfig.parameters)) {
                        allParams.add(paramName);
                      }
                      if (samplingConfig.time_series_parameters) {
                        for (const paramName of Object.keys(samplingConfig.time_series_parameters)) {
                          allParams.add(paramName);
                        }
                      }
                      setEnabledSamplingParams(allParams);
                    }}
                    className="px-2 py-0.5 bg-green-600 hover:bg-green-500 text-[9px] font-semibold rounded uppercase"
                  >
                    All On
                  </button>
                  <button
                    onClick={() => setEnabledSamplingParams(new Set())}
                    className="px-2 py-0.5 bg-red-600 hover:bg-red-500 text-[9px] font-semibold rounded uppercase"
                  >
                    All Off
                  </button>
                  <button
                    onClick={() => {
                      // Clear samples and re-fetch with new settings
                      setSampleTrajectories([]);
                      setSamplingDebugLog([]);
                      setResampleTrigger(prev => prev + 1);
                    }}
                    className="px-2 py-0.5 bg-blue-600 hover:bg-blue-500 text-[9px] font-semibold rounded uppercase"
                  >
                    Resample
                  </button>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-1 text-[10px] font-mono">
                {/* Model parameters */}
                {Object.entries(samplingConfig.parameters)
                  .filter(([, config]) => config.dist !== 'fixed')
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([paramName, config]) => {
                    const isEnabled = enabledSamplingParams.has(paramName);
                    const medianValue = getDistributionMedian(config);
                    return (
                      <label 
                        key={paramName} 
                        className={`flex items-center gap-1.5 p-1 rounded cursor-pointer hover:bg-gray-700/50 ${
                          isEnabled ? 'text-green-400' : 'text-gray-500'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isEnabled}
                          onChange={(e) => {
                            setEnabledSamplingParams(prev => {
                              const next = new Set(prev);
                              if (e.target.checked) {
                                next.add(paramName);
                              } else {
                                next.delete(paramName);
                              }
                              return next;
                            });
                          }}
                          className="w-3 h-3"
                        />
                        <span className="truncate flex-1" title={paramName}>{paramName}</span>
                        <span className="text-gray-600 text-[8px]">
                          {typeof medianValue === 'number' 
                            ? (Math.abs(medianValue) > 1000 || (Math.abs(medianValue) < 0.01 && medianValue !== 0)
                                ? medianValue.toExponential(1) 
                                : medianValue.toFixed(2))
                            : String(medianValue)}
                        </span>
                      </label>
                    );
                  })}
                {/* Time series parameters */}
                {samplingConfig.time_series_parameters && Object.entries(samplingConfig.time_series_parameters)
                  .filter(([, config]) => config.dist !== 'fixed')
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([paramName, config]) => {
                    const isEnabled = enabledSamplingParams.has(paramName);
                    const medianValue = getDistributionMedian(config);
                    return (
                      <label 
                        key={paramName} 
                        className={`flex items-center gap-1.5 p-1 rounded cursor-pointer hover:bg-gray-700/50 ${
                          isEnabled ? 'text-yellow-400' : 'text-gray-500'
                        }`}
                        title="Time series parameter"
                      >
                        <input
                          type="checkbox"
                          checked={isEnabled}
                          onChange={(e) => {
                            setEnabledSamplingParams(prev => {
                              const next = new Set(prev);
                              if (e.target.checked) {
                                next.add(paramName);
                              } else {
                                next.delete(paramName);
                              }
                              return next;
                            });
                          }}
                          className="w-3 h-3"
                        />
                        <span className="truncate flex-1" title={`${paramName} (time series)`}>📊 {paramName}</span>
                        <span className="text-gray-600 text-[8px]">
                          {typeof medianValue === 'number' 
                            ? (Math.abs(medianValue) > 1000 || (Math.abs(medianValue) < 0.01 && medianValue !== 0)
                                ? medianValue.toExponential(1) 
                                : medianValue.toFixed(2))
                            : String(medianValue)}
                        </span>
                      </label>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Profiling Results Section */}
          {profilingEnabled && (
            <div className="p-2 bg-purple-900/30 border-b border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-purple-300 font-semibold">Server Profiling</span>
                {profilingData && (
                  <span className="text-xs text-purple-400">
                    Total: {profilingData.totalTimeSeconds.toFixed(3)}s @ {profilingData.timestamp.toLocaleTimeString()}
                  </span>
                )}
              </div>
              {profilingData ? (
                <>
                  {/* Timing Breakdown */}
                  <div className="bg-gray-950 rounded p-2 mb-2">
                    <div className="text-[10px] text-purple-300 font-semibold mb-1">Timing Breakdown:</div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[9px] font-mono">
                      {Object.entries(profilingData.timingBreakdown)
                        .filter(([key]) => key !== 'model_internal')
                        .sort(([,a], [,b]) => (b as number) - (a as number))
                        .map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-gray-400">{key}:</span>
                            <span className="text-green-400">{(value as number).toFixed(3)}s</span>
                          </div>
                        ))}
                    </div>
                  </div>
                  {/* cProfile Stats */}
                  <details className="bg-gray-950 rounded">
                    <summary className="p-2 text-[10px] text-gray-400 cursor-pointer hover:text-gray-300">
                      cProfile Details (click to expand)
                    </summary>
                    <div className="p-2 max-h-[150px] overflow-auto">
                      <pre className="text-[9px] text-gray-300 whitespace-pre font-mono leading-tight">
                        {profilingData.stats}
                      </pre>
                    </div>
                  </details>
                </>
              ) : (
                <p className="text-gray-500 text-xs">
                  Profiling enabled. Move a slider to trigger a request and see results.
                </p>
              )}
            </div>
          )}

          <div className="overflow-auto max-h-[280px] p-2">
            <div className="space-y-1 font-mono text-xs">
              {samplingDebugLog.length === 0 ? (
                <p className="text-gray-500 p-2">No sampling requests yet. Waiting for samplingConfig...</p>
              ) : (
                samplingDebugLog.map((log, idx) => (
                  <div 
                    key={idx}
                    className={`p-2 rounded ${
                      log.type === 'request' ? 'bg-blue-900/50' : 
                      log.type === 'response' ? 'bg-green-900/50' : 
                      'bg-red-900/50'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className={`px-1 rounded text-[10px] uppercase ${
                        log.type === 'request' ? 'bg-blue-600' :
                        log.type === 'response' ? 'bg-green-600' :
                        'bg-red-600'
                      }`}>
                        {log.type}
                      </span>
                      <span className="text-gray-400">#{log.id}</span>
                      <span className="text-gray-500">{log.timestamp.toISOString().split('T')[1].split('.')[0]}</span>
                    </div>
                    {log.params && (
                      <div className="mt-1 text-gray-300 text-[10px]">
                        {Object.entries(log.params).map(([k, v]) => (
                          <div key={k}>
                            <span className="text-gray-500">{k}:</span>{' '}
                            {typeof v === 'number' ? v.toExponential(2) : String(v)}
                          </div>
                        ))}
                      </div>
                    )}
                    {log.dataPoints !== undefined && (
                      <div className="mt-1 text-green-400">✓ {log.dataPoints} data points received</div>
                    )}
                    {log.error && (
                      <div className="mt-1 text-red-400">Error: {log.error}</div>
                    )}
                  </div>
                ))
              )}
            </div>
            {sampleTrajectories.length > 0 && (
              <div className="mt-3 p-2 bg-gray-800 rounded">
                <div className="text-xs text-gray-400 mb-2">Current Trajectories (click row to expand, Apply to use):</div>
                {sampleTrajectories.map((sample, idx) => {
                  const traj = sample.trajectory;
                  const horizonMinutes = sample.params.ac_time_horizon_minutes;
                  const isExpanded = expandedTrajectoryIndex === idx;
                  return (
                    <div key={idx} className="mb-1">
                      <div 
                        className="flex items-center justify-between p-1.5 bg-gray-700/50 rounded hover:bg-gray-600/50 cursor-pointer transition-colors"
                        onClick={() => setExpandedTrajectoryIndex(isExpanded ? null : idx)}
                      >
                        <div className="text-[10px] text-gray-300 flex items-center gap-1">
                          <span className={`transition-transform ${isExpanded ? 'rotate-90' : ''}`}>▶</span>
                          <span className="font-semibold text-white">#{idx + 1}</span>
                          {' • '}{traj.length} pts
                          {' • '}SC: {typeof horizonMinutes === 'number' ? formatWorkTimeDuration(horizonMinutes) : 'N/A'}
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            applySampleTrajectory(sample);
                          }}
                          className="px-2 py-0.5 bg-green-600 hover:bg-green-500 text-white text-[9px] font-semibold rounded uppercase tracking-wider"
                        >
                          Apply
                        </button>
                      </div>
                      {isExpanded && (
                        <div className="mt-1 ml-4 p-2 bg-gray-900/50 rounded text-[9px] font-mono max-h-[200px] overflow-auto">
                          <div className="text-gray-400 mb-1 font-semibold">Full Parameters:</div>
                          {Object.entries(sample.params)
                            .sort(([a], [b]) => a.localeCompare(b))
                            .map(([key, value]) => (
                              <div key={key} className="flex gap-2">
                                <span className="text-gray-500 min-w-[200px]">{key}:</span>
                                <span className="text-gray-300">
                                  {typeof value === 'number' 
                                    ? (Math.abs(value) > 1000 || Math.abs(value) < 0.01 
                                        ? value.toExponential(3) 
                                        : value.toFixed(4))
                                    : String(value)}
                                </span>
                              </div>
                            ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </ParameterHoverProvider>
  );
}
