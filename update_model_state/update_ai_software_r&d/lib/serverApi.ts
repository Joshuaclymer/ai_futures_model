import { cacheLife } from 'next/cache';
import { promises as fs } from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import { DEFAULT_PARAMETERS, ParametersType } from '@/constants/parameters';
import { convertParametersToAPIFormat, convertSampledParametersToAPIFormat, ParameterRecord } from '@/utils/monteCarlo';
import { 
  SamplingConfig, 
  SeededRandom, 
  setSamplingRng, 
  generateParameterSample,
  initializeCorrelationSampling 
} from '@/utils/sampling';

const SIMULATION_END_YEAR = 2045;
const VISIBLE_CHART_START_YEAR = 2018;
const NUM_SAMPLES = 10;

const API_BASE_URL = process.env.NODE_ENV === 'development'
  ? 'http://127.0.0.1:5328'
  : 'https://ai-rates-calculator.vercel.app';

export interface TimeSeriesPoint {
  year: number;
  horizonLength: number;
  effectiveCompute: number;
  automationFraction: number;
  trainingCompute?: number;
  experimentCapacity?: number;
  aiResearchTaste?: number;
  aiSoftwareProgressMultiplier?: number;
  aiSwProgressMultRefPresentDay?: number;
  serialCodingLaborMultiplier?: number;
  humanLabor?: number;
  inferenceCompute?: number;
  experimentCompute?: number;
  researchEffort?: number;
  researchStock?: number;
  softwareProgressRate?: number;
  softwareEfficiency?: number;
  aiCodingLaborMultiplier?: number;
  aiCodingLaborMultRefPresentDay?: number;
}

export interface ComputeApiResponse {
  success: boolean;
  summary?: Record<string, unknown>;
  time_series?: TimeSeriesPoint[];
  milestones?: Record<string, unknown>;
  horizon_params?: {
    uses_shifted_form: boolean;
    anchor_progress: number | null;
  };
  exp_capacity_params?: {
    rho: number | null;
    alpha: number | null;
    experiment_compute_exponent: number | null;
  };
  error?: string;
}

export interface SampleTrajectory {
  trajectory: TimeSeriesPoint[];
  params: Record<string, number | string | boolean>;
}

// Load sampling config from YAML file
async function loadSamplingConfig(): Promise<SamplingConfig | null> {
  try {
    const configPath = path.join(process.cwd(), 'config/sampling_config.yaml');
    const fileContents = await fs.readFile(configPath, 'utf8');
    const config = yaml.load(fileContents) as SamplingConfig;
    return config;
  } catch (error) {
    console.error('Failed to load sampling config:', error);
    return null;
  }
}

// Fetch a single trajectory from the API
async function fetchSingleTrajectory(
  params: Record<string, number | string | boolean>
): Promise<TimeSeriesPoint[] | null> {
  try {
    const apiParameters = convertSampledParametersToAPIFormat(params as unknown as ParameterRecord);
    
    const response = await fetch(`${API_BASE_URL}/api/compute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        parameters: apiParameters,
        time_range: [2012, SIMULATION_END_YEAR],
        initial_progress: 0.0
      }),
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    if (!data.success || !data.time_series) {
      return null;
    }

    return data.time_series.filter((point: TimeSeriesPoint) => point.year >= VISIBLE_CHART_START_YEAR);
  } catch {
    return null;
  }
}

// Cached function to fetch compute data - cache key includes parameters
export async function fetchComputeData(parameters: ParametersType): Promise<ComputeApiResponse> {
  'use cache';
  cacheLife('hours');

  try {
    const apiParameters = convertParametersToAPIFormat(parameters as unknown as ParameterRecord);
    
    const response = await fetch(`${API_BASE_URL}/api/compute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        parameters: apiParameters,
        time_range: [2012, SIMULATION_END_YEAR],
        initial_progress: 0.0
      }),
    });

    if (!response.ok) {
      console.error(`Failed to fetch compute data: HTTP ${response.status}`);
      return { success: false, error: `HTTP ${response.status}` };
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to fetch compute data:', error);
    return { success: false, error: String(error) };
  }
}

// Cached function to fetch sample trajectories with a seed
export async function fetchSampleTrajectories(
  baseParameters: ParametersType,
  seed: number
): Promise<SampleTrajectory[]> {
  'use cache';
  cacheLife('hours');

  const samplingConfig = await loadSamplingConfig();
  if (!samplingConfig) {
    console.error('[fetchSampleTrajectories] Failed to load sampling config');
    return [];
  }

  // Initialize correlation sampling if configured
  if (samplingConfig.correlation_matrix) {
    initializeCorrelationSampling(samplingConfig.correlation_matrix);
  }

  // Set up seeded RNG
  const rng = new SeededRandom(seed);
  setSamplingRng(rng);

  try {
    const samples: SampleTrajectory[] = [];

    // Generate and fetch samples sequentially to ensure deterministic ordering
    for (let i = 0; i < NUM_SAMPLES; i++) {
      const sampleParams = generateParameterSample(samplingConfig, baseParameters as Record<string, number | string | boolean>);
      const trajectory = await fetchSingleTrajectory(sampleParams);
      
      if (trajectory) {
        samples.push({
          trajectory,
          params: sampleParams
        });
      }
    }

    return samples;
  } finally {
    // Reset RNG to not affect other code
    setSamplingRng(null);
  }
}

// Convenience function for default parameters
export async function fetchInitialComputeData(): Promise<ComputeApiResponse> {
  return fetchComputeData({ ...DEFAULT_PARAMETERS });
}
