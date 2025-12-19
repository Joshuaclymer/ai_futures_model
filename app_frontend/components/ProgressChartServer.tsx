import { cacheLife } from 'next/cache';
import { loadBenchmarkData } from '../utils/benchmarkLoader';
import ProgressChartClient from './ProgressChartClient';
import { DEFAULT_PARAMETERS } from '@/constants/parameters';
import type { ComputeApiResponse } from '@/lib/serverApi';
import { GDocContent } from './GDocContent';
import { convertParametersToAPIFormat, ParameterRecord } from '@/utils/monteCarlo';

const DEFAULT_SEED = 12345;
const SIMULATION_END_YEAR = 2045;

// Use local API in development, production API in production
const API_URL = process.env.NODE_ENV === 'development'
  ? 'http://127.0.0.1:5329/api/run-sw-progress-simulation'
  : 'https://ai-rates-calculator.vercel.app/api/run-sw-progress-simulation';

async function fetchInitialChartData(): Promise<ComputeApiResponse> {
  'use cache';
  cacheLife('hours');

  try {
    const parameters = { ...DEFAULT_PARAMETERS };
    const apiParameters = convertParametersToAPIFormat(parameters as unknown as ParameterRecord);
    
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        parameters: apiParameters,
        time_range: [2012, SIMULATION_END_YEAR],
        initial_progress: 0.0
      }),
    });

    if (!response.ok) {
      console.error(`Failed to fetch initial chart data: HTTP ${response.status}`);
      return { success: false, error: `HTTP ${response.status}` };
    }

    const data = await response.json();
    // Log raw milestone times for debugging (server-side)
    if (data.milestones) {
      console.log('[Server] Raw AC time:', data.milestones['AC']?.time);
      console.log('[Server] Raw SAR time:', data.milestones['SAR-level-experiment-selection-skill']?.time);
    }
    return data;
  } catch (error) {
    console.error('Failed to fetch initial chart data:', error);
    return { success: false, error: String(error) };
  }
}

// Async server component - fetches from production API with caching
export default async function ProgressChartServer() {
  const [benchmarkData, initialComputeData] = await Promise.all([
    Promise.resolve(loadBenchmarkData()),
    fetchInitialChartData(),
  ]);

  const parameters = { ...DEFAULT_PARAMETERS };

  return (
    <ProgressChartClient 
      benchmarkData={benchmarkData} 
      modelDescriptionGDocPortionMarkdown={<GDocContent />}
      initialComputeData={initialComputeData}
      initialParameters={parameters}
      initialSampleTrajectories={[]}
      initialSeed={DEFAULT_SEED}
    />
  );
}
