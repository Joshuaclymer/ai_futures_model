import { Suspense } from 'react';
import { cacheLife } from 'next/cache';
import { loadBenchmarkData } from '@/utils/benchmarkLoader';
import { fetchComputeData } from '@/lib/serverApi';
import { DEFAULT_PARAMETERS } from '@/constants/parameters';
import UnifiedAppClient from '@/components/UnifiedAppClient';

const DEFAULT_SEED = 12345;

export default async function BlackProjectsPage() {
  'use cache';
  cacheLife('hours');

  const benchmarkData = loadBenchmarkData();
  const parameters = { ...DEFAULT_PARAMETERS };
  const seed = DEFAULT_SEED;

  // Only fetch compute data server-side - sample trajectories are loaded progressively on the client
  const initialComputeData = await fetchComputeData(parameters);

  return (
    <Suspense fallback={null}>
      <UnifiedAppClient
        initialTab="black-projects"
        benchmarkData={benchmarkData}
        initialComputeData={initialComputeData}
        initialParameters={parameters}
        initialSampleTrajectories={[]}
        initialSeed={seed}
      />
    </Suspense>
  );
}

export const metadata = {
  title: 'Black Projects - AI Futures Model',
  description: 'Monte Carlo simulation of covert compute production capabilities',
};
