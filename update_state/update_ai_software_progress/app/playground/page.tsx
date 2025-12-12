import { Suspense } from 'react';
import { cacheLife } from 'next/cache';
import { loadBenchmarkData } from '@/utils/benchmarkLoader';
import { fetchComputeData } from '@/lib/serverApi';
import { DEFAULT_PARAMETERS } from '@/constants/parameters';
import PlaygroundClient from '@/components/PlaygroundClient';

const DEFAULT_SEED = 12345;

export default async function PlaygroundPage() {
  'use cache';
  cacheLife('hours');

  const benchmarkData = loadBenchmarkData();
  const parameters = { ...DEFAULT_PARAMETERS };
  const seed = DEFAULT_SEED;
  
  // Only fetch compute data server-side - sample trajectories are loaded progressively on the client
  // This ensures the main charts render as quickly as possible
  const initialComputeData = await fetchComputeData(parameters);

  return (
    <Suspense fallback={null}>
      <PlaygroundClient 
        benchmarkData={benchmarkData}
        initialComputeData={initialComputeData}
        initialParameters={parameters}
        initialSampleTrajectories={[]}
        initialSeed={seed}
      />
    </Suspense>
  );
}

