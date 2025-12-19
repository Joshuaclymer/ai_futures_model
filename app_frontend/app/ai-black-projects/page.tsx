import { Suspense } from 'react';
import { loadBenchmarkData } from '@/utils/benchmarkLoader';
import { DEFAULT_PARAMETERS } from '@/constants/parameters';
import UnifiedAppClient from '@/components/UnifiedAppClient';

const DEFAULT_SEED = 12345;

export default function BlackProjectsPage() {
  // Don't do server-side data fetching for black projects page
  // The black projects tab fetches its own data from a different API
  const benchmarkData = loadBenchmarkData();
  const parameters = { ...DEFAULT_PARAMETERS };
  const seed = DEFAULT_SEED;

  return (
    <Suspense fallback={null}>
      <UnifiedAppClient
        initialTab="black-projects"
        benchmarkData={benchmarkData}
        initialComputeData={null}
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
