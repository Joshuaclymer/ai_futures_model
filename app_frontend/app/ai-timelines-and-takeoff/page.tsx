import { Suspense } from 'react';
import { loadBenchmarkData } from './utils/benchmarkLoader';
import UnifiedAppClient from '@/components/UnifiedAppClient';

const DEFAULT_SEED = 12345;

export default function TimelinesPage() {
  const benchmarkData = loadBenchmarkData();
  const seed = DEFAULT_SEED;

  return (
    <Suspense fallback={null}>
      <UnifiedAppClient
        initialTab="timelines"
        benchmarkData={benchmarkData}
        initialComputeData={null}
        initialParameters={null}
        initialSampleTrajectories={[]}
        initialSeed={seed}
      />
    </Suspense>
  );
}
