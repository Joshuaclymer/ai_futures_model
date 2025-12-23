import { Suspense } from 'react';
import { loadBenchmarkData } from '@/utils/benchmarkLoader';
import { DEFAULT_PARAMETERS } from '@/constants/parameters';
import UnifiedAppClient from '@/components/UnifiedAppClient';

const DEFAULT_SEED = 12345;

export default function TimelinesPage() {
  const benchmarkData = loadBenchmarkData();
  const parameters = { ...DEFAULT_PARAMETERS };
  const seed = DEFAULT_SEED;

  return (
    <Suspense fallback={null}>
      <UnifiedAppClient
        initialTab="timelines"
        benchmarkData={benchmarkData}
        initialComputeData={null}
        initialParameters={parameters}
        initialSampleTrajectories={[]}
        initialSeed={seed}
      />
    </Suspense>
  );
}
