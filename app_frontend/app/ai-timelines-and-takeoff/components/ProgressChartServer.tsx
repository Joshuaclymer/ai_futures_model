import { loadBenchmarkData } from '../utils/benchmarkLoader';
import ProgressChartClient from './ProgressChartClient';
import { GDocContent } from './sections/GDocContent';

const DEFAULT_SEED = 12345;

// Async server component - client will fetch initial data and defaults
export default async function ProgressChartServer() {
  const benchmarkData = loadBenchmarkData();

  return (
    <ProgressChartClient
      benchmarkData={benchmarkData}
      modelDescriptionGDocPortionMarkdown={<GDocContent />}
      initialSampleTrajectories={[]}
      initialSeed={DEFAULT_SEED}
    />
  );
}
