import { cacheLife } from 'next/cache';
import { loadBenchmarkData } from '../utils/benchmarkLoader';
import ProgressChartClient from './ProgressChartClient';
import { googleDocToMarkdown } from '@/utils/googleDocToMarkdown';
import MarkdownRenderer from './MarkdownRenderer';
import { fetchComputeData } from '@/lib/serverApi';
import { DEFAULT_PARAMETERS } from '@/constants/parameters';

const GOOGLE_DOC_ID = '1aMgKau-Wmq2dCMEIHDanDqUnAI4_2R12yZw2WJ58_-M';
const DEFAULT_SEED = 12345;

async function CachedModelDescriptionGDocPortionMarkdown() {
  'use cache';
  cacheLife('hours');

  const markdown = await googleDocToMarkdown(GOOGLE_DOC_ID);
  return <MarkdownRenderer markdown={markdown} />;
}

export default async function ProgressChartServer() {
  'use cache';
  cacheLife('hours');

  const benchmarkData = loadBenchmarkData();
  const modelDescriptionGDocPortionMarkdown = await CachedModelDescriptionGDocPortionMarkdown();
  
  // Use default parameters for initial server-side fetch
  // URL state is handled by the client
  const parameters = { ...DEFAULT_PARAMETERS };
  const seed = DEFAULT_SEED;
  
  // Only fetch compute data server-side - sample trajectories are loaded progressively on the client
  // This ensures the main charts render as quickly as possible
  const initialComputeData = await fetchComputeData(parameters);

  return (
    <ProgressChartClient 
      benchmarkData={benchmarkData} 
      modelDescriptionGDocPortionMarkdown={modelDescriptionGDocPortionMarkdown}
      initialComputeData={initialComputeData}
      initialParameters={parameters}
      initialSampleTrajectories={[]}
      initialSeed={seed}
    />
  );
}
