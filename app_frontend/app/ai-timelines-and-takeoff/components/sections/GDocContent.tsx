import { Suspense } from 'react';
import { cacheLife } from 'next/cache';
import { googleDocToMarkdown } from '../../utils/googleDocToMarkdown';
import MarkdownRenderer from '../ui/MarkdownRenderer';

const GOOGLE_DOC_ID = '1aMgKau-Wmq2dCMEIHDanDqUnAI4_2R12yZw2WJ58_-M';

// Async component for Google Doc content - cached for performance
async function CachedGDocMarkdown() {
  'use cache';
  cacheLife('hours');

  const markdown = await googleDocToMarkdown(GOOGLE_DOC_ID);
  return <MarkdownRenderer markdown={markdown} />;
}

// Wrapper component with Suspense - can be rendered synchronously
export function GDocContent() {
  return (
    <Suspense fallback={<div className="animate-pulse h-96 bg-gray-100 rounded" />}>
      <CachedGDocMarkdown />
    </Suspense>
  );
}

