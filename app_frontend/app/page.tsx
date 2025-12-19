import { Suspense } from 'react';
import ProgressChartServer from '@/components/ProgressChartServer';

// Suspense with null fallback satisfies Next.js 16's requirement that async
// operations be wrapped in Suspense, while showing no loading state.
// The charts render instantly via pre-computed static data.
// Only the Google Doc content (below the fold) loads asynchronously.
export default function Home() {
  return (
    <div className="bg-vivid-background text-vivid-foreground min-h-screen">
      <Suspense fallback={null}>
        <ProgressChartServer />
      </Suspense>
    </div>
  );
}
