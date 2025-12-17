import { Suspense } from 'react';
import ProgressChartServer from '@/components/ProgressChartServer';

export default function Home() {
  return (
    <div className="bg-vivid-background text-vivid-foreground min-h-screen">
      <Suspense fallback={null}>
      <ProgressChartServer />
      </Suspense>
    </div>
  );
}
