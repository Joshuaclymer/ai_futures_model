import { Suspense } from 'react';
import UnifiedAppClient from '@/components/UnifiedAppClient';

export default function BlackProjectsPage() {
  return (
    <Suspense fallback={null}>
      <UnifiedAppClient initialTab="black-projects" />
    </Suspense>
  );
}

export const metadata = {
  title: 'Black Projects - AI Futures Model',
  description: 'Monte Carlo simulation of covert compute production capabilities',
};
