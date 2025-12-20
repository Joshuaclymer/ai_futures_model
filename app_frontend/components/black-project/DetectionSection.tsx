'use client';

import { BlackProjectData } from '@/types/blackProject';

interface DetectionSectionProps {
  data: BlackProjectData | null;
  isLoading?: boolean;
}

export function DetectionSection({ data, isLoading }: DetectionSectionProps) {
  if (isLoading) {
    return (
      <section className="space-y-6">
        <h2 className="text-xl font-semibold text-gray-800">Detection Model</h2>
        <div className="h-48 bg-gray-100 animate-pulse rounded" />
      </section>
    );
  }

  if (!data?.black_project_model) {
    return (
      <section className="space-y-6">
        <h2 className="text-xl font-semibold text-gray-800">Detection Model</h2>
        <p className="text-gray-500">No detection data available</p>
      </section>
    );
  }

  return (
    <section className="space-y-6">
      <h2 className="text-xl font-semibold text-gray-800">Detection Model</h2>
      <p className="text-gray-600">
        Detection is modeled using likelihood ratios that update the US intelligence community&apos;s 
        belief that PRC is running a covert AI project.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 bg-white border border-gray-200 rounded shadow-sm">
          <h3 className="font-medium text-gray-700 mb-2">Posterior Probability</h3>
          <p className="text-sm text-gray-500">
            Time series of P(covert project exists) based on cumulative evidence
          </p>
        </div>
        <div className="p-4 bg-white border border-gray-200 rounded shadow-sm">
          <h3 className="font-medium text-gray-700 mb-2">Cumulative Likelihood Ratio</h3>
          <p className="text-sm text-gray-500">
            Combined LR from all intelligence sources
          </p>
        </div>
      </div>
    </section>
  );
}
