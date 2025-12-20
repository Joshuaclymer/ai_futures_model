'use client';

import { BlackProjectData } from '@/types/blackProject';

interface CovertFabSectionProps {
  data: BlackProjectData | null;
  isLoading?: boolean;
}

export function CovertFabSection({ data, isLoading }: CovertFabSectionProps) {
  if (isLoading) {
    return (
      <section className="space-y-6">
        <h2 className="text-xl font-semibold text-gray-800">Covert Fab</h2>
        <div className="h-48 bg-gray-100 animate-pulse rounded" />
      </section>
    );
  }

  if (!data?.black_fab) {
    return (
      <section className="space-y-6">
        <h2 className="text-xl font-semibold text-gray-800">Covert Fab</h2>
        <p className="text-gray-500">No fab data available</p>
      </section>
    );
  }

  const fabData = data.black_fab;
  const isOperationalProportion = fabData.is_operational?.proportion || [];
  const lastProportion = isOperationalProportion[isOperationalProportion.length - 1] || 0;

  return (
    <section className="space-y-6">
      <h2 className="text-xl font-semibold text-gray-800">Covert Fab</h2>
      <p className="text-gray-600">
        A covert semiconductor fabrication facility producing AI accelerator chips.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-white border border-gray-200 rounded shadow-sm">
          <h3 className="font-medium text-gray-700 mb-2">Fab Operational</h3>
          <p className="text-2xl font-bold text-[#E9A842]">
            {(lastProportion * 100).toFixed(0)}%
          </p>
          <p className="text-sm text-gray-500">of simulations</p>
        </div>
        <div className="p-4 bg-white border border-gray-200 rounded shadow-sm">
          <h3 className="font-medium text-gray-700 mb-2">Process Node</h3>
          <p className="text-lg font-semibold text-gray-700">
            Best Indigenous
          </p>
        </div>
        <div className="p-4 bg-white border border-gray-200 rounded shadow-sm">
          <h3 className="font-medium text-gray-700 mb-2">Chip Production</h3>
          <p className="text-sm text-gray-500">
            H100-equivalent chips per year
          </p>
        </div>
      </div>
    </section>
  );
}
