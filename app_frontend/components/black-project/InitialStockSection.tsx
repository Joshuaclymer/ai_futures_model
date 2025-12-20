'use client';

import { BlackProjectData } from '@/types/blackProject';

interface InitialStockSectionProps {
  data: BlackProjectData | null;
  isLoading?: boolean;
  diversionProportion?: number;
}

export function InitialStockSection({ data, isLoading, diversionProportion = 0.1 }: InitialStockSectionProps) {
  if (isLoading) {
    return (
      <section className="space-y-6">
        <h2 className="text-xl font-semibold text-gray-800">Initial Diverted Compute</h2>
        <div className="h-48 bg-gray-100 animate-pulse rounded" />
      </section>
    );
  }

  return (
    <section className="space-y-6">
      <h2 className="text-xl font-semibold text-gray-800">Initial Diverted Compute</h2>
      <p className="text-gray-600">
        A proportion of PRC&apos;s existing compute stock is diverted to the covert project at the start.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 bg-white border border-gray-200 rounded shadow-sm">
          <h3 className="font-medium text-gray-700 mb-2">Diversion Proportion</h3>
          <p className="text-2xl font-bold text-[#5E6FB8]">
            {(diversionProportion * 100).toFixed(0)}%
          </p>
          <p className="text-sm text-gray-500">of PRC compute stock</p>
        </div>
        <div className="p-4 bg-white border border-gray-200 rounded shadow-sm">
          <h3 className="font-medium text-gray-700 mb-2">Accounting LR</h3>
          <p className="text-sm text-gray-500">
            Likelihood ratio from PRC compute accounting discrepancies
          </p>
        </div>
      </div>
    </section>
  );
}
