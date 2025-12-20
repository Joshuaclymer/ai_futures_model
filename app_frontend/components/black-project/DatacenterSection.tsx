'use client';

import { BlackProjectData } from '@/types/blackProject';

interface DatacenterSectionProps {
  data: BlackProjectData | null;
  isLoading?: boolean;
}

export function DatacenterSection({ data, isLoading }: DatacenterSectionProps) {
  if (isLoading) {
    return (
      <section className="space-y-6">
        <h2 className="text-xl font-semibold text-gray-800">Covert Datacenters</h2>
        <div className="h-48 bg-gray-100 animate-pulse rounded" />
      </section>
    );
  }

  if (!data?.black_datacenters) {
    return (
      <section className="space-y-6">
        <h2 className="text-xl font-semibold text-gray-800">Covert Datacenters</h2>
        <p className="text-gray-500">No datacenter data available</p>
      </section>
    );
  }

  const dcData = data.black_datacenters;
  const capacityMedian = dcData.datacenter_capacity?.median || [];
  const lastCapacity = capacityMedian[capacityMedian.length - 1] || 0;

  return (
    <section className="space-y-6">
      <h2 className="text-xl font-semibold text-gray-800">Covert Datacenters</h2>
      <p className="text-gray-600">
        Concealed datacenter capacity constructed to house covert compute infrastructure.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 bg-white border border-gray-200 rounded shadow-sm">
          <h3 className="font-medium text-gray-700 mb-2">Final Capacity (median)</h3>
          <p className="text-2xl font-bold text-[#4AA896]">
            {lastCapacity.toFixed(1)} GW
          </p>
        </div>
        <div className="p-4 bg-white border border-gray-200 rounded shadow-sm">
          <h3 className="font-medium text-gray-700 mb-2">Construction Model</h3>
          <p className="text-sm text-gray-500">
            Linear growth based on construction labor allocation
          </p>
        </div>
      </div>
    </section>
  );
}
