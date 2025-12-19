'use client';

import { CCDFChart } from './charts/CCDFChart';
import { TimeSeriesChart } from './charts/TimeSeriesChart';
import { BlackProjectData, COLOR_PALETTE } from '@/types/blackProject';

interface DatacenterSectionProps {
  data: BlackProjectData | null;
  isLoading: boolean;
}

export function DatacenterSection({ data, isLoading }: DatacenterSectionProps) {
  const datacenters = data?.black_datacenters;
  const model = data?.black_project_model;

  // Calculate dashboard values
  const medianCapacityAtDetection = datacenters?.individual_capacity_before_detection?.length
    ? datacenters.individual_capacity_before_detection.sort((a, b) => a - b)[
        Math.floor(datacenters.individual_capacity_before_detection.length / 2)
      ]
    : 0;

  const medianTimeAtDetection = datacenters?.individual_time_before_detection?.length
    ? datacenters.individual_time_before_detection.sort((a, b) => a - b)[
        Math.floor(datacenters.individual_time_before_detection.length / 2)
      ]
    : 0;

  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-800 mb-2">
          Covert data centers
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          A covert AI project requires datacenter capacity to power its compute.
          The PRC may build concealed datacenters or divert capacity from existing
          facilities. This section models datacenter capacity growth and its detectability.
        </p>
      </div>

      {/* Dashboard and main CCDF */}
      <div className="flex flex-wrap gap-4">
        {/* Dashboard */}
        <div className="bg-white rounded-lg border border-gray-200 p-4 w-60 flex-shrink-0">
          <div className="text-sm font-medium text-gray-700 mb-3">
            Datacenter outcomes at detection
          </div>
          <div className="space-y-4">
            <div>
              <div className="text-xl font-bold text-[#4AA896]">
                {medianCapacityAtDetection.toFixed(1)} GW
              </div>
              <div className="text-xs text-gray-500">median total capacity</div>
            </div>
            <div>
              <div className="text-xl font-bold text-[#4AA896]">
                {medianTimeAtDetection.toFixed(1)} years
              </div>
              <div className="text-xs text-gray-500">median operational time</div>
            </div>
          </div>
        </div>

        {/* Capacity CCDF */}
        <div className="bg-white rounded-lg border border-gray-200 p-4 flex-1 min-w-[300px]">
          <div className="text-sm font-medium text-gray-700 mb-2">
            Total datacenter capacity at detection (GW)
          </div>
          <div className="h-64">
            <CCDFChart
              data={datacenters?.capacity_ccdfs?.[1]}
              color={COLOR_PALETTE.datacenters_and_energy}
              xLabel="GW"
              isLoading={isLoading}
            />
          </div>
        </div>
      </div>

      {/* Capacity breakdown */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-800">
          Capacity breakdown
        </h3>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-3">
            Total capacity = Unconcealed (diverted) + Concealed (built for project)
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Datacenter capacity over time */}
            <div>
              <div className="text-sm text-gray-600 mb-2">Total datacenter capacity over time</div>
              <div className="h-48">
                <TimeSeriesChart
                  years={datacenters?.years}
                  median={datacenters?.datacenter_capacity?.median}
                  p25={datacenters?.datacenter_capacity?.p25}
                  p75={datacenters?.datacenter_capacity?.p75}
                  color={COLOR_PALETTE.datacenters_and_energy}
                  yLabel="GW"
                  isLoading={isLoading}
                />
              </div>
            </div>

            {/* Operational compute over time */}
            <div>
              <div className="text-sm text-gray-600 mb-2">Operational compute over time</div>
              <div className="h-48">
                <TimeSeriesChart
                  years={model?.years}
                  median={model?.operational_compute?.median}
                  p25={model?.operational_compute?.p25}
                  p75={model?.operational_compute?.p75}
                  color={COLOR_PALETTE.chip_stock}
                  yLabel="H100e"
                  isLoading={isLoading}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Energy consumption */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="text-sm font-medium text-gray-700 mb-3">
          Energy consumption breakdown
        </div>
        <div className="text-xs text-gray-500 mb-4">
          Energy consumption varies by compute source due to different chip energy efficiencies.
          Initial (diverted) chips use state-of-the-art efficiency while fab-produced chips
          use the efficiency of their process node.
        </div>
        {model?.energy_source_labels && (
          <div className="flex flex-wrap gap-4">
            {model.energy_source_labels.map((label, i) => (
              <div key={i} className="flex items-center gap-2">
                <div
                  className="w-4 h-4 rounded"
                  style={{
                    backgroundColor: i === 0 ? COLOR_PALETTE.chip_stock : COLOR_PALETTE.fab,
                  }}
                />
                <span className="text-xs text-gray-600">{label}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* PRC Capacity Trajectory */}
      {datacenters?.prc_capacity_years && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-3">
            PRC datacenter capacity trajectory (2025 to agreement)
          </div>
          <div className="h-48">
            <TimeSeriesChart
              years={datacenters.prc_capacity_years}
              median={datacenters.prc_capacity_gw?.median}
              p25={datacenters.prc_capacity_gw?.p25}
              p75={datacenters.prc_capacity_gw?.p75}
              color={COLOR_PALETTE.datacenters_and_energy}
              yLabel="GW"
              isLoading={isLoading}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-2">
            <span>
              Capacity at agreement: {datacenters.prc_capacity_at_agreement_year_gw?.toFixed(1)} GW
            </span>
            <span>
              Fraction diverted: {((datacenters.fraction_diverted || 0) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* Datacenter Detection */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-800">
          Datacenter detection
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* LR from datacenters */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="text-sm font-medium text-gray-700 mb-2">
              LR from concealed datacenters
            </div>
            <div className="h-48">
              <TimeSeriesChart
                years={datacenters?.years}
                median={datacenters?.lr_datacenters?.median}
                p25={datacenters?.lr_datacenters?.p25}
                p75={datacenters?.lr_datacenters?.p75}
                color={COLOR_PALETTE.datacenters_and_energy}
                yLabel="LR"
                isLoading={isLoading}
                yLogScale={true}
              />
            </div>
          </div>

          {/* Detection probability */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="text-sm font-medium text-gray-700 mb-2">
              Detection probability over time
            </div>
            <div className="h-48">
              <TimeSeriesChart
                years={datacenters?.years}
                median={datacenters?.datacenter_detection_prob}
                color={COLOR_PALETTE.detection}
                yLabel="P(detected)"
                isLoading={isLoading}
                showBand={false}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default DatacenterSection;
