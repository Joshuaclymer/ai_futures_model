'use client';

import { CCDFChart } from './charts/CCDFChart';
import { TimeSeriesChart } from './charts/TimeSeriesChart';
import { BlackProjectData, COLOR_PALETTE } from '@/types/blackProject';

interface CovertFabSectionProps {
  data: BlackProjectData | null;
  isLoading: boolean;
}

export function CovertFabSection({ data, isLoading }: CovertFabSectionProps) {
  const fab = data?.black_fab;
  const model = data?.black_project_model;
  const probFabBuilt = data?.prob_fab_built ?? 0;

  // Format numbers for display
  const formatNumber = (n: number) => {
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return n.toFixed(0);
  };

  // Calculate dashboard values
  const medianH100eAtDetection = fab?.individual_h100e_before_detection?.length
    ? fab.individual_h100e_before_detection.sort((a, b) => a - b)[
        Math.floor(fab.individual_h100e_before_detection.length / 2)
      ]
    : 0;

  const medianTimeAtDetection = fab?.individual_time_before_detection?.length
    ? fab.individual_time_before_detection.sort((a, b) => a - b)[
        Math.floor(fab.individual_time_before_detection.length / 2)
      ]
    : 0;

  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-800 mb-2">
          Covert fab
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          The PRC may attempt to build a covert semiconductor fabrication facility
          to produce chips domestically. This section models the production capacity
          and detection probability of such a facility.
        </p>
      </div>

      {/* Probability fab built */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="text-sm text-blue-800">
          <span className="font-semibold">{(probFabBuilt * 100).toFixed(0)}%</span> of simulations
          include a covert fab being built (based on process node localization probability)
        </div>
      </div>

      {/* Dashboard and main CCDF */}
      <div className="flex flex-wrap gap-4">
        {/* Dashboard */}
        <div className="bg-white rounded-lg border border-gray-200 p-4 w-60 flex-shrink-0">
          <div className="text-sm font-medium text-gray-700 mb-3">
            Fab outcomes at detection
          </div>
          <div className="space-y-4">
            <div>
              <div className="text-xl font-bold text-[#E9A842]">
                {formatNumber(medianH100eAtDetection)} H100e
              </div>
              <div className="text-xs text-gray-500">median chips produced</div>
            </div>
            <div>
              <div className="text-xl font-bold text-[#E9A842]">
                {medianTimeAtDetection.toFixed(1)} years
              </div>
              <div className="text-xs text-gray-500">median operational time</div>
            </div>
          </div>
        </div>

        {/* Compute at Detection CCDF */}
        <div className="bg-white rounded-lg border border-gray-200 p-4 flex-1 min-w-[300px]">
          <div className="text-sm font-medium text-gray-700 mb-2">
            H100e produced by detection
          </div>
          <div className="h-64">
            <CCDFChart
              data={fab?.compute_ccdf}
              color={COLOR_PALETTE.fab}
              xLabel="H100-equivalents"
              xLogScale={true}
              isLoading={isLoading}
            />
          </div>
        </div>

        {/* Operational Time CCDF */}
        <div className="bg-white rounded-lg border border-gray-200 p-4 flex-1 min-w-[300px]">
          <div className="text-sm font-medium text-gray-700 mb-2">
            Operational time before detection
          </div>
          <div className="h-64">
            <CCDFChart
              data={fab?.op_time_ccdf}
              color={COLOR_PALETTE.fab}
              xLabel="Years"
              isLoading={isLoading}
            />
          </div>
        </div>
      </div>

      {/* Production Breakdown */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-800">
          Production breakdown
        </h3>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-3">
            Production formula: H100e/month = Wafers/month × Chips/wafer × Architecture efficiency
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Wafer starts */}
            <div>
              <div className="text-sm text-gray-600 mb-2">Wafer starts per month</div>
              <div className="h-48">
                <TimeSeriesChart
                  years={fab?.years}
                  median={fab?.wafer_starts?.median}
                  p25={fab?.wafer_starts?.p25}
                  p75={fab?.wafer_starts?.p75}
                  color={COLOR_PALETTE.fab}
                  yLabel="Wafers"
                  isLoading={isLoading}
                />
              </div>
            </div>

            {/* Chips per wafer */}
            <div>
              <div className="text-sm text-gray-600 mb-2">Chips per wafer</div>
              <div className="h-48">
                <TimeSeriesChart
                  years={fab?.years}
                  median={fab?.chips_per_wafer?.median}
                  p25={fab?.chips_per_wafer?.p25}
                  p75={fab?.chips_per_wafer?.p75}
                  color={COLOR_PALETTE.fab}
                  yLabel="Chips"
                  isLoading={isLoading}
                />
              </div>
            </div>

            {/* Architecture efficiency */}
            <div>
              <div className="text-sm text-gray-600 mb-2">Architecture efficiency</div>
              <div className="h-48">
                <TimeSeriesChart
                  years={fab?.years}
                  median={fab?.architecture_efficiency?.median}
                  p25={fab?.architecture_efficiency?.p25}
                  p75={fab?.architecture_efficiency?.p75}
                  color={COLOR_PALETTE.fab}
                  yLabel="Relative to H100"
                  isLoading={isLoading}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Fab is operational */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="text-sm font-medium text-gray-700 mb-2">
          Proportion of simulations where fab is operational
        </div>
        <div className="h-48">
          <TimeSeriesChart
            years={fab?.years}
            median={fab?.is_operational?.proportion}
            color={COLOR_PALETTE.fab}
            yLabel="Proportion"
            isLoading={isLoading}
            showBand={false}
          />
        </div>
        <div className="text-xs text-gray-500 mt-2">
          Fab becomes operational after construction duration (varies by process node and productivity)
        </div>
      </div>

      {/* Fab Detection LR Components */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-800">
          Fab detection likelihood ratios
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Combined LR */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="text-sm font-medium text-gray-700 mb-2">
              Combined fab LR over time
            </div>
            <div className="h-48">
              <TimeSeriesChart
                years={fab?.years}
                median={fab?.lr_combined?.median}
                p25={fab?.lr_combined?.p25}
                p75={fab?.lr_combined?.p75}
                color={COLOR_PALETTE.fab}
                yLabel="LR"
                isLoading={isLoading}
                yLogScale={true}
              />
            </div>
          </div>

          {/* Other Intel LR */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="text-sm font-medium text-gray-700 mb-2">
              LR from other intelligence (worker detection)
            </div>
            <div className="h-48">
              <TimeSeriesChart
                years={fab?.years}
                median={fab?.lr_other?.median}
                p25={fab?.lr_other?.p25}
                p75={fab?.lr_other?.p75}
                color={COLOR_PALETTE.gray}
                yLabel="LR"
                isLoading={isLoading}
                yLogScale={true}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default CovertFabSection;
