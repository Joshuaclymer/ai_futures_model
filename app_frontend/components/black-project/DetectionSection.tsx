'use client';

import { TimeSeriesChart } from './charts/TimeSeriesChart';
import { BlackProjectData, COLOR_PALETTE, hexToRgba } from '@/types/blackProject';

interface DetectionSectionProps {
  data: BlackProjectData | null;
  isLoading: boolean;
}

export function DetectionSection({ data, isLoading }: DetectionSectionProps) {
  const model = data?.black_project_model;

  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-800 mb-2">
          Strength of evidence breakdown
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          The US intelligence community updates its probability assessment that the PRC
          is running a covert AI project based on multiple sources of evidence. Each source
          contributes a likelihood ratio (LR) that multiplicatively updates the prior.
        </p>
      </div>

      {/* LR Equation Breakdown */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="text-sm font-medium text-gray-700 mb-3">
          Total evidence = LR<sub>initial stock</sub> × LR<sub>SME</sub> × LR<sub>other intel</sub>
        </div>
        <div className="text-xs text-gray-500">
          Each LR represents how much more likely the observed evidence is if the PRC
          has a covert project compared to if they don&apos;t.
        </div>
      </div>

      {/* LR Components Over Time */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Cumulative LR */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-2">
            Cumulative likelihood ratio over time
          </div>
          <div className="h-64">
            <TimeSeriesChart
              years={model?.years}
              median={model?.cumulative_lr?.median}
              p25={model?.cumulative_lr?.p25}
              p75={model?.cumulative_lr?.p75}
              color={COLOR_PALETTE.detection}
              yLabel="Likelihood ratio"
              isLoading={isLoading}
              yLogScale={true}
            />
          </div>
        </div>

        {/* Posterior Probability */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-2">
            Posterior probability of covert project
          </div>
          <div className="h-64">
            <TimeSeriesChart
              years={model?.years}
              median={model?.posterior_prob_project?.median}
              p25={model?.posterior_prob_project?.p25}
              p75={model?.posterior_prob_project?.p75}
              color={COLOR_PALETTE.detection}
              yLabel="P(covert project exists)"
              isLoading={isLoading}
            />
          </div>
        </div>
      </div>

      {/* Individual LR Components */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-800">
          LR component breakdown
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* LR Initial Stock */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="text-sm font-medium text-gray-700 mb-2">
              LR from initial chip stock diversion
            </div>
            <div className="h-48">
              <TimeSeriesChart
                years={model?.years}
                median={model?.lr_initial_stock?.median}
                p25={model?.lr_initial_stock?.p25}
                p75={model?.lr_initial_stock?.p75}
                color={COLOR_PALETTE.chip_stock}
                yLabel="LR"
                isLoading={isLoading}
              />
            </div>
            <div className="text-xs text-gray-500 mt-2">
              Based on PRC compute accounting discrepancies
            </div>
          </div>

          {/* LR SME */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="text-sm font-medium text-gray-700 mb-2">
              LR from SME equipment analysis
            </div>
            <div className="h-48">
              <TimeSeriesChart
                years={model?.years}
                median={model?.lr_diverted_sme?.median}
                p25={model?.lr_diverted_sme?.p25}
                p75={model?.lr_diverted_sme?.p75}
                color={COLOR_PALETTE.fab}
                yLabel="LR"
                isLoading={isLoading}
              />
            </div>
            <div className="text-xs text-gray-500 mt-2">
              Based on semiconductor manufacturing equipment tracking
            </div>
          </div>

          {/* LR Other Intel */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="text-sm font-medium text-gray-700 mb-2">
              LR from other intelligence sources
            </div>
            <div className="h-48">
              <TimeSeriesChart
                years={model?.years}
                median={model?.lr_other_intel?.median}
                p25={model?.lr_other_intel?.p25}
                p75={model?.lr_other_intel?.p75}
                color={COLOR_PALETTE.gray}
                yLabel="LR"
                isLoading={isLoading}
                yLogScale={true}
              />
            </div>
            <div className="text-xs text-gray-500 mt-2">
              Worker detection model (time-varying)
            </div>
          </div>
        </div>
      </div>

      {/* Additional LR Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* LR Energy */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-2">
            LR from reported energy consumption
          </div>
          <div className="h-48">
            <TimeSeriesChart
              years={model?.years}
              median={model?.lr_reported_energy?.median}
              p25={model?.lr_reported_energy?.p25}
              p75={model?.lr_reported_energy?.p75}
              color={COLOR_PALETTE.datacenters_and_energy}
              yLabel="LR"
              isLoading={isLoading}
            />
          </div>
          <div className="text-xs text-gray-500 mt-2">
            Grows over time as covert datacenter capacity increases
          </div>
        </div>

        {/* LR Satellite */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-2">
            LR from satellite surveillance
          </div>
          <div className="h-48 flex items-center justify-center">
            {isLoading ? (
              <div className="animate-pulse bg-gray-200 w-full h-full rounded" />
            ) : model?.lr_satellite_datacenter?.individual ? (
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-800">
                  {model.lr_satellite_datacenter.individual.length > 0
                    ? (model.lr_satellite_datacenter.individual.reduce((a, b) => a + b, 0) /
                       model.lr_satellite_datacenter.individual.length).toFixed(2)
                    : '1.00'}
                </div>
                <div className="text-sm text-gray-500">median LR</div>
              </div>
            ) : (
              <div className="text-gray-400 text-sm">No data</div>
            )}
          </div>
          <div className="text-xs text-gray-500 mt-2">
            Constant over time (based on diverted datacenter capacity)
          </div>
        </div>
      </div>
    </section>
  );
}

export default DetectionSection;
