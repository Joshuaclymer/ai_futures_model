'use client';

import { TimeSeriesChart } from './charts/TimeSeriesChart';
import { PlotlyChart } from './charts/PlotlyChart';
import { BlackProjectData, COLOR_PALETTE, hexToRgba } from '@/types/blackProject';

interface InitialStockSectionProps {
  data: BlackProjectData | null;
  isLoading: boolean;
  diversionProportion: number;
}

export function InitialStockSection({ data, isLoading, diversionProportion }: InitialStockSectionProps) {
  const model = data?.black_project_model;
  const initialStock = data?.initial_stock;
  const initialProject = data?.initial_black_project;

  // Format numbers for display
  const formatNumber = (n: number) => {
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return n.toFixed(0);
  };

  // Calculate median initial stock
  const medianInitialPRCStock = initialStock?.initial_prc_stock_samples?.length
    ? initialStock.initial_prc_stock_samples.sort((a, b) => a - b)[
        Math.floor(initialStock.initial_prc_stock_samples.length / 2)
      ]
    : 0;

  const medianInitialDiverted = initialStock?.initial_compute_stock_samples?.length
    ? initialStock.initial_compute_stock_samples.sort((a, b) => a - b)[
        Math.floor(initialStock.initial_compute_stock_samples.length / 2)
      ]
    : 0;

  // Histogram data for initial diverted compute
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const histogramData: any[] = initialStock?.initial_compute_stock_samples
    ? [
        {
          x: initialStock.initial_compute_stock_samples,
          type: 'histogram',
          marker: { color: COLOR_PALETTE.chip_stock },
          nbinsx: 30,
          hovertemplate: '%{x:.2f}: %{y} samples<extra></extra>',
        },
      ]
    : [];

  // LR distribution histogram
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const lrHistogramData: any[] = initialStock?.lr_prc_accounting_samples
    ? [
        {
          x: initialStock.lr_prc_accounting_samples,
          type: 'histogram',
          marker: { color: COLOR_PALETTE.detection },
          nbinsx: 30,
          hovertemplate: 'LR=%{x:.2f}: %{y} samples<extra></extra>',
        },
      ]
    : [];

  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-800 mb-2">
          Initial dark compute stock
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          At the start of the agreement, the PRC diverts a fraction of its existing AI
          compute stock to the covert project. This section models the amount of compute
          diverted and its detectability through accounting discrepancies.
        </p>
      </div>

      {/* Diversion equation */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="text-sm font-medium text-gray-700 mb-3">
          Initial diverted compute = PRC total compute × Diversion proportion
        </div>
        <div className="flex items-center gap-4 text-sm text-gray-600">
          <div className="bg-gray-50 px-3 py-2 rounded">
            {formatNumber(medianInitialDiverted)} H100e
          </div>
          <span>=</span>
          <div className="bg-gray-50 px-3 py-2 rounded">
            {formatNumber(medianInitialPRCStock)} H100e
          </div>
          <span>×</span>
          <div className="bg-gray-50 px-3 py-2 rounded">
            {(diversionProportion * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Distribution of initial compute */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Initial PRC compute distribution */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-2">
            Distribution of initial diverted compute
          </div>
          <div className="h-48">
            <PlotlyChart
              data={histogramData}
              layout={{
                margin: { l: 50, r: 20, t: 10, b: 50 },
                xaxis: {
                  title: { text: 'H100-equivalents', font: { size: 11 } },
                  tickfont: { size: 10 },
                },
                yaxis: {
                  title: { text: 'Count', font: { size: 11 } },
                  tickfont: { size: 10 },
                },
                showlegend: false,
              }}
              isLoading={isLoading}
              isEmpty={!initialStock?.initial_compute_stock_samples?.length}
            />
          </div>
          <div className="text-xs text-gray-500 mt-2">
            Uncertainty in PRC compute stock at agreement year
          </div>
        </div>

        {/* LR from PRC accounting */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-2">
            LR from PRC compute accounting discrepancy
          </div>
          <div className="h-48">
            <PlotlyChart
              data={lrHistogramData}
              layout={{
                margin: { l: 50, r: 20, t: 10, b: 50 },
                xaxis: {
                  title: { text: 'Likelihood ratio', font: { size: 11 } },
                  tickfont: { size: 10 },
                },
                yaxis: {
                  title: { text: 'Count', font: { size: 11 } },
                  tickfont: { size: 10 },
                },
                showlegend: false,
              }}
              isLoading={isLoading}
              isEmpty={!initialStock?.lr_prc_accounting_samples?.length}
            />
          </div>
          <div className="text-xs text-gray-500 mt-2">
            Based on US intelligence estimate of PRC compute stock
          </div>
        </div>
      </div>

      {/* Detection probabilities */}
      {initialStock?.initial_black_project_detection_probs && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-3">
            Probability of detection from initial stock diversion alone
          </div>
          <div className="flex flex-wrap gap-6">
            {Object.entries(initialStock.initial_black_project_detection_probs).map(
              ([threshold, prob]) => (
                <div key={threshold} className="text-center">
                  <div className="text-xl font-bold text-gray-800">
                    {((prob as number) * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">
                    LR &gt;= {threshold}
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      )}

      {/* Survival over time */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-800">
          Chip survival and compute decay
        </h3>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-3">
            Dark compute = Initial diverted × Survival rate + Fab production × Survival rate
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Survival rate */}
            <div>
              <div className="text-sm text-gray-600 mb-2">Chip survival rate over time</div>
              <div className="h-48">
                <TimeSeriesChart
                  years={model?.years}
                  median={model?.survival_rate?.median}
                  p25={model?.survival_rate?.p25}
                  p75={model?.survival_rate?.p75}
                  color={COLOR_PALETTE.survival_rate}
                  yLabel="Survival rate"
                  isLoading={isLoading}
                />
              </div>
              <div className="text-xs text-gray-500 mt-2">
                Chips fail over time due to attrition (maintenance, heat, etc.)
              </div>
            </div>

            {/* Surviving dark compute */}
            <div>
              <div className="text-sm text-gray-600 mb-2">Total surviving dark compute</div>
              <div className="h-48">
                <TimeSeriesChart
                  years={model?.years}
                  median={model?.total_black_project?.median}
                  p25={model?.total_black_project?.p25}
                  p75={model?.total_black_project?.p75}
                  color={COLOR_PALETTE.chip_stock}
                  yLabel="H100e"
                  isLoading={isLoading}
                />
              </div>
              <div className="text-xs text-gray-500 mt-2">
                Before capacity limit from datacenters
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Compute stock breakdown */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="text-sm font-medium text-gray-700 mb-3">
          Dark compute stock by source
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Initial stock contribution */}
          <div>
            <div className="text-sm text-gray-600 mb-2">From initial diversion</div>
            <div className="h-48">
              <TimeSeriesChart
                years={model?.years}
                median={model?.initial_black_project?.median}
                p25={model?.initial_black_project?.p25}
                p75={model?.initial_black_project?.p75}
                color={COLOR_PALETTE.chip_stock}
                yLabel="H100e"
                isLoading={isLoading}
              />
            </div>
          </div>

          {/* Fab production contribution */}
          <div>
            <div className="text-sm text-gray-600 mb-2">From covert fab production</div>
            <div className="h-48">
              <TimeSeriesChart
                years={model?.years}
                median={model?.black_fab_flow_all_sims?.median}
                p25={model?.black_fab_flow_all_sims?.p25}
                p75={model?.black_fab_flow_all_sims?.p75}
                color={COLOR_PALETTE.fab}
                yLabel="H100e (cumulative)"
                isLoading={isLoading}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default InitialStockSection;
