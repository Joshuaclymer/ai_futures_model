'use client';

import { useMemo } from 'react';
import { CCDFChart } from '../charts';
import { COLOR_PALETTE } from '../colors';
import { SimulationData, MultiThresholdCCDF } from '../../types';

interface TopChartsSectionProps {
  data: SimulationData | null;
  isLoading: boolean;
  agreementYear: number;
}

interface DashboardValues {
  medianH100Years: string;
  medianTimeToDetection: string;
  aiRdReduction: string;
  chipsProduced: string;
}

const CHART_HEIGHT = 280;

export function TopChartsSection({ data, isLoading, agreementYear }: TopChartsSectionProps) {
  const dashboardValues = useDashboardValues(data);

  return (
    <div className="top-charts-section">
      {/* Title and description */}
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-gray-800 mb-1" style={{ fontFamily: 'et-book, Georgia, serif' }}>
          Black project model
        </h1>
        <p className="text-sm text-gray-600">
          This is a model of how much compute a covert AI project might operate if the PRC cheats an AI slowdown agreement.
          Predictions assume an agreement goes into force in{' '}
          <span className="font-semibold text-[#5E6FB8]">{agreementYear}</span>.
        </p>
      </div>

      {/* Top row: Dashboard + 2 main charts */}
      <div className="top-charts-row">
        {/* Dashboard */}
        <Dashboard values={dashboardValues} />

        {/* Covert Compute CCDF */}
        <ChartContainer title="Covert compute">
          <CCDFChart
            data={data?.black_project_model?.h100_years_ccdf}
            color={COLOR_PALETTE.chip_stock}
            xLabel="Covert computation (H100-years)"
            yLabel="P(compute > x)"
            xLogScale={true}
            showArea={false}
            isLoading={isLoading}
            height={CHART_HEIGHT}
            thresholdLabels={{
              '1': 'Detection is >1× LR update        ',
              '2': 'Detection is >2× LR update        ',
              '4': 'Detection is >4× LR update        ',
            }}
          />
        </ChartContainer>

        {/* Time to Detection CCDF */}
        <ChartContainer title="Time to detection (after which agreement ends)">
          <CCDFChart
            data={data?.black_project_model?.time_to_detection_ccdf}
            color={COLOR_PALETTE.detection}
            xLabel="Time to detection (years)"
            yLabel="P(time > x)"
            showArea={false}
            isLoading={isLoading}
            height={CHART_HEIGHT}
            thresholdLabels={{
              '1': 'Detection is >1× LR update        ',
              '2': 'Detection is >2× LR update        ',
              '4': 'Detection is >4× LR update        ',
            }}
          />
        </ChartContainer>
      </div>

      {/* Bottom row: 2 comparison charts */}
      <div className="top-charts-bottom-row">
        {/* Chip Production CCDF */}
        <ChartContainer title="Covert AI chip production relative to no slowdown production*">
          <CCDFChart
            data={data?.black_project_model?.chip_production_reduction_ccdf}
            color={COLOR_PALETTE.fab}
            xLabel="Covert chip production during agreement / chip production if there was no agreement"
            yLabel="P(Ratio < x)"
            xAsInverseFraction
            showArea={false}
            isLoading={isLoading}
            height={CHART_HEIGHT}
            thresholdLabels={{
              'global': 'Relative to global production (no slowdown)    ',
              'prc': 'Relative to PRC production (no slowdown)    ',
            }}
            thresholdColors={{
              'global': '#9B8AC4',
              'prc': '#5E6FB8',
            }}
            referenceLine={{
              y: 0.85,
              label: 'No covert chip production    ',
              color: '#888888',
              dash: 'dot',
            }}
          />
        </ChartContainer>

        {/* AI R&D Reduction CCDF */}
        <ChartContainer title="Covert AI R&D computation relative to no slowdown computation*">
          <CCDFChart
            data={data?.black_project_model?.ai_rd_reduction_ccdf}
            color={COLOR_PALETTE.datacenters_and_energy}
            xLabel="Covert computation during agreement / computation if there was no agreement"
            yLabel="P(Ratio < x)"
            xAsInverseFraction
            showArea={false}
            isLoading={isLoading}
            height={CHART_HEIGHT}
            thresholdLabels={{
              'largest_ai_company': 'Relative to largest AI company (no slowdown)    ',
              'prc': 'Relative to PRC (no slowdown)    ',
            }}
            thresholdColors={{
              'largest_ai_company': '#7A9EC2',
              'prc': '#5E6FB8',
            }}
            referenceLine={{
              y: 0.20,
              label: 'No covert computation    ',
              color: '#888888',
              dash: 'dot',
            }}
          />
        </ChartContainer>
      </div>

      <p className="text-xs text-gray-500 italic mt-2">
        *Unless otherwise specified, US intelligence &apos;detects&apos; a covert project after it receives &gt;4x update that the project exists, after which, USG exits the AI slowdown agreement.
      </p>
    </div>
  );
}

function Dashboard({ values }: { values: DashboardValues }) {
  return (
    <div className="top-charts-dashboard">
      <div className="top-charts-title">Median outcome</div>
      <DashboardItem value={values.medianH100Years} label="Covert computation*" />
      <DashboardItem value={values.medianTimeToDetection} label="Time to detection*" />
      <DashboardItem
        value={values.aiRdReduction}
        label="Reduction in AI R&D computation of largest company*"
      />
      <DashboardItem value={values.chipsProduced} label="Chips covertly produced*" isLast />
    </div>
  );
}

function DashboardItem({
  value,
  label,
  isLast = false,
}: {
  value: string;
  label: string;
  isLast?: boolean;
}) {
  return (
    <div className={`top-charts-dashboard-item ${isLast ? 'last' : ''}`}>
      <div className="top-charts-dashboard-value">{value}</div>
      <div className="top-charts-dashboard-label">{label}</div>
    </div>
  );
}

function ChartContainer({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="top-charts-chart-container">
      <div className="top-charts-title">{title}</div>
      <div className="top-charts-chart">
        {children}
      </div>
    </div>
  );
}

function useDashboardValues(data: SimulationData | null): DashboardValues {
  return useMemo(() => {
    if (!data?.black_project_model) {
      return {
        medianH100Years: '--',
        medianTimeToDetection: '--',
        aiRdReduction: '--',
        chipsProduced: '--',
      };
    }

    const model = data.black_project_model;

    const formatNumber = (n: number) => {
      if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
      if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
      if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
      return n.toFixed(1);
    };

    const getMedian = (arr: number[] | undefined): number | null => {
      if (!arr || arr.length === 0) return null;
      const sorted = [...arr].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };

    const h100YearsMedian = getMedian(model.individual_project_h100_years_before_detection);
    const timeMedian = getMedian(model.individual_project_time_before_detection);
    const h100eMedian = getMedian(model.individual_project_h100e_before_detection);
    const aiRdReductionData = model.ai_rd_reduction_median;

    return {
      medianH100Years: h100YearsMedian !== null
        ? `${formatNumber(h100YearsMedian)} H100-years`
        : '--',
      medianTimeToDetection: timeMedian !== null
        ? `${timeMedian.toFixed(1)} years`
        : '--',
      aiRdReduction: aiRdReductionData !== undefined
        ? `${(aiRdReductionData * 100).toFixed(1)}%`
        : '~5%',
      chipsProduced: h100eMedian !== null
        ? formatNumber(h100eMedian)
        : '--',
    };
  }, [data]);
}
