'use client';

import { useMemo } from 'react';
import { CCDFChart } from '../charts';
import { COLOR_PALETTE } from '../colors';
import { SimulationData, MultiThresholdCCDF, CCDFPoint } from '../../types';
import { formatNumber, formatSigFigs, toSigFigs } from '../../utils/formatters';
import './TopChartsSection.css';

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

// Get the minimum y value from CCDF data (the plateau representing fraction with extreme values)
// For CCDF display: use last point's y (minimum raw y)
// For CDF display (showAsCDF=true): use first point's y, which after 1-y transform gives minimum displayed y
function getMinYFromCCDF(ccdfData: MultiThresholdCCDF | undefined, forCDF: boolean = false): number | null {
  if (!ccdfData) return null;

  if (forCDF) {
    // For CDF display, find the first point's y (highest raw CCDF y)
    // After transformation (1 - y), this becomes the minimum displayed y
    let maxFirstY = 0;
    for (const key of Object.keys(ccdfData)) {
      const points = ccdfData[key] as CCDFPoint[];
      if (points && points.length > 0) {
        const firstPoint = points[0];
        if (firstPoint && firstPoint.y > maxFirstY) {
          maxFirstY = firstPoint.y;
        }
      }
    }
    // Return the raw y value - CCDFChart will transform it with (1 - y)
    return maxFirstY > 0 ? maxFirstY : null;
  } else {
    // For CCDF display, find the last point's y (minimum raw y)
    let minY = 1;
    for (const key of Object.keys(ccdfData)) {
      const points = ccdfData[key] as CCDFPoint[];
      if (points && points.length > 0) {
        const lastPoint = points[points.length - 1];
        if (lastPoint && lastPoint.y < minY) {
          minY = lastPoint.y;
        }
      }
    }
    return minY < 1 ? minY : null;
  }
}

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
        <ChartContainer title="Average covert compute operated before detection">
          <CCDFChart
            data={data?.black_project_model?.average_covert_compute_ccdf}
            color={COLOR_PALETTE.chip_stock}
            xLabel="Average H100e operated by covert project before detection"
            yLabel="P(compute > x)"
            xLogScale={true}
            showArea={false}
            isLoading={isLoading}
            height={CHART_HEIGHT}
            creamBackground={true}
            thresholdLabels={{
              '1': '"Detection" is a >1x update    ',
              '2': '"Detection" is a >2x update    ',
              '4': '"Detection" is a >4x update    ',
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
            creamBackground={true}
            xMax={10}
            thresholdLabels={{
              '1': '"Detection" is a >1x update    ',
              '2': '"Detection" is a >2x update    ',
              '4': '"Detection" is a >4x update    ',
            }}
          />
        </ChartContainer>
      </div>

      {/* Bottom row: 2 comparison charts */}
      <div className="top-charts-bottom-row">
        {/* Chip Production CCDF */}
        <ChartContainer title="Covert AI chip production relative to no slowdown production*">
          <CCDFChart
            data={data?.black_project_model?.chip_production_reduction_ccdf_flat}
            color={COLOR_PALETTE.fab}
            xLabel="Covert chip production during agreement / chip production if there was no agreement"
            yLabel="P(ratio < x)"
            xAsInverseFraction
            xReverse
            showAsCDF
            showArea={false}
            isLoading={isLoading}
            height={CHART_HEIGHT}
            legendPosition="bottom-left"
            creamBackground={true}
            thresholdLabels={{
              'global': 'Relative to global production (no slowdown)       ',
              'prc': 'Relative to PRC production (no slowdown)        ',
            }}
            thresholdColors={{
              'global': '#9B8AC4',
              'prc': '#5E6FB8',
            }}
            referenceLine={
              getMinYFromCCDF(data?.black_project_model?.chip_production_reduction_ccdf_flat, true) !== null
                ? {
                    y: getMinYFromCCDF(data?.black_project_model?.chip_production_reduction_ccdf_flat, true)!,
                    label: 'No covert chip production',
                    color: '#888888',
                    dash: 'dot',
                  }
                : undefined
            }
          />
        </ChartContainer>

        {/* AI R&D Reduction CCDF */}
        <ChartContainer title="Covert AI research computation relative to no slowdown computation*">
          <CCDFChart
            data={data?.black_project_model?.ai_rd_reduction_ccdf_flat}
            color={COLOR_PALETTE.datacenters_and_energy}
            xLabel="Covert computation during agreement / computation if there was no agreement"
            yLabel="P(ratio < x)"
            xAsInverseFraction
            xReverse
            showAsCDF
            showArea={false}
            isLoading={isLoading}
            height={CHART_HEIGHT}
            creamBackground={true}
            thresholdLabels={{
              'global': 'Relative to total global research computation (no slowdown)        ',
              'prc': 'Relative to total PRC research computation (no slowdown)        ',
            }}
            thresholdColors={{
              'global': '#7A9EC2',
              'prc': '#5E6FB8',
            }}
            referenceLine={
              getMinYFromCCDF(data?.black_project_model?.ai_rd_reduction_ccdf_flat, true) !== null
                ? {
                    y: getMinYFromCCDF(data?.black_project_model?.ai_rd_reduction_ccdf_flat, true)!,
                    label: 'No covert computation',
                    color: '#888888',
                    dash: 'dot',
                  }
                : undefined
            }
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
        label="Reduction in total global AI R&D computation*"
      />
      <DashboardItem value={values.chipsProduced} label="H100 equivalents covertly produced*" isLast />
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

    const getMedian = (arr: number[] | undefined): number | null => {
      if (!arr || arr.length === 0) return null;
      const sorted = [...arr].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };

    // Get median from CCDF data (find x where y = 0.5)
    const getMedianFromCcdf = (ccdfPoints: CCDFPoint[] | undefined): number | null => {
      if (!ccdfPoints || ccdfPoints.length === 0) return null;
      // Find the point where y crosses 0.5
      for (let i = 0; i < ccdfPoints.length - 1; i++) {
        const p1 = ccdfPoints[i];
        const p2 = ccdfPoints[i + 1];
        if (p1.y >= 0.5 && p2.y <= 0.5) {
          // Linear interpolation to find x at y = 0.5
          if (p1.y === p2.y) return p1.x;
          const t = (0.5 - p1.y) / (p2.y - p1.y);
          return p1.x + t * (p2.x - p1.x);
        }
      }
      // If y never crosses 0.5, return first or last x
      if (ccdfPoints[0].y < 0.5) return ccdfPoints[0].x;
      return ccdfPoints[ccdfPoints.length - 1].x;
    };

    const h100YearsMedian = getMedian(model.individual_project_h100_years_before_detection);
    const timeMedian = getMedian(model.individual_project_time_before_detection);
    const h100eMedian = getMedian(model.individual_project_h100e_before_detection);

    // Get AI R&D reduction median from CCDF data (global comparison)
    // Use the flat version which has direct CCDFPoint[] arrays
    const aiRdCcdfFlat = model.ai_rd_reduction_ccdf_flat as Record<string, CCDFPoint[]> | undefined;
    const globalCcdf = aiRdCcdfFlat?.global;
    const aiRdReductionMedian = getMedianFromCcdf(globalCcdf);

    return {
      medianH100Years: h100YearsMedian !== null
        ? `${formatNumber(h100YearsMedian)} H100-years`
        : '--',
      medianTimeToDetection: timeMedian !== null
        ? `${formatSigFigs(timeMedian)} years`
        : '--',
      aiRdReduction: aiRdReductionMedian !== null && aiRdReductionMedian > 0
        ? `${toSigFigs(1 / aiRdReductionMedian, 1)}x`
        : '--',
      chipsProduced: h100eMedian !== null
        ? formatNumber(h100eMedian)
        : '--',
    };
  }, [data]);
}
