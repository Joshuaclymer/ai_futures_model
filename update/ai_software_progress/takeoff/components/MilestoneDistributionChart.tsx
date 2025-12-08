'use client';

import { memo, useMemo, ReactNode } from 'react';
import { CustomLineChart, DataPoint } from './CustomLineChart';
import { formatTo3SigFigs, formatYearMonth } from '@/utils/formatting';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';

export interface MilestoneDistributionPoint {
  year: number;
  probabilityDensity: number;
}

export interface MilestoneDistributionData {
  milestones: { [key: string]: MilestoneDistributionPoint[] };
  statistics: {
    [key: string]: {
      achievementRate: number;
      mode: number;
      p10: number;
      p50: number;
      p90: number;
    };
  };
}

interface MilestoneDistributionChartProps {
  data: MilestoneDistributionData;
  className?: string;
  title?: string;
  height?: number;
}

// Color palette for different milestones
const MILESTONE_COLORS: { [key: string]: string } = {
  'AC': '#2A623D',
  'AI2027-SC': '#af1e86ff',
  'SAR-level-experiment-selection-skill': '#000090',
  'SIAR-level-experiment-selection-skill': '#900000',
};

// Display names for milestones
const MILESTONE_DISPLAY_NAMES: { [key: string]: string } = {
  'AC': 'AC (Automated Coder)',
  'AI2027-SC': 'SC',
  'SAR-level-experiment-selection-skill': 'SAR (Superhuman AI Researcher)',
  'SIAR-level-experiment-selection-skill': 'SIAR (Superintelligent AI Researcher)',
};

const getNormalizedCdfAtYear = (
  targetYear: number,
  points: MilestoneDistributionPoint[],
  normalizedValues: number[],
): number => {
  if (points.length === 0) {
    return 0;
  }

  if (targetYear <= points[0].year) {
    return normalizedValues[0];
  }

  for (let i = 1; i < points.length; i += 1) {
    const currentPoint = points[i];
    if (targetYear <= currentPoint.year) {
      const previousPoint = points[i - 1];
      const span = currentPoint.year - previousPoint.year;

      if (span <= 0) {
        return normalizedValues[i];
      }

      const ratio = (targetYear - previousPoint.year) / span;
      return normalizedValues[i - 1] + (normalizedValues[i] - normalizedValues[i - 1]) * ratio;
    }
  }

  return 1;
};

const MilestoneDistributionChart = memo(({ data, className, title = 'Milestone Probability Density', height = 400 }: MilestoneDistributionChartProps) => {
  const { chartData, lines } = useMemo(() => {
    // Combine all milestone data into a single dataset
    const allYears = new Set<number>();

    // Collect all unique years
    for (const milestoneName of Object.keys(data.milestones)) {
      for (const point of data.milestones[milestoneName]) {
        allYears.add(point.year);
      }
    }

    // Create a data point for each year with all milestone densities
    const sortedYears = Array.from(allYears).sort((a, b) => a - b);
    const combinedData = sortedYears.map(year => {
      const dataPoint: DataPoint = { x: year };

      for (const milestoneName of Object.keys(data.milestones)) {
        const point = data.milestones[milestoneName].find(p => p.year === year);
        dataPoint[milestoneName] = point?.probabilityDensity ?? 0;
      }

      return dataPoint;
    });

    // Create line configurations for each milestone
    const lineConfigs = Object.keys(data.milestones).map(milestoneName => ({
      dataKey: milestoneName,
      stroke: MILESTONE_COLORS[milestoneName] || '#666666',
      strokeWidth: 2.5,
      name: MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName,
    }));

    return { chartData: combinedData, lines: lineConfigs };
  }, [data]);

  const xDomain = useMemo<[number, number]>(() => {
    if (chartData.length === 0) {
      return [2020, 2030];
    }

    const years = chartData.map(point => point.x);
    const minYear = Math.min(...years);
    const dataMaxYear = Math.max(...years);

    // Cap the display at 2050, but use actual data range if it's smaller
    const maxYear = Math.min(dataMaxYear, 2050);
    const span = Math.max(maxYear - minYear, 1);
    const padding = span * 0.02;

    return [minYear - padding, Math.min(maxYear + padding, 2050)];
  }, [chartData]);

  const yDomain = useMemo<[number, number]>(() => {
    if (chartData.length === 0) {
      return [0, 1];
    }

    let maxDensity = 0;

    for (const point of chartData) {
      for (const milestoneName of Object.keys(data.milestones)) {
        const density = point[milestoneName] as number;
        if (Number.isFinite(density) && density > maxDensity) {
          maxDensity = density;
        }
      }
    }

    const headroom = maxDensity === 0 ? 0.1 : maxDensity * 0.1;

    return [0, maxDensity + headroom];
  }, [chartData, data.milestones]);

  const cumulativeProbabilityLookup = useMemo(() => {
    const lookup: Record<string, Map<number, number>> = {};

    for (const milestoneName of Object.keys(data.milestones)) {
      const stats = data.statistics[milestoneName];
      const points = [...data.milestones[milestoneName]].sort((a, b) => a.year - b.year);

      if (points.length === 0) {
        lookup[milestoneName] = new Map();
        continue;
      }

      const totalDensity = points.reduce((sum, point) => sum + point.probabilityDensity, 0);

      if (!Number.isFinite(totalDensity) || totalDensity <= 0) {
        lookup[milestoneName] = new Map();
        continue;
      }

      let running = 0;
      const normalizedCdf: number[] = [];

      for (const point of points) {
        running += point.probabilityDensity;
        normalizedCdf.push(running / totalDensity);
      }

      const medianYear = stats?.p50;
      const hasMedian = typeof medianYear === 'number' && Number.isFinite(medianYear);
      const cdfAtMedian = hasMedian ? getNormalizedCdfAtYear(medianYear, points, normalizedCdf) : null;

      let exponent = 1;
      if (cdfAtMedian !== null && cdfAtMedian > 0 && cdfAtMedian < 1) {
        exponent = Math.log(0.5) / Math.log(cdfAtMedian);
      }

      const scale = stats?.achievementRate ?? 1;
      const probabilityMap = new Map<number, number>();

      points.forEach((point, index) => {
        const normalizedValue = Math.min(Math.max(normalizedCdf[index], 0), 1);
        const adjustedValue = normalizedValue === 0 ? 0 : Math.pow(normalizedValue, exponent);
        probabilityMap.set(point.year, Math.min(1, adjustedValue * scale));
      });

      lookup[milestoneName] = probabilityMap;
    }

    return lookup;
  }, [data]);

  const formatProbabilityPercent = (value: number): string => {
    if (!Number.isFinite(value)) {
      return '0.00';
    }

    const text = value.toFixed(2);

    return text === '-0.00' ? '0.00' : text;
  };

  const tooltip = (point: DataPoint): ReactNode => {
    return (
      <div style={tooltipBoxStyle}>
        <span style={tooltipHeaderStyle}>{formatYearMonth(point.x)}</span>
        <div style={{ fontSize: '10px', color: '#666', marginBottom: '4px' }}>
          Cumulative Probability
        </div>
        {Object.keys(data.milestones).map(milestoneName => {
          const displayName = MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName;
          const color = MILESTONE_COLORS[milestoneName] || '#666666';

          const probabilityMap = cumulativeProbabilityLookup[milestoneName];
          let probability = probabilityMap?.get(point.x);

          if (probability === undefined) {
            const milestoneData = data.milestones[milestoneName];
            probability = milestoneData
              .filter(p => p.year <= point.x)
              .reduce((sum, p) => sum + p.probabilityDensity, 0);
          }

          return (
            <div key={milestoneName} style={{ ...tooltipValueStyle, color }}>
              {`${displayName}: ${formatProbabilityPercent((probability ?? 0) * 100)}%`}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className={className || 'flex-1'}>
      <div className="flex gap-2 items-center mb-4">
        <span className="text-[10px] font-semibold text-primary text-left font-system-mono">{title}</span>
        <div className="flex-1 border-t border-gray-500/30" />
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 mb-4 text-xs">
        {Object.keys(data.milestones).map(milestoneName => {
          const displayName = MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName;
          const color = MILESTONE_COLORS[milestoneName] || '#666666';
          const stats = data.statistics[milestoneName];

          return (
            <div key={milestoneName} className="flex items-center gap-2">
              <div className="w-8 h-0.5" style={{ backgroundColor: color }} />
              <span className="font-medium" style={{ color }}>
                {displayName}
              </span>
              {stats && (
                <span className="text-gray-500">
                  (p10: {formatYearMonth(stats.p10)}, p50: {formatYearMonth(stats.p50)}, p90: {formatYearMonth(stats.p90)})
                </span>
              )}
            </div>
          );
        })}
      </div>

      <CustomLineChart
        data={chartData}
        height={height}
        width={1000}
        margin={{ top: 0, right: 40, left: 60, bottom: 60 }}
        xDomain={xDomain}
        xTickCount={6}
        xTickFormatter={(value) => formatYearMonth(value)}
        yDomain={yDomain}
        yTickFormatter={(value) => formatTo3SigFigs(value)}
        showYAxis={false}
        lines={lines}
        tooltip={tooltip}
      />
    </div>
  );
});

MilestoneDistributionChart.displayName = 'MilestoneDistributionChart';

export default MilestoneDistributionChart;
