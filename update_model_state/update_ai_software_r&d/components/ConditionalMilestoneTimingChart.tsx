'use client';

import { memo, useMemo, ReactNode } from 'react';
import { CustomLineChart, DataPoint } from './CustomLineChart';
import { formatTo3SigFigs } from '@/utils/formatting';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';

export interface ConditionalTimingPoint {
  timeFromAC: number;
  probabilityDensity: number;
}

export interface ConditionalTimingData {
  milestones: { [key: string]: ConditionalTimingPoint[] };
  statistics: {
    [key: string]: {
      achievementRate: number;
      mode: number;
      p10: number;
      p50: number;
      p90: number;
    };
  };
  conditionDescription: string;
}

interface ConditionalMilestoneTimingChartProps {
  data: ConditionalTimingData;
  className?: string;
  title?: string;
  height?: number;
  maxTimeYears?: number;
  width?: number;
  sharedYDomain?: [number, number];
  showLegend?: boolean;
}

// Color palette for different milestones (matching MilestoneDistributionChart)
const MILESTONE_COLORS: { [key: string]: string } = {
  'AI2027-SC': '#af1e86ff',
  'AIR-5x': '#FF6B35',
  'AIR-25x': '#FF8C42',
  'AIR-250x': '#FFAD5A',
  'AIR-2000x': '#FFCE73',
  'AIR-10000x': '#FFE98C',
  'SAR-level-experiment-selection-skill': '#000090',
  'SIAR-level-experiment-selection-skill': '#900000',
  'STRAT-AI': '#00A896',
  'TED-AI': '#2A623D',
  'ASI': '#af1e86ff',
};

// Display names for milestones
const MILESTONE_DISPLAY_NAMES: { [key: string]: string } = {
  'AI2027-SC': 'SC',
  'AIR-5x': 'AIR 5x',
  'AIR-25x': 'AIR 25x',
  'AIR-250x': 'AIR 250x',
  'AIR-2000x': 'AIR 2000x',
  'AIR-10000x': 'AIR 10000x',
  'SAR-level-experiment-selection-skill': 'SAR (Superhuman AI Researcher)',
  'SIAR-level-experiment-selection-skill': 'SIAR (Superintelligent AI Researcher)',
  'STRAT-AI': 'STRAT-AI',
  'TED-AI': 'TED-AI (Top Expert Dominating AI)',
  'ASI': 'ASI (Artifical Superintelligence)',
};

const ConditionalMilestoneTimingChart = memo(({
  data,
  className,
  title = 'Time Until Milestones (Conditional)',
  height = 400,
  maxTimeYears,
  width = 1000,
  sharedYDomain,
  showLegend = true
}: ConditionalMilestoneTimingChartProps) => {
  const { chartData, lines } = useMemo(() => {
    // Calculate scaled densities for each milestone to match p50 medians
    const scaledDensities: Record<string, Map<number, number>> = {};

    for (const milestoneName of Object.keys(data.milestones)) {
      const stats = data.statistics[milestoneName];
      const points = [...data.milestones[milestoneName]].sort((a, b) => a.timeFromAC - b.timeFromAC);

      if (points.length === 0) {
        scaledDensities[milestoneName] = new Map();
        continue;
      }

      const totalDensity = points.reduce((sum, point) => sum + point.probabilityDensity, 0);

      if (!Number.isFinite(totalDensity) || totalDensity <= 0) {
        scaledDensities[milestoneName] = new Map();
        continue;
      }

      // Find the empirical median from the raw PDF
      let running = 0;
      let empiricalMedianTime = points[0].timeFromAC;
      for (const point of points) {
        running += point.probabilityDensity / totalDensity;
        if (running >= 0.5) {
          empiricalMedianTime = point.timeFromAC;
          break;
        }
      }

      // Calculate scaling factor to shift the median
      const targetMedian = stats?.p50;
      const hasMedian = typeof targetMedian === 'number' && Number.isFinite(targetMedian);

      if (!hasMedian || empiricalMedianTime === 0) {
        // No scaling needed, just normalize
        const densityMap = new Map<number, number>();
        for (const point of points) {
          densityMap.set(point.timeFromAC, point.probabilityDensity / totalDensity);
        }
        scaledDensities[milestoneName] = densityMap;
        continue;
      }

      // Scale the time axis so empirical median becomes target median
      const timeScaleFactor = targetMedian / empiricalMedianTime;

      // Create new scaled density map
      const densityMap = new Map<number, number>();
      for (const point of points) {
        const scaledTime = point.timeFromAC * timeScaleFactor;
        // Also need to scale the density by the inverse to preserve total probability
        const scaledDensity = (point.probabilityDensity / totalDensity) / timeScaleFactor;
        densityMap.set(scaledTime, scaledDensity);
      }

      scaledDensities[milestoneName] = densityMap;
    }

    // Combine all milestone data into a single dataset
    const allTimes = new Set<number>();
    for (const milestoneName of Object.keys(data.milestones)) {
      const densityMap = scaledDensities[milestoneName];
      if (densityMap) {
        for (const time of densityMap.keys()) {
          allTimes.add(time);
        }
      }
    }

    // Create a data point for each time with scaled densities
    const sortedTimes = Array.from(allTimes).sort((a, b) => a - b);
    const combinedData = sortedTimes.map(time => {
      const dataPoint: DataPoint = { x: time };

      for (const milestoneName of Object.keys(data.milestones)) {
        const densityMap = scaledDensities[milestoneName];
        const value = densityMap?.get(time);
        // Only set the value if it exists, otherwise leave as null
        // This allows the smooth curve to only render where there's actual data
        dataPoint[milestoneName] = value !== undefined ? value : null;
      }

      return dataPoint;
    });

    // Create line configurations for each milestone
    const lineConfigs = Object.keys(data.milestones).map(milestoneName => ({
      dataKey: milestoneName,
      stroke: MILESTONE_COLORS[milestoneName] || '#666666',
      strokeWidth: 2.5,
      name: MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName,
      smooth: true,
    }));

    return { chartData: combinedData, lines: lineConfigs };
  }, [data]);

  const xDomain = useMemo<[number, number]>(() => {
    if (chartData.length === 0) {
      return [0, maxTimeYears ?? 10];
    }

    const times = chartData.map(point => point.x);
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);

    // Cap the display at specified max or 50 years by default
    const cappedMaxTime = Math.min(maxTime, maxTimeYears ?? 50);
    const span = Math.max(cappedMaxTime - minTime, 1);
    const padding = span * 0.02;

    return [Math.max(minTime - padding, 0), cappedMaxTime + padding];
  }, [chartData, maxTimeYears]);

  const yDomain = useMemo<[number, number]>(() => {
    // Use shared domain if provided, otherwise calculate local domain
    if (sharedYDomain) {
      return sharedYDomain;
    }

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
  }, [chartData, data.milestones, sharedYDomain]);

  const getNormalizedCdfAtTime = (
    targetTime: number,
    points: ConditionalTimingPoint[],
    normalizedValues: number[],
  ): number => {
    if (points.length === 0) {
      return 0;
    }

    if (targetTime <= points[0].timeFromAC) {
      return normalizedValues[0];
    }

    for (let i = 1; i < points.length; i += 1) {
      const currentPoint = points[i];
      if (targetTime <= currentPoint.timeFromAC) {
        const previousPoint = points[i - 1];
        const span = currentPoint.timeFromAC - previousPoint.timeFromAC;

        if (span <= 0) {
          return normalizedValues[i];
        }

        const ratio = (targetTime - previousPoint.timeFromAC) / span;
        return normalizedValues[i - 1] + (normalizedValues[i] - normalizedValues[i - 1]) * ratio;
      }
    }

    return 1;
  };

  const cumulativeProbabilityLookup = useMemo(() => {
    const lookup: Record<string, Map<number, number>> = {};

    for (const milestoneName of Object.keys(data.milestones)) {
      const stats = data.statistics[milestoneName];
      const points = [...data.milestones[milestoneName]].sort((a, b) => a.timeFromAC - b.timeFromAC);

      if (points.length === 0) {
        lookup[milestoneName] = new Map();
        continue;
      }

      const totalDensity = points.reduce((sum, point) => sum + point.probabilityDensity, 0);

      if (!Number.isFinite(totalDensity) || totalDensity <= 0) {
        lookup[milestoneName] = new Map();
        continue;
      }

      // Find the empirical median from the raw PDF
      let running = 0;
      let empiricalMedianTime = points[0].timeFromAC;
      for (const point of points) {
        running += point.probabilityDensity / totalDensity;
        if (running >= 0.5) {
          empiricalMedianTime = point.timeFromAC;
          break;
        }
      }

      // Calculate scaling factor to shift the median
      const targetMedian = stats?.p50;
      const hasMedian = typeof targetMedian === 'number' && Number.isFinite(targetMedian);

      if (!hasMedian || empiricalMedianTime === 0) {
        // No scaling, just build normalized CDF
        const probabilityMap = new Map<number, number>();
        let cumulativeProb = 0;
        for (const point of points) {
          cumulativeProb += point.probabilityDensity / totalDensity;
          probabilityMap.set(point.timeFromAC, cumulativeProb);
        }
        lookup[milestoneName] = probabilityMap;
        continue;
      }

      // Scale the time axis and build CDF from scaled distribution
      const timeScaleFactor = targetMedian / empiricalMedianTime;
      const probabilityMap = new Map<number, number>();
      let cumulativeProb = 0;

      for (const point of points) {
        const scaledTime = point.timeFromAC * timeScaleFactor;
        cumulativeProb += point.probabilityDensity / totalDensity;
        probabilityMap.set(scaledTime, cumulativeProb);
      }

      lookup[milestoneName] = probabilityMap;
    }

    return lookup;
  }, [data]);

  const formatYears = (value: number): string => {
    if (!Number.isFinite(value)) {
      return '0';
    }

    // For values less than 1, show one decimal place
    if (value < 1) {
      return value.toFixed(1);
    }

    return Math.round(value).toString();
  };

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
        <span style={tooltipHeaderStyle}>{formatYears(point.x)} years from AC</span>
        <div style={{ fontSize: '10px', color: '#666', marginBottom: '4px' }}>
          Cumulative Probability
        </div>
        {Object.keys(data.milestones).map(milestoneName => {
          const displayName = MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName;
          const color = MILESTONE_COLORS[milestoneName] || '#666666';

          const probabilityMap = cumulativeProbabilityLookup[milestoneName];
          let probability = probabilityMap?.get(point.x);

          if (probability === undefined && probabilityMap) {
            // Interpolate between the closest points in the scaled lookup
            const sortedTimes = Array.from(probabilityMap.keys()).sort((a, b) => a - b);

            if (sortedTimes.length === 0) {
              probability = 0;
            } else if (point.x <= sortedTimes[0]) {
              probability = probabilityMap.get(sortedTimes[0]) ?? 0;
            } else if (point.x >= sortedTimes[sortedTimes.length - 1]) {
              probability = probabilityMap.get(sortedTimes[sortedTimes.length - 1]) ?? 1;
            } else {
              // Find the two closest points and interpolate
              let lowerTime = sortedTimes[0];
              let upperTime = sortedTimes[0];

              for (let i = 0; i < sortedTimes.length - 1; i++) {
                if (sortedTimes[i] <= point.x && sortedTimes[i + 1] >= point.x) {
                  lowerTime = sortedTimes[i];
                  upperTime = sortedTimes[i + 1];
                  break;
                }
              }

              const lowerProb = probabilityMap.get(lowerTime) ?? 0;
              const upperProb = probabilityMap.get(upperTime) ?? 1;

              if (upperTime - lowerTime > 0) {
                const ratio = (point.x - lowerTime) / (upperTime - lowerTime);
                probability = lowerProb + (upperProb - lowerProb) * ratio;
              } else {
                probability = lowerProb;
              }
            }
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

      <div className="text-sm text-gray-600 mb-4">
        {data.conditionDescription}
      </div>

      {/* Legend */}
      {showLegend && (
        <div className="flex flex-wrap gap-4 mb-4 text-xs">
          {Object.keys(data.milestones).map(milestoneName => {
            const displayName = MILESTONE_DISPLAY_NAMES[milestoneName] || milestoneName;
            const color = MILESTONE_COLORS[milestoneName] || '#666666';

            return (
              <div key={milestoneName} className="flex items-center gap-2">
                <div className="w-8 h-0.5" style={{ backgroundColor: color }} />
                <span className="font-medium" style={{ color }}>
                  {displayName}
                </span>
              </div>
            );
          })}
        </div>
      )}

      <div style={{ overflow: 'hidden' }}>
        <CustomLineChart
          data={chartData}
          height={height}
          width={width}
          margin={{ top: 0, right: 40, left: 60, bottom: 60 }}
          xDomain={xDomain}
          xTickCount={6}
          xTickFormatter={(value) => `${formatYears(value)} yr`}
          yDomain={yDomain}
          yTickFormatter={(value) => formatTo3SigFigs(value)}
          showYAxis={false}
          lines={lines}
          tooltip={tooltip}
        />
      </div>
    </div>
  );
});

ConditionalMilestoneTimingChart.displayName = 'ConditionalMilestoneTimingChart';

export default ConditionalMilestoneTimingChart;
