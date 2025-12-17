import { memo, useMemo, ReactNode, useState, useCallback } from "react";
import { BenchmarkPoint, ChartDataPoint } from "@/app/types";
import { CustomLineChart, DataPoint, CustomElementsContext } from './CustomLineChart';
import {
  createDiamondShape,
  createSquareShape,
  createTriangleShape,
  createStarShape,
  createCircleShape
} from './ChartShapes';
import { createScale, calculateTimeLogTicks, calculateDynamicXDomain, clamp } from '@/utils/chartUtils';
import { formatLogWorkTick, formatWorkTimeDurationDetailed } from '@/utils/formatting';
import { tooltipBoxStyle, tooltipHeaderStyle, tooltipValueStyle } from './chartTooltipStyle';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import { HORIZON_LENGTH_EXPLANATION } from '@/constants/chartExplanations';
import { WithChartTooltip } from './ChartTitleTooltip';

const CODING_AUTOMATION_MARKER_COLOR = '#6b7280';

const CustomHorizonChart = memo(({
  chartData,
  scHorizonMinutes,
  tooltip,
  formatTimeDuration,
  benchmarkData,
  className,
  height = CHART_LAYOUT.primary.height,
  width,
  displayEndYear,
  sampleTrajectories = []
}: {
  chartData: ChartDataPoint[];
  scHorizonMinutes: number;
  tooltip: (point: DataPoint) => ReactNode;
  formatTimeDuration: (minutes: number) => string;
  benchmarkData: BenchmarkPoint[];
  className?: string;
  height?: number;
  width?: number;
  displayEndYear: number;
  sampleTrajectories?: ChartDataPoint[][];
}) => {
  const [hoveredBenchmark, setHoveredBenchmark] = useState<BenchmarkPoint | null>(null);
  const [benchmarkMousePos, setBenchmarkMousePos] = useState<{ x: number; y: number } | null>(null);
  // Calculate Y-axis domain and y-axis ticks based on main trajectory data
  const { horizonDomain, yAxisTicks } = useMemo(() => {
    const validData = chartData.filter(d => d.horizonLength != null && !isNaN(d.horizonLength));
    let domain: [number, number];

    if (validData.length === 0) {
      domain = [
        Math.max(0.1, scHorizonMinutes * 0.1),
        Math.max(scHorizonMinutes * 10, 5.256e9)
      ];
    } else {
      const horizonValues = validData.map(d => d.horizonLength);
      const dataMin = Math.min(...horizonValues);
      const dataMax = Math.max(...horizonValues);
      const tenMillenniaInMinutes = 5.256e9;

      domain = [
        Math.max(0.1, Math.min(dataMin, scHorizonMinutes * 0.1)),
        Math.min(tenMillenniaInMinutes, Math.max(scHorizonMinutes * 10, dataMax))
      ];
    }

    // Calculate y-axis ticks using the new function
    const yTicks = calculateTimeLogTicks(domain, 5, true);

    return { horizonDomain: domain, yAxisTicks: yTicks };
  }, [chartData, scHorizonMinutes]);

  // Calculate X-axis domain from main trajectory
  const xDomain = useMemo((): [number, number] => {
    const years = chartData
      .filter(d => d.horizonLength != null && !isNaN(d.horizonLength))
      .map(d => d.year)
      .filter(y => !isNaN(y));

    if (years.length === 0) return [2020, displayEndYear];

    // Find intersection year where horizon crosses SC threshold
    const intersectionPoint = chartData.find(d =>
      d.horizonLength != null &&
      typeof d.horizonLength === 'number' &&
      d.horizonLength >= scHorizonMinutes
    );
    const intersectionYear = intersectionPoint?.year ?? null;

    return calculateDynamicXDomain(years, {
      intersectionPoints: [intersectionYear],
      minPadding: 0.05,
      maxPadding: 0.15,
      minRange: 3,
      maxDomain: displayEndYear,
    });
  }, [chartData, scHorizonMinutes, displayEndYear]);

  // Generate x-axis ticks at whole year boundaries, limited to ~4 ticks
  const xTicks = useMemo(() => {
    const [start, end] = xDomain;
    const targetTickCount = 4;
    
    // Find all whole years in range
    const firstYear = Math.ceil(start);
    const lastYear = Math.floor(end);
    const yearCount = lastYear - firstYear + 1;
    
    if (yearCount <= 0) {
      return [Math.round(start), Math.round(end)];
    }
    
    if (yearCount <= targetTickCount) {
      // If few enough years, show them all
      const ticks: number[] = [];
      for (let year = firstYear; year <= lastYear; year++) {
        ticks.push(year);
      }
      return ticks;
    }
    
    // Otherwise, sample evenly to get ~targetTickCount ticks
    const step = Math.ceil(yearCount / targetTickCount);
    const ticks: number[] = [];
    for (let year = firstYear; year <= lastYear; year += step) {
      ticks.push(year);
    }
    // Always include the last year if not already included
    if (ticks[ticks.length - 1] !== lastYear) {
      ticks.push(lastYear);
    }
    return ticks;
  }, [xDomain]);

  // Prepare trajectory lines (from main chartData)
  const trajectoryKeys = useMemo(() => {
    if (chartData.length === 0) return [];
    const sample = chartData[0];
    return Object.keys(sample).filter(key => key.startsWith('trajectory'));
  }, [chartData]);

  // Generate keys for sample trajectories
  const sampleKeys = useMemo(() => {
    return sampleTrajectories.map((_, index) => `sample_horizon_${index}`);
  }, [sampleTrajectories]);

  // Merge chartData with sample trajectories
  const mergedData = useMemo<DataPoint[]>(() => {
    // Only use years from main chart data (don't add years from samples)
    return chartData.map(point => {
      const newPoint: DataPoint = { ...point, x: point.year };
      
      // Initialize all sample keys and add their values if available
      sampleTrajectories.forEach((trajectory, sampleIndex) => {
        const key = `sample_horizon_${sampleIndex}`;
        const samplePoint = trajectory.find(p => p.year === point.year);
        // Always set the key, use null if no matching point found
        newPoint[key] = samplePoint?.horizonLength ?? null;
      });
      
      return newPoint;
    });
  }, [chartData, sampleTrajectories]);

  const sanitizedData = useMemo<DataPoint[]>(() => {
    const [xMin, xMax] = xDomain;
    const [yMin, yMax] = horizonDomain;
    const allTrajectoryKeys = [...trajectoryKeys, ...sampleKeys];

    // Track whether each series has already gone above yMax (to stop drawing after first out-of-range point)
    const seriesExceededMax: Record<string, boolean> = {};
    allTrajectoryKeys.forEach(key => { seriesExceededMax[key] = false; });
    let horizonExceededMax = false;

    return mergedData
      .filter(point => {
        const xValue = point.x;
        return typeof xValue === 'number' && Number.isFinite(xValue) && xValue >= xMin && xValue <= xMax;
      })
      .map(point => {
        const sanitizedPoint: DataPoint = { ...point };

        allTrajectoryKeys.forEach(key => {
          const rawValue = point[key];
          if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
            sanitizedPoint[key] = null;
          } else if (rawValue < yMin) {
            sanitizedPoint[key] = null;
          } else if (rawValue > yMax) {
            if (seriesExceededMax[key]) {
              // Already had an out-of-range point, stop drawing
              sanitizedPoint[key] = null;
            } else {
              // First out-of-range point: clamp to yMax so line extends to top
              sanitizedPoint[key] = yMax;
              seriesExceededMax[key] = true;
            }
          }
        });

        const horizonValue = point.horizonLength;
        if (typeof horizonValue !== 'number' || !Number.isFinite(horizonValue)) {
          sanitizedPoint.horizonLength = null;
        } else if (horizonValue < yMin) {
          sanitizedPoint.horizonLength = null;
        } else if (horizonValue > yMax) {
          if (horizonExceededMax) {
            // Already had an out-of-range point, stop drawing
            sanitizedPoint.horizonLength = null;
          } else {
            // First out-of-range point: clamp to yMax so line extends to top
            sanitizedPoint.horizonLength = yMax;
            horizonExceededMax = true;
          }
        }

        return sanitizedPoint;
      })
      .filter(point => {
        const horizonValue = point.horizonLength;
        if (typeof horizonValue === 'number' && Number.isFinite(horizonValue)) {
          return true;
        }

        return allTrajectoryKeys.some(key => {
          const value = point[key];
          return typeof value === 'number' && Number.isFinite(value);
        });
      });
  }, [mergedData, xDomain, horizonDomain, trajectoryKeys, sampleKeys]);

  // Build line configs
  const lines = useMemo(() => {
    const result = [];

    // Add sample trajectory lines first (so they're behind everything)
    // They fade in on hover
    for (const key of sampleKeys) {
      result.push({
        dataKey: key,
        stroke: '#2A623D',
        strokeWidth: 1,
        strokeOpacity: 0.15,
      });
    }

    // Add Monte Carlo trajectory lines (also fade in on hover)
    for (const key of trajectoryKeys) {
      result.push({
        dataKey: key,
        stroke: '#2A623D',
        strokeWidth: 1,
        strokeOpacity: 0.3,
      });
    }

    // Add main line on top
    result.push({
      dataKey: 'horizonLength',
      stroke: '#2A623D',
      strokeWidth: 3,
      strokeOpacity: 1,
      name: 'Horizon Length',
    });

    return result;
  }, [trajectoryKeys, sampleKeys]);

  const automationCrossing = useMemo(() => {
    const ordered = chartData
      .filter(d => d.horizonLength != null && !isNaN(d.horizonLength) && Number.isFinite(d.horizonLength as number))
      .map(d => ({ year: d.year, horizonLength: d.horizonLength as number }))
      .sort((a, b) => a.year - b.year);

    if (ordered.length === 0) {
      return null;
    }

    let previous: { year: number; horizonLength: number } | null = null;

    for (const point of ordered) {
      const currentValue = point.horizonLength;
      if (!Number.isFinite(currentValue)) {
        continue;
      }

      if (currentValue >= scHorizonMinutes) {
        if (!previous || !Number.isFinite(previous.horizonLength)) {
          return { year: point.year, horizonLength: scHorizonMinutes };
        }

        const delta = currentValue - previous.horizonLength;
        const ratio = delta === 0 ? 0 : clamp((scHorizonMinutes - previous.horizonLength) / delta, 0, 1);
        const interpolatedYear = previous.year + ratio * (point.year - previous.year);

        return { year: interpolatedYear, horizonLength: scHorizonMinutes };
      }

      previous = point;
    }

    return null;
  }, [chartData, scHorizonMinutes]);

  // Custom elements for benchmarks - now a render function
  const visibleBenchmarks = useMemo(() => {
    const [xMin, xMax] = xDomain;
    const [yMin, yMax] = horizonDomain;

    return benchmarkData.filter(benchmark => {
      const { year, horizonLength } = benchmark;
      return (
        Number.isFinite(year) &&
        Number.isFinite(horizonLength) &&
        horizonLength > 0 &&
        year >= xMin &&
        year <= xMax &&
        horizonLength >= yMin &&
        horizonLength <= yMax
      );
    });
  }, [benchmarkData, xDomain, horizonDomain]);

  const customElements = useCallback((context: CustomElementsContext) => {
    const { chartWidth, chartHeight, xScale, yScale } = context;

    const handleMouseMove = (e: React.MouseEvent<SVGRectElement>) => {
      const rect = (e.currentTarget as SVGElement).ownerSVGElement?.getBoundingClientRect();
      if (!rect) return;

      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Find the closest benchmark to the cursor
      let closestBenchmark: BenchmarkPoint | null = null;
      let minDistance = Infinity;
      const maxDistance = 15; // Don't highlight if cursor is too far from any point

      for (const benchmark of visibleBenchmarks) {
        const cx = xScale(benchmark.year);
        const cy = yScale(benchmark.horizonLength);
        const distance = Math.sqrt(Math.pow(mouseX - cx, 2) + Math.pow(mouseY - cy, 2));

        if (distance < minDistance) {
          minDistance = distance;
          closestBenchmark = benchmark;
        }
      }

      if (closestBenchmark && minDistance <= maxDistance) {
        setBenchmarkMousePos({ x: mouseX, y: mouseY });
        setHoveredBenchmark(closestBenchmark);
      } else {
        setBenchmarkMousePos(null);
        setHoveredBenchmark(null);
      }
    };

    const handleMouseLeave = () => {
      setHoveredBenchmark(null);
      setBenchmarkMousePos(null);
    };

    const markerX = automationCrossing ? xScale(automationCrossing.year) : null;
    const markerY = automationCrossing ? yScale(automationCrossing.horizonLength) : null;

    const hasMarker = Number.isFinite(markerX) && Number.isFinite(markerY);

    const lineHalfSpan = 10;
    const circleX = hasMarker ? clamp(markerX as number, 0, chartWidth) : null;
    const circleY = hasMarker ? clamp(markerY as number, 0, chartHeight) : null;
    const markerStartX = hasMarker ? clamp((circleX as number) - lineHalfSpan, 0, chartWidth) : null;
    const markerEndX = hasMarker ? clamp((circleX as number) + lineHalfSpan, 0, chartWidth) : null;

    let labelAnchorX: number | null = null;
    let labelTextAnchor: 'start' | 'end' = 'end';
    if (hasMarker && circleX != null) {
      const offset = Math.min(14, circleX);
      labelAnchorX = clamp(circleX - offset, 0, chartWidth);
      if (labelAnchorX === 0 && circleX <= 6) {
        // Avoid clipping at extreme left while keeping the label generally to the left
        labelTextAnchor = 'start';
        labelAnchorX = clamp(circleX + 8, 0, chartWidth);
      }
    }

    const topLabelY = hasMarker && circleY != null ? clamp(circleY - 10, 10, chartHeight - 10) : null;
    const bottomLabelY = hasMarker && circleY != null ? clamp(circleY + 12, 12, chartHeight - 12) : null;

    return (
      <g>
        {/* Transparent overlay to capture all mouse movements */}
        <rect
          x={0}
          y={0}
          width={chartWidth}
          height={chartHeight}
          fill="transparent"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ pointerEvents: 'all' }}
        />

        {/* Render benchmarks */}
        {visibleBenchmarks.map((benchmark, index) => {
          const label = benchmark.label?.toLowerCase() || '';
          let fill = '#6b7280';

          let shapeFunc = createCircleShape;
          if (label.includes('claude')) {
            fill = '#dc2626';
            shapeFunc = createDiamondShape;
          } else if (label.includes('gpt') || label.includes('o1')) {
            fill = '#059669';
            shapeFunc = createSquareShape;
          } else if (label.includes('gemini')) {
            fill = '#2563eb';
            shapeFunc = createTriangleShape;
          } else if (label.includes('deepseek')) {
            fill = '#7c3aed';
            shapeFunc = createStarShape;
          }

          const cx = xScale(benchmark.year);
          const cy = yScale(benchmark.horizonLength);
          const isHovered = hoveredBenchmark === benchmark;
          const stroke = isHovered ? '#000000' : '#ffffff';
          const ShapeComponent = shapeFunc(fill, stroke);

          return (
            <g key={`benchmark-${index}`}>
              <ShapeComponent
                cx={cx}
                cy={cy}
                style={{ pointerEvents: 'none' }}
              />
            </g>
          );
        })}

        {hasMarker && circleX != null && circleY != null && markerStartX != null && markerEndX != null && labelAnchorX != null && topLabelY != null && bottomLabelY != null && (
          <g style={{ pointerEvents: 'none' }}>
            <line
              x1={markerStartX}
              y1={circleY}
              x2={markerEndX}
              y2={circleY}
              stroke={CODING_AUTOMATION_MARKER_COLOR}
              strokeWidth={2}
              strokeLinecap="round"
            />
            <circle
              cx={circleX}
              cy={circleY}
              r={3}
              fill={CODING_AUTOMATION_MARKER_COLOR}
            />
            <text
              x={labelAnchorX}
              y={topLabelY}
              textAnchor={labelTextAnchor}
              fontSize={11}
              fill={CODING_AUTOMATION_MARKER_COLOR}
            >
              Coding automation horizon
            </text>
            <text
              x={labelAnchorX}
              y={bottomLabelY}
              textAnchor={labelTextAnchor}
              fontSize={11}
              fontWeight={600}
              fill={CODING_AUTOMATION_MARKER_COLOR}
            >
              {formatTimeDuration(scHorizonMinutes)}
            </text>
          </g>
        )}
      </g>
    );
  }, [visibleBenchmarks, scHorizonMinutes, formatTimeDuration, automationCrossing, hoveredBenchmark]);

  return (
    <div 
      className={className || "flex-1"} 
      style={{ position: 'relative' }}
    >
      <div className="flex gap-2 items-center">
        
          <span className="text-[12px] font-semibold text-primary text-left font-system-mono">Coding Time Horizon</span>
        <div className="flex-1 border-t border-gray-500/30" />
        <WithChartTooltip explanation={HORIZON_LENGTH_EXPLANATION} className="!gap-0"><></></WithChartTooltip>
      </div>
      <CustomLineChart
        data={sanitizedData}
        height={height}
        margin={{ top: 0, right: 0, left: 0, bottom: 35 }}
        xDomain={xDomain}
        xTicks={xTicks}
        xTickFormatter={(value) => Math.round(value).toString()}
        animateXDomain={false}
        yDomain={horizonDomain}
        yScale="log"
        yTicks={yAxisTicks}
        yTickFormatter={formatLogWorkTick}
        showYAxis={true}
        lines={lines}
        customElements={customElements}
        tooltip={tooltip}
        width={width}
      />
      {/* Benchmark tooltip */}
      {hoveredBenchmark && benchmarkMousePos && (
        <div
          style={{
            position: 'absolute',
            left: benchmarkMousePos.x + 10,
            top: benchmarkMousePos.y + 10,
            pointerEvents: 'none',
            zIndex: 1000,
          }}
        >
          <div style={tooltipBoxStyle}>
            <span style={tooltipHeaderStyle}>{hoveredBenchmark.label}</span>
            <span>{Math.floor(hoveredBenchmark.year)}</span>
            <span style={tooltipValueStyle}>
              {formatWorkTimeDurationDetailed(hoveredBenchmark.horizonLength)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
});

CustomHorizonChart.displayName = 'CustomHorizonChart';

export default CustomHorizonChart;
