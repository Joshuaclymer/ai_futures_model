'use client';
import { ChartDataPoint } from '@/app/types';
import { memo, useMemo, useEffect, useRef, useState, useCallback, ReactNode } from 'react';
import { CustomLineChart, DataPoint, CustomElementsContext } from './CustomLineChart';
import { formatTo3SigFigs, formatAsPowerOfTenText, formatCompactNumberNode } from '@/utils/formatting';
import { calculateDynamicXDomain, clamp, ChartMargin } from '@/utils/chartUtils';
import { CHART_LAYOUT } from '@/constants/chartLayout';
import {
  XYPoint,
  PixelSegment,
  interpolateAtX,
  getVisibleEndpoints,
  estimateLabelWidth,
  selectLabelPosition,
  pointsToPixelSegments,
  LABEL_HEIGHT,
  SEGMENTS_TO_CHECK,
} from '@/utils/chartLabelUtils';

export interface CustomMetricChartProps {
  chartData: ChartDataPoint[];
  tooltip: (point: DataPoint) => ReactNode;
  className?: string;
  title: string;
  dataKey: string;
  yFormatter?: (value: number) => string | ReactNode;
  yScale?: 'linear' | 'log';
  color?: string;
  domainOverride?: [number, number] | ((observed: number[]) => [number, number]);
  verticalReferenceLine?: {
    x: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  };
  displayEndYear: number;
  height?: number;
  logSuffix?: string;
  width?: number;
  showXAxis?: boolean;
  xDomainOverride?: [number, number];
  xTickCount?: number;
  xTickFormatter?: (value: number) => string;
  xLabel?: string;
  margin?: ChartMargin;
  showTitle?: boolean;
  /** If true, the data values are already in log form (OOMs) and should be converted via Math.pow(10, v) for display */
  isDataInLogForm?: boolean;
}

function CustomMetricChartComponent({
  chartData,
  tooltip,
  className,
  title,
  dataKey,
  yFormatter,
  yScale = 'linear',
  color = '#2A623D',
  domainOverride,
  verticalReferenceLine,
  displayEndYear,
  height,
  logSuffix,
  width,
  showXAxis,
  xDomainOverride,
  xTickCount,
  xTickFormatter,
  xLabel,
  margin,
  showTitle = true,
  isDataInLogForm = false,
}: CustomMetricChartProps) {
  const resolvedYFormatter = useMemo<(value: number) => string | ReactNode>(() => {
    if (yFormatter) {
      return yFormatter;
    }

    return yScale === 'log'
      ? (value: number) => formatAsPowerOfTenText(value)
      : (value: number) => formatTo3SigFigs(value);
  }, [yFormatter, yScale]);

  const containerRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const [chartHeight, setChartHeight] = useState<number>(() => height ?? 0);

  useEffect(() => {
    if (height != null) {
      setChartHeight(height);
      return;
    }

    const updateHeight = () => {
      if (!containerRef.current) return;

      const containerHeight = containerRef.current.clientHeight;
      setChartHeight(containerHeight);
    };

    updateHeight();

    const observer = new ResizeObserver(updateHeight);
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      observer.disconnect();
    };
  }, [height]);

  const yDomain = useMemo(() => {
    const observed: number[] = [];

    // Collect all values for this metric
    chartData.forEach(point => {
      const value = point[dataKey as keyof ChartDataPoint];
      if (typeof value === 'number' && Number.isFinite(value) && point.year <= displayEndYear) {
        observed.push(value);
      }
    });

    if (observed.length === 0) {
      return [0, 1] as [number, number];
    }

    const minObserved = Math.min(...observed);
    const maxObserved = Math.max(...observed);

    let domain: [number, number];

    if (domainOverride) {
      domain = typeof domainOverride === 'function' ? domainOverride(observed) : domainOverride;
    } else if (yScale === 'log' && isDataInLogForm) {
      // Data is already in log form (OOMs), so use linear domain on the log values
      // Add some padding in log space
      const range = maxObserved - minObserved;
      const padding = Math.max(range * 0.1, 0.5); // At least 0.5 OOM padding
      domain = [minObserved - padding, maxObserved + padding];
    } else if (yScale === 'log') {
      const positiveValues = observed.filter(v => v > 0);
      if (positiveValues.length === 0) {
        domain = [1, 10];
      } else {
        const minPositive = Math.min(...positiveValues);
        const maxPositive = Math.max(...positiveValues);
        let lower = minPositive * 0.9;
        const upper = maxPositive * 1.1;

        if (lower <= 0) {
          lower = minPositive * 0.1;
        }

        const safeLower = Math.max(lower, Number.MIN_VALUE);
        let safeUpper = Math.max(upper, safeLower * 1.1);

        if (!Number.isFinite(safeUpper) || safeUpper <= safeLower) {
          safeUpper = safeLower * 10;
        }

        domain = [safeLower, safeUpper];
      }
    } else {
      const suggestedMax = maxObserved > 1 ? Math.floor(Math.max(1, Math.ceil(maxObserved * 1.1))) : 1;
      domain = [Math.min(0, minObserved), suggestedMax];
    }

    return domain;
  }, [chartData, dataKey, yScale, domainOverride, displayEndYear, isDataInLogForm]);

  const xDomain = useMemo((): [number, number] => {
    if (xDomainOverride && xDomainOverride[1] > xDomainOverride[0]) {
      return xDomainOverride;
    }

    const years = chartData
      .filter(d => {
        const value = d[dataKey as keyof ChartDataPoint];
        return value != null && !isNaN(value as number);
      })
      .map(d => d.year)
      .filter(y => !isNaN(y));

    if (years.length === 0) return [2020, 2030];

    // Use dynamic domain calculation that considers vertical reference line (intersection)
    const intersectionX = verticalReferenceLine?.x ?? null;
    return calculateDynamicXDomain(years, {
      intersectionPoints: [intersectionX],
      minPadding: 0.01,
      maxPadding: 0.15,
      minRange: 3, // Minimum 3 years visible
      maxDomain: displayEndYear // Constrain to smart display end year
    });
  }, [chartData, dataKey, verticalReferenceLine, displayEndYear, xDomainOverride]);

  // Prepare trajectory lines
  const trajectoryKeys = useMemo(() => {
    if (chartData.length === 0) return [];
    const sample = chartData[0];
    return Object.keys(sample).filter(key => key.startsWith(`${dataKey}_trajectory_`));
  }, [chartData, dataKey]);

  // Add x property to chartData
  const mergedData = useMemo<(ChartDataPoint & { x: number })[]>(() => {
    const withinDisplay = chartData.filter(point => point.year <= displayEndYear);
    const source = withinDisplay.length > 0 ? withinDisplay : chartData;
    return source.map(point => ({ ...point, x: point.year }));
  }, [chartData, displayEndYear]);

  // Build line configs
  const lines = useMemo(() => {
    const result = [];

    // Add trajectory lines first (so they're behind)
    for (const key of trajectoryKeys) {
      result.push({
        dataKey: key,
        stroke: color,
        strokeWidth: 1,
        strokeOpacity: 0.3,
      });
    }

    // Add main line
    result.push({
      dataKey: dataKey,
      stroke: color,
      strokeWidth: 3,
      strokeOpacity: 1,
      name: title,
    });

    return result;
  }, [trajectoryKeys, dataKey, color, title]);

  type EndpointLabelRenderer = ((context: CustomElementsContext) => ReactNode) & { displayName?: string };

  const formatLogEndpoint = useCallback((value: number): ReactNode => {
    // If data is already in log form (OOMs), convert to actual value first
    if (isDataInLogForm) {
      const actualValue = Math.pow(10, value);
      return formatCompactNumberNode(actualValue, {
        renderMode: 'svg',
        mantissaOpacity: 0.7,
        mantissaFontSize: 12,
        exponentFontSize: 12,
        suffix: logSuffix,
      });
    }

    if (!Number.isFinite(value) || value <= 0) {
      return '';
    }

    // Use formatCompactNumberNode to properly handle mantissas (e.g., 3×10^-8)
    return formatCompactNumberNode(value, {
      renderMode: 'svg',
      mantissaOpacity: 0.7,
      mantissaFontSize: 12,
      exponentFontSize: 12,
      suffix: logSuffix,
    });

  }, [logSuffix, isDataInLogForm]);

  const formatEndpointValue = useCallback((value: number): ReactNode => {
    if (!Number.isFinite(value)) {
      return '';
    }

    // For log-scale charts (including those with data already in log form)
    if (yScale === 'log') {
      return formatLogEndpoint(value);
    }
    
    // For data in log form but not explicitly marked as log scale
    if (isDataInLogForm) {
      return formatLogEndpoint(value);
    }

    if (yFormatter) {
      const formatted = yFormatter(value);
      if (typeof formatted === 'string' || typeof formatted === 'number') {
        return formatted;
      }
      return formatted;
    }

    return formatTo3SigFigs(value);
  }, [yFormatter, yScale, formatLogEndpoint, isDataInLogForm]);

  const customElements = useMemo(() => {
    const seriesPoints: XYPoint[] = mergedData
      .filter(point => {
        const value = point[dataKey as keyof ChartDataPoint];
        return typeof value === 'number' && Number.isFinite(value) && point.x <= displayEndYear;
      })
      .map(point => ({ x: point.x as number, y: point[dataKey as keyof ChartDataPoint] as number }))
      .sort((a, b) => a.x - b.x);

    if (seriesPoints.length === 0) {
      return undefined;
    }

    const EndpointLabels: EndpointLabelRenderer = (context) => {
      const { chartWidth, chartHeight, xScale, yScale: yScaleFunc, xDomain: renderedXDomain } = context;

      const { left, right } = getVisibleEndpoints(seriesPoints, renderedXDomain[0], renderedXDomain[1]);
      const elements: ReactNode[] = [];

      let visibleSeries = seriesPoints.filter(
        point => point.x >= renderedXDomain[0] - 1e-6 && point.x <= renderedXDomain[1] + 1e-6
      );
      if (visibleSeries.length === 0) {
        visibleSeries = [...seriesPoints];
      }
      if (left && (visibleSeries.length === 0 || Math.abs(visibleSeries[0].x - left.x) > 1e-6)) {
        visibleSeries = [left, ...visibleSeries];
      }
      if (right && (visibleSeries.length === 0 || Math.abs(visibleSeries[visibleSeries.length - 1].x - right.x) > 1e-6)) {
        visibleSeries = [...visibleSeries, right];
      }

      const segments = pointsToPixelSegments(visibleSeries, xScale, yScaleFunc, chartHeight);
      const leftSegments = segments.slice(0, SEGMENTS_TO_CHECK);
      const rightSegments = segments.slice(Math.max(segments.length - SEGMENTS_TO_CHECK, 0));

      const renderEndpoint = (endpoint: XYPoint | null, align: 'left' | 'right', key: string) => {
        if (!endpoint) {
          return;
        }

        const xPos = xScale(endpoint.x);
        const yPosRaw = yScaleFunc(endpoint.y);

        if (!Number.isFinite(xPos) || !Number.isFinite(yPosRaw)) {
          return;
        }

        const yPos = clamp(yPosRaw, 0, chartHeight);

        const label = formatEndpointValue(endpoint.y);
        if (label == null || label === '') {
          return;
        }

        const labelWidth = estimateLabelWidth(label);
        const placement = selectLabelPosition(
          align,
          xPos,
          yPos,
          labelWidth,
          LABEL_HEIGHT,
          chartWidth,
          chartHeight,
          align === 'left' ? leftSegments : rightSegments
        );

        elements.push(
          <g key={key}>
            <circle cx={xPos} cy={yPos} r={3} fill={color} />
            <text
              x={placement.textX}
              y={placement.textY}
              textAnchor={placement.anchor}
              alignmentBaseline="middle"
              fontSize={11}
              fill={color}
              fontWeight="500"
            >
              {label}
            </text>
          </g>
        );
      };

      renderEndpoint(left, 'left', 'metric-left-endpoint');
      renderEndpoint(right, 'right', 'metric-right-endpoint');

      return <g>{elements}</g>;
    };

    EndpointLabels.displayName = 'CustomMetricChartEndpointLabels';

    return EndpointLabels;
  }, [mergedData, dataKey, displayEndYear, formatEndpointValue, color]);

  return (
    <div
      ref={containerRef}
      className={`flex min-h-0 flex-col relative ${className ? className : ''}`.trim()}
      style={{ height: '100%' }}
    >
      {showTitle ? (
        <h3
          ref={headerRef}
          className="text-[9px] uppercase font-light text-gray-700 font-system-mono absolute left-0 max-w-40"
        >
          {title}
        </h3>
      ) : null}
      <div className="flex-1 min-h-0">
        <CustomLineChart
          data={mergedData}
          height={chartHeight}
          margin={margin ?? { top: 3, right: 10, left: 4, bottom: 3 }}
          xDomain={xDomain}
          xTickCount={xTickCount ?? 5}
          xTickFormatter={xTickFormatter ?? ((value) => Math.round(value).toString())}
          yDomain={yDomain}
          // When data is already in log form, use linear scale on the y-axis
          // (the data values are already logarithms, so linear spacing = log spacing of actual values)
          yScale={isDataInLogForm ? 'linear' : yScale}
          yTickFormatter={resolvedYFormatter}
          lines={lines}
          tooltip={tooltip}
          width={width}
          showXAxis={showXAxis ?? false}
          xLabel={xLabel}
          customElements={customElements}
        />
      </div>
    </div>
  );
}

export const CustomMetricChart = memo(CustomMetricChartComponent);

CustomMetricChart.displayName = 'CustomMetricChart';
