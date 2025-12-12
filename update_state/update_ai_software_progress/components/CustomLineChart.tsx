'use client';

import { useState, useRef, useCallback, ReactNode, useEffect, useMemo } from 'react';
import { createScale, generateLinePath, calculateTicks, ChartMargin, clamp } from '@/utils/chartUtils';
import { useChartSync } from './ChartSyncContext';

const DEFAULT_X_DOMAIN_ANIMATION_DURATION_MS = 450;

export interface DataPoint {
  x: number;
  y?: number | null;
  [key: string]: number | null | undefined | string;
}

export interface CustomElementsContext {
  chartWidth: number;
  chartHeight: number;
  xDomain: [number, number];
  yDomain: [number, number];
  xScale: (value: number) => number;
  yScale: (value: number) => number;
}

export interface CustomLineChartProps {
  data: DataPoint[];
  width?: number;
  height: number;
  margin?: ChartMargin;

  // X axis config
  xDomain: [number, number];
  showXAxis?: boolean;
  xTicks?: number[];
  xTickStepSize?: number;
  xTickCount?: number;
  xTickFormatter?: (value: number) => string;
  xLabel?: string;
  xPadding?: {
    start?: number;
    end?: number;
  };

  // Y axis config
  yDomain: [number, number];
  yScale?: 'linear' | 'log';
  yTicks?: number[];
  yTickFormatter?: (value: number, workTimeLabel?: boolean) => string | ReactNode;
  showYAxis?: boolean;

  // Line configs
  lines: Array<{
    dataKey: string;
    stroke: string;
    strokeWidth?: number;
    strokeOpacity?: number;
    name?: string;
    smooth?: boolean;
  }>;

  // Optional features
  verticalReferenceLine?: {
    x: number;
    stroke: string;
    strokeDasharray?: string;
    strokeWidth?: number;
    label?: string;
    strokeOpacity?: number;
  };
  customElements?: (context: CustomElementsContext) => ReactNode;
  tooltip?: (point: DataPoint) => ReactNode;
  animateXDomain?: boolean;
  xDomainAnimationDurationMs?: number;
}

export function CustomLineChart({
  data,
  width,
  height,
  margin = { top: 20, right: 0, left: 0, bottom: 60 },
  xDomain,
  xTicks: explicitXTicks,
  xTickStepSize,
  xTickCount,
  xTickFormatter = (v) => v.toString(),
  xLabel,
  yDomain,
  yScale = 'linear',
  yTicks,
  yTickFormatter,
  showYAxis = false,
  showXAxis = true,
  lines,
  verticalReferenceLine,
  customElements,
  tooltip,
  xPadding,
  animateXDomain = false,
  xDomainAnimationDurationMs = DEFAULT_X_DOMAIN_ANIMATION_DURATION_MS,
}: CustomLineChartProps) {
  const [hoveredPoint, setHoveredPoint] = useState<DataPoint | null>(null);
  const [mousePosition, setMousePosition] = useState<{ x: number; y: number } | null>(null);
  const [containerWidth, setContainerWidth] = useState(width || 250);
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const { hoveredX, setHoveredX } = useChartSync();
  const [animatedXDomain, setAnimatedXDomain] = useState<[number, number]>(xDomain);
  const animatedXDomainRef = useRef(animatedXDomain);
  const xDomainAnimationFrameRef = useRef<number | null>(null);
  const xDomainAnimationStartRef = useRef(0);
  const xDomainFromRef = useRef<[number, number]>(xDomain);
  const xDomainToRef = useRef<[number, number]>(xDomain);
  const [animatedXTicks, setAnimatedXTicks] = useState<number[]>(() =>
    calculateTicks(
      xDomain,
      xTickCount ? { targetCount: xTickCount } : xTickStepSize ? { stepSize: xTickStepSize } : { targetCount: 3 },
      'linear'
    )
  );
  const animatedXTicksRef = useRef(animatedXTicks);
  const xTicksFromRef = useRef<number[]>(animatedXTicks);
  const xTicksToRef = useRef<number[]>(animatedXTicks);

  useEffect(() => {
    animatedXDomainRef.current = animatedXDomain;
  }, [animatedXDomain]);

  useEffect(() => {
    animatedXTicksRef.current = animatedXTicks;
  }, [animatedXTicks]);

  const xTickOptions = useMemo(() => {
    if (xTickCount) {
      return { targetCount: xTickCount };
    }
    if (xTickStepSize) {
      return { stepSize: xTickStepSize };
    }
    return { targetCount: 3 };
  }, [xTickCount, xTickStepSize]);

  const generateTicksForDomain = useCallback(
    (domain: [number, number]) => calculateTicks(domain, xTickOptions, 'linear'),
    [xTickOptions]
  );

  const sampleTickArray = useCallback((ticks: number[], normalizedIndex: number) => {
    if (!ticks.length) {
      return NaN;
    }
    if (ticks.length === 1) {
      return ticks[0];
    }
    const clampedNorm = clamp(normalizedIndex, 0, 1);
    const scaledIndex = clampedNorm * (ticks.length - 1);
    const lowerIndex = Math.floor(scaledIndex);
    const upperIndex = Math.min(ticks.length - 1, Math.ceil(scaledIndex));
    if (lowerIndex === upperIndex) {
      return ticks[lowerIndex];
    }
    const fraction = scaledIndex - lowerIndex;
    const lowerValue = ticks[lowerIndex];
    const upperValue = ticks[upperIndex];
    return lowerValue + (upperValue - lowerValue) * fraction;
  }, []);

  const interpolateTickArrays = useCallback(
    (from: number[], to: number[], progress: number) => {
      if (!from.length && !to.length) {
        return [];
      }

      if (progress >= 1) {
        return [...to];
      }

      if (progress <= 0) {
        return [...from];
      }

      const maxLength = Math.max(from.length, to.length, 1);
      const result: number[] = [];

      for (let i = 0; i < maxLength; i += 1) {
        const normalizedIndex = maxLength === 1 ? 0 : i / (maxLength - 1);
        const startValueRaw = sampleTickArray(from, normalizedIndex);
        const endValueRaw = sampleTickArray(to, normalizedIndex);
        const fallback = Number.isNaN(startValueRaw) ? endValueRaw : startValueRaw;
        const startValue = Number.isNaN(startValueRaw) ? fallback : startValueRaw;
        const endValue = Number.isNaN(endValueRaw) ? fallback : endValueRaw;
        const safeStart = Number.isFinite(startValue) ? startValue : 0;
        const safeEnd = Number.isFinite(endValue) ? endValue : safeStart;
        result.push(safeStart + (safeEnd - safeStart) * progress);
      }

      return result;
    },
    [sampleTickArray]
  );

  useEffect(() => {
    const arraysEqual = (a: number[], b: number[]) =>
      a.length === b.length && a.every((value, idx) => value === b[idx]);

    const nextTicks = generateTicksForDomain(xDomain);

    if (!animateXDomain) {
      if (
        animatedXDomainRef.current[0] !== xDomain[0] ||
        animatedXDomainRef.current[1] !== xDomain[1]
      ) {
        setAnimatedXDomain(xDomain);
        animatedXDomainRef.current = xDomain;
      }
      xDomainFromRef.current = xDomain;
      xDomainToRef.current = xDomain;
      if (xDomainAnimationFrameRef.current !== null) {
        cancelAnimationFrame(xDomainAnimationFrameRef.current);
        xDomainAnimationFrameRef.current = null;
      }
      if (!arraysEqual(animatedXTicksRef.current, nextTicks)) {
        setAnimatedXTicks(nextTicks);
        animatedXTicksRef.current = nextTicks;
        xTicksFromRef.current = nextTicks;
        xTicksToRef.current = nextTicks;
      }
      return;
    }

    const fromDomain = animatedXDomainRef.current;
    const domainUnchanged =
      fromDomain[0] === xDomain[0] && fromDomain[1] === xDomain[1];

    if (domainUnchanged) {
      if (!arraysEqual(animatedXTicksRef.current, nextTicks)) {
        setAnimatedXTicks(nextTicks);
        animatedXTicksRef.current = nextTicks;
        xTicksFromRef.current = nextTicks;
        xTicksToRef.current = nextTicks;
      }
      return;
    }

    xDomainFromRef.current = fromDomain;
    xDomainToRef.current = xDomain;
    xDomainAnimationStartRef.current = performance.now();
    xTicksFromRef.current = animatedXTicksRef.current;
    xTicksToRef.current = nextTicks;

    if (xDomainAnimationFrameRef.current !== null) {
      cancelAnimationFrame(xDomainAnimationFrameRef.current);
    }

    const duration = Math.max(0, xDomainAnimationDurationMs);

    const step = (timestamp: number) => {
      const elapsed = timestamp - xDomainAnimationStartRef.current;
      const progress = duration === 0 ? 1 : Math.min(1, elapsed / duration);
      const nextDomain: [number, number] = [
        xDomainFromRef.current[0] + (xDomainToRef.current[0] - xDomainFromRef.current[0]) * progress,
        xDomainFromRef.current[1] + (xDomainToRef.current[1] - xDomainFromRef.current[1]) * progress,
      ];
      setAnimatedXDomain(nextDomain);
      animatedXDomainRef.current = nextDomain;

      const interpolatedTicks = interpolateTickArrays(
        xTicksFromRef.current,
        xTicksToRef.current,
        progress
      );
      setAnimatedXTicks(interpolatedTicks);
      animatedXTicksRef.current = interpolatedTicks;

      if (progress < 1) {
        xDomainAnimationFrameRef.current = requestAnimationFrame(step);
      } else {
        setAnimatedXTicks(xTicksToRef.current);
        animatedXTicksRef.current = xTicksToRef.current;
        xDomainAnimationFrameRef.current = null;
      }
    };

    xDomainAnimationFrameRef.current = requestAnimationFrame(step);

    return () => {
      if (xDomainAnimationFrameRef.current !== null) {
        cancelAnimationFrame(xDomainAnimationFrameRef.current);
        xDomainAnimationFrameRef.current = null;
      }
    };
  }, [xDomain, animateXDomain, xDomainAnimationDurationMs, generateTicksForDomain, interpolateTickArrays]);

  // Measure container width for responsive sizing
  useEffect(() => {
    if (!containerRef.current || width) return; // Skip if width is explicitly provided

    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, [width]);

  const actualWidth = width || containerWidth;
  const chartWidth = actualWidth - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  

  const currentXDomain = animateXDomain ? animatedXDomain : xDomain;
  
  // Debug: log when there's a mismatch between xDomain prop and currentXDomain
  if (Math.abs(currentXDomain[1] - xDomain[1]) > 0.1) {
    console.log('[CustomLineChart] Domain mismatch - prop xDomain:', xDomain, 'currentXDomain:', currentXDomain, 'animateXDomain:', animateXDomain);
  }

  // Create scales
  const paddingStart = clamp(xPadding?.start ?? 0, 0, chartWidth);
  const paddingEnd = clamp(xPadding?.end ?? 0, 0, chartWidth - paddingStart);
  const xRangeStart = paddingStart;
  const xRangeEnd = Math.max(xRangeStart + 1, chartWidth - paddingEnd);

  const xScale = createScale({
    domain: currentXDomain,
    range: [xRangeStart, xRangeEnd],
    type: 'linear',
  });

  const yScaleFunc = createScale({
    domain: yDomain,
    range: [chartHeight, 0], // SVG Y is inverted
    type: yScale,
  });

  const customElementGroup = customElements
    ? customElements({
        chartWidth,
        chartHeight,
        xDomain: currentXDomain,
        yDomain,
        xScale,
        yScale: yScaleFunc,
      })
    : null;

  

  // Calculate ticks - use explicit ticks if provided
  const xTicks = explicitXTicks ?? (animateXDomain ? animatedXTicks : generateTicksForDomain(currentXDomain));

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;

    const rect = svgRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left - margin.left;

    setMousePosition({ x: e.clientX, y: e.clientY });

    // Find closest point based on X-axis distance only (better for time-series)
    const mainDataKey = lines[lines.length - 1]?.dataKey; // Use last line (main line)
    if (!mainDataKey) return;

    const dataWithMainLine = data.filter(d => d[mainDataKey] != null);

    // Find closest point by X-axis distance only
    let closest: DataPoint | null = null;
    let minXDistance = Infinity;

    for (const point of dataWithMainLine) {
      const px = xScale(point.x);
      const xDistance = Math.abs(mouseX - px);

      if (xDistance < minXDistance) {
        minXDistance = xDistance;
        closest = point;
      }
    }

    if (tooltip) {
      setHoveredPoint(closest);
    }

    // Update shared hover X position for crosshair sync
    if (closest) {
      setHoveredX(closest.x);
    }
  }, [data, lines, margin, xScale, tooltip, setHoveredX]);

  const handleMouseLeave = useCallback(() => {
    setHoveredPoint(null);
    setMousePosition(null);
    setHoveredX(null);
  }, [setHoveredX]);

  return (
    <div ref={containerRef} style={{ position: 'relative', width: width || '100%', height }}>
      <svg
        ref={svgRef}
        width={actualWidth}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ userSelect: 'none' }}
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          {/* Vertical reference line */}
          {verticalReferenceLine && (
            <g>
              <line
                x1={xScale(verticalReferenceLine.x)}
                y1={0}
                x2={xScale(verticalReferenceLine.x)}
                y2={chartHeight}
                stroke={verticalReferenceLine.stroke}
                strokeDasharray={verticalReferenceLine.strokeDasharray || '2 10'}
                strokeWidth={verticalReferenceLine.strokeWidth || 1}
                strokeOpacity={verticalReferenceLine.strokeOpacity ?? 0.5}
                strokeLinecap="round"
              />
              {verticalReferenceLine.label && (
                <text
                  x={xScale(verticalReferenceLine.x) + 5}
                  y={15}
                  textAnchor="start"
                  fontSize={11}
                  fill={verticalReferenceLine.stroke}
                  transform={`rotate(-90, ${xScale(verticalReferenceLine.x) + 5}, 15)`}
                >
                  {verticalReferenceLine.label}
                </text>
              )}
            </g>
          )}

          {/* Lines */}
          {lines.map((line, i) => {
            const lineData = data.map(d => ({
              x: d.x,
              y: d[line.dataKey] as number | null | undefined,
            }));

            const path = generateLinePath(lineData, xScale, yScaleFunc, line.smooth);

            return (
              <path
                id={`line-${i}-${line.dataKey}`}
                key={`line-${i}-${line.dataKey}`}
                d={path}
                fill="none"
                stroke={line.stroke}
                strokeWidth={line.strokeWidth ?? 2}
                strokeOpacity={line.strokeOpacity ?? 1}
                style={{ transition: 'stroke-opacity 0.6s ease-in-out' }}
              />
            );
          })}

          {customElementGroup}

          {/* Synchronized crosshair */}
          {hoveredX !== null && (
            <line
              x1={xScale(hoveredX)}
              y1={0}
              x2={xScale(hoveredX)}
              y2={chartHeight}
              stroke="#999"
              strokeWidth={1}
              strokeDasharray="4 4"
              opacity={0.6}
              pointerEvents="none"
            />
          )}

          {/* Y Axis */}
          {showYAxis && yTicks && yTicks.length > 0 && (
            <g>
              {yTicks.map((tick, i) => {
                const yPos = yScaleFunc(tick);
                const labelContent = yTickFormatter ? yTickFormatter(tick) : tick.toString();
                const isReactNode = typeof labelContent !== 'string';
                return (
                  <g key={`y-tick-${i}`} transform={`translate(0,${yPos})`}>
                    {isReactNode ? (
                      <foreignObject x={10} y={-10} width={80} height={20} style={{ overflow: 'visible' }}>
                        <div style={{ fontSize: '11px', color: '#666', fontWeight: 500, fontFamily: 'inherit' }}>
                          {labelContent as unknown as ReactNode}
                        </div>
                      </foreignObject>
                    ) : (
                      <text
                        x={10}
                        textAnchor="start"
                        alignmentBaseline="middle"
                        fontSize={11}
                        fill="#666"
                      >
                        {labelContent as string}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          )}

          {/* X Axis */}
          {showXAxis && (
          <g transform={`translate(0,${chartHeight})`}>
            {xTicks.map((tick, i) => {
              const xPos = xScale(tick);
              // Skip ticks that are too close to edges to be centered properly
              // A year label like "2024" is ~30px wide, so need ~15px clearance on each side
              const minEdgeDistance = 20;
              if (xPos < minEdgeDistance || xPos > chartWidth - minEdgeDistance) {
                return null;
              }
              
              return (
                <g key={`x-tick-${i}`} transform={`translate(${xPos},0)`}>
                  <text
                    y={20}
                    textAnchor="middle"
                    fontSize={12}
                    fill="#666"
                  >
                    {xTickFormatter(tick)}
                  </text>
                </g>
              );
            })}
            {xLabel && (
              <text
                x={chartWidth / 2}
                y={50}
                textAnchor="middle"
                fontSize={14}
                fill="#666"
              >
                {xLabel}
                </text>
              )}
            </g>
          )}

        </g>
      </svg>

      {/* Tooltip */}
      {tooltip && hoveredPoint && mousePosition && (
        <div
          style={{
            position: 'fixed',
            left: mousePosition.x + 12,
            top: mousePosition.y + 12,
            pointerEvents: 'none',
            zIndex: 1000,
          }}
        >
          {tooltip(hoveredPoint)}
        </div>
      )}
    </div>
  );
}
