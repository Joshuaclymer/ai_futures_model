'use client';

import { PlotlyChart } from './PlotlyChart';
import { hexToRgba } from '../colors';
import { CHART_FONT_SIZES } from '../chartConfig';

interface CCDFPoint {
  x: number;
  y: number;
}

// Multi-threshold data format: { "1": [...], "2": [...], "4": [...] }
type MultiThresholdData = Record<string | number, CCDFPoint[]>;

interface ReferenceLine {
  y: number;  // y-value for horizontal line
  label: string;
  color?: string;
  dash?: 'solid' | 'dot' | 'dash' | 'longdash' | 'dashdot';
}

interface CCDFChartProps {
  data?: CCDFPoint[] | MultiThresholdData;
  color: string;
  xLabel?: string;
  yLabel?: string;
  xAsPercent?: boolean;
  xLogScale?: boolean;
  xAsInverseFraction?: boolean;  // Show x-axis as "1/Nx" format
  isLoading?: boolean;
  title?: string;
  fillAlpha?: number;
  showArea?: boolean;
  thresholdLabels?: Record<string | number, string>;
  thresholdColors?: Record<string | number, string>;
  referenceLine?: ReferenceLine;
  height?: number;
}

// Default colors for different thresholds - purple palette
const DEFAULT_THRESHOLD_COLORS: Record<string, string> = {
  '1': '#5E6FB8',  // 1x update (primary purple)
  '2': '#9B8AC4',  // 2x update (lighter purple)
  '4': '#7A9EC2',  // 4x update (purple-blue)
};

// Trailing spaces are a workaround for Plotly bug that clips legend text
// See: https://github.com/plotly/documentation/issues/1195
const DEFAULT_THRESHOLD_LABELS: Record<string, string> = {
  '1': '1× LR update    ',
  '2': '2× LR update    ',
  '4': '4× LR update    ',
};

function isMultiThresholdData(data: CCDFPoint[] | MultiThresholdData | undefined): data is MultiThresholdData {
  if (!data) return false;
  if (Array.isArray(data)) return false;
  if (typeof data !== 'object') return false;
  const keys = Object.keys(data);
  if (keys.length === 0) return false;
  // Allow both numeric keys (1, 2, 4) and string keys (global, prc, largest_ai_company)
  const hasArrayValues = keys.every(k => Array.isArray((data as MultiThresholdData)[k]));
  return hasArrayValues;
}

// Convert fraction to inverse fraction string for log scale (e.g., 0.1 -> "1/10x", 0.001 -> "1/1,000x")
function formatInverseFraction(value: number): string {
  if (value >= 1) return `${Math.round(value)}x`;
  if (value === 0) return '0x';

  // Log scale fractions (powers of 10)
  const fractions: [number, string][] = [
    [1, '1x'],
    [0.1, '1/10x'],
    [0.01, '1/100x'],
    [0.001, '1/1,000x'],
    [0.0001, '1/10,000x'],
    [0.00001, '1/100,000x'],
  ];

  for (const [frac, label] of fractions) {
    if (Math.abs(value - frac) < frac * 0.1) return label;
  }

  // For non-standard values, compute the inverse
  const inverse = 1 / value;
  if (inverse >= 1000) {
    return `1/${(inverse / 1000).toFixed(0)},000x`;
  } else if (inverse >= 1) {
    return `1/${Math.round(inverse)}x`;
  }

  return `${value.toFixed(2)}x`;
}

export function CCDFChart({
  data,
  color,
  xLabel = '',
  yLabel = 'P(X > x)',
  xAsPercent = false,
  xLogScale = false,
  xAsInverseFraction = false,
  isLoading = false,
  title,
  fillAlpha = 0.1,
  showArea = true,
  thresholdLabels = DEFAULT_THRESHOLD_LABELS,
  thresholdColors = DEFAULT_THRESHOLD_COLORS,
  referenceLine,
  height,
}: CCDFChartProps) {
  // Handle multi-threshold data
  if (isMultiThresholdData(data)) {
    const thresholds = Object.keys(data).sort((a, b) => Number(a) - Number(b));
    const isEmpty = thresholds.length === 0 || thresholds.every(t => !data[t] || data[t].length === 0);

    const thresholdTraces: Plotly.Data[] = isEmpty ? [] : thresholds.map((threshold, idx) => {
      const thresholdData = data[threshold];
      if (!thresholdData || thresholdData.length === 0) return null;

      const lineColor = thresholdColors[threshold] || DEFAULT_THRESHOLD_COLORS[threshold] || color;

      return {
        x: thresholdData.map(d => xAsPercent ? d.x * 100 : d.x),
        y: thresholdData.map(d => d.y),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: thresholdLabels[threshold] || `${threshold}× update`,
        line: { color: lineColor, width: 2 },
        fill: idx === 0 && showArea ? 'tozeroy' : 'none',
        fillcolor: idx === 0 && showArea ? hexToRgba(lineColor, fillAlpha) : undefined,
        hovertemplate: xAsPercent
          ? `${thresholdLabels[threshold] || threshold}: %{x:.1f}%, P=%{y:.2f}<extra></extra>`
          : `${thresholdLabels[threshold] || threshold}: %{x:.2f}, P=%{y:.2f}<extra></extra>`,
      };
    }).filter(Boolean) as Plotly.Data[];

    // Add horizontal reference line trace if provided
    const plotData: Plotly.Data[] = [...thresholdTraces];
    if (referenceLine && !isEmpty) {
      // Get x range from data for horizontal line
      const allXValues = thresholds.flatMap(t => (data[t] || []).map(d => xAsPercent ? d.x * 100 : d.x));
      const xMin = Math.min(...allXValues);
      const xMax = Math.max(...allXValues);

      plotData.push({
        x: [xMin, xMax],
        y: [referenceLine.y, referenceLine.y],
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: referenceLine.label,
        line: {
          color: referenceLine.color || '#888888',
          width: 1.5,
          dash: referenceLine.dash || 'dot',
        },
        hoverinfo: 'name' as const,
      });
    }

    const showLegend = thresholds.length > 1 || !!referenceLine;

    // Generate tick values and labels for inverse fraction format
    let xaxisConfig: Partial<Plotly.LayoutAxis> = {
      title: { text: xLabel, font: { size: CHART_FONT_SIZES.axisTitle } },
      tickfont: { size: CHART_FONT_SIZES.tickLabel },
      ticksuffix: xAsPercent ? '%' : '',
      type: xLogScale ? 'log' : 'linear',
    };

    if (xAsInverseFraction && !isEmpty) {
      // Log scale tick values: 1x, 1/10x, 1/100x, 1/1,000x, 1/10,000x
      const tickVals = [1, 0.1, 0.01, 0.001, 0.0001];
      const tickText = tickVals.map(formatInverseFraction);

      xaxisConfig = {
        ...xaxisConfig,
        type: 'log' as const,
        tickmode: 'array' as const,
        tickvals: tickVals,
        ticktext: tickText,
        ticksuffix: '',
        autorange: 'reversed' as const,  // Reverse so 1x is on left, 1/10,000x on right
      };
    }

    const layout: Partial<Plotly.Layout> = {
      margin: showLegend ? { l: 50, r: 20, t: 10, b: 50 } : undefined,
      xaxis: xaxisConfig,
      yaxis: {
        title: { text: yLabel, font: { size: CHART_FONT_SIZES.axisTitle } },
        tickfont: { size: CHART_FONT_SIZES.tickLabel },
        range: [0, 1.05],
      },
      showlegend: showLegend,
      legend: {
        x: 0.98,
        y: 0.98,
        xanchor: 'right',
        yanchor: 'top',
        font: { size: CHART_FONT_SIZES.legend },
        bgcolor: 'rgba(255,255,248,0.9)',
        borderwidth: 0,
      },
      title: title ? { text: title, font: { size: CHART_FONT_SIZES.plotTitle } } : undefined,
    };

    return (
      <PlotlyChart
        data={plotData}
        layout={layout}
        isLoading={isLoading}
        isEmpty={isEmpty}
        height={height}
      />
    );
  }

  // Single threshold data handling - must be an array
  if (!Array.isArray(data)) {
    return (
      <PlotlyChart
        data={[]}
        layout={{}}
        isLoading={isLoading}
        isEmpty={true}
        height={height}
      />
    );
  }

  const singleData = data as CCDFPoint[];
  const isEmpty = singleData.length === 0;

  const plotData: Plotly.Data[] = isEmpty ? [] : [
    {
      x: singleData.map(d => xAsPercent ? d.x * 100 : d.x),
      y: singleData.map(d => d.y),
      type: 'scatter',
      mode: 'lines',
      line: { color, width: 2 },
      fill: showArea ? 'tozeroy' : 'none',
      fillcolor: showArea ? hexToRgba(color, fillAlpha) : undefined,
      hovertemplate: xAsPercent
        ? '%{x:.1f}%: P=%{y:.2f}<extra></extra>'
        : '%{x:.2f}: P=%{y:.2f}<extra></extra>',
    },
  ];

  const layout: Partial<Plotly.Layout> = {
    xaxis: {
      title: { text: xLabel, font: { size: CHART_FONT_SIZES.axisTitle } },
      tickfont: { size: CHART_FONT_SIZES.tickLabel },
      ticksuffix: xAsPercent ? '%' : '',
      type: xLogScale ? 'log' : 'linear',
    },
    yaxis: {
      title: { text: yLabel, font: { size: 11 } },
      tickfont: { size: 10 },
      range: [0, 1.05],
    },
    showlegend: false,
    title: title ? { text: title, font: { size: 12 } } : undefined,
  };

  return (
    <PlotlyChart
      data={plotData}
      layout={layout}
      isLoading={isLoading}
      isEmpty={isEmpty}
      height={height}
    />
  );
}

export default CCDFChart;
