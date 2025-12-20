'use client';

import { PlotlyChart } from './PlotlyChart';

interface CCDFPoint {
  x: number;
  y: number;
}

// Multi-threshold data format: { "1": [...], "2": [...], "4": [...] }
type MultiThresholdData = Record<string | number, CCDFPoint[]>;

interface CCDFChartProps {
  data?: CCDFPoint[] | MultiThresholdData;
  color: string;
  xLabel?: string;
  yLabel?: string;
  xAsPercent?: boolean;
  xLogScale?: boolean;
  isLoading?: boolean;
  title?: string;
  fillAlpha?: number;
  showArea?: boolean;
  thresholdLabels?: Record<string | number, string>;
  thresholdColors?: Record<string | number, string>;
}

// Default colors for different thresholds (matching black_project_frontend)
const DEFAULT_THRESHOLD_COLORS: Record<string, string> = {
  '1': '#5E6FB8',  // 1x update (blue)
  '2': '#E9A842',  // 2x update (orange)
  '4': '#4AA896',  // 4x update (green)
};

const DEFAULT_THRESHOLD_LABELS: Record<string, string> = {
  '1': '1× LR update',
  '2': '2× LR update',
  '4': '4× LR update',
};

// Helper to convert hex to rgba
function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function isMultiThresholdData(data: CCDFPoint[] | MultiThresholdData | undefined): data is MultiThresholdData {
  if (!data) return false;
  if (Array.isArray(data)) return false;
  if (typeof data !== 'object') return false;
  const keys = Object.keys(data);
  if (keys.length === 0) return false;
  const hasNumericKeys = keys.every(k => !isNaN(Number(k)));
  const hasArrayValues = keys.every(k => Array.isArray((data as MultiThresholdData)[k]));
  return hasNumericKeys && hasArrayValues;
}

export function CCDFChart({
  data,
  color,
  xLabel = '',
  yLabel = 'P(X > x)',
  xAsPercent = false,
  xLogScale = false,
  isLoading = false,
  title,
  fillAlpha = 0.1,
  showArea = true,
  thresholdLabels = DEFAULT_THRESHOLD_LABELS,
  thresholdColors = DEFAULT_THRESHOLD_COLORS,
}: CCDFChartProps) {
  // Handle multi-threshold data
  if (isMultiThresholdData(data)) {
    const thresholds = Object.keys(data).sort((a, b) => Number(a) - Number(b));
    const isEmpty = thresholds.length === 0 || thresholds.every(t => !data[t] || data[t].length === 0);

    const plotData: Plotly.Data[] = isEmpty ? [] : thresholds.map((threshold, idx) => {
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

    const layout: Partial<Plotly.Layout> = {
      xaxis: {
        title: { text: xLabel, font: { size: 11 } },
        tickfont: { size: 10 },
        ticksuffix: xAsPercent ? '%' : '',
        type: xLogScale ? 'log' : 'linear',
      },
      yaxis: {
        title: { text: yLabel, font: { size: 11 } },
        tickfont: { size: 10 },
        range: [0, 1.05],
      },
      showlegend: thresholds.length > 1,
      legend: {
        x: 0.98,
        y: 0.98,
        xanchor: 'right',
        yanchor: 'top',
        font: { size: 10 },
        bgcolor: 'rgba(255,255,255,0.8)',
      },
      title: title ? { text: title, font: { size: 12 } } : undefined,
    };

    return (
      <PlotlyChart
        data={plotData}
        layout={layout}
        isLoading={isLoading}
        isEmpty={isEmpty}
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
      title: { text: xLabel, font: { size: 11 } },
      tickfont: { size: 10 },
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
    />
  );
}

export default CCDFChart;
