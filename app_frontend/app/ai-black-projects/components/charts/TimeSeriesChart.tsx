'use client';

import { PlotlyChart } from './PlotlyChart';

interface TimeSeriesChartProps {
  years?: number[];
  median?: number[];
  p25?: number[];
  p75?: number[];
  color: string;
  yLabel?: string;
  xLabel?: string;
  isLoading?: boolean;
  title?: string;
  showBand?: boolean;
  bandAlpha?: number;
  yLogScale?: boolean;
  additionalTraces?: Plotly.Data[];
}

// Helper to convert hex to rgba
function hexToRgba(hex: string | undefined, alpha: number): string {
  if (!hex) {
    return `rgba(128, 128, 128, ${alpha})`; // fallback gray
  }
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function TimeSeriesChart({
  years,
  median,
  p25,
  p75,
  color = '#666666',
  yLabel = '',
  xLabel = 'Year',
  isLoading = false,
  title,
  showBand = true,
  bandAlpha = 0.2,
  yLogScale = false,
  additionalTraces = [],
}: TimeSeriesChartProps) {
  const isEmpty = !years || !median || years.length === 0;

  const traces: Plotly.Data[] = [];

  if (!isEmpty) {
    // Percentile band
    if (showBand && p25 && p75) {
      traces.push({
        x: [...years, ...years.slice().reverse()],
        y: [...p75, ...p25.slice().reverse()],
        type: 'scatter',
        mode: 'lines',
        fill: 'toself',
        fillcolor: hexToRgba(color, bandAlpha),
        line: { color: 'transparent' },
        showlegend: false,
        hoverinfo: 'skip',
        name: 'IQR',
      });
    }

    // Median line
    traces.push({
      x: years,
      y: median,
      type: 'scatter',
      mode: 'lines',
      line: { color, width: 2 },
      name: 'Median',
      hovertemplate: '%{x}: %{y:.2f}<extra></extra>',
    });

    // Add any additional traces
    traces.push(...additionalTraces);
  }

  const layout: Partial<Plotly.Layout> = {
    margin: { l: 60, r: 20, t: title ? 30 : 10, b: 50 },
    xaxis: {
      title: { text: xLabel, font: { size: 11 } },
      tickfont: { size: 10 },
    },
    yaxis: {
      title: { text: yLabel, font: { size: 11 } },
      tickfont: { size: 10 },
      type: yLogScale ? 'log' : 'linear',
    },
    showlegend: additionalTraces.length > 0,
    legend: {
      x: 0,
      y: 1,
      xanchor: 'left',
      yanchor: 'top',
      bgcolor: 'rgba(255,255,255,0.8)',
      font: { size: 10 },
    },
    hovermode: 'x unified',
    title: title ? { text: title, font: { size: 12 } } : undefined,
  };

  return (
    <PlotlyChart
      data={traces}
      layout={layout}
      isLoading={isLoading}
      isEmpty={isEmpty}
    />
  );
}

export default TimeSeriesChart;
