'use client';

import { PlotlyChart } from './PlotlyChart';
import { hexToRgba } from '../colors';
import { CHART_FONT_SIZES, CHART_MARGINS } from '../chartConfig';

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
  yRange?: [number, number];
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
  yRange,
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
      name: 'Median    ',
      hovertemplate: '%{x}: %{y:.2f}<extra></extra>',
    });

    // Add any additional traces
    traces.push(...additionalTraces);
  }

  const layout: Partial<Plotly.Layout> = {
    margin: title ? CHART_MARGINS.withTitle : CHART_MARGINS.noTitle,
    xaxis: {
      title: { text: xLabel, font: { size: CHART_FONT_SIZES.axisTitle } },
      tickfont: { size: CHART_FONT_SIZES.tickLabel },
    },
    yaxis: {
      title: { text: yLabel, font: { size: CHART_FONT_SIZES.axisTitle } },
      tickfont: { size: CHART_FONT_SIZES.tickLabel },
      type: yLogScale ? 'log' : 'linear',
      ...(yRange ? { range: yRange } : {}),
    },
    showlegend: additionalTraces.length > 0,
    legend: {
      x: 0,
      y: 1,
      xanchor: 'left',
      yanchor: 'top',
      bgcolor: 'rgba(255,255,255,0.9)',
      borderwidth: 0,
      font: { size: CHART_FONT_SIZES.legend },
    },
    hovermode: 'closest',
    title: title ? { text: title, font: { size: CHART_FONT_SIZES.plotTitle } } : undefined,
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
