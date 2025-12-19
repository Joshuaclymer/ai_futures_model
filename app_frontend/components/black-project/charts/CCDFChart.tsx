'use client';

import { PlotlyChart } from './PlotlyChart';
import { hexToRgba, ColorName } from '@/types/blackProject';

interface CCDFPoint {
  x: number;
  y: number;
}

interface CCDFChartProps {
  data?: CCDFPoint[];
  color: string;
  xLabel?: string;
  yLabel?: string;
  xAsPercent?: boolean;
  xLogScale?: boolean;
  isLoading?: boolean;
  title?: string;
  fillAlpha?: number;
  showArea?: boolean;
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
}: CCDFChartProps) {
  const isEmpty = !data || data.length === 0;

  const plotData: Plotly.Data[] = isEmpty ? [] : [
    {
      x: data.map(d => xAsPercent ? d.x * 100 : d.x),
      y: data.map(d => d.y),
      type: 'scatter',
      mode: 'lines',
      line: { color, width: 2 },
      fill: showArea ? 'tozeroy' : 'none',
      fillcolor: showArea ? hexToRgba('chip_stock' as ColorName, fillAlpha) : undefined,
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
