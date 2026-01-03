'use client';

import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';
import { CHART_MARGINS } from '../chartConfig';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface PlotlyChartProps {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  config?: Partial<Plotly.Config>;
  style?: React.CSSProperties;
  className?: string;
  isLoading?: boolean;
  loadingMessage?: string;
  emptyMessage?: string;
  isEmpty?: boolean;
  height?: number;
}

export function PlotlyChart({
  data,
  layout = {},
  config = {},
  style,
  className = '',
  isLoading = false,
  loadingMessage = 'Loading...',
  emptyMessage = 'No data available',
  isEmpty = false,
  height,
}: PlotlyChartProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const placeholderStyle: React.CSSProperties = height ? { height: `${height}px` } : {};

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-full text-gray-400 text-sm ${className}`} style={placeholderStyle}>
        {loadingMessage}
      </div>
    );
  }

  if (isEmpty || !data || data.length === 0) {
    return (
      <div className={`flex items-center justify-center h-full text-gray-400 text-sm ${className}`} style={placeholderStyle}>
        {emptyMessage}
      </div>
    );
  }

  if (!mounted) {
    return null;
  }

  const defaultLayout: Partial<Plotly.Layout> = {
    margin: CHART_MARGINS.default,
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'inherit', size: 11 },
    hovermode: 'closest',
    hoverlabel: {
      bgcolor: '#ffffff',
      bordercolor: '#ffffff',
      font: {
        family: 'system-ui, -apple-system, sans-serif',
        size: 12,
        color: '#333',
      },
    },
    ...layout,
  };

  const defaultConfig: Partial<Plotly.Config> = {
    displayModeBar: false,
    responsive: true,
    ...config,
  };

  const computedStyle: React.CSSProperties = {
    width: '100%',
    height: height ? `${height}px` : '100%',
    ...style,
  };

  return (
    <Plot
      data={data}
      layout={defaultLayout}
      config={defaultConfig}
      style={computedStyle}
      className={className}
    />
  );
}

export default PlotlyChart;
