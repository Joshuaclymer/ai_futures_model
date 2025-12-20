'use client';

import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';

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
}: PlotlyChartProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <div className="flex flex-col items-center gap-2">
          <div className="w-8 h-8 border-3 border-gray-200 border-t-blue-500 rounded-full animate-spin" />
          <span className="text-xs text-gray-400">{loadingMessage}</span>
        </div>
      </div>
    );
  }

  if (isEmpty || !data || data.length === 0) {
    return (
      <div className={`flex items-center justify-center h-full text-gray-400 text-sm ${className}`}>
        {emptyMessage}
      </div>
    );
  }

  if (!mounted) {
    return null;
  }

  const defaultLayout: Partial<Plotly.Layout> = {
    margin: { l: 50, r: 20, t: 10, b: 50 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'inherit', size: 11 },
    hovermode: 'closest',
    ...layout,
  };

  const defaultConfig: Partial<Plotly.Config> = {
    displayModeBar: false,
    responsive: true,
    ...config,
  };

  return (
    <Plot
      data={data}
      layout={defaultLayout}
      config={defaultConfig}
      style={{ width: '100%', height: '100%', ...style }}
      className={className}
    />
  );
}

export default PlotlyChart;
