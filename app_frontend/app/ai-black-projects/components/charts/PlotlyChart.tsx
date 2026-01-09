'use client';

import dynamic from 'next/dynamic';
import { useEffect, useState, useRef } from 'react';
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
  // Keep track of the last valid data to show while loading
  const lastDataRef = useRef<Plotly.Data[]>(data);
  const lastLayoutRef = useRef<Partial<Plotly.Layout>>(layout);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Update refs when we have new valid data
  useEffect(() => {
    if (!isLoading && data && data.length > 0) {
      lastDataRef.current = data;
      lastLayoutRef.current = layout;
    }
  }, [data, layout, isLoading]);

  const placeholderStyle: React.CSSProperties = height ? { height: `${height}px` } : {};

  // Show empty state only if we have no data at all (not while loading with previous data)
  const hasNoData = isEmpty || !data || data.length === 0;
  const hasPreviousData = lastDataRef.current && lastDataRef.current.length > 0;

  if (hasNoData && !hasPreviousData) {
    return (
      <div className={`flex items-center justify-center h-full text-gray-400 text-sm ${className}`} style={placeholderStyle}>
        {isLoading ? loadingMessage : emptyMessage}
      </div>
    );
  }

  if (!mounted) {
    return null;
  }

  // Use current data if available, otherwise use last valid data
  const displayData = (hasNoData ? lastDataRef.current : data) || [];
  const displayLayout = hasNoData ? lastLayoutRef.current : layout;

  const defaultLayout: Partial<Plotly.Layout> = {
    margin: CHART_MARGINS.default,
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'inherit', size: 11 },
    hovermode: 'closest',
    autosize: true,
    hoverlabel: {
      bgcolor: '#ffffff',
      bordercolor: '#ffffff',
      font: {
        family: 'system-ui, -apple-system, sans-serif',
        size: 12,
        color: '#333',
      },
    },
    ...displayLayout,
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
    <div className="relative w-full h-full">
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/70 backdrop-blur-[1px] rounded">
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>{loadingMessage}</span>
          </div>
        </div>
      )}
      <Plot
        data={displayData}
        layout={defaultLayout}
        config={defaultConfig}
        style={computedStyle}
        className={className}
        useResizeHandler={true}
      />
    </div>
  );
}

export default PlotlyChart;
