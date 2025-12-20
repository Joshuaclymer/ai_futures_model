'use client';

import { PlotlyChart } from './PlotlyChart';

interface PDFChartProps {
  samples: number[];
  color: string;
  xLabel?: string;
  yLabel?: string;
  logScale?: boolean;
  numBins?: number;
  isLoading?: boolean;
}

/**
 * PDFChart - Displays a probability density function (histogram) from sample data.
 * Used for visualizing likelihood ratio distributions.
 */
export function PDFChart({
  samples,
  color,
  xLabel = 'Value',
  yLabel = 'Density',
  logScale = true,
  numBins = 12,
  isLoading = false,
}: PDFChartProps) {
  if (!samples || samples.length === 0) {
    return <PlotlyChart data={[]} isEmpty emptyMessage="No data" />;
  }

  // Filter positive values for log scale
  const validSamples = logScale ? samples.filter(v => v > 0) : samples;

  if (validSamples.length === 0) {
    return <PlotlyChart data={[]} isEmpty emptyMessage="No valid data" />;
  }

  // Calculate bin edges
  let binEdges: number[] = [];
  let binCenters: number[] = [];
  let binWidths: number[] = [];

  if (logScale) {
    // Log-scale binning
    const minVal = Math.min(...validSamples);
    const maxVal = Math.max(...validSamples);
    const logMin = Math.log10(minVal);
    const logMax = Math.log10(maxVal);
    const logRange = logMax - logMin;

    // Add padding
    const paddedLogMin = logMin - logRange * 0.05;
    const paddedLogMax = logMax + logRange * 0.05;

    for (let i = 0; i <= numBins; i++) {
      binEdges.push(Math.pow(10, paddedLogMin + i * (paddedLogMax - paddedLogMin) / numBins));
    }

    // Calculate bin centers and widths
    for (let i = 0; i < binEdges.length - 1; i++) {
      binCenters.push((binEdges[i] + binEdges[i + 1]) / 2);
      binWidths.push(binEdges[i + 1] - binEdges[i]);
    }
  } else {
    // Linear binning
    const minVal = Math.min(...validSamples);
    const maxVal = Math.max(...validSamples);
    const range = maxVal - minVal;
    const binWidth = range / numBins;

    for (let i = 0; i <= numBins; i++) {
      binEdges.push(minVal + i * binWidth);
    }

    for (let i = 0; i < binEdges.length - 1; i++) {
      binCenters.push((binEdges[i] + binEdges[i + 1]) / 2);
      binWidths.push(binWidth);
    }
  }

  // Bin the data
  const binCounts = new Array(numBins).fill(0);
  validSamples.forEach(v => {
    for (let i = 0; i < binEdges.length - 1; i++) {
      if (v >= binEdges[i] && (i === binEdges.length - 2 ? v <= binEdges[i + 1] : v < binEdges[i + 1])) {
        binCounts[i]++;
        break;
      }
    }
  });

  // Convert to probabilities
  const probabilities = binCounts.map(count => count / validSamples.length);

  const data: Plotly.Data[] = [
    {
      type: 'bar',
      x: binCenters,
      y: probabilities,
      width: binWidths,
      marker: {
        color: color,
        line: { width: 0.5, color: 'white' },
      },
      hovertemplate: `${xLabel}: %{x:.2f}<br>Probability: %{y:.1%}<extra></extra>`,
    },
  ];

  const layout: Partial<Plotly.Layout> = {
    xaxis: {
      title: { text: xLabel, font: { size: 10 } },
      type: logScale ? 'log' : 'linear',
      tickfont: { size: 9 },
      gridcolor: 'rgba(128, 128, 128, 0.2)',
    },
    yaxis: {
      title: { text: yLabel, font: { size: 10 } },
      tickfont: { size: 9 },
      tickformat: '.0%',
      gridcolor: 'rgba(128, 128, 128, 0.2)',
    },
    margin: { l: 45, r: 10, t: 10, b: 35 },
    bargap: 0.05,
  };

  return (
    <PlotlyChart
      data={data}
      layout={layout}
      isLoading={isLoading}
    />
  );
}

export default PDFChart;
