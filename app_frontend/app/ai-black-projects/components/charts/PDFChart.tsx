'use client';

import { PlotlyChart } from './PlotlyChart';
import { hexToRgba } from '../colors';
import { CHART_FONT_SIZES, CHART_MARGINS } from '../chartConfig';

interface PDFChartProps {
  samples: number[];
  color: string;
  xLabel?: string;
  yLabel?: string;
  logScale?: boolean;
  numBins?: number;
  isLoading?: boolean;
  /** 'probability' normalizes so bars sum to 1; 'density' divides by bin width too */
  normalize?: 'probability' | 'density';
}

/**
 * PDFChart - Displays a probability density function (histogram) from sample data.
 * Used for visualizing likelihood ratio distributions.
 */
export function PDFChart({
  samples,
  color,
  xLabel = 'Value',
  yLabel,
  logScale = true,
  numBins = 12,
  isLoading = false,
  normalize = 'probability',
}: PDFChartProps) {
  // Default yLabel based on normalization
  const effectiveYLabel = yLabel ?? (normalize === 'density' ? 'Density' : 'Probability');

  if (!samples || samples.length === 0) {
    return <PlotlyChart data={[]} isEmpty emptyMessage="No data" />;
  }

  // Filter positive values for log scale
  const validSamples = logScale ? samples.filter(v => v > 0) : samples;

  if (validSamples.length === 0) {
    return <PlotlyChart data={[]} isEmpty emptyMessage="No valid data" />;
  }

  // Calculate bin edges
  const binEdges: number[] = [];
  const binCenters: number[] = [];
  const binWidths: number[] = [];

  if (logScale) {
    // Log-scale binning
    const minVal = Math.min(...validSamples);
    const maxVal = Math.max(...validSamples);
    const logMin = Math.log10(minVal);
    const logMax = Math.log10(maxVal);
    let logRange = logMax - logMin;

    // Handle case where all samples are identical (logRange = 0)
    if (logRange === 0 || !isFinite(logRange)) {
      // Create a range of ±0.5 log units around the value
      logRange = 1.0;
    }

    // Add padding
    const paddedLogMin = logMin - logRange * 0.1;
    const paddedLogMax = logMax + logRange * 0.1;

    for (let i = 0; i <= numBins; i++) {
      binEdges.push(Math.pow(10, paddedLogMin + i * (paddedLogMax - paddedLogMin) / numBins));
    }

    // Calculate bin centers (geometric mean for log scale) and widths
    for (let i = 0; i < binEdges.length - 1; i++) {
      binCenters.push(Math.sqrt(binEdges[i] * binEdges[i + 1]));
      binWidths.push(binEdges[i + 1] - binEdges[i]);
    }
  } else {
    // Linear binning
    const minVal = Math.min(...validSamples);
    const maxVal = Math.max(...validSamples);
    let range = maxVal - minVal;

    // Handle case where all samples are identical
    if (range === 0) {
      range = Math.abs(minVal) * 0.2 || 1.0; // Use 20% of value as range, or 1.0 if value is 0
    }

    const binWidth = range / numBins;

    for (let i = 0; i <= numBins; i++) {
      binEdges.push(minVal - range * 0.1 + i * (range * 1.2) / numBins);
    }

    for (let i = 0; i < binEdges.length - 1; i++) {
      binCenters.push((binEdges[i] + binEdges[i + 1]) / 2);
      binWidths.push(binEdges[i + 1] / numBins);
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

  // Normalize based on mode
  const yValues = normalize === 'density'
    ? binCounts.map((count, i) => count / (validSamples.length * binWidths[i]))
    : binCounts.map(count => count / validSamples.length);

  // Format hover and y-axis based on normalization
  const isProbability = normalize === 'probability';
  const hoverYFormat = isProbability ? '%{y:.1%}' : '%{y:.4f}';
  const hovertemplate = `${xLabel}: %{x:.2f}<br>${effectiveYLabel}: ${hoverYFormat}<extra></extra>`;

  const data: Plotly.Data[] = [
    {
      type: 'bar',
      x: binCenters,
      y: yValues,
      width: binWidths,
      marker: {
        color: isProbability ? hexToRgba(color, 0.7) : color,
        line: { width: isProbability ? 1 : 0.5, color: isProbability ? color : 'white' },
      },
      hovertemplate,
    },
  ];

  const layout: Partial<Plotly.Layout> = {
    xaxis: {
      title: { text: xLabel, font: { size: CHART_FONT_SIZES.axisTitle } },
      type: logScale ? 'log' : 'linear',
      tickfont: { size: CHART_FONT_SIZES.tickLabel },
      gridcolor: 'rgba(128, 128, 128, 0.2)',
    },
    yaxis: {
      title: { text: effectiveYLabel, font: { size: CHART_FONT_SIZES.axisTitle } },
      tickfont: { size: CHART_FONT_SIZES.tickLabel },
      tickformat: isProbability ? '.0%' : undefined,
      gridcolor: 'rgba(128, 128, 128, 0.2)',
    },
    margin: CHART_MARGINS.compact,
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
