'use client';

import { PlotlyChart } from './PlotlyChart';
import { COLOR_PALETTE, hexToRgba } from '../colors';
import { CHART_FONT_SIZES, CHART_MARGINS } from '../chartConfig';

interface EnergyStackedAreaChartProps {
  years: number[];
  /** Energy data per year: array of [initialStockEnergy, fabProducedEnergy] */
  energyData: number[][];
  /** Labels for the two energy sources */
  sourceLabels?: string[];
  /** Datacenter capacity (GW) per year */
  datacenterCapacity: number[];
  isLoading?: boolean;
}

/**
 * EnergyStackedAreaChart - Shows chip stock energy usage with:
 * 1. Stacked areas for energy from Initial Stock and Fab-Produced sources
 * 2. A datacenter capacity line
 * 3. A hash pattern for energy that exceeds capacity ("cannot be operated")
 */
export function EnergyStackedAreaChart({
  years,
  energyData,
  sourceLabels = ['Initial Stock', 'Fab-Produced'],
  datacenterCapacity,
  isLoading = false,
}: EnergyStackedAreaChartProps) {
  if (!years || years.length === 0 || !energyData || energyData.length === 0) {
    return <PlotlyChart data={[]} isEmpty emptyMessage="No data" />;
  }

  // Two shades of teal for energy sources (lighter for initial, darker for fab)
  const colors = ['#7DD4C0', '#3D9E8A'];

  // Calculate total energy at each time point
  const totalEnergy = energyData.map(yearData => yearData[0] + yearData[1]);

  // Check if fab has any energy
  const fabEnergy = energyData.map(yearData => yearData[1]);
  const hasFabEnergy = fabEnergy.some(val => val > 0);

  // Calculate shared Y-axis max
  const allYValues = [...totalEnergy, ...datacenterCapacity];
  const maxY = Math.max(...allYValues);
  const yMax = maxY * 1.3;

  const traces: Plotly.Data[] = [];

  // LAYER 1: Stacked area traces for each source
  for (let i = 0; i < sourceLabels.length; i++) {
    const energyAtSource = energyData.map(yearData => yearData[i]);
    const sourceLabel = sourceLabels[i];

    // Skip fab compute if there's no fab energy
    if (i === 1 && !hasFabEnergy) {
      continue;
    }

    traces.push({
      x: years,
      y: energyAtSource,
      type: 'scatter',
      mode: 'lines',
      stackgroup: 'energy',
      fillcolor: colors[i],
      line: { width: 0 },
      name: sourceLabel,
      hovertemplate: `${sourceLabel}<br>Energy: %{y:.2f} GW<extra></extra>`,
    });
  }

  // LAYER 2: Hash pattern for energy that exceeds capacity
  const unpoweredEnergy = years.map((_, i) =>
    Math.max(0, totalEnergy[i] - datacenterCapacity[i])
  );

  // Create the hash region as a shape that sits above the capacity line
  const hashX = [...years, ...years.slice().reverse()];
  const hashYTop = years.map((_, i) => datacenterCapacity[i] + unpoweredEnergy[i]);
  const hashY = [...hashYTop, ...datacenterCapacity.slice().reverse()];

  traces.push({
    x: hashX,
    y: hashY,
    type: 'scatter',
    mode: 'none',
    fill: 'toself',
    fillcolor: 'rgba(0, 0, 0, 0)',
    fillpattern: {
      shape: '/',
      fgcolor: 'rgba(100, 100, 100, 0.8)',
      bgcolor: 'rgba(0, 0, 0, 0)',
      size: 8,
      solidity: 0.7,
    },
    showlegend: false,
    hoverinfo: 'skip',
    line: { width: 0 },
  } as Plotly.Data);

  // LAYER 3: Datacenter capacity line
  traces.push({
    x: years,
    y: datacenterCapacity,
    type: 'scatter',
    mode: 'lines',
    line: { color: COLOR_PALETTE.datacenters_and_energy, width: 3 },
    name: 'Covert Datacenter Capacity',
    hovertemplate: 'Capacity: %{y:.1f} GW<extra></extra>',
  });

  // Add legend entry for the hash pattern
  traces.push({
    x: [years[0], years[1]],
    y: [null, null],
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    fillcolor: 'rgba(0, 0, 0, 0)',
    fillpattern: {
      shape: '/',
      fgcolor: 'rgba(100, 100, 100, 0.8)',
      bgcolor: 'rgba(0, 0, 0, 0)',
      size: 8,
      solidity: 0.7,
    },
    line: { width: 0 },
    name: 'Cannot be operated',
    showlegend: true,
    hoverinfo: 'skip',
  } as Plotly.Data);

  const layout: Partial<Plotly.Layout> = {
    xaxis: {
      title: { text: 'Year', font: { size: CHART_FONT_SIZES.axisTitle } },
      tickfont: { size: CHART_FONT_SIZES.tickLabel },
      range: [years[0], years[years.length - 1]],
      automargin: true,
    },
    yaxis: {
      title: { text: 'GW', font: { size: CHART_FONT_SIZES.axisTitle } },
      tickfont: { size: CHART_FONT_SIZES.tickLabel },
      range: [0, yMax],
      automargin: true,
    },
    margin: CHART_MARGINS.compact,
    hovermode: 'x unified',
    showlegend: true,
    legend: {
      x: 0.02,
      y: 0.98,
      xanchor: 'left',
      yanchor: 'top',
      bgcolor: 'rgba(255,255,255,0.9)',
      borderwidth: 0,
      font: { size: CHART_FONT_SIZES.legend },
    },
  };

  return (
    <PlotlyChart
      data={traces}
      layout={layout}
      isLoading={isLoading}
    />
  );
}

export default EnergyStackedAreaChart;
