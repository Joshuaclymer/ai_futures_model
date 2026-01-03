'use client';

import { useMemo } from 'react';
import { COLOR_PALETTE } from '@/types/blackProject';
import { PlotlyChart } from '../../charts';
import { hexToRgba, DETECTION_THRESHOLD_COLORS } from '../../colors';
import { CHART_FONT_SIZES } from '../../chartConfig';
import { ParamValue, Dashboard, DashboardItem } from '../../ui';
import { Parameters, SimulationData } from '../../../types';
import { parseInitialStockData, InitialStockData } from '../../../utils/typeGuards';
import { formatH100e, formatEnergy, formatPercent } from '../../../utils/formatters';

interface InitialStockSectionProps {
  data: SimulationData | null;
  isLoading?: boolean;
  parameters: Parameters;
}

// Create histogram trace with probability normalization
// For log-scale x-axis, we need to bin in log space
function createHistogramTrace(
  samples: number[],
  color: string,
  nbins: number = 30,
  useLogBins: boolean = false
): Plotly.Data | null {
  if (!samples || samples.length === 0) return null;

  // Filter out non-positive values for log binning
  const validSamples = useLogBins
    ? samples.filter(s => s > 0)
    : samples;

  if (validSamples.length === 0) return null;

  if (useLogBins) {
    // Create log-spaced bins manually
    const logSamples = validSamples.map(s => Math.log10(s));
    const minLog = Math.min(...logSamples);
    const maxLog = Math.max(...logSamples);
    const binWidth = (maxLog - minLog) / nbins;

    // Count samples in each bin
    const counts = new Array(nbins).fill(0);
    const binEdges: number[] = [];

    for (let i = 0; i <= nbins; i++) {
      binEdges.push(Math.pow(10, minLog + i * binWidth));
    }

    for (const sample of validSamples) {
      const binIndex = Math.min(
        Math.floor((Math.log10(sample) - minLog) / binWidth),
        nbins - 1
      );
      counts[binIndex]++;
    }

    // Convert to probability
    const total = validSamples.length;
    const probs = counts.map(c => c / total);

    // Create bin centers (geometric mean of edges)
    const binCenters = binEdges.slice(0, -1).map((edge, i) =>
      Math.sqrt(edge * binEdges[i + 1])
    );

    // Calculate bar widths for log scale (as fraction of bin center)
    const barWidths = binCenters.map((center, i) =>
      binEdges[i + 1] - binEdges[i]
    );

    return {
      x: binCenters,
      y: probs,
      type: 'bar',
      width: barWidths,
      marker: {
        color: hexToRgba(color, 0.7),
        line: { color: color, width: 1 },
      },
      hovertemplate: 'Value: %{x:.2s}<br>Probability: %{y:.3f}<extra></extra>',
    } as Plotly.Data;
  }

  return {
    x: samples,
    type: 'histogram',
    nbinsx: nbins,
    histnorm: 'probability',
    marker: {
      color: hexToRgba(color, 0.7),
      line: { color: color, width: 1 },
    },
    hovertemplate: 'Value: %{x:.2s}<br>Probability: %{y:.3f}<extra></extra>',
  } as Plotly.Data;
}

export function InitialStockSection({
  data,
  isLoading,
  parameters
}: InitialStockSectionProps) {
  // Extract values from parameters
  const agreementYear = parameters.agreementYear;

  // Constants
  const H100_POWER_KW = 0.7; // 700W = 0.7kW

  // Use API data from data.initial_stock, with null fallback when not available
  const initialStock: InitialStockData | null = useMemo(() => {
    if (data?.initial_stock) {
      return parseInitialStockData(data.initial_stock);
    }
    return null;
  }, [data]);

  // Get state of the art efficiency
  const sotaEfficiency = initialStock?.state_of_the_art_energy_efficiency_relative_to_h100 || 6.35;

  // Calculate dashboard values
  const dashboardValues = useMemo(() => {
    const computeSamples = initialStock?.initial_compute_stock_samples;
    const energySamples = initialStock?.initial_energy_samples;
    if (!computeSamples?.length || !energySamples?.length) {
      return { medianDarkCompute: '--', medianEnergy: '--', detectionProb: '--' };
    }

    const sortedCompute = [...computeSamples].sort((a, b) => a - b);
    const medianCompute = sortedCompute[Math.floor(sortedCompute.length * 0.5)];

    const sortedEnergy = [...energySamples].sort((a, b) => a - b);
    const medianEnergy = sortedEnergy[Math.floor(sortedEnergy.length * 0.5)];

    // Get detection probability for 4x threshold
    const detectionProb = initialStock?.initial_black_project_detection_probs?.['4x'];

    return {
      medianDarkCompute: formatH100e(medianCompute),
      medianEnergy: formatEnergy(medianEnergy),
      detectionProb: detectionProb !== undefined ? formatPercent(detectionProb) : '--',
    };
  }, [initialStock]);

  // Create detection probability bar chart data
  const detectionBarData = useMemo(() => {
    const probs = initialStock?.initial_black_project_detection_probs;
    if (!probs) return [];

    const thresholds = ['1x', '2x', '4x'];
    const colors = [DETECTION_THRESHOLD_COLORS['1x'], DETECTION_THRESHOLD_COLORS['2x'], DETECTION_THRESHOLD_COLORS['4x']];

    const validData = thresholds
      .map((t, i) => ({ threshold: t, prob: probs[t], color: colors[i] }))
      .filter(d => d.prob !== undefined);

    if (validData.length === 0) return [];

    return [{
      x: validData.map(d => `Detection means<br>\u2265${d.threshold} LR`),
      y: validData.map(d => d.prob),
      type: 'bar' as const,
      marker: { color: validData.map(d => d.color) },
      hovertemplate: 'P(LR \u2265 %{x}): %{y:.2%}<extra></extra>',
    }];
  }, [initialStock]);

  // Create histogram data for initial compute stock (log-binned for log scale x-axis)
  const computeHistogramData = useMemo(() => {
    const samples = initialStock?.initial_compute_stock_samples;
    if (!samples?.length) return [];
    const trace = createHistogramTrace(samples, COLOR_PALETTE.chip_stock, 25, true);
    return trace ? [trace] : [];
  }, [initialStock]);

  // Create histogram data for initial PRC stock (log-binned for log scale x-axis)
  const prcStockHistogramData = useMemo(() => {
    const samples = initialStock?.initial_prc_stock_samples;
    if (!samples?.length) return [];
    const trace = createHistogramTrace(samples, COLOR_PALETTE.chip_stock, 25, true);
    return trace ? [trace] : [];
  }, [initialStock]);

  // Create energy requirements histogram (from API-provided samples)
  const energyRequirementsData = useMemo(() => {
    const energySamples = initialStock?.initial_energy_samples;
    if (!energySamples?.length) return [];

    // Filter out any invalid values
    const validEnergySamples = energySamples.filter(e =>
      typeof e === 'number' && !isNaN(e) && isFinite(e) && e > 0
    );

    if (validEnergySamples.length === 0) return [];

    const trace = createHistogramTrace(validEnergySamples, COLOR_PALETTE.datacenters_and_energy, 20);
    return trace ? [trace] : [];
  }, [initialStock]);

  // Create PRC compute over time time series data
  const prcComputeOverTimeData = useMemo(() => {
    const years = initialStock?.prc_compute_years;
    const overTime = initialStock?.prc_compute_over_time;
    const domestic = initialStock?.prc_domestic_compute_over_time;
    const proportionDomestic = initialStock?.proportion_domestic_by_year;
    const largestCompany = initialStock?.largest_company_compute_over_time;

    console.log('[prcComputeOverTimeData] years length:', years?.length);
    console.log('[prcComputeOverTimeData] overTime:', overTime);
    console.log('[prcComputeOverTimeData] overTime.median length:', overTime?.median?.length);

    if (!years?.length || !overTime?.median?.length) {
      console.log('[prcComputeOverTimeData] Returning empty - missing data');
      return [];
    }

    const traces: Plotly.Data[] = [];

    // Shaded area for 25-75 percentile range
    traces.push({
      x: [...years, ...years.slice().reverse()],
      y: [...overTime.p75, ...overTime.p25.slice().reverse()],
      fill: 'toself',
      fillcolor: hexToRgba(COLOR_PALETTE.chip_stock, 0.2),
      line: { color: 'transparent' },
      type: 'scatter',
      showlegend: false,
      hoverinfo: 'skip',
      name: '25th-75th percentile',
    });

    // 25th percentile line
    traces.push({
      x: years,
      y: overTime.p25,
      mode: 'lines',
      line: { color: COLOR_PALETTE.chip_stock, width: 1, dash: 'dash' },
      type: 'scatter',
      showlegend: false,
      hovertemplate: 'Year: %{x}<br>25th percentile: %{y:.2s} H100e<extra></extra>',
    });

    // Median line
    traces.push({
      x: years,
      y: overTime.median,
      mode: 'lines',
      line: { color: COLOR_PALETTE.chip_stock, width: 2 },
      type: 'scatter',
      name: 'Median    ',
      hovertemplate: 'Year: %{x}<br>Median: %{y:.2s} H100e<extra></extra>',
    });

    // 75th percentile line
    traces.push({
      x: years,
      y: overTime.p75,
      mode: 'lines',
      line: { color: COLOR_PALETTE.chip_stock, width: 1, dash: 'dash' },
      type: 'scatter',
      showlegend: false,
      hovertemplate: 'Year: %{x}<br>75th percentile: %{y:.2s} H100e<extra></extra>',
    });

    // Dummy trace for percentile range legend entry
    traces.push({
      x: [years[0], years[1]],
      y: [null, null],
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      fillcolor: hexToRgba(COLOR_PALETTE.chip_stock, 0.2),
      line: { color: 'transparent' },
      name: '25th-75th %tile    ',
      showlegend: true,
      hoverinfo: 'skip',
    });

    // Domestic production line
    if (domestic?.median?.length && proportionDomestic) {
      const domesticHoverText = years.map((year, idx) =>
        `Year: ${year}<br>Domestically produced: ${domestic.median[idx]?.toExponential(2)} H100e<br>(${(proportionDomestic[idx] * 100).toFixed(1)}% of total)`
      );

      traces.push({
        x: years,
        y: domestic.median,
        mode: 'lines',
        line: { color: COLOR_PALETTE.datacenters_and_energy, width: 2, dash: 'dot' },
        type: 'scatter',
        name: 'Domestically produced (median)    ',
        text: domesticHoverText,
        hovertemplate: '%{text}<extra></extra>',
      });
    }

    // Largest AI company line
    if (largestCompany?.length) {
      traces.push({
        x: years,
        y: largestCompany,
        mode: 'lines',
        line: { color: COLOR_PALETTE.fab, width: 2, dash: 'dash' },
        type: 'scatter',
        name: 'Largest AI Company    ',
        hovertemplate: 'Year: %{x}<br>Largest AI Company: %{y:.2s} H100e<extra></extra>',
      });
    }

    return traces;
  }, [initialStock]);

  if (isLoading) {
    return (
      <section className="space-y-6">
        <h1 className="text-2xl font-bold text-gray-800">Initial stock</h1>
        <div className="h-48 bg-gray-100 animate-pulse rounded" />
      </section>
    );
  }

  return (
    <section>
      {/* Title */}
      <h1 id="initialStockSection" className="text-[32px] font-bold text-gray-800 mb-5 mt-10">
        Initial stock
      </h1>

      {/* Description */}
      <p className="section-description">
        States might stash away compute for a covert project before an international AI agreement is in force.
        How much compute could the PRC stash away, and how energy efficient would it be?
      </p>

      {/* Dashboard and Detection Plots */}
      <div className="flex flex-wrap gap-5 items-stretch mb-2">
        {/* Dashboard */}
        <Dashboard style={{ flex: '0 0 auto' }}>
          <DashboardItem
            value={dashboardValues.medianDarkCompute}
            secondary={dashboardValues.medianEnergy}
            label="Initial PRC dark compute stock"
          />
          <DashboardItem
            value={dashboardValues.detectionProb}
            label="Probability of detection"
            sublabel={`Detection means a \u22654x likelihood ratio`}
            size="small"
          />
        </Dashboard>

        {/* Detection Probability Bar Chart */}
        <div className="plot-container" style={{ flex: '1 1 200px', minWidth: 200, padding: 20 }}>
          <div className="plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: 8, marginBottom: 10 }}>
            Detection probability by likelihood ratio
          </div>
          <div style={{ height: 240 }}>
            <PlotlyChart
              data={detectionBarData}
              layout={{
                xaxis: {
                  title: { text: 'Probability of detection', font: { size: CHART_FONT_SIZES.axisTitle } },
                  tickfont: { size: CHART_FONT_SIZES.tickLabel },
                  automargin: true,
                },
                yaxis: {
                  title: { text: 'P(Detection)', standoff: 15, font: { size: CHART_FONT_SIZES.axisTitle } },
                  tickfont: { size: CHART_FONT_SIZES.tickLabel },
                  range: [0, 1],
                  tickformat: '.0%',
                  automargin: true,
                },
                margin: { l: 55, r: 0, t: 0, b: 60 },
              }}
              isLoading={isLoading}
              isEmpty={detectionBarData.length === 0}
            />
          </div>
        </div>

        {/* Initial Compute Stock Distribution */}
        <div className="plot-container" style={{ flex: '1 1 200px', minWidth: 200, padding: 20 }}>
          <div className="plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: 8, marginBottom: 10 }}>
            Initial PRC dark compute distribution
          </div>
          <div style={{ height: 240 }}>
            <PlotlyChart
              data={computeHistogramData}
              layout={{
                xaxis: {
                  title: { text: 'PRC Dark Compute Stock (H100 equivalents)', font: { size: CHART_FONT_SIZES.axisTitle } },
                  tickfont: { size: CHART_FONT_SIZES.tickLabel },
                  type: 'log',
                  automargin: true,
                },
                yaxis: {
                  title: { text: 'Probability', font: { size: CHART_FONT_SIZES.axisTitle } },
                  tickfont: { size: CHART_FONT_SIZES.tickLabel },
                },
                margin: { l: 50, r: 10, t: 10, b: 55 },
              }}
              isLoading={isLoading}
              isEmpty={computeHistogramData.length === 0}
            />
          </div>
        </div>
      </div>

      {/* Breakdown Section - Compute Stock */}
      <div className="breakdown-section">
        <h3 className="breakdown-title">Breaking down initial unreported chip stock</h3>
        <p className="section-description">
          The PRC&apos;s stock of covert compute depends on total PRC compute stock at the agreement start
          and the proportion it diverts to a covert project.
        </p>

        {/* Breakdown Row - Compute Stock with Time Series */}
        <div className="breakdown-plots-row">
          {/* PRC Compute Over Time Plot */}
          <div className="breakdown-item">
            <div className="breakdown-plot">
              <PlotlyChart
                data={prcComputeOverTimeData}
                layout={{
                  xaxis: {
                    title: { text: 'Year', font: { size: CHART_FONT_SIZES.axisTitle } },
                    tickfont: { size: CHART_FONT_SIZES.tickLabel },
                    automargin: true,
                    range: initialStock?.prc_compute_years?.length ? [
                      initialStock.prc_compute_years[0],
                      initialStock.prc_compute_years[initialStock.prc_compute_years.length - 1]
                    ] : undefined,
                  },
                  yaxis: {
                    title: { text: 'PRC chip stock (H100e)', font: { size: CHART_FONT_SIZES.axisTitle } },
                    tickfont: { size: CHART_FONT_SIZES.tickLabel },
                    type: 'log',
                    automargin: true,
                  },
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
                  margin: { l: 50, r: 20, t: 10, b: 55, pad: 10 },
                }}
                isLoading={isLoading}
                isEmpty={prcComputeOverTimeData.length === 0}
              />
            </div>
            <div className="breakdown-label">PRC chip stock over time</div>
            <div className="breakdown-description">
              Assuming the PRC has <ParamValue paramKey="totalPrcComputeTppH100eIn2025" parameters={parameters} /> H100e in 2025, and its chip stock grows by <ParamValue paramKey="annualGrowthRateOfPrcComputeStock" parameters={parameters} /> per year.
            </div>
          </div>

          {/* Arrow with dynamic text */}
          <div className="operator arrow-operator" style={{ height: 240, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 5 }}>
            <div style={{ fontSize: 13, color: '#666', textAlign: 'center', whiteSpace: 'nowrap' }}>
              Plug in {agreementYear}
            </div>
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#666' }}>{'\u2192'}</div>
          </div>

          {/* Initial PRC compute stock */}
          <div className="breakdown-item">
            <div className="breakdown-plot">
              <PlotlyChart
                data={prcStockHistogramData}
                layout={{
                  xaxis: {
                    title: { text: 'H100 equivalents', font: { size: CHART_FONT_SIZES.axisTitle } },
                    tickfont: { size: CHART_FONT_SIZES.tickLabel },
                    type: 'log',
                  },
                  yaxis: {
                    title: { text: 'Probability', font: { size: CHART_FONT_SIZES.axisTitle } },
                    tickfont: { size: CHART_FONT_SIZES.tickLabel },
                  },
                  margin: { l: 45, r: 10, t: 10, b: 45 },
                }}
                isLoading={isLoading}
                isEmpty={prcStockHistogramData.length === 0}
              />
            </div>
            <div className="breakdown-label">PRC chip stock at agreement start</div>
            <div className="breakdown-description">
              The total compute stock owned by the PRC at the time the agreement goes into force.
            </div>
          </div>

          {/* Multiply operator */}
          <div className="operator">{'\u00D7'}</div>

          {/* Diversion Proportion Box */}
          <div className="breakdown-box-item">
            <div className="breakdown-box">
              <div className="breakdown-box-inner">
                <ParamValue paramKey="proportionOfInitialChipStockToDivert" parameters={parameters} />
              </div>
            </div>
            <div className="breakdown-label" style={{ marginTop: 5 }}>
              Proportion diverted<br />to covert project
            </div>
          </div>

          {/* Equals operator */}
          <div className="operator">=</div>

          {/* Dark Compute Result Plot */}
          <div className="breakdown-item">
            <div className="breakdown-plot">
              <PlotlyChart
                data={computeHistogramData}
                layout={{
                  xaxis: {
                    title: { text: 'H100 equivalents', font: { size: CHART_FONT_SIZES.axisTitle } },
                    tickfont: { size: CHART_FONT_SIZES.tickLabel },
                    type: 'log',
                  },
                  yaxis: {
                    title: { text: 'Probability', font: { size: CHART_FONT_SIZES.axisTitle } },
                    tickfont: { size: CHART_FONT_SIZES.tickLabel },
                  },
                  margin: { l: 45, r: 10, t: 10, b: 45 },
                }}
                isLoading={isLoading}
                isEmpty={computeHistogramData.length === 0}
              />
            </div>
            <div className="breakdown-label">Initial PRC unreported chip stock</div>
            <div className="breakdown-description">
              The resulting compute stock diverted to the covert project, which is hidden from international monitoring.
            </div>
          </div>
        </div>

        {/* Energy Efficiency Description */}
        <p className="section-description" style={{ marginTop: 40 }}>
          The energy efficiency of these AI chips is important because it determines how much covert
          datacenter capacity the PRC needs to operate them.
        </p>

        {/* Energy Breakdown Row */}
        <div className="breakdown-plots-row">
          {/* H100 Energy Box */}
          <div className="breakdown-box-item">
            <div className="breakdown-box">
              <div className="breakdown-box-inner">
                {H100_POWER_KW.toFixed(2)} kW
              </div>
            </div>
            <div className="breakdown-label" style={{ marginTop: 5 }}>
              Energy consumption of H100
            </div>
          </div>

          {/* Multiply operator */}
          <div className="operator">{'\u00D7'}</div>

          {/* Dark Compute Stock Plot */}
          <div className="breakdown-item">
            <div className="breakdown-plot">
              <PlotlyChart
                data={computeHistogramData}
                layout={{
                  xaxis: {
                    title: { text: 'Dark Compute Stock (H100e)', font: { size: CHART_FONT_SIZES.axisTitle } },
                    tickfont: { size: CHART_FONT_SIZES.tickLabel },
                    type: 'log',
                  },
                  yaxis: {
                    title: { text: 'Probability', font: { size: CHART_FONT_SIZES.axisTitle } },
                    tickfont: { size: CHART_FONT_SIZES.tickLabel },
                  },
                  margin: { l: 40, r: 5, t: 5, b: 45 },
                }}
                isLoading={isLoading}
                isEmpty={computeHistogramData.length === 0}
              />
            </div>
            <div className="breakdown-label">Initial PRC unreported chip stock</div>
            <div className="breakdown-description">
              The compute stock diverted to the covert project.
            </div>
          </div>

          {/* Divide operator */}
          <div className="operator">{'\u00F7'}</div>

          {/* Open parenthesis - hidden on mobile */}
          <div className="operator desktop-only" style={{ fontSize: 120, fontWeight: 300, lineHeight: '250px' }}>(</div>

          {/* SOTA Efficiency Box */}
          <div className="breakdown-box-item">
            <div className="breakdown-box">
              <div className="breakdown-box-inner">
                {sotaEfficiency.toFixed(2)}
              </div>
            </div>
            <div className="breakdown-label" style={{ marginTop: 5, maxWidth: 140 }}>
              State of the art energy efficiency relative to H100 in {agreementYear}
            </div>
          </div>

          {/* Multiply operator */}
          <div className="operator">{'\u00D7'}</div>

          {/* PRC Efficiency Box */}
          <div className="breakdown-box-item">
            <div className="breakdown-box">
              <div className="breakdown-box-inner">
                <ParamValue paramKey="energyEfficiencyOfComputeStockRelativeToStateOfTheArt" parameters={parameters} />
              </div>
            </div>
            <div className="breakdown-label" style={{ marginTop: 5, maxWidth: 140 }}>
              PRC energy efficiency relative to state of the art
            </div>
          </div>

          {/* Close parenthesis - hidden on mobile */}
          <div className="operator desktop-only" style={{ fontSize: 120, fontWeight: 300, lineHeight: '250px' }}>)</div>

          {/* Equals operator */}
          <div className="operator">=</div>

          {/* Energy Requirements Plot */}
          <div className="breakdown-item">
            <div className="breakdown-plot">
              <PlotlyChart
                data={energyRequirementsData}
                layout={{
                  xaxis: {
                    title: { text: 'Energy Requirements (GW)', font: { size: CHART_FONT_SIZES.axisTitle } },
                    tickfont: { size: CHART_FONT_SIZES.tickLabel },
                  },
                  yaxis: {
                    title: { text: 'Probability', font: { size: CHART_FONT_SIZES.axisTitle } },
                    tickfont: { size: CHART_FONT_SIZES.tickLabel },
                  },
                  margin: { l: 40, r: 5, t: 5, b: 45 },
                }}
                isLoading={isLoading}
                isEmpty={energyRequirementsData.length === 0}
              />
            </div>
            <div className="breakdown-label">Energy requirements of initial stock (GW)</div>
            <div className="breakdown-description">
              Total energy required to power the initial unreported chip stock.
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
