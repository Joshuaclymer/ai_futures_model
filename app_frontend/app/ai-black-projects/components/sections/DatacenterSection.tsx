'use client';

import { useMemo } from 'react';
import { BlackProjectData, COLOR_PALETTE } from '@/types/blackProject';
import { CCDFChart, TimeSeriesChart, PlotlyChart } from '../charts';
import { useTooltip, Tooltip, TOOLTIP_DOCS } from '../ui/Tooltip';

interface DatacenterSectionProps {
  data: BlackProjectData | null;
  isLoading?: boolean;
  agreementYear?: number;
}

// Helper to format capacity values
function formatCapacity(gw: number): string {
  if (gw >= 1) {
    return `${gw.toFixed(1)} GW`;
  } else if (gw >= 0.001) {
    return `${(gw * 1000).toFixed(0)} MW`;
  } else {
    return `${(gw * 1000).toFixed(1)} MW`;
  }
}

// Helper to convert hex to rgba
function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Generate dummy data for PRC datacenter capacity over time
function generateDummyPrcCapacity(agreementYear: number) {
  const years = [];
  const median = [];
  const p25 = [];
  const p75 = [];

  for (let year = 2020; year <= agreementYear; year++) {
    years.push(year);
    // Simulate exponential growth in datacenter capacity
    const baseValue = 50 * Math.pow(1.15, year - 2020); // ~15% annual growth
    median.push(baseValue);
    p25.push(baseValue * 0.8);
    p75.push(baseValue * 1.2);
  }

  return { years, median, p25, p75 };
}

// Generate dummy data for datacenter capacity built for concealment
function generateDummyDatacenterCapacity(agreementYear: number) {
  const years = [];
  const median = [];
  const p25 = [];
  const p75 = [];

  for (let year = agreementYear; year <= agreementYear + 10; year++) {
    years.push(year);
    // Capacity grows linearly from construction
    const yearsFromStart = year - agreementYear;
    const baseValue = yearsFromStart * 5; // 5 GW per year
    median.push(Math.min(baseValue, 50)); // Cap at 50 GW
    p25.push(Math.min(baseValue * 0.7, 40));
    p75.push(Math.min(baseValue * 1.3, 60));
  }

  return { years, median, p25, p75 };
}

export function DatacenterSection({ data, isLoading, agreementYear = 2027 }: DatacenterSectionProps) {
  const { tooltipState, showTooltip, hideTooltip } = useTooltip();

  // Helper to create tooltip handlers - pass the markdown doc name as string
  const createTooltipHandlers = (docName: keyof typeof TOOLTIP_DOCS) => ({
    onMouseEnter: (e: React.MouseEvent) => showTooltip(TOOLTIP_DOCS[docName], e),
    onMouseLeave: hideTooltip,
  });

  // Calculate dashboard values from individual simulation data
  const dashboardValues = useMemo(() => {
    if (!data?.black_datacenters) {
      return { capacity: '--', time: '--' };
    }

    const dc = data.black_datacenters;
    const capacities = dc.individual_capacity_before_detection || [];
    const times = dc.individual_time_before_detection || [];

    if (capacities.length === 0) {
      return { capacity: '--', time: '--' };
    }

    // Calculate median (50th percentile)
    const sortedCapacities = [...capacities].sort((a, b) => a - b);
    const sortedTimes = [...times].sort((a, b) => a - b);
    const medianIdx = Math.floor(sortedCapacities.length * 0.5);
    const medianCapacity = sortedCapacities[medianIdx] || 0;
    const medianTime = sortedTimes[medianIdx] || 0;

    return {
      capacity: formatCapacity(medianCapacity),
      time: medianTime.toFixed(1),
    };
  }, [data]);

  // Build combined plot data (capacity + LR over time)
  const combinedPlotData = useMemo((): Plotly.Data[] => {
    if (!data?.black_datacenters || !data?.black_project_model) {
      return [];
    }

    const dc = data.black_datacenters;
    const model = data.black_project_model;
    const years = dc.years || model.years || [];

    if (years.length === 0) return [];

    const capacityColor = COLOR_PALETTE.datacenters_and_energy;
    const lrColor = COLOR_PALETTE.detection;

    const traces: Plotly.Data[] = [];

    // LR percentile band
    if (dc.lr_datacenters?.p25 && dc.lr_datacenters?.p75) {
      traces.push({
        x: years,
        y: dc.lr_datacenters.p25,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'transparent' },
        showlegend: false,
        hoverinfo: 'skip',
        yaxis: 'y',
      });
      traces.push({
        x: years,
        y: dc.lr_datacenters.p75,
        type: 'scatter',
        mode: 'lines',
        fill: 'tonexty',
        fillcolor: hexToRgba(lrColor, 0.2),
        line: { color: 'transparent' },
        showlegend: false,
        hoverinfo: 'skip',
        yaxis: 'y',
      });
    }

    // LR median line
    if (dc.lr_datacenters?.median) {
      traces.push({
        x: years,
        y: dc.lr_datacenters.median,
        type: 'scatter',
        mode: 'lines',
        line: { color: lrColor, width: 3 },
        name: 'Evidence of Datacenters',
        hovertemplate: 'LR: %{y:.2f}<extra></extra>',
        yaxis: 'y',
      });
    }

    // Capacity percentile band
    if (dc.datacenter_capacity?.p25 && dc.datacenter_capacity?.p75) {
      traces.push({
        x: years,
        y: dc.datacenter_capacity.p75,
        type: 'scatter',
        mode: 'lines',
        line: { width: 0 },
        showlegend: false,
        hoverinfo: 'skip',
        yaxis: 'y2',
      });
      traces.push({
        x: years,
        y: dc.datacenter_capacity.p25,
        type: 'scatter',
        mode: 'lines',
        fill: 'tonexty',
        fillcolor: hexToRgba(capacityColor, 0.2),
        line: { width: 0 },
        showlegend: false,
        hoverinfo: 'skip',
        yaxis: 'y2',
      });
    }

    // Capacity median line
    if (dc.datacenter_capacity?.median) {
      traces.push({
        x: years,
        y: dc.datacenter_capacity.median,
        type: 'scatter',
        mode: 'lines',
        line: { color: capacityColor, width: 3 },
        name: 'Covert Datacenter capacity',
        hovertemplate: 'Capacity: %{y:.2f} GW<extra></extra>',
        yaxis: 'y2',
      });
    }

    return traces;
  }, [data]);

  const combinedPlotLayout: Partial<Plotly.Layout> = {
    xaxis: {
      title: { text: 'Year', font: { size: 13 } },
      tickfont: { size: 10 },
    },
    yaxis: {
      title: { text: 'Evidence (LR)', font: { size: 13 } },
      tickfont: { size: 10 },
      side: 'left',
      type: 'log',
    },
    yaxis2: {
      title: { text: 'Capacity (GW)', font: { size: 13 } },
      tickfont: { size: 10 },
      overlaying: 'y',
      side: 'right',
    },
    showlegend: true,
    legend: {
      x: 0.98,
      y: 0.02,
      xanchor: 'right',
      yanchor: 'bottom',
      orientation: 'v',
      font: { size: 10 },
      bgcolor: 'rgba(255,255,255,0.8)',
      borderwidth: 1,
      bordercolor: '#ddd',
    },
    hovermode: 'x unified',
    margin: { l: 55, r: 55, t: 10, b: 50 },
  };

  // Get CCDF data for capacity - handle both [{x, y}, ...] and {x: [], y: []} formats
  const capacityCcdfData = useMemo(() => {
    if (!data?.black_datacenters?.capacity_ccdfs) return undefined;
    const ccdfs = data.black_datacenters.capacity_ccdfs;

    // Helper to normalize CCDF data to [{x, y}, ...] format
    const normalizeCcdf = (ccdf: unknown): { x: number; y: number }[] | undefined => {
      if (!ccdf) return undefined;

      // Already in [{x, y}, ...] format
      if (Array.isArray(ccdf) && ccdf.length > 0 && 'x' in ccdf[0] && 'y' in ccdf[0]) {
        return ccdf as { x: number; y: number }[];
      }

      // In {x: [...], y: [...]} format - transform it
      const obj = ccdf as { x?: number[]; y?: number[] };
      if (obj.x && obj.y && Array.isArray(obj.x) && Array.isArray(obj.y)) {
        return obj.x.map((xVal, i) => ({ x: xVal, y: obj.y![i] }));
      }

      return undefined;
    };

    // Try threshold 4 first (used for "detection"), then 1
    // Keys can be strings or numbers depending on API
    const ccdfRecord = ccdfs as Record<string, unknown>;
    if (ccdfRecord['4'] || ccdfRecord[4]) return normalizeCcdf(ccdfRecord['4'] || ccdfRecord[4]);
    if (ccdfRecord['1'] || ccdfRecord[1]) return normalizeCcdf(ccdfRecord['1'] || ccdfRecord[1]);
    const firstKey = Object.keys(ccdfRecord)[0];
    if (firstKey) return normalizeCcdf(ccdfRecord[firstKey]);
    return undefined;
  }, [data]);

  if (isLoading) {
    return (
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-800">Covert Datacenters</h2>
        <div className="h-48 bg-gray-100 animate-pulse rounded" />
      </section>
    );
  }

  if (!data?.black_datacenters) {
    return (
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-800">Covert Datacenters</h2>
        <p className="text-gray-500">No datacenter data available</p>
      </section>
    );
  }

  return (
    <section className="space-y-6">
      {/* Title */}
      <h2 className="text-2xl font-bold text-gray-800">Covert Datacenters</h2>

      {/* Description */}
      <p className="text-gray-600">
        The typical datacenter is visible from satellites because of its cooling and electrical equipment.
        However, states could design datacenters to be much more stealthy. For example, the PRC might build
        datacenters in ordinary warehouse buildings, power them with shipments of natural gas, and diffuse
        thermal output underneath bodies of water. If the PRC builds datacenters designed for stealth, how
        much energy capacity could they provide?
      </p>

      {/* Top Section: Dashboard + Plots */}
      <div className="flex flex-wrap gap-5 items-start">
        {/* Dashboard */}
        <div className="bp-dashboard w-60 flex-shrink-0">
          <div className="bp-plot-title">Median outcome</div>
          <div className="space-y-5 mt-4">
            <div className="bp-dashboard-item">
              <div className="bp-dashboard-value" style={{ color: COLOR_PALETTE.datacenters_and_energy }}>
                {dashboardValues.capacity}
              </div>
              <div className="bp-dashboard-label">Datacenter capacity built before detection</div>
              <div className="text-xs text-gray-400">Detection means a ≥4x update</div>
            </div>
            <div className="bp-dashboard-item">
              <div className="bp-dashboard-value-small">{dashboardValues.time}</div>
              <div className="bp-dashboard-label-light">Years operational before detection</div>
            </div>
          </div>
        </div>

        {/* Capacity CCDF Plot */}
        <div className="bp-plot-container flex-1 min-w-[280px]">
          <div className="bp-plot-title">Datacenter capacity built before detection</div>
          <div className="h-[300px]">
            <CCDFChart
              data={capacityCcdfData}
              color={COLOR_PALETTE.datacenters_and_energy}
              xLabel="Datacenter Capacity (GW)"
              yLabel="P(capacity > x)"
              xLogScale={true}
              isLoading={isLoading}
            />
          </div>
        </div>

        {/* Combined Simulation Runs Plot */}
        <div className="bp-plot-container flex-1 min-w-[280px]">
          <div className="bp-plot-title">Simulation runs</div>
          <div className="h-[300px]">
            <PlotlyChart
              data={combinedPlotData}
              layout={combinedPlotLayout}
              isLoading={isLoading}
              isEmpty={combinedPlotData.length === 0}
            />
          </div>
        </div>
      </div>

      {/* Breaking Down Section */}
      <div className="breakdown-section" style={{ marginTop: '30px' }}>
        <h3 className="text-xl font-semibold text-gray-800 mb-4" style={{ textAlign: 'left' }}>
          Breaking down covert datacenter capacity
        </h3>
        <div className="section-description">
          <p>There are two ways to create covert data center capacity:</p>
          <ol style={{ margin: '10px 0 10px 20px' }}>
            <li>Retrofit existing datacenters that were not built with concealment in mind.</li>
            <li>Build datacenters that are designed to be stealthy.</li>
          </ol>
          <p style={{ marginBottom: '10px' }}>
            Retrofitting unconcealed datacenters isn&apos;t a good strategy, since the vast majority of
            these datacenters will be easy to spot with satellites.
          </p>
        </div>

        {/* Mobile message */}
        <p className="mobile-only text-gray-500 text-sm italic" style={{ marginTop: '10px' }}>
          View on desktop for a detailed breakdown.
        </p>

        {/* Detailed breakdown - hidden on mobile */}
        <div className="detailed-breakdown-sections">

        {/* Unconcealed datacenter capacity breakdown row */}
        <div className="breakdown-plots-row" style={{ marginBottom: '30px' }}>
          {/* PRC datacenter capacity over time plot */}
          <div className="breakdown-item">
            <div className="breakdown-plot">
              {(() => {
                // Use real data if available, otherwise use dummy data
                const hasRealData = data.black_datacenters.prc_capacity_years?.length > 0 &&
                                    data.black_datacenters.prc_capacity_gw?.median?.length > 0;
                const plotData = hasRealData
                  ? {
                      years: data.black_datacenters.prc_capacity_years,
                      median: data.black_datacenters.prc_capacity_gw?.median,
                      p25: data.black_datacenters.prc_capacity_gw?.p25,
                      p75: data.black_datacenters.prc_capacity_gw?.p75,
                    }
                  : generateDummyPrcCapacity(agreementYear);
                return (
                  <TimeSeriesChart
                    years={plotData.years}
                    median={plotData.median}
                    p25={plotData.p25}
                    p75={plotData.p75}
                    color={COLOR_PALETTE.datacenters_and_energy}
                    yLabel="GW"
                    isLoading={isLoading}
                  />
                );
              })()}
            </div>
            <div className="breakdown-label">Unconcealed PRC datacenter capacity</div>
            <div className="breakdown-description">The capacity of datacenters not built with concealment in mind.</div>
          </div>

          {/* Arrow with "Plug in year" */}
          <div className="operator" style={{ position: 'relative', height: '240px', width: '100px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#666' }}>→</div>
            <div style={{ position: 'absolute', top: '80px', fontSize: '12px', color: '#666', whiteSpace: 'nowrap' }}>
              Plug in {agreementYear}
            </div>
          </div>

          {/* Total PRC datacenter capacity at agreement year */}
          <div
            className="breakdown-box-item has-tooltip"
            style={{ cursor: 'pointer' }}
            {...createTooltipHandlers('prc_capacity')}
          >
            <div className="breakdown-box">
              <div className="breakdown-box-inner" style={{ color: COLOR_PALETTE.datacenters_and_energy }}>
                {data.black_datacenters.prc_capacity_at_agreement_year_gw
                  ? formatCapacity(data.black_datacenters.prc_capacity_at_agreement_year_gw)
                  : formatCapacity(generateDummyPrcCapacity(agreementYear).median.slice(-1)[0])}
              </div>
            </div>
            <div className="breakdown-label">Unconcealed PRC datacenter<br/>capacity in {agreementYear}</div>
          </div>

          <div className="operator">×</div>

          {/* Proportion diverted */}
          <div
            className="breakdown-box-item has-tooltip"
            style={{ cursor: 'pointer' }}
            {...createTooltipHandlers('fraction_diverted')}
          >
            <div className="breakdown-box">
              <div className="breakdown-box-inner" style={{ color: COLOR_PALETTE.datacenters_and_energy }}>
                {`${((data.black_datacenters.fraction_diverted || 0.01) * 100).toFixed(0)}%`}
              </div>
            </div>
            <div className="breakdown-label">Proportion diverted<br/>to covert project</div>
          </div>

          <div className="operator">=</div>

          {/* Covert datacenter capacity not built for concealment */}
          <div
            className="breakdown-box-item has-tooltip"
            style={{ cursor: 'pointer' }}
            {...createTooltipHandlers('retrofitted_capacity')}
          >
            <div className="breakdown-box">
              <div className="breakdown-box-inner" style={{ color: COLOR_PALETTE.datacenters_and_energy }}>
                {(() => {
                  const prcCapacity = data.black_datacenters.prc_capacity_at_agreement_year_gw || generateDummyPrcCapacity(agreementYear).median.slice(-1)[0];
                  const fractionDiverted = data.black_datacenters.fraction_diverted || 0.01;
                  return formatCapacity(prcCapacity * fractionDiverted);
                })()}
              </div>
            </div>
            <div className="breakdown-label">Capacity of datacenters<br/>retrofitted for secrecy</div>
          </div>
        </div>

        {/* Second explanation */}
        <div className="section-description" style={{ marginTop: '30px' }}>
          <p>
            A much better strategy is to build datacenters that are designed for stealth. The PRC builds
            these datacenters as quickly as it can do so with a small workforce. However, if the PRC builds
            too much covert datacenter capacity, energy consumption becomes hard to hide. So the PRC only
            builds covert datacenters up to some maximum proportion of total PRC energy consumption.
          </p>
        </div>

        {/* Second breakdown row: Min[] formula for datacenters built for concealment */}
        <div className="breakdown-plots-row">
          {/* Min function start bracket */}
          <div className="operator" style={{ position: 'relative', height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#666' }}>Min[</div>
          </div>

          {/* Total PRC energy consumption box */}
          <div
            className="breakdown-box-item has-tooltip"
            style={{ cursor: 'pointer' }}
            {...createTooltipHandlers('prc_energy')}
          >
            <div className="breakdown-box">
              <div className="breakdown-box-inner" style={{ color: COLOR_PALETTE.datacenters_and_energy, fontSize: '20px' }}>
                {data.black_datacenters.total_prc_energy_gw
                  ? `${(data.black_datacenters.total_prc_energy_gw / 1000).toFixed(0)}K GW`
                  : '1K GW'}
              </div>
            </div>
            <div className="breakdown-label">PRC energy<br/>consumption</div>
          </div>

          <div className="operator">×</div>

          {/* Max proportion box */}
          <div
            className="breakdown-box-item has-tooltip"
            style={{ cursor: 'pointer' }}
            {...createTooltipHandlers('max_energy_proportion')}
          >
            <div className="breakdown-box">
              <div className="breakdown-box-inner" style={{ color: COLOR_PALETTE.datacenters_and_energy, fontSize: '20px' }}>
                {data.black_datacenters.max_proportion_energy
                  ? `${(data.black_datacenters.max_proportion_energy * 100).toFixed(0)}%`
                  : '1%'}
              </div>
            </div>
            <div className="breakdown-label">Max % energy</div>
          </div>

          <div className="operator">−</div>

          {/* Covert unconcealed capacity box */}
          <div
            className="breakdown-box-item has-tooltip"
            style={{ cursor: 'pointer' }}
            {...createTooltipHandlers('covert_unconcealed')}
          >
            <div className="breakdown-box">
              <div className="breakdown-box-inner" style={{ color: COLOR_PALETTE.datacenters_and_energy, fontSize: '20px' }}>
                {(() => {
                  const prcCapacity = data.black_datacenters.prc_capacity_at_agreement_year_gw || generateDummyPrcCapacity(agreementYear).median.slice(-1)[0];
                  const fractionDiverted = data.black_datacenters.fraction_diverted || 0.01;
                  return formatCapacity(prcCapacity * fractionDiverted);
                })()}
              </div>
            </div>
            <div className="breakdown-label">Covert unconcealed<br/>capacity</div>
          </div>

          <div className="operator">,</div>

          {/* Number of workers box */}
          <div
            className="breakdown-box-item has-tooltip"
            style={{ cursor: 'pointer' }}
            {...createTooltipHandlers('construction_workers')}
          >
            <div className="breakdown-box">
              <div className="breakdown-box-inner" style={{ color: COLOR_PALETTE.datacenters_and_energy, fontSize: '20px' }}>
                {data.black_datacenters.construction_workers
                  ? `${(data.black_datacenters.construction_workers / 1000).toFixed(0)}K`
                  : '10K'}
              </div>
            </div>
            <div className="breakdown-label">Construction<br/>workers</div>
          </div>

          <div className="operator">×</div>

          {/* MW per worker box */}
          <div
            className="breakdown-box-item has-tooltip"
            style={{ cursor: 'pointer' }}
            {...createTooltipHandlers('mw_per_worker')}
          >
            <div className="breakdown-box">
              <div className="breakdown-box-inner" style={{ color: COLOR_PALETTE.datacenters_and_energy, fontSize: '20px' }}>
                {data.black_datacenters.mw_per_worker_per_year?.toFixed(2) || '0.20'}
              </div>
            </div>
            <div className="breakdown-label">MW built per<br/>worker per year</div>
          </div>

          <div className="operator">× (year −</div>

          {/* Start year box */}
          <div
            className="breakdown-box-item has-tooltip"
            style={{ cursor: 'pointer' }}
            {...createTooltipHandlers('datacenter_start_year')}
          >
            <div className="breakdown-box">
              <div className="breakdown-box-inner" style={{ color: COLOR_PALETTE.datacenters_and_energy, fontSize: '20px' }}>
                {data.black_datacenters.datacenter_start_year || (agreementYear - 2)}
              </div>
            </div>
            <div className="breakdown-label">Year PRC starts building<br/>covert datacenters</div>
          </div>

          <div className="operator">)</div>

          {/* Min function end bracket */}
          <div className="operator" style={{ position: 'relative', height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#666' }}>]</div>
          </div>

          <div className="operator">=</div>

          {/* Datacenter capacity plot */}
          <div className="breakdown-item">
            <div className="breakdown-plot">
              {(() => {
                // Use real data if available, otherwise use dummy data
                const hasRealData = data.black_datacenters.years?.length > 0 &&
                                    data.black_datacenters.datacenter_capacity?.median?.length > 0;
                const plotData = hasRealData
                  ? {
                      years: data.black_datacenters.years,
                      median: data.black_datacenters.datacenter_capacity?.median,
                      p25: data.black_datacenters.datacenter_capacity?.p25,
                      p75: data.black_datacenters.datacenter_capacity?.p75,
                    }
                  : generateDummyDatacenterCapacity(agreementYear);
                return (
                  <TimeSeriesChart
                    years={plotData.years}
                    median={plotData.median}
                    p25={plotData.p25}
                    p75={plotData.p75}
                    color={COLOR_PALETTE.datacenters_and_energy}
                    yLabel="GW"
                    isLoading={isLoading}
                  />
                );
              })()}
            </div>
            <div className="breakdown-label">Capacity of datacenters built for concealment</div>
            <div className="breakdown-description">The cumulative capacity of covert datacenters built by the construction workforce, capped by maximum energy consumption limits.</div>
          </div>
        </div>
        </div>{/* End detailed-breakdown-sections */}
      </div>

      {/* Global tooltip */}
      <Tooltip
        content={tooltipState.content}
        visible={tooltipState.visible}
        position={tooltipState.position}
      />
    </section>
  );
}
