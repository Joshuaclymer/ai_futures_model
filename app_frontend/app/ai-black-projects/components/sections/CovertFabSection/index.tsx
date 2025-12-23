'use client';

import { TimeSeriesChart, CCDFChart, PlotlyChart, PDFChart } from '../../charts';
import { COLOR_PALETTE, DETECTION_THRESHOLD_COLORS, hexToRgba } from '../../colors';
import { CHART_FONT_SIZES } from '../../chartConfig';
import { useTooltip, Tooltip, TOOLTIP_DOCS, ParamLink, ParamValue, Dashboard, DashboardItem } from '../../ui';
import { Parameters, SimulationData } from '../../../types';

// Consistent loading indicator style (matches "No data available" style)
function LoadingIndicator() {
  return <div className="flex items-center justify-center h-full text-gray-400 text-sm">Loading...</div>;
}

// Types for the covert fab data from API
interface TimeSeriesData {
  years: number[];
  median: number[];
  p25: number[];
  p75: number[];
}

interface DashboardData {
  production: string;
  energy: string;
  probFabBuilt: string;
  yearsOperational: string;
  processNode: string;
}

interface CCDFPoint {
  x: number;
  y: number;
}

export interface CovertFabApiData {
  dashboard?: DashboardData;
  compute_ccdf?: Record<number | string, CCDFPoint[]>;
  time_series_data?: {
    years: number[];
    lr_combined: TimeSeriesData;
    h100e_flow: TimeSeriesData;
  };
  is_operational?: TimeSeriesData;
  wafer_starts_samples?: number[];
  chips_per_wafer?: number;
  architecture_efficiency?: number;
  h100_power?: number;
  transistor_density?: { node: string; density: number; wattsPerTpp?: number; probability?: number }[];
  compute_per_month?: TimeSeriesData;
  watts_per_tpp_curve?: {
    densityRelative: number[];
    wattsPerTppRelative: number[];
  };
  energy_per_month?: TimeSeriesData;
}

interface CovertFabSectionProps {
  data: SimulationData | null;
  isLoading?: boolean;
  parameters: Parameters;
  covertFabData?: CovertFabApiData | null;
}


// CovertFab Dashboard wrapper using shared Dashboard component
function CovertFabDashboard({ dashboard }: { dashboard?: DashboardData }) {
  if (!dashboard) return <div className="bp-dashboard" style={{ width: '240px' }}><LoadingIndicator /></div>;
  return (
    <Dashboard>
      <DashboardItem
        value={dashboard.production}
        secondary={dashboard.energy}
        label="Production before detection"
        sublabel="Detection means ≥5x update"
      />
      <DashboardItem
        value={dashboard.probFabBuilt}
        label="Probability covert fab is built"
        size="small"
      />
      <DashboardItem
        value={dashboard.yearsOperational}
        label="Years operational before detection"
        size="small"
      />
      <DashboardItem
        value={dashboard.processNode}
        label="Process node"
        size="small"
      />
    </Dashboard>
  );
}

// CCDF Chart for compute produced before detection
function ComputeCCDFChart({ ccdfData }: { ccdfData?: Record<number, CCDFPoint[]> }) {
  const thresholds = [
    { value: 4, label: '"Detection" = >4x update        ', color: DETECTION_THRESHOLD_COLORS['4'] },
    { value: 2, label: '"Detection" = >2x update        ', color: DETECTION_THRESHOLD_COLORS['2'] },
    { value: 1, label: '"Detection" = >1x update        ', color: DETECTION_THRESHOLD_COLORS['1'] },
  ];

  // Filter out thresholds that don't have data
  const validThresholds = ccdfData ? thresholds.filter(t => ccdfData[t.value]?.length > 0) : [];

  const traces: Plotly.Data[] = validThresholds.map(t => ({
    x: ccdfData![t.value].map(d => d.x),
    y: ccdfData![t.value].map(d => d.y),
    type: 'scatter' as const,
    mode: 'lines' as const,
    line: { color: t.color, width: 2 },
    name: t.label,
    hovertemplate: 'H100e: %{x:.0f}<br>P(\u2265x): %{y:.3f}<extra></extra>',
  }));

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: {
          title: { text: "H100 equivalents (FLOPS) Produced by Covert Fab Before 'Detection'", font: { size: CHART_FONT_SIZES.axisTitle } },
          type: 'log',
          tickfont: { size: CHART_FONT_SIZES.tickLabel },
        },
        yaxis: {
          title: { text: 'P(covert compute > x)', font: { size: CHART_FONT_SIZES.axisTitle } },
          range: [0, 1],
          tickfont: { size: CHART_FONT_SIZES.tickLabel },
        },
        showlegend: true,
        legend: {
          x: 0.98,
          y: 0.98,
          xanchor: 'right',
          yanchor: 'top',
          bgcolor: 'rgba(255,255,248,0.9)',
          borderwidth: 0,
          font: { size: CHART_FONT_SIZES.legend },
        },
        margin: { l: 55, r: 10, t: 0, b: 60 },
      }}
    />
  );
}

// Time series chart for simulation runs
function SimulationRunsChart({ timeSeriesData }: { timeSeriesData?: CovertFabApiData['time_series_data'] }) {
  if (!timeSeriesData) return <LoadingIndicator />;
  const traces: Plotly.Data[] = [
    // LR percentile band
    {
      x: [...timeSeriesData.years, ...timeSeriesData.years.slice().reverse()],
      y: [...timeSeriesData.lr_combined.p75, ...timeSeriesData.lr_combined.p25.slice().reverse()],
      type: 'scatter' as const,
      mode: 'lines' as const,
      fill: 'toself',
      fillcolor: hexToRgba(COLOR_PALETTE.detection, 0.2),
      line: { color: 'transparent' },
      showlegend: false,
      hoverinfo: 'skip' as const,
      name: 'LR IQR',
    },
    // LR median
    {
      x: timeSeriesData.years,
      y: timeSeriesData.lr_combined.median,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: COLOR_PALETTE.detection, width: 2 },
      name: 'LR: Median    ',
      hovertemplate: 'LR: %{y:.2f}<extra></extra>',
    },
    // H100e percentile band (secondary axis)
    {
      x: [...timeSeriesData.years, ...timeSeriesData.years.slice().reverse()],
      y: [...timeSeriesData.h100e_flow.p75, ...timeSeriesData.h100e_flow.p25.slice().reverse()],
      type: 'scatter' as const,
      mode: 'lines' as const,
      fill: 'toself',
      fillcolor: hexToRgba(COLOR_PALETTE.fab, 0.15),
      line: { color: 'transparent' },
      showlegend: false,
      hoverinfo: 'skip' as const,
      yaxis: 'y2',
      name: 'H100e IQR',
    },
    // H100e median (secondary axis)
    {
      x: timeSeriesData.years,
      y: timeSeriesData.h100e_flow.median,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: COLOR_PALETTE.fab, width: 2 },
      name: 'H100e: Median    ',
      yaxis: 'y2',
      hovertemplate: 'H100e: %{y:,.0f}<extra></extra>',
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: {
          title: { text: 'Year', font: { size: CHART_FONT_SIZES.axisTitle } },
          tickfont: { size: CHART_FONT_SIZES.tickLabel },
        },
        yaxis: {
          title: { text: 'Evidence of Covert Fab (LR)', font: { size: CHART_FONT_SIZES.axisTitle } },
          type: 'log',
          side: 'left',
          tickfont: { size: CHART_FONT_SIZES.tickLabel },
        },
        yaxis2: {
          title: { text: 'H100 equivalents (FLOPS) Produced by Fab', font: { size: CHART_FONT_SIZES.axisTitle } },
          overlaying: 'y',
          side: 'right',
          tickfont: { size: CHART_FONT_SIZES.tickLabel },
          tickformat: '.2s',
        },
        showlegend: false,
        hovermode: 'closest',
        margin: { l: 55, r: 55, t: 0, b: 50 },
      }}
    />
  );
}

// Breakdown item component
interface BreakdownItemProps {
  title: string;
  description?: React.ReactNode;
  children: React.ReactNode;
  tooltipKey?: string;
  onMouseEnter?: (e: React.MouseEvent) => void;
  onMouseLeave?: () => void;
}

function BreakdownItem({ title, description, children, tooltipKey, onMouseEnter, onMouseLeave }: BreakdownItemProps) {
  const hasTooltip = tooltipKey && tooltipKey in TOOLTIP_DOCS;

  return (
    <div
      className={`breakdown-item ${hasTooltip ? 'has-tooltip' : ''}`}
      style={hasTooltip ? { cursor: 'pointer' } : undefined}
      onMouseEnter={hasTooltip && onMouseEnter ? onMouseEnter : undefined}
      onMouseLeave={hasTooltip && onMouseLeave ? onMouseLeave : undefined}
    >
      <div className="breakdown-plot">
        {children}
      </div>
      <div className="breakdown-label">
        {title}
      </div>
      {description && (
        <div className="breakdown-description">
          {description}
        </div>
      )}
    </div>
  );
}

// Box item (for constant values)
interface BreakdownBoxProps {
  title: string;
  description?: React.ReactNode;
  value: string | number;
  tooltipKey?: string;
  onMouseEnter?: (e: React.MouseEvent) => void;
  onMouseLeave?: () => void;
}

function BreakdownBox({ title, description, value, tooltipKey, onMouseEnter, onMouseLeave }: BreakdownBoxProps) {
  const hasTooltip = tooltipKey && tooltipKey in TOOLTIP_DOCS;

  return (
    <div
      className={`breakdown-box-item ${hasTooltip ? 'has-tooltip' : ''}`}
      style={{ flex: '0 0 auto', marginTop: '20px', marginBottom: '10px', cursor: hasTooltip ? 'pointer' : 'default' }}
      onMouseEnter={hasTooltip && onMouseEnter ? onMouseEnter : undefined}
      onMouseLeave={hasTooltip && onMouseLeave ? onMouseLeave : undefined}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
        <div className="breakdown-box" style={{ maxWidth: '170px', width: '100%' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
            <div className="breakdown-box-inner" style={{
              background: 'white',
              borderRadius: '4px',
              padding: '20px',
              fontSize: '28px',
              fontWeight: 'bold',
              color: COLOR_PALETTE.chip_stock,
              textAlign: 'center',
              width: '100%',
            }}>
              {value}
            </div>
            <div className="breakdown-label">
              {title}
            </div>
            {description && (
              <div className="breakdown-description">
                {description}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Operator component
function Operator({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div className="operator" style={{
      fontSize: '24px',
      fontWeight: 'bold',
      color: '#666',
      padding: '0 3px',
      height: '240px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      marginTop: '20px',
      ...style,
    }}>
      {children}
    </div>
  );
}

// Is Operational Plot (proportion over time)
function IsOperationalPlot({ data }: { data?: TimeSeriesData }) {
  if (!data) return <LoadingIndicator />;
  return (
    <TimeSeriesChart
      years={data.years}
      median={data.median}
      p25={data.p25}
      p75={data.p75}
      color={COLOR_PALETTE.fab}
      yLabel="Probability"
      showBand={false}
    />
  );
}

// Wafer Starts PDF Plot - uses shared PDFChart component
function WaferStartsPlot({ samples }: { samples?: number[] }) {
  if (!samples || samples.length === 0) return <LoadingIndicator />;
  return (
    <PDFChart
      samples={samples}
      color={COLOR_PALETTE.chip_stock}
      xLabel="Wafers/Month"
      logScale={false}
      numBins={20}
      normalize="density"
    />
  );
}

// Transistor Density Bar Plot by Process Node - shows probability distribution
function TransistorDensityPlot({ data }: { data?: { node: string; density: number; probability?: number }[] }) {
  if (!data || data.length === 0) return <LoadingIndicator />;

  // If probability data is available, show probability distribution
  const hasProbability = data.some(d => d.probability !== undefined && d.probability > 0);

  const traces: Plotly.Data[] = [
    {
      // X-axis: show density with node label (e.g., "0.06x (28nm)")
      x: data.map(d => hasProbability ? `${d.density.toFixed(2)}x (${d.node})` : d.node),
      // Y-axis: show probability if available, otherwise density
      y: data.map(d => hasProbability ? (d.probability || 0) : d.density),
      type: 'bar' as const,
      marker: { color: COLOR_PALETTE.fab },
      hovertemplate: hasProbability
        ? '%{x}: %{y:.1%} probability<extra></extra>'
        : '%{x}: %{y:.2f}x H100<extra></extra>',
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: {
          title: { text: hasProbability ? '' : 'Process Node', font: { size: CHART_FONT_SIZES.axisTitle } },
          tickfont: { size: CHART_FONT_SIZES.tickLabel },
          tickangle: hasProbability ? -30 : 0,
        },
        yaxis: {
          title: { text: hasProbability ? 'Probability' : 'Density (rel. to H100)', font: { size: CHART_FONT_SIZES.axisTitle } },
          tickfont: { size: CHART_FONT_SIZES.tickLabel },
          range: hasProbability ? [0, 1] : undefined,
        },
        bargap: 0.3,
        margin: { l: 50, r: 10, t: 10, b: hasProbability ? 70 : 50 },
      }}
    />
  );
}

// Compute per Month Plot
function ComputePerMonthPlot({ data }: { data?: TimeSeriesData }) {
  if (!data) return <LoadingIndicator />;
  return (
    <TimeSeriesChart
      years={data.years}
      median={data.median}
      p25={data.p25}
      p75={data.p75}
      color={COLOR_PALETTE.fab}
      yLabel="H100e/Month"
      showBand={true}
      bandAlpha={0.15}
    />
  );
}

// H100 reference transistor density (M/mm²)
const H100_TRANSISTOR_DENSITY = 98.28;

// Watts per TPP Curve Plot - computes curve dynamically from parameters
function WattsPerTppPlot({
  simDensities,
  transistorDensityAtEndOfDennard,
  exponentBeforeDennard,
  exponentAfterDennard,
}: {
  simDensities?: { node: string; density: number; wattsPerTpp?: number }[];
  transistorDensityAtEndOfDennard: number; // M/mm²
  exponentBeforeDennard: number; // e.g., -1.0
  exponentAfterDennard: number; // e.g., -0.33
}) {
  if (!simDensities) return <div className="text-gray-400 text-sm">No data available</div>;

  // Filter points that have wattsPerTpp data
  const validPoints = simDensities.filter(d => d.wattsPerTpp !== undefined);

  // Convert Dennard scaling end from M/mm² to relative density (rel. to H100)
  const dennardScalingEnd = transistorDensityAtEndOfDennard / H100_TRANSISTOR_DENSITY;

  // Compute curve dynamically based on parameters
  // W/TPP = density^exponent, normalized so W/TPP = 1 at density = 1 (H100)
  const densityPoints = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0];

  const computeWattsPerTpp = (density: number): number => {
    if (density < dennardScalingEnd) {
      // Before Dennard scaling: steeper exponent
      // Need to ensure continuity at dennardScalingEnd
      const wattsAtDennard = Math.pow(dennardScalingEnd, exponentAfterDennard);
      return wattsAtDennard * Math.pow(density / dennardScalingEnd, exponentBeforeDennard);
    } else {
      // After Dennard scaling: shallower exponent
      return Math.pow(density, exponentAfterDennard);
    }
  };

  const wattsPerTppPoints = densityPoints.map(computeWattsPerTpp);

  const traces: Plotly.Data[] = [
    // Curve
    {
      x: densityPoints,
      y: wattsPerTppPoints,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: '#7DD4C0', width: 2 },
      name: 'Power Law',
      hovertemplate: 'Density: %{x:.2g}x<br>W/TPP: %{y:.2g}x<extra></extra>',
    },
    // Simulation points (using pre-computed wattsPerTpp from API) - only if we have valid points
    ...(validPoints.length > 0 ? [{
      x: validPoints.map(d => d.density),
      y: validPoints.map(d => d.wattsPerTpp!),
      type: 'scatter' as const,
      mode: 'markers' as const,
      marker: { size: 8, color: COLOR_PALETTE.fab },
      name: 'Simulations',
      hovertemplate: 'Density: %{x:.2g}x<br>W/TPP: %{y:.2g}x<extra></extra>',
    }] : []),
  ];

  // Compute y-axis range based on actual data
  const yMin = Math.min(...wattsPerTppPoints.filter(y => y > 0));
  const yMax = Math.max(...wattsPerTppPoints);

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: {
          title: { text: 'Transistor Density (rel. to H100)', font: { size: CHART_FONT_SIZES.axisTitle } },
          type: 'log',
          range: [Math.log10(0.001), Math.log10(100)],
          tickfont: { size: CHART_FONT_SIZES.tickLabel },
        },
        yaxis: {
          title: { text: 'Watts per TPP (rel. to H100)', font: { size: CHART_FONT_SIZES.axisTitle } },
          type: 'log',
          range: [Math.log10(Math.max(0.01, yMin * 0.5)), Math.log10(yMax * 2)],
          tickfont: { size: CHART_FONT_SIZES.tickLabel },
        },
        showlegend: false,
        margin: { l: 60, r: 10, t: 10, b: 50 },
        shapes: [
          // Vertical dashed line for "End of Dennard Scaling"
          {
            type: 'line',
            x0: dennardScalingEnd,
            x1: dennardScalingEnd,
            y0: yMin * 0.5,
            y1: yMax * 2,
            xref: 'x',
            yref: 'y',
            line: {
              color: '#888888',
              width: 1.5,
              dash: 'dash',
            },
          },
        ],
        annotations: [
          // Label for "End of Dennard Scaling"
          {
            x: Math.log10(dennardScalingEnd),
            y: Math.log10(yMax * 0.8),
            xref: 'x',
            yref: 'y',
            text: 'End of Dennard<br>Scaling',
            showarrow: false,
            font: {
              size: 10,
              color: '#666666',
            },
            xanchor: 'right',
            xshift: -5,
          },
        ],
      }}
    />
  );
}

// Energy per Month Plot
function EnergyPerMonthPlot({ data }: { data?: TimeSeriesData }) {
  if (!data) return <LoadingIndicator />;
  return (
    <TimeSeriesChart
      years={data.years}
      median={data.median}
      p25={data.p25}
      p75={data.p75}
      color={COLOR_PALETTE.datacenters_and_energy}
      yLabel="GW/Month"
      showBand={true}
      bandAlpha={0.15}
    />
  );
}

export function CovertFabSection({ data, isLoading, parameters, covertFabData }: CovertFabSectionProps) {
  const agreementYear = parameters.agreementYear;
  const { tooltipState, showTooltip, hideTooltip, onTooltipMouseEnter, onTooltipMouseLeave } = useTooltip();

  if (isLoading) {
    return (
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-800">Covert fab</h2>
        <div className="h-48 bg-gray-100 animate-pulse rounded" />
      </section>
    );
  }

  // Extract data from API response (no client-side fallbacks - data comes from API)
  const dashboard = covertFabData?.dashboard;
  const ccdfData = covertFabData?.compute_ccdf;
  const timeSeriesData = covertFabData?.time_series_data;
  const isOperationalData = covertFabData?.is_operational;
  const waferStartsSamples = covertFabData?.wafer_starts_samples;
  const transistorDensityData = covertFabData?.transistor_density;
  const computePerMonthData = covertFabData?.compute_per_month;
  // wattsPerTppCurveData no longer needed - curve computed dynamically from parameters
  const energyPerMonthData = covertFabData?.energy_per_month;
  const chipsPerWafer = covertFabData?.chips_per_wafer;
  const archEfficiency = covertFabData?.architecture_efficiency;
  const h100Power = covertFabData?.h100_power;

  // Helper to create tooltip handlers - pass markdown doc name as string
  const createTooltipHandlers = (docName: keyof typeof TOOLTIP_DOCS) => ({
    onMouseEnter: (e: React.MouseEvent) => showTooltip(TOOLTIP_DOCS[docName], e),
    onMouseLeave: hideTooltip,
  });

  return (
    <section id="covertFabSection">
      <h1 style={{ fontSize: '32px', fontWeight: 'bold', marginBottom: '20px', marginTop: '40px', color: '#333' }}>
        Covert fab
      </h1>
      <div className="section-description" style={{ marginBottom: '20px', fontSize: '14px', color: '#666', lineHeight: 1.6 }}>
        The semiconductor industry is complex and would be nearly impossible to hide. But the PRC might still build a covert fab if it stashes away the necessary semiconductor manufacturing equipment (SME) <em>before</em> an AI agreement begins. The plots below show simulation runs where the PRC chooses to build a covert fab, which happens in <span style={{ fontWeight: 'bold' }}>56.2%</span> of all simulations. These are the cases where two conditions are met: the PRC can source most equipment (&gt;90%) indigenously (since imports are more traceable), and the PRC can achieve the <span style={{ fontWeight: 'bold' }}>28</span>nm process node or better, which makes chips powerful enough to be worth producing.
      </div>

      {/* Top section: Dashboard + 2 plots */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', marginBottom: '10px', alignItems: 'stretch' }}>
        <CovertFabDashboard dashboard={dashboard} />

        <div className="bp-plot-container" style={{ flex: '1 1 200px', minWidth: '200px', padding: '20px', display: 'flex', flexDirection: 'column' }}>
          <div className="plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: 8, marginBottom: 10 }}>
            Covert compute produced before detection
          </div>
          <div style={{ flex: 1, minHeight: '250px' }}>
            <ComputeCCDFChart ccdfData={ccdfData} />
          </div>
        </div>

        <div className="bp-plot-container" style={{ flex: '1 1 200px', minWidth: '200px', padding: '20px', display: 'flex', flexDirection: 'column' }}>
          <div className="plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: 8, marginBottom: 10 }}>
            Simulation runs
          </div>
          <div style={{ flex: 1, minHeight: '250px' }}>
            <SimulationRunsChart timeSeriesData={timeSeriesData} />
          </div>
        </div>
      </div>

      {/* Breaking down fab production */}
      <div className="bp-breakdown-section" style={{ marginTop: '30px', padding: '40px 30px', background: '#f9f9f9', borderRadius: '8px' }}>
        <h2 style={{ fontSize: '28px', fontWeight: 700, marginBottom: '10px', color: '#222' }}>
          Breaking down fab production
        </h2>
        <div className="section-description" style={{ marginBottom: '20px', fontSize: '15px', color: '#666' }}>
          The compute produced by a covert fab depends on when the fab finishes construction, the production capacity, its yield, and the compute density of the chips.
        </div>

        {/* First row: compute production breakdown */}
        <div className="breakdown-plots-row" style={{ display: 'flex', alignItems: 'flex-start', gap: '5px', flexWrap: 'wrap' }}>
          <BreakdownItem
            title="Probability construction has finished"
            tooltipKey="fab_construction_time"
            {...createTooltipHandlers('fab_construction_time')}
            description={
              <>Construction time depends on fab capacity (wafers/month) and available workers, based on <ParamLink paramId="param-construction-time-5k">5k wafer/month</ParamLink> and <ParamLink paramId="param-construction-time-100k">100k wafer/month</ParamLink> benchmarks.</>
            }
          >
            <IsOperationalPlot data={isOperationalData} />
          </BreakdownItem>

          <Operator>&times;</Operator>

          <BreakdownItem
            title="Production capacity (wafer starts per month)"
            tooltipKey="operating_labor_production"
            {...createTooltipHandlers('operating_labor_production')}
            description={
              <>Production capacity is the minimum of labor constraints (<ParamValue paramKey="fabWafersPerMonthPerOperatingWorker" parameters={parameters} /> wafers/month per operating worker) and SME constraints (scanners × <ParamValue paramKey="wafersPerMonthPerLithographyScanner" parameters={parameters} /> wafers/month).</>
            }
          >
            <WaferStartsPlot samples={waferStartsSamples} />
          </BreakdownItem>

          <Operator>&times;</Operator>

          <BreakdownBox
            title="Working H100-sized chips per wafer"
            tooltipKey="chips_per_wafer"
            {...createTooltipHandlers('chips_per_wafer')}
            value={chipsPerWafer ?? '--'}
          />

          <Operator>&times;</Operator>

          <BreakdownItem
            title="Transistor density relative to H100"
            tooltipKey="transistor_density"
            {...createTooltipHandlers('transistor_density')}
            description={
              <>Transistor density scales with process node improvement. Density increases <ParamValue paramKey="transistorDensityScalingExponent" parameters={parameters} /> for every halving of process node (e.g., 28nm → 14nm).</>
            }
          >
            <TransistorDensityPlot data={transistorDensityData} />
          </BreakdownItem>

          <Operator>&times;</Operator>

          <BreakdownBox
            title="Architecture efficiency (relative to H100)"
            tooltipKey="architecture_efficiency"
            {...createTooltipHandlers('architecture_efficiency')}
            description={
              <>Assuming architecture improves <ParamValue paramKey="stateOfTheArtArchitectureEfficiencyImprovementPerYear" parameters={parameters} /> per year.</>
            }
            value={archEfficiency?.toFixed(2) ?? '--'}
          />

          <Operator>=</Operator>

          <BreakdownItem
            title="Compute produced per month (H100e / month)"
            description="Total monthly compute production in H100-equivalent units, combining production capacity, yield, transistor density, and architecture improvements."
          >
            <ComputePerMonthPlot data={computePerMonthData} />
          </BreakdownItem>
        </div>

        {/* Second row: energy efficiency breakdown */}
        <div className="section-description" style={{ marginTop: '40px', marginBottom: '20px', fontSize: '15px', color: '#666' }}>
          The energy efficiency of the chips is just as important as the volume of compute. Energy efficiency is a direct function of the process node of the fab.
        </div>

        <div className="breakdown-plots-row" style={{ display: 'flex', alignItems: 'flex-start', gap: '5px', flexWrap: 'wrap' }}>
          <BreakdownItem
            title="Transistor density relative to H100"
            tooltipKey="transistor_density"
            {...createTooltipHandlers('transistor_density')}
            description={
              <>Same as compute production: density increases <ParamValue paramKey="transistorDensityScalingExponent" parameters={parameters} /> for every halving of process node.</>
            }
          >
            <TransistorDensityPlot data={transistorDensityData} />
          </BreakdownItem>

          <Operator style={{ flexDirection: 'column' }}>
            <div style={{ fontSize: '9px', color: '#888', marginBottom: '5px' }}>Plug into</div>
            <div style={{ fontSize: '20px' }}>&rarr;</div>
          </Operator>

          <BreakdownItem
            title="Watts per performance relative to H100"
            tooltipKey="watts_per_tpp"
            {...createTooltipHandlers('watts_per_tpp')}
            description={
              <>Energy efficiency follows W/TPP ∝ (density)^exponent. Exponent is <ParamValue paramKey="wattsTppDensityExponentBeforeDennard" parameters={parameters} /> before Dennard scaling ended (2006) and <ParamValue paramKey="wattsTppDensityExponentAfterDennard" parameters={parameters} /> after.</>
            }
          >
            <WattsPerTppPlot
              simDensities={transistorDensityData}
              transistorDensityAtEndOfDennard={parameters.transistorDensityAtEndOfDennardScaling}
              exponentBeforeDennard={parameters.wattsTppDensityExponentBeforeDennard}
              exponentAfterDennard={parameters.wattsTppDensityExponentAfterDennard}
            />
          </BreakdownItem>

          <Operator>&times;</Operator>

          <BreakdownBox
            title="Energy requirements of H100"
            tooltipKey="h100_power"
            {...createTooltipHandlers('h100_power')}
            value={h100Power ? `${(h100Power / 1000).toFixed(2)} kW` : '--'}
          />

          <Operator>&times;</Operator>

          <BreakdownItem
            title="Compute produced per month (H100e / month)"
            description="Same as the compute production result above, showing monthly H100-equivalent production."
          >
            <ComputePerMonthPlot data={computePerMonthData} />
          </BreakdownItem>

          <Operator>=</Operator>

          <BreakdownItem
            title="Energy requirements per month (GW / month)"
            description="Total energy required to produce chips each month, combining compute output, efficiency scaling, and H100 power requirements."
          >
            <EnergyPerMonthPlot data={energyPerMonthData} />
          </BreakdownItem>
        </div>
      </div>

      {/* Global tooltip */}
      <Tooltip
        content={tooltipState.content}
        visible={tooltipState.visible}
        triggerRect={tooltipState.triggerRect}
        onMouseEnter={onTooltipMouseEnter}
        onMouseLeave={onTooltipMouseLeave}
      />
    </section>
  );
}

export default CovertFabSection;
