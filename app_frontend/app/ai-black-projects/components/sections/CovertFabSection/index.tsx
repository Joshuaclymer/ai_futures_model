'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { BlackProjectData } from '@/types/blackProject';
import { TimeSeriesChart, CCDFChart, PlotlyChart } from '../../charts';
import { COLOR_PALETTE } from '../../colors';
import { useTooltip, Tooltip, TOOLTIP_DOCS } from '../../ui/Tooltip';
import {
  getDummyDashboard,
  getDummyComputeCCDF,
  getDummyTimeSeriesData,
  getDummyIsOperational,
  getDummyWaferStarts,
  getDummyChipsPerWafer,
  getDummyTransistorDensity,
  getDummyArchitectureEfficiency,
  getDummyComputePerMonth,
  getDummyWattsPerTppCurve,
  getDummyH100Power,
  getDummyEnergyPerMonth,
} from './DUMMY_DATA';

interface CovertFabSectionProps {
  data: BlackProjectData | null;
  isLoading?: boolean;
  agreementYear?: number;
}

// Helper function to convert hex to rgba
function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ParamLink component for clickable parameter values
function ParamLink({ children }: { children: React.ReactNode }) {
  return (
    <span className="param-link" style={{
      fontWeight: 'bold',
      color: COLOR_PALETTE.chip_stock,
      cursor: 'pointer',
      textDecoration: 'underline',
      textDecorationStyle: 'dotted',
    }}>
      {children}
    </span>
  );
}


// Dashboard component
function Dashboard() {
  const dashboard = getDummyDashboard();

  return (
    <div className="bp-dashboard" style={{ width: '240px', flexShrink: 0, display: 'flex', flexDirection: 'column', padding: '20px' }}>
      <div className="bp-plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: '8px', marginBottom: '20px' }}>
        Median outcome
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        <div className="bp-dashboard-item">
          <div className="bp-dashboard-value">{dashboard.energy}</div>
          <div className="bp-dashboard-label" style={{ fontWeight: 600 }}>Production before detection</div>
          <div style={{ fontSize: '20px', color: COLOR_PALETTE.fab, marginTop: '4px' }}>{dashboard.production}</div>
          <div className="bp-dashboard-sublabel" style={{ fontSize: '10px', color: '#777' }}>Detection means &ge;5x update</div>
        </div>
        <div className="bp-dashboard-item">
          <div className="bp-dashboard-value-small">{dashboard.probFabBuilt}</div>
          <div className="bp-dashboard-label-light">Probability covert fab is built</div>
        </div>
        <div className="bp-dashboard-item">
          <div className="bp-dashboard-value-small">{dashboard.yearsOperational}</div>
          <div className="bp-dashboard-label-light">Years operational before detection</div>
        </div>
        <div className="bp-dashboard-item">
          <div className="bp-dashboard-value-small">{dashboard.processNode}</div>
          <div className="bp-dashboard-label-light">Process node</div>
        </div>
      </div>
    </div>
  );
}

// CCDF Chart for compute produced before detection
function ComputeCCDFChart() {
  const ccdfData = getDummyComputeCCDF();
  const thresholds = [
    { value: 4, label: '"Detection" = >4x update', color: '#2E7D32' },
    { value: 2, label: '"Detection" = >2x update', color: '#4AA896' },
    { value: 1, label: '"Detection" = >1x update', color: '#7BA3C4' },
  ];

  const traces: Plotly.Data[] = thresholds.map(t => ({
    x: ccdfData[t.value].map(d => d.x),
    y: ccdfData[t.value].map(d => d.y),
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
          title: { text: "H100 equivalents (FLOPS) Produced by Covert Fab Before 'Detection'", font: { size: 11 } },
          type: 'log',
          tickfont: { size: 10 },
        },
        yaxis: {
          title: { text: 'P(covert compute > x)', font: { size: 11 } },
          range: [0, 1],
          tickfont: { size: 10 },
        },
        showlegend: true,
        legend: {
          x: 0.98,
          y: 0.98,
          xanchor: 'right',
          yanchor: 'top',
          bgcolor: 'rgba(255,255,255,0.8)',
          bordercolor: '#ccc',
          borderwidth: 1,
          font: { size: 10 },
        },
        margin: { l: 55, r: 10, t: 0, b: 60 },
      }}
    />
  );
}

// Time series chart for simulation runs
function SimulationRunsChart({ agreementYear = 2030 }: { agreementYear?: number }) {
  const data = getDummyTimeSeriesData(agreementYear);

  const traces: Plotly.Data[] = [
    // LR percentile band
    {
      x: [...data.years, ...data.years.slice().reverse()],
      y: [...data.lrCombined.p75, ...data.lrCombined.p25.slice().reverse()],
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
      x: data.years,
      y: data.lrCombined.median,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: COLOR_PALETTE.detection, width: 2 },
      name: 'LR: Median',
      hovertemplate: 'LR: %{y:.2f}<extra></extra>',
    },
    // H100e percentile band (secondary axis)
    {
      x: [...data.years, ...data.years.slice().reverse()],
      y: [...data.h100eFlow.p75, ...data.h100eFlow.p25.slice().reverse()],
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
      x: data.years,
      y: data.h100eFlow.median,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: COLOR_PALETTE.fab, width: 2 },
      name: 'H100e: Median',
      yaxis: 'y2',
      hovertemplate: 'H100e: %{y:,.0f}<extra></extra>',
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: {
          title: { text: 'Year', font: { size: 11 } },
          tickfont: { size: 10 },
        },
        yaxis: {
          title: { text: 'Evidence of Covert Fab (LR)', font: { size: 11 } },
          type: 'log',
          side: 'left',
          tickfont: { size: 10 },
        },
        yaxis2: {
          title: { text: 'H100 equivalents (FLOPS) Produced by Fab', font: { size: 11 } },
          overlaying: 'y',
          side: 'right',
          tickfont: { size: 10 },
          tickformat: '.2s',
        },
        showlegend: false,
        hovermode: 'x unified',
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
      style={{
        flex: '1 1 170px',
        minWidth: '170px',
        marginTop: '20px',
        marginBottom: '10px',
        cursor: hasTooltip ? 'pointer' : 'default',
      }}
      onMouseEnter={hasTooltip && onMouseEnter ? onMouseEnter : undefined}
      onMouseLeave={hasTooltip && onMouseLeave ? onMouseLeave : undefined}
    >
      <div className="breakdown-plot" style={{ height: '240px', background: 'white', borderRadius: '4px' }}>
        {children}
      </div>
      <div className="breakdown-label" style={{ fontSize: '11px', fontWeight: 'bold', marginTop: '5px', color: '#555', textAlign: 'center' }}>
        {title}
      </div>
      {description && (
        <div className="breakdown-description" style={{ fontSize: '9px', color: '#777', marginTop: '5px', lineHeight: 1.3, fontStyle: 'italic', textAlign: 'center' }}>
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
            <div className="breakdown-label" style={{ fontSize: '11px', fontWeight: 'bold', color: '#555', textAlign: 'center' }}>
              {title}
            </div>
            {description && (
              <div className="breakdown-description" style={{ fontSize: '9px', color: '#777', lineHeight: 1.3, fontStyle: 'italic', textAlign: 'center' }}>
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
function IsOperationalPlot({ agreementYear = 2030 }: { agreementYear?: number }) {
  const data = getDummyIsOperational(agreementYear);
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

// Wafer Starts PDF Plot
function WaferStartsPlot() {
  const values = getDummyWaferStarts();
  const bins = 20;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binWidth = (max - min) / bins;

  // Create histogram bins
  const binCounts = new Array(bins).fill(0);
  const binCenters: number[] = [];
  for (let i = 0; i < bins; i++) {
    binCenters.push(min + (i + 0.5) * binWidth);
  }
  for (const v of values) {
    const binIndex = Math.min(bins - 1, Math.floor((v - min) / binWidth));
    binCounts[binIndex]++;
  }

  // Normalize to density
  const density = binCounts.map(c => c / (values.length * binWidth));

  const traces: Plotly.Data[] = [
    {
      x: binCenters,
      y: density,
      type: 'bar' as const,
      marker: { color: COLOR_PALETTE.chip_stock },
      hovertemplate: 'Wafers/Month: %{x:.0f}<br>Density: %{y:.4f}<extra></extra>',
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: {
          title: { text: 'Wafers/Month', font: { size: 10 } },
          tickfont: { size: 9 },
        },
        yaxis: {
          title: { text: 'Density', font: { size: 10 } },
          tickfont: { size: 9 },
        },
        bargap: 0.05,
        margin: { l: 50, r: 10, t: 10, b: 50 },
      }}
    />
  );
}

// Transistor Density Bar Plot by Process Node
function TransistorDensityPlot() {
  const data = getDummyTransistorDensity();

  const traces: Plotly.Data[] = [
    {
      x: data.map(d => d.node),
      y: data.map(d => d.density),
      type: 'bar' as const,
      marker: { color: COLOR_PALETTE.fab },
      hovertemplate: '%{x}: %{y:.2f}x H100<extra></extra>',
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: {
          title: { text: 'Process Node', font: { size: 10 } },
          tickfont: { size: 9 },
        },
        yaxis: {
          title: { text: 'Density (rel. to H100)', font: { size: 10 } },
          tickfont: { size: 9 },
        },
        bargap: 0.3,
        margin: { l: 50, r: 10, t: 10, b: 50 },
      }}
    />
  );
}

// Compute per Month Plot
function ComputePerMonthPlot({ agreementYear = 2030 }: { agreementYear?: number }) {
  const data = getDummyComputePerMonth(agreementYear);
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

// Watts per TPP Curve Plot
function WattsPerTppPlot() {
  const data = getDummyWattsPerTppCurve();
  const simDensities = getDummyTransistorDensity();

  const traces: Plotly.Data[] = [
    // Curve
    {
      x: data.densityRelative,
      y: data.wattsPerTppRelative,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: '#7DD4C0', width: 2 },
      name: 'Power Law',
      hovertemplate: 'Density: %{x:.2g}x<br>W/TPP: %{y:.2g}x<extra></extra>',
    },
    // Simulation points
    {
      x: simDensities.map(d => d.density),
      y: simDensities.map(d => {
        // Calculate watts per TPP for each density
        const dennardThreshold = 0.02;
        if (d.density < dennardThreshold) {
          return Math.pow(d.density, -0.5);
        } else {
          const wattsAtThreshold = Math.pow(dennardThreshold, -0.5);
          return wattsAtThreshold * Math.pow(d.density / dennardThreshold, -0.15);
        }
      }),
      type: 'scatter' as const,
      mode: 'markers' as const,
      marker: { size: 8, color: COLOR_PALETTE.fab },
      name: 'Simulations',
      hovertemplate: 'Density: %{x:.2g}x<br>W/TPP: %{y:.2g}x<extra></extra>',
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: {
          title: { text: 'Transistor Density (rel. to H100)', font: { size: 10 } },
          type: 'log',
          range: [Math.log10(0.001), Math.log10(100)],
          tickfont: { size: 9 },
        },
        yaxis: {
          title: { text: 'Watts per TPP (rel. to H100)', font: { size: 10 } },
          type: 'log',
          tickfont: { size: 9 },
        },
        showlegend: false,
        margin: { l: 60, r: 10, t: 10, b: 50 },
      }}
    />
  );
}

// Energy per Month Plot
function EnergyPerMonthPlot({ agreementYear = 2030 }: { agreementYear?: number }) {
  const data = getDummyEnergyPerMonth(agreementYear);
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

export function CovertFabSection({ data, isLoading, agreementYear = 2030 }: CovertFabSectionProps) {
  const { tooltipState, showTooltip, hideTooltip } = useTooltip();

  if (isLoading) {
    return (
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-800">Covert fab</h2>
        <div className="h-48 bg-gray-100 animate-pulse rounded" />
      </section>
    );
  }

  const chipsPerWafer = getDummyChipsPerWafer();
  const archEfficiency = getDummyArchitectureEfficiency();
  const h100Power = getDummyH100Power();

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
        <Dashboard />

        <div className="bp-plot-container" style={{ flex: '1 1 200px', minWidth: '200px', padding: '20px', display: 'flex', flexDirection: 'column' }}>
          <div className="bp-plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: '8px', marginBottom: '10px' }}>
            Covert compute produced before detection
          </div>
          <div style={{ flex: 1, minHeight: '250px' }}>
            <ComputeCCDFChart />
          </div>
        </div>

        <div className="bp-plot-container" style={{ flex: '1 1 200px', minWidth: '200px', padding: '20px', display: 'flex', flexDirection: 'column' }}>
          <div className="bp-plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: '8px', marginBottom: '10px' }}>
            Simulation runs
          </div>
          <div style={{ flex: 1, minHeight: '250px' }}>
            <SimulationRunsChart agreementYear={agreementYear} />
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
              <>Construction time depends on fab capacity (wafers/month) and available workers, based on <ParamLink>5k wafer/month</ParamLink> and <ParamLink>100k wafer/month</ParamLink> benchmarks.</>
            }
          >
            <IsOperationalPlot agreementYear={agreementYear} />
          </BreakdownItem>

          <Operator>&times;</Operator>

          <BreakdownItem
            title="Production capacity (wafer starts per month)"
            tooltipKey="operating_labor_production"
            {...createTooltipHandlers('operating_labor_production')}
            description={
              <>Production capacity is the minimum of labor constraints (<ParamLink>24.6</ParamLink> wafers/month per operating worker) and <ParamLink>SME constraints</ParamLink> (scanners × 1000 wafers/month).</>
            }
          >
            <WaferStartsPlot />
          </BreakdownItem>

          <Operator>&times;</Operator>

          <BreakdownBox
            title="Working H100-sized chips per wafer"
            tooltipKey="chips_per_wafer"
            {...createTooltipHandlers('chips_per_wafer')}
            description={
              <>Yield based on <ParamLink>{chipsPerWafer}</ParamLink> H100-sized chips per wafer (TSMC's yield for H100s).</>
            }
            value={chipsPerWafer}
          />

          <Operator>&times;</Operator>

          <BreakdownItem
            title="Transistor density relative to H100"
            tooltipKey="transistor_density"
            {...createTooltipHandlers('transistor_density')}
            description={
              <>Transistor density scales with process node improvement. Density increases <ParamLink>2x</ParamLink> for every halving of process node (e.g., 28nm → 14nm).</>
            }
          >
            <TransistorDensityPlot />
          </BreakdownItem>

          <Operator>&times;</Operator>

          <BreakdownBox
            title="Architecture efficiency (relative to H100)"
            tooltipKey="architecture_efficiency"
            {...createTooltipHandlers('architecture_efficiency')}
            description={
              <>Architecture improves at <ParamLink>1.23x</ParamLink> per year based on historical chip performance improvements beyond transistor density gains.</>
            }
            value={archEfficiency.toFixed(2)}
          />
        </div>

        {/* Result: Compute produced per month - large plot at bottom */}
        <div style={{ marginTop: '20px', width: '100%' }}>
          <div style={{ height: '300px', background: 'white', borderRadius: '4px', marginBottom: '10px' }}>
            <ComputePerMonthPlot agreementYear={agreementYear} />
          </div>
          <div className="breakdown-label" style={{ fontSize: '13px', fontWeight: 'bold', color: '#555', textAlign: 'center' }}>
            Compute produced per month (H100e / month)
          </div>
          <div className="breakdown-description" style={{ fontSize: '10px', color: '#777', marginTop: '5px', lineHeight: 1.3, fontStyle: 'italic', textAlign: 'center' }}>
            Total monthly compute production in H100-equivalent units, combining production capacity, yield, transistor density, and architecture improvements.
          </div>
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
              <>Same as compute production: density increases <ParamLink>2x</ParamLink> for every halving of process node.</>
            }
          >
            <TransistorDensityPlot />
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
              <>Energy efficiency follows W/TPP ∝ (density)^exponent. Exponent is <ParamLink>-0.5</ParamLink> before Dennard scaling ended (2006) and <ParamLink>-0.15</ParamLink> after.</>
            }
          >
            <WattsPerTppPlot />
          </BreakdownItem>

          <Operator>&times;</Operator>

          <BreakdownBox
            title="Energy requirements of H100"
            tooltipKey="h100_power"
            {...createTooltipHandlers('h100_power')}
            description={
              <>H100 power consumption is <ParamLink>{h100Power}</ParamLink>W per chip.</>
            }
            value={`${(h100Power / 1000).toFixed(2)} kW`}
          />

          <Operator>&times;</Operator>

          <BreakdownItem
            title="Compute produced per month (H100e / month)"
            description="Same as the compute production result above, showing monthly H100-equivalent production."
          >
            <ComputePerMonthPlot agreementYear={agreementYear} />
          </BreakdownItem>
        </div>

        {/* Result: Energy requirements per month - large plot at bottom */}
        <div style={{ marginTop: '20px', width: '100%' }}>
          <div style={{ height: '300px', background: 'white', borderRadius: '4px', marginBottom: '10px' }}>
            <EnergyPerMonthPlot agreementYear={agreementYear} />
          </div>
          <div className="breakdown-label" style={{ fontSize: '13px', fontWeight: 'bold', color: '#555', textAlign: 'center' }}>
            Energy requirements per month (GW / month)
          </div>
          <div className="breakdown-description" style={{ fontSize: '10px', color: '#777', marginTop: '5px', lineHeight: 1.3, fontStyle: 'italic', textAlign: 'center' }}>
            Total energy required to produce chips each month, combining compute output, efficiency scaling, and H100 power requirements.
          </div>
        </div>
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

export default CovertFabSection;
