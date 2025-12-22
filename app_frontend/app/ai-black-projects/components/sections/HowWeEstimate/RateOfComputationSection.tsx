'use client';

import { TimeSeriesChart, PDFChart, EnergyStackedAreaChart } from '../../charts';
import { COLOR_PALETTE } from '../../colors';
import { ParamLink, ParamValue } from '../../ui';
import { Parameters } from '../../../types';

// Types for the rate of computation data from API
interface TimeSeriesData {
  years: number[];
  median: number[];
  p25: number[];
  p75: number[];
}

export interface RateOfComputationData {
  years: number[];
  initial_chip_stock_samples: number[];
  acquired_hardware: TimeSeriesData;
  surviving_fraction: TimeSeriesData;
  covert_chip_stock: TimeSeriesData;
  datacenter_capacity: TimeSeriesData;
  energy_usage: TimeSeriesData;
  energy_stacked_data?: [number, number][];
  energy_source_labels?: [string, string];
  operating_chips: TimeSeriesData;
  covert_computation: TimeSeriesData;
}

interface SubsectionItemProps {
  children: React.ReactNode;
}

function SubsectionItem({ children }: SubsectionItemProps) {
  return (
    <div className="relative">
      <span className="absolute -left-[26px] top-1/2 -translate-y-1/2 text-gray-300 text-[22px] leading-none">
        &mdash;
      </span>
      {children}
    </div>
  );
}


interface BreakdownChartProps {
  title: string;
  description?: string;
  descriptionNode?: React.ReactNode;
  data: { years: number[]; median: number[]; p25: number[]; p75: number[] };
  color: string;
  yLabel?: string;
  onClick?: () => void;
  tooltip?: string;
}

function BreakdownChart({ title, description, descriptionNode, data, color, yLabel, onClick, tooltip }: BreakdownChartProps) {
  const isClickable = !!onClick;

  return (
    <div
      className={`breakdown-item ${isClickable ? 'clickable' : ''}`}
      onClick={onClick}
      title={tooltip}
    >
      <div className="breakdown-plot">
        <TimeSeriesChart
          years={data.years}
          median={data.median}
          p25={data.p25}
          p75={data.p75}
          color={color}
          yLabel={yLabel}
          showBand={true}
          bandAlpha={0.15}
        />
      </div>
      <div className="breakdown-label">{title}</div>
      {(description || descriptionNode) && (
        <div className="breakdown-description" onClick={(e) => e.stopPropagation()}>
          {descriptionNode || description}
        </div>
      )}
    </div>
  );
}

function Operator({ children }: { children: React.ReactNode }) {
  return <div className="breakdown-operator">{children}</div>;
}

function Bracket({ children }: { children: React.ReactNode }) {
  return <div className="breakdown-bracket">{children}</div>;
}

interface PDFBreakdownChartProps {
  title: string;
  description?: string;
  descriptionNode?: React.ReactNode;
  samples: number[];
  color: string;
  xLabel?: string;
  onClick?: () => void;
  tooltip?: string;
}

function PDFBreakdownChart({ title, description, descriptionNode, samples, color, xLabel = 'H100e', onClick, tooltip }: PDFBreakdownChartProps) {
  const isClickable = !!onClick;

  return (
    <div
      className={`breakdown-item ${isClickable ? 'clickable' : ''}`}
      onClick={onClick}
      title={tooltip}
    >
      <div className="breakdown-plot">
        <PDFChart
          samples={samples}
          color={color}
          xLabel={xLabel}
          yLabel="Prob"
          logScale={true}
          numBins={15}
        />
      </div>
      <div className="breakdown-label">{title}</div>
      {(description || descriptionNode) && (
        <div className="breakdown-description" onClick={(e) => e.stopPropagation()}>
          {descriptionNode || description}
        </div>
      )}
    </div>
  );
}

// Helper function to scroll to a section by ID
function scrollToSection(sectionId: string) {
  const element = document.getElementById(sectionId);
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

interface RateOfComputationSectionProps {
  agreementYear?: number;
  data?: RateOfComputationData | null;
  parameters?: Parameters;
}

export function RateOfComputationSection({ agreementYear = 2030, data, parameters }: RateOfComputationSectionProps) {
  // Use data from API, with fallback empty arrays if not available
  const initialStockSamples = data?.initial_chip_stock_samples || [];
  const acquiredHardware = data?.acquired_hardware || { years: [], median: [], p25: [], p75: [] };
  const survivingFraction = data?.surviving_fraction || { years: [], median: [], p25: [], p75: [] };
  const covertChipStock = data?.covert_chip_stock || { years: [], median: [], p25: [], p75: [] };
  const datacenterCapacity = data?.datacenter_capacity || { years: [], median: [], p25: [], p75: [] };
  const energyUsage = data?.energy_usage || { years: [], median: [], p25: [], p75: [] };
  const energyStackedData = data?.energy_stacked_data || [];
  const energySourceLabels = data?.energy_source_labels || ['Initial Stock', 'Fab-Produced'] as [string, string];
  const operatingChips = data?.operating_chips || { years: [], median: [], p25: [], p75: [] };
  const covertComputation = data?.covert_computation || { years: [], median: [], p25: [], p75: [] };

  return (
    <div className="pt-5">
      <h3 className="text-xl font-semibold text-[#333] mb-4">
        Predicting the rate of computation
      </h3>
      <p className="text-[15px] text-[#555] leading-[1.7] max-w-[900px] mb-5">
        The rate of computation depends on a covert project&apos;s{' '}
        <strong>stock of AI chips</strong> and the{' '}
        <strong>capacity of covert datacenters</strong>.
      </p>

      {/* Bordered subsections with vertical line */}
      <div className="border-l-2 border-gray-300 pl-6 ml-1 space-y-5">
        {/* AI Chips subsection */}
        <SubsectionItem>
          <p className="text-[14px] text-[#666] leading-[1.6] max-w-[900px]">
            The PRC&apos;s <strong>stock of AI chips</strong> is the{' '}
            <em>initial stock</em> the PRC chooses to divert to the covert project,
            plus the chips the PRC <em>acquires</em> after the agreement is in force,
            multiplied by the fraction that survives <em>degradation and failure</em> over time.
          </p>
        </SubsectionItem>

        {/* Equation visualization - chips */}
        <div className="breakdown-equation-row">
          <Bracket>[</Bracket>

          <PDFBreakdownChart
            title="Initial covert chip stock"
            descriptionNode={
              <>
                Assuming the PRC initially has ~1M H100e, and diverts{' '}
                <ParamLink paramId="param-fraction-compute-divert">{parameters ? `${(parameters.proportionOfInitialChipStockToDivert * 100).toFixed(0)}%` : '5%'}</ParamLink>{' '}
                before the agreement goes into force.
              </>
            }
            samples={initialStockSamples}
            color={COLOR_PALETTE.chip_stock}
            xLabel="H100e"
            onClick={() => scrollToSection('initialStockSection')}
            tooltip="Click for more details"
          />

          <Operator>+</Operator>

          <BreakdownChart
            title="Acquired hardware"
            descriptionNode={
              <>
                Assuming the PRC diverts{' '}
                <ParamLink paramId="param-fraction-scanners-divert">{parameters ? `${(parameters.fractionOfLithographyScannersToDivert * 100).toFixed(0)}%` : '10%'}</ParamLink>{' '}
                of indigenous ≤<ParamLink paramId="param-fab-node">{parameters?.blackFabMaxProcessNode || '28'}</ParamLink>nm SME before the agreement goes into force.
              </>
            }
            data={acquiredHardware}
            color={COLOR_PALETTE.fab}
            yLabel="H100e"
            onClick={() => scrollToSection('covertFabSection')}
            tooltip="Click for more details"
          />

          <Bracket>]</Bracket>

          <Operator>&times;</Operator>

          <BreakdownChart
            title="Surviving fraction of compute"
            descriptionNode={
              <>
                The fraction of compute that remains operational, accounting for degradation, assuming a{' '}
                <ParamLink paramId="param-initial-hazard-rate">{parameters ? `${(parameters.initialAnnualHazardRate * 100).toFixed(0)}%` : '5%'}</ParamLink> initial annual hazard rate that increases by{' '}
                <ParamLink paramId="param-hazard-rate-increase">{parameters ? `${(parameters.annualHazardRateIncreasePerYear * 100).toFixed(0)}%` : '2%'}</ParamLink> per year.
              </>
            }
            data={survivingFraction}
            color={COLOR_PALETTE.survival_rate}
            yLabel="Fraction"
          />

          <Operator>=</Operator>

          <BreakdownChart
            title="Covert chip stock"
            description="Total untraced compute available to a covert PRC project."
            data={covertChipStock}
            color={COLOR_PALETTE.chip_stock}
            yLabel="H100e"
          />
        </div>

        {/* Datacenter subsection */}
        <SubsectionItem>
          <p className="text-[14px] text-[#666] leading-[1.6] max-w-[900px]">
            The PRC can only operate these AI chips if there&apos;s enough{' '}
            <strong>covert datacenter capacity</strong>, which grows linearly as more
            datacenters are constructed by a small and stealthy workforce.
          </p>
        </SubsectionItem>

        {/* Equation visualization - datacenters */}
        <div className="breakdown-equation-row">
          <BreakdownChart
            title="Unreported datacenter capacity"
            descriptionNode={
              <>
                Assuming{' '}
                {parameters ? <ParamValue paramKey="workersInCovertProject" parameters={parameters} /> : '10,000'}{' '}
                workers are involved in the covert project and{' '}
                {parameters ? <ParamValue paramKey="fractionOfLaborDevotedToDatacenterConstruction" parameters={parameters} /> : '89%'}{' '}
                construct datacenters.
              </>
            }
            data={datacenterCapacity}
            color={COLOR_PALETTE.datacenters_and_energy}
            yLabel="GW"
            onClick={() => scrollToSection('covertDataCentersSection')}
            tooltip="Click for more details"
          />

          <div className="breakdown-limits-arrow">
            <span className="breakdown-limits-label">Limits</span>
            <span className="breakdown-limits-symbol">&rarr;</span>
          </div>

          <div className="breakdown-item">
            <div className="breakdown-plot">
              <EnergyStackedAreaChart
                years={datacenterCapacity.years}
                energyData={energyStackedData}
                sourceLabels={energySourceLabels}
                datacenterCapacity={datacenterCapacity.median}
              />
            </div>
            <div className="breakdown-label">Chip stock energy usage (medians)</div>
            <div className="breakdown-description">
              Energy required to run untraced AI chips at full capacity.
            </div>
          </div>

          <Operator>=</Operator>

          <BreakdownChart
            title="Operating AI chips"
            description="Quantity of compute operated in PRC covert datacenters."
            data={operatingChips}
            color={COLOR_PALETTE.datacenters_and_energy}
            yLabel="H100e"
          />

          <Operator>=</Operator>

          <BreakdownChart
            title="Covert computation"
            description="H100-years of computation calculated cumulatively."
            data={covertComputation}
            color={COLOR_PALETTE.datacenters_and_energy}
            yLabel="H100-years"
          />
        </div>
      </div>
    </div>
  );
}
