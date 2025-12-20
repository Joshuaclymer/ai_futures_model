'use client';

import { TimeSeriesChart } from '../../charts';
import { COLOR_PALETTE } from '../../colors';
import {
  getDummyInitialChipStock,
  getDummyAcquiredHardware,
  getDummySurvivingFraction,
  getDummyCovertChipStock,
  getDummyDatacenterCapacity,
  getDummyEnergyUsage,
  getDummyOperatingChips,
  getDummyCovertComputation,
} from './DUMMY_DATA';

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

interface ParamInputProps {
  label: string;
  value: string | number;
  paramId: string;
  suffix?: string;
}

function ParamInput({ label, value, paramId, suffix = '' }: ParamInputProps) {
  return (
    <div
      className="param-input-row"
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '4px',
        fontSize: '9px',
        color: '#666',
        marginTop: '2px'
      }}
    >
      <span>{label}:</span>
      <input
        type="text"
        value={`${value}${suffix}`}
        readOnly
        onClick={(e) => {
          e.stopPropagation();
          scrollToParameter(paramId);
        }}
        style={{
          width: '50px',
          padding: '2px 4px',
          fontSize: '9px',
          border: '1px solid #ddd',
          borderRadius: '3px',
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: '#f9f9f9'
        }}
        title={`Click to edit ${label} in sidebar`}
      />
    </div>
  );
}

interface BreakdownChartProps {
  title: string;
  description?: string;
  data: { years: number[]; median: number[]; p25: number[]; p75: number[] };
  color: string;
  yLabel?: string;
  onClick?: () => void;
  tooltip?: string;
  params?: ParamInputProps[];
}

function BreakdownChart({ title, description, data, color, yLabel, onClick, tooltip, params }: BreakdownChartProps) {
  const isClickable = !!onClick;

  return (
    <div
      className={`breakdown-item ${isClickable ? 'clickable' : ''}`}
      onClick={onClick}
      title={tooltip}
      style={isClickable ? { cursor: 'pointer' } : undefined}
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
      {description && <div className="breakdown-description">{description}</div>}
      {params && params.length > 0 && (
        <div style={{ marginTop: '4px' }}>
          {params.map((param, idx) => (
            <ParamInput key={idx} {...param} />
          ))}
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


// Helper function to scroll to a section by ID
function scrollToSection(sectionId: string) {
  const element = document.getElementById(sectionId);
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// Helper function to scroll to a sidebar parameter and highlight it
function scrollToParameter(paramId: string) {
  const input = document.getElementById(paramId);
  if (input) {
    input.scrollIntoView({ behavior: 'smooth', block: 'center' });
    input.focus();
    if (input instanceof HTMLInputElement) {
      input.select();
    }
  }
}

// ParamLink component for clickable parameter references
interface ParamLinkProps {
  paramId: string;
  children: React.ReactNode;
}

function ParamLink({ paramId, children }: ParamLinkProps) {
  return (
    <span
      className="param-link"
      onClick={(e) => {
        e.stopPropagation();
        scrollToParameter(paramId);
      }}
      style={{
        color: '#5E6FB8',
        textDecoration: 'underline',
        cursor: 'pointer',
        textDecorationStyle: 'dotted'
      }}
    >
      {children}
    </span>
  );
}

interface RateOfComputationSectionProps {
  agreementYear?: number;
}

export function RateOfComputationSection({ agreementYear = 2030 }: RateOfComputationSectionProps) {
  // Get dummy data (clearly marked as fake)
  const initialStock = getDummyInitialChipStock(agreementYear);
  const acquiredHardware = getDummyAcquiredHardware(agreementYear);
  const survivingFraction = getDummySurvivingFraction(agreementYear);
  const covertChipStock = getDummyCovertChipStock(agreementYear);
  const datacenterCapacity = getDummyDatacenterCapacity(agreementYear);
  const energyUsage = getDummyEnergyUsage(agreementYear);
  const operatingChips = getDummyOperatingChips(agreementYear);
  const covertComputation = getDummyCovertComputation(agreementYear);

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

          <BreakdownChart
            title="Initial covert chip stock"
            data={initialStock}
            color={COLOR_PALETTE.chip_stock}
            yLabel="H100e"
            onClick={() => scrollToSection('initialStockSection')}
            tooltip="Click for more details"
            params={[
              { label: 'Divert', value: '5', paramId: 'param-proportion-chips-divert', suffix: '%' }
            ]}
          />

          <Operator>+</Operator>

          <BreakdownChart
            title="Acquired hardware"
            data={acquiredHardware}
            color={COLOR_PALETTE.fab}
            yLabel="H100e"
            onClick={() => scrollToSection('covertFabSection')}
            tooltip="Click for more details"
            params={[
              { label: 'Fab', value: 'Yes', paramId: 'param-build-covert-fab' }
            ]}
          />

          <Bracket>]</Bracket>

          <Operator>&times;</Operator>

          <BreakdownChart
            title="Surviving fraction of compute"
            description="The fraction of compute that remains operational, accounting for degradation."
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
            data={datacenterCapacity}
            color={COLOR_PALETTE.datacenters_and_energy}
            yLabel="GW"
            onClick={() => scrollToSection('covertDataCentersSection')}
            tooltip="Click for more details"
            params={[
              { label: 'Workers', value: '10K', paramId: 'param-datacenter-construction-labor' }
            ]}
          />

          <div className="breakdown-limits-arrow">
            <span className="breakdown-limits-label">Limits</span>
            <span className="breakdown-limits-symbol">&rarr;</span>
          </div>

          <BreakdownChart
            title="Chip stock energy usage (medians)"
            description="Energy required to run untraced AI chips at full capacity."
            data={energyUsage}
            color={COLOR_PALETTE.datacenters_and_energy}
            yLabel="GW"
          />

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
