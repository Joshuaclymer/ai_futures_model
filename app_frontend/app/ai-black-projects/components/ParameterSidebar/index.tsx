'use client';

import Link from 'next/link';
import { Parameters } from '../../types';
import { Slider, CollapsibleSection } from '../ui';
import { HEADER_HEIGHT } from '../Header';
import './ParameterSidebar.css';

interface ParameterSidebarProps {
  parameters: Parameters;
  onParameterChange: <K extends keyof Parameters>(key: K, value: Parameters[K]) => void;
  isOpen: boolean;
  hideHeader?: boolean;
}

export function ParameterSidebar({
  parameters,
  onParameterChange,
  isOpen,
  hideHeader = false,
}: ParameterSidebarProps) {
  return (
    <aside
      className={`bp-sidebar ${isOpen ? 'bp-sidebar-open' : ''}`}
      style={{ top: `${HEADER_HEIGHT}px`, height: `calc(100vh - ${HEADER_HEIGHT}px)` }}
    >
      <div className="bp-sidebar-content">
        {/* Sidebar Header */}
        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontWeight: 600, fontSize: '14px', color: '#333', marginBottom: '4px' }}>
            Parameters
          </div>
          <Link
            href="/ai-black-project-parameters"
            className="text-xs hover:underline"
            style={{ color: '#5E6FB8' }}
            target="_blank"
          >
            View documentation →
          </Link>
        </div>

        {/* Key parameters section */}
        <KeyParametersSection parameters={parameters} onChange={onParameterChange} />

        {/* Black project properties */}
        <CollapsibleSection title="Black project properties">
          <BlackProjectPropertiesSection parameters={parameters} onChange={onParameterChange} />
        </CollapsibleSection>

        {/* Detection parameters */}
        <CollapsibleSection title="Detection parameters">
          <DetectionParametersSection parameters={parameters} onChange={onParameterChange} />
        </CollapsibleSection>

        {/* PRC compute */}
        <CollapsibleSection title="PRC compute">
          <PRCComputeSection parameters={parameters} onChange={onParameterChange} />
        </CollapsibleSection>

        {/* PRC data centers and energy */}
        <CollapsibleSection title="PRC data centers and energy">
          <PRCDataCentersSection parameters={parameters} onChange={onParameterChange} />
        </CollapsibleSection>

        {/* US compute */}
        <CollapsibleSection title="US compute">
          <USComputeSection parameters={parameters} onChange={onParameterChange} />
        </CollapsibleSection>

        {/* Compute survival */}
        <CollapsibleSection title="Compute survival">
          <ComputeSurvivalSection parameters={parameters} onChange={onParameterChange} />
        </CollapsibleSection>

        {/* Exogenous compute trends */}
        <CollapsibleSection title="Exogenous compute trends">
          <ExogenousComputeSection parameters={parameters} onChange={onParameterChange} />
        </CollapsibleSection>
      </div>
    </aside>
  );
}

// Section component props
interface SectionProps {
  parameters: Parameters;
  onChange: <K extends keyof Parameters>(key: K, value: Parameters[K]) => void;
}

function KeyParametersSection({ parameters, onChange }: SectionProps) {
  return (
    <div className="mb-2">
      <Slider
        label="Number of simulations"
        value={parameters.numSimulations}
        onChange={(v) => onChange('numSimulations', v)}
        min={10}
        max={1000}
        step={10}
      />
      <Slider
        label="Slowdown agreement start year"
        value={parameters.agreementYear}
        onChange={(v) => onChange('agreementYear', v)}
        min={2026}
        max={2035}
        step={1}
      />
      <Slider
        label="Black project start year"
        value={parameters.blackProjectStartYear}
        onChange={(v) => onChange('blackProjectStartYear', v)}
        min={2024}
        max={2035}
        step={1}
      />
      <Slider
        id="param-workers-in-project"
        label="Workers involved in black project"
        value={parameters.workersInCovertProject}
        onChange={(v) => onChange('workersInCovertProject', v)}
        min={1000}
        max={100000}
        step={1000}
        formatValue={(v) => v.toLocaleString()}
      />
      <Slider
        label="Mean time to detect 100 worker project"
        value={parameters.meanDetectionTime100}
        onChange={(v) => onChange('meanDetectionTime100', v)}
        min={1}
        max={20}
        step={0.5}
        formatValue={(v) => `${v} years`}
        tooltipDoc="detection_time"
      />
      <Slider
        label="Mean time to detect 1000 worker project"
        value={parameters.meanDetectionTime1000}
        onChange={(v) => onChange('meanDetectionTime1000', v)}
        min={1}
        max={20}
        step={0.5}
        formatValue={(v) => `${v} years`}
        tooltipDoc="detection_time"
      />
      <Slider
        label="Detection time variance"
        value={parameters.varianceDetectionTime}
        onChange={(v) => onChange('varianceDetectionTime', v)}
        min={0.1}
        max={10}
        step={0.1}
        tooltipDoc="detection_time"
      />
      <Slider
        id="param-fraction-compute-divert"
        label="Fraction of compute to divert"
        value={parameters.proportionOfInitialChipStockToDivert}
        onChange={(v) => onChange('proportionOfInitialChipStockToDivert', v)}
        min={0}
        max={0.5}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="fraction_diverted"
      />
      <Slider
        label="Median error of intelligence estimate of compute stock"
        value={parameters.intelligenceMedianError}
        onChange={(v) => onChange('intelligenceMedianError', v)}
        min={0.01}
        max={0.5}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="chip_stock_detection"
      />
    </div>
  );
}

function BlackProjectPropertiesSection({ parameters, onChange }: SectionProps) {
  return (
    <>
      <Slider
        label="Total labor"
        value={parameters.totalLabor}
        onChange={(v) => onChange('totalLabor', v)}
        min={1000}
        max={100000}
        step={1000}
        formatValue={(v) => v.toLocaleString()}
      />
      <Slider
        id="param-fraction-labor-datacenter"
        label="Fraction of labor devoted to datacenter construction"
        value={parameters.fractionOfLaborDevotedToDatacenterConstruction}
        onChange={(v) => onChange('fractionOfLaborDevotedToDatacenterConstruction', v)}
        min={0}
        max={1}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="construction_workers"
      />
      <Slider
        label="Fraction of labor devoted to black fab construction"
        value={parameters.fractionOfLaborDevotedToBlackFabConstruction}
        onChange={(v) => onChange('fractionOfLaborDevotedToBlackFabConstruction', v)}
        min={0}
        max={1}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="fab_construction_time"
      />
      <Slider
        label="Fraction of labor devoted to black fab operation"
        value={parameters.fractionOfLaborDevotedToBlackFabOperation}
        onChange={(v) => onChange('fractionOfLaborDevotedToBlackFabOperation', v)}
        min={0}
        max={1}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="operating_labor_production"
      />
      <Slider
        label="Fraction of labor devoted to AI research"
        value={parameters.fractionOfLaborDevotedToAiResearch}
        onChange={(v) => onChange('fractionOfLaborDevotedToAiResearch', v)}
        min={0}
        max={1}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
      />
      <Slider
        id="param-fraction-datacenter-divert"
        label="Fraction of datacenter capacity to divert"
        value={parameters.fractionOfDatacenterCapacityToDivert}
        onChange={(v) => onChange('fractionOfDatacenterCapacityToDivert', v)}
        min={0}
        max={1}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="fraction_diverted"
      />
      <Slider
        id="param-fraction-scanners-divert"
        label="Fraction of lithography scanners to divert"
        value={parameters.fractionOfLithographyScannersToDivert}
        onChange={(v) => onChange('fractionOfLithographyScannersToDivert', v)}
        min={0}
        max={1}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="sme_inventory_detection"
      />
      <Slider
        id="param-max-energy-fraction"
        label="Max fraction of total national energy consumption"
        value={parameters.maxFractionOfTotalNationalEnergyConsumption}
        onChange={(v) => onChange('maxFractionOfTotalNationalEnergyConsumption', v)}
        min={0}
        max={0.2}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="max_energy_proportion"
      />
      <div className="bp-checkbox-group">
        <input
          type="checkbox"
          id="param-build-covert-fab"
          checked={parameters.buildCovertFab}
          onChange={(e) => onChange('buildCovertFab', e.target.checked)}
        />
        <label htmlFor="param-build-covert-fab">Build a black fab</label>
      </div>
      <div className="bp-slider-group">
        <label htmlFor="param-fab-node">
          Black fab max process node
        </label>
        <select
          id="param-fab-node"
          value={parameters.blackFabMaxProcessNode}
          onChange={(e) => onChange('blackFabMaxProcessNode', e.target.value)}
          style={{
            width: '100%',
            padding: '4px 8px',
            fontSize: '10px',
            border: '1px solid #ccc',
            borderRadius: '3px',
            backgroundColor: 'transparent',
            cursor: 'pointer',
            marginTop: '4px',
          }}
        >
          <option value="130">130nm</option>
          <option value="28">28nm</option>
          <option value="14">14nm</option>
          <option value="7">7nm</option>
        </select>
      </div>
    </>
  );
}

function DetectionParametersSection({ parameters, onChange }: SectionProps) {
  return (
    <>
      <Slider
        id="param-median-error-chip-stock"
        label="Intelligence median error in estimate of fab stock"
        value={parameters.intelligenceMedianErrorInEstimateOfFabStock}
        onChange={(v) => onChange('intelligenceMedianErrorInEstimateOfFabStock', v)}
        min={0.01}
        max={0.5}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="chip_stock_detection"
      />
      <Slider
        id="param-median-error-energy"
        label="Intelligence median error in energy consumption estimate"
        value={parameters.intelligenceMedianErrorInEnergyConsumptionEstimate}
        onChange={(v) => onChange('intelligenceMedianErrorInEnergyConsumptionEstimate', v)}
        min={0.01}
        max={0.5}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="energy_accounting_detection"
      />
      <Slider
        id="param-median-error-satellite"
        label="Intelligence median error in satellite estimate"
        value={parameters.intelligenceMedianErrorInSatelliteEstimate}
        onChange={(v) => onChange('intelligenceMedianErrorInSatelliteEstimate', v)}
        min={0.001}
        max={0.1}
        step={0.001}
        formatValue={(v) => `${(v * 100).toFixed(1)}%`}
        tooltipDoc="satellite_datacenter_detection"
      />
      <Slider
        id="param-detection-threshold"
        label="Detection threshold"
        value={parameters.detectionThreshold}
        onChange={(v) => onChange('detectionThreshold', v)}
        min={1}
        max={1000}
        step={10}
      />
      <Slider
        id="param-prior-odds"
        label="Prior odds of covert project"
        value={parameters.priorOddsOfCovertProject}
        onChange={(v) => onChange('priorOddsOfCovertProject', v)}
        min={0.01}
        max={1}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="prior_odds"
      />
    </>
  );
}

function PRCComputeSection({ parameters, onChange }: SectionProps) {
  return (
    <>
      <Slider
        label="Total PRC compute (TPP H100e) in 2025"
        value={parameters.totalPrcComputeTppH100eIn2025}
        onChange={(v) => onChange('totalPrcComputeTppH100eIn2025', v)}
        min={10000}
        max={500000}
        step={10000}
        formatValue={(v) => v.toLocaleString()}
        tooltipDoc="prc_capacity"
      />
      <Slider
        label="Annual growth rate of PRC compute stock"
        value={parameters.annualGrowthRateOfPrcComputeStock}
        onChange={(v) => onChange('annualGrowthRateOfPrcComputeStock', v)}
        min={1}
        max={5}
        step={0.1}
        formatValue={(v) => `${v}x`}
      />
      <Slider
        label="PRC architecture efficiency relative to state of the art"
        value={parameters.prcArchitectureEfficiencyRelativeToStateOfTheArt}
        onChange={(v) => onChange('prcArchitectureEfficiencyRelativeToStateOfTheArt', v)}
        min={0.5}
        max={1.5}
        step={0.1}
        tooltipDoc="architecture_efficiency"
      />
      <Slider
        label="Proportion of PRC chip stock produced domestically (2026)"
        value={parameters.proportionOfPrcChipStockProducedDomestically2026}
        onChange={(v) => onChange('proportionOfPrcChipStockProducedDomestically2026', v)}
        min={0}
        max={1}
        step={0.05}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="prc_sme_indigenization"
      />
      <Slider
        label="Proportion of PRC chip stock produced domestically (2030)"
        value={parameters.proportionOfPrcChipStockProducedDomestically2030}
        onChange={(v) => onChange('proportionOfPrcChipStockProducedDomestically2030', v)}
        min={0}
        max={1}
        step={0.05}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="prc_sme_indigenization"
      />
      <Slider
        label="PRC lithography scanners produced in first year"
        value={parameters.prcLithographyScannersProducedInFirstYear}
        onChange={(v) => onChange('prcLithographyScannersProducedInFirstYear', v)}
        min={0}
        max={100}
        step={5}
        tooltipDoc="prc_scanner_rampup"
      />
      <Slider
        label="PRC additional lithography scanners produced per year"
        value={parameters.prcAdditionalLithographyScannersProducedPerYear}
        onChange={(v) => onChange('prcAdditionalLithographyScannersProducedPerYear', v)}
        min={0}
        max={100}
        step={2}
        tooltipDoc="prc_scanner_rampup"
      />
      <Slider
        label="Probability of 28nm localization by 2030"
        value={parameters.pLocalization28nm2030}
        onChange={(v) => onChange('pLocalization28nm2030', v)}
        min={0}
        max={1}
        step={0.05}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="prc_sme_indigenization"
      />
      <Slider
        label="Probability of 14nm localization by 2030"
        value={parameters.pLocalization14nm2030}
        onChange={(v) => onChange('pLocalization14nm2030', v)}
        min={0}
        max={1}
        step={0.05}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="prc_sme_indigenization"
      />
      <Slider
        label="Probability of 7nm localization by 2030"
        value={parameters.pLocalization7nm2030}
        onChange={(v) => onChange('pLocalization7nm2030', v)}
        min={0}
        max={1}
        step={0.05}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="prc_sme_indigenization"
      />
      <Slider
        id="param-chips-per-wafer"
        label="H100-sized chips per wafer"
        value={parameters.h100SizedChipsPerWafer}
        onChange={(v) => onChange('h100SizedChipsPerWafer', v)}
        min={10}
        max={100}
        step={1}
        tooltipDoc="chips_per_wafer"
      />
      <Slider
        id="param-wafers-per-scanner"
        label="Wafers per month per lithography scanner"
        value={parameters.wafersPerMonthPerLithographyScanner}
        onChange={(v) => onChange('wafersPerMonthPerLithographyScanner', v)}
        min={100}
        max={5000}
        step={100}
        tooltipDoc="scanner_production_capacity"
      />
      <Slider
        id="param-construction-time-5k"
        label="Construction time for 5k wafers per month"
        value={parameters.constructionTimeFor5kWafersPerMonth}
        onChange={(v) => onChange('constructionTimeFor5kWafersPerMonth', v)}
        min={0.5}
        max={5}
        step={0.1}
        formatValue={(v) => `${v} years`}
        tooltipDoc="fab_construction_time"
      />
      <Slider
        id="param-construction-time-100k"
        label="Construction time for 100k wafers per month"
        value={parameters.constructionTimeFor100kWafersPerMonth}
        onChange={(v) => onChange('constructionTimeFor100kWafersPerMonth', v)}
        min={0.5}
        max={5}
        step={0.1}
        formatValue={(v) => `${v} years`}
        tooltipDoc="fab_construction_time"
      />
      <Slider
        id="param-wafers-per-operating-worker"
        label="Fab wafers per month per operating worker"
        value={parameters.fabWafersPerMonthPerOperatingWorker}
        onChange={(v) => onChange('fabWafersPerMonthPerOperatingWorker', v)}
        min={1}
        max={100}
        step={1}
        tooltipDoc="operating_labor_production"
      />
      <Slider
        label="Fab wafers per month per construction worker"
        value={parameters.fabWafersPerMonthPerConstructionWorker}
        onChange={(v) => onChange('fabWafersPerMonthPerConstructionWorker', v)}
        min={1}
        max={50}
        step={1}
        tooltipDoc="fab_construction_time"
      />
    </>
  );
}

function PRCDataCentersSection({ parameters, onChange }: SectionProps) {
  return (
    <>
      <Slider
        label="Energy efficiency of compute stock relative to state of the art"
        value={parameters.energyEfficiencyOfComputeStockRelativeToStateOfTheArt}
        onChange={(v) => onChange('energyEfficiencyOfComputeStockRelativeToStateOfTheArt', v)}
        min={0.1}
        max={1}
        step={0.05}
        tooltipDoc="energy_efficiency"
      />
      <Slider
        label="Total PRC energy consumption (GW)"
        value={parameters.totalPrcEnergyConsumptionGw}
        onChange={(v) => onChange('totalPrcEnergyConsumptionGw', v)}
        min={500}
        max={2000}
        step={50}
        formatValue={(v) => `${v} GW`}
        tooltipDoc="prc_energy_consumption"
      />
      <Slider
        label="Data center MW per year per construction worker"
        value={parameters.dataCenterMwPerYearPerConstructionWorker}
        onChange={(v) => onChange('dataCenterMwPerYearPerConstructionWorker', v)}
        min={0.1}
        max={5}
        step={0.1}
        formatValue={(v) => `${v} MW`}
        tooltipDoc="mw_per_worker"
      />
      <Slider
        label="Data center MW per operating worker"
        value={parameters.dataCenterMwPerOperatingWorker}
        onChange={(v) => onChange('dataCenterMwPerOperatingWorker', v)}
        min={1}
        max={50}
        step={1}
        formatValue={(v) => `${v} MW`}
      />
    </>
  );
}

function USComputeSection({ parameters, onChange }: SectionProps) {
  return (
    <>
      <Slider
        label="US frontier project compute (TPP H100e) in 2025"
        value={parameters.usFrontierProjectComputeTppH100eIn2025}
        onChange={(v) => onChange('usFrontierProjectComputeTppH100eIn2025', v)}
        min={10000}
        max={500000}
        step={10000}
        formatValue={(v) => v.toLocaleString()}
        tooltipDoc="largest_ai_project"
      />
      <Slider
        label="US frontier project compute annual growth rate"
        value={parameters.usFrontierProjectComputeAnnualGrowthRate}
        onChange={(v) => onChange('usFrontierProjectComputeAnnualGrowthRate', v)}
        min={1}
        max={10}
        step={0.5}
        formatValue={(v) => `${v}x`}
      />
    </>
  );
}

function ComputeSurvivalSection({ parameters, onChange }: SectionProps) {
  return (
    <>
      <Slider
        id="param-initial-hazard-rate"
        label="Initial annual hazard rate"
        value={parameters.initialAnnualHazardRate}
        onChange={(v) => onChange('initialAnnualHazardRate', v)}
        min={0}
        max={0.2}
        step={0.01}
        formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        tooltipDoc="ai_chip_lifespan"
      />
      <Slider
        id="param-hazard-rate-increase"
        label="Annual hazard rate increase per year"
        value={parameters.annualHazardRateIncreasePerYear}
        onChange={(v) => onChange('annualHazardRateIncreasePerYear', v)}
        min={0}
        max={0.1}
        step={0.005}
        formatValue={(v) => `${(v * 100).toFixed(1)}%`}
        tooltipDoc="ai_chip_lifespan"
      />
    </>
  );
}

function ExogenousComputeSection({ parameters, onChange }: SectionProps) {
  return (
    <>
      <Slider
        id="param-transistor-density-scaling"
        label="Transistor density scaling exponent"
        value={parameters.transistorDensityScalingExponent}
        onChange={(v) => onChange('transistorDensityScalingExponent', v)}
        min={1}
        max={2}
        step={0.01}
        tooltipDoc="transistor_density"
      />
      <Slider
        id="param-architecture-improvement"
        label="State of the art architecture efficiency improvement per year"
        value={parameters.stateOfTheArtArchitectureEfficiencyImprovementPerYear}
        onChange={(v) => onChange('stateOfTheArtArchitectureEfficiencyImprovementPerYear', v)}
        min={1}
        max={2}
        step={0.01}
        formatValue={(v) => `${v}x`}
        tooltipDoc="architecture_efficiency"
      />
      <Slider
        label="Transistor density at end of Dennard scaling (M/mm²)"
        value={parameters.transistorDensityAtEndOfDennardScaling}
        onChange={(v) => onChange('transistorDensityAtEndOfDennardScaling', v)}
        min={1}
        max={50}
        step={1}
        tooltipDoc="dennard_scaling_end"
      />
      <Slider
        id="param-watts-tpp-before-dennard"
        label="Watts/TPP vs transistor density exponent (before Dennard)"
        value={parameters.wattsTppDensityExponentBeforeDennard}
        onChange={(v) => onChange('wattsTppDensityExponentBeforeDennard', v)}
        min={-2}
        max={0}
        step={0.1}
        tooltipDoc="watts_per_tpp_before_dennard"
      />
      <Slider
        id="param-watts-tpp-after-dennard"
        label="Watts/TPP vs transistor density exponent (after Dennard)"
        value={parameters.wattsTppDensityExponentAfterDennard}
        onChange={(v) => onChange('wattsTppDensityExponentAfterDennard', v)}
        min={-1}
        max={0}
        step={0.05}
        tooltipDoc="watts_per_tpp_after_dennard"
      />
      <Slider
        label="State of the art energy efficiency improvement per year"
        value={parameters.stateOfTheArtEnergyEfficiencyImprovementPerYear}
        onChange={(v) => onChange('stateOfTheArtEnergyEfficiencyImprovementPerYear', v)}
        min={1}
        max={2}
        step={0.01}
        formatValue={(v) => `${v}x`}
        tooltipDoc="energy_efficiency_improvement"
      />
    </>
  );
}
