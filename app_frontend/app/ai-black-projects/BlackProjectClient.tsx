'use client';

import { useState, useCallback, useEffect, useMemo } from 'react';
import Link from 'next/link';
import './ai-black-projects.css';

// Import page-specific components
import {
  CCDFChart,
  CovertFabSection,
  DatacenterSection,
  InitialStockSection,
  HowWeEstimateSection,
  Slider,
  CollapsibleSection,
} from './components';
import { BlackProjectData } from '@/types/blackProject';


// Re-export from shared colors for backwards compatibility
import { COLOR_PALETTE as IMPORTED_COLORS, rgba } from './components/colors';
export const COLOR_PALETTE = IMPORTED_COLORS;


interface BlackProjectClientProps {
  initialData: SimulationData | null;
  hideHeader?: boolean;
}

// Parameter types
interface Parameters {
  // Key Parameters
  numYearsToSimulate: number;
  numSimulations: number;
  agreementYear: number;
  blackProjectStartYear: number;
  workersInCovertProject: number;
  meanDetectionTime100: number;
  meanDetectionTime1000: number;
  varianceDetectionTime: number;
  proportionOfInitialChipStockToDivert: number;
  intelligenceMedianError: number;

  // Black Project Properties (from BlackProjectProperties class)
  totalLabor: number;
  fractionOfLaborDevotedToDatacenterConstruction: number;
  fractionOfLaborDevotedToBlackFabConstruction: number;
  fractionOfLaborDevotedToBlackFabOperation: number;
  fractionOfLaborDevotedToAiResearch: number;
  fractionOfDatacenterCapacityToDivert: number;
  fractionOfLithographyScannersToDivert: number;
  maxFractionOfTotalNationalEnergyConsumption: number;
  buildCovertFab: boolean;
  blackFabMaxProcessNode: string;

  // Detection Parameters (from BlackProjectPerceptionsParameters class)
  intelligenceMedianErrorInEstimateOfFabStock: number;
  intelligenceMedianErrorInEnergyConsumptionEstimate: number;
  intelligenceMedianErrorInSatelliteEstimate: number;
  detectionThreshold: number;

  // PRC Compute (from PRCComputeParameters class)
  totalPrcComputeTppH100eIn2025: number;
  annualGrowthRateOfPrcComputeStock: number;
  prcArchitectureEfficiencyRelativeToStateOfTheArt: number;
  proportionOfPrcChipStockProducedDomestically2026: number;
  proportionOfPrcChipStockProducedDomestically2030: number;
  prcLithographyScannersProducedInFirstYear: number;
  prcAdditionalLithographyScannersProducedPerYear: number;
  pLocalization28nm2030: number;
  pLocalization14nm2030: number;
  pLocalization7nm2030: number;
  h100SizedChipsPerWafer: number;
  wafersPerMonthPerLithographyScanner: number;
  constructionTimeFor5kWafersPerMonth: number;
  constructionTimeFor100kWafersPerMonth: number;
  fabWafersPerMonthPerOperatingWorker: number;
  fabWafersPerMonthPerConstructionWorker: number;

  // PRC Data Centers and Energy (from PRCDataCenterAndEnergyParameters class)
  energyEfficiencyOfComputeStockRelativeToStateOfTheArt: number;
  totalPrcEnergyConsumptionGw: number;
  dataCenterMwPerYearPerConstructionWorker: number;
  dataCenterMwPerOperatingWorker: number;

  // US Compute (from USComputeParameters class)
  usFrontierProjectComputeTppH100eIn2025: number;
  usFrontierProjectComputeAnnualGrowthRate: number;

  // Exogenous Compute Trends (from ExogenousComputeTrends class)
  transistorDensityScalingExponent: number;
  stateOfTheArtArchitectureEfficiencyImprovementPerYear: number;
  transistorDensityAtEndOfDennardScaling: number;
  wattsTppDensityExponentBeforeDennard: number;
  wattsTppDensityExponentAfterDennard: number;
  stateOfTheArtEnergyEfficiencyImprovementPerYear: number;
}

// CCDF point type
interface CCDFPoint {
  x: number;
  y: number;
}

// Multi-threshold CCDF data: { "1": [...], "2": [...], "4": [...] }
type MultiThresholdCCDF = Record<string | number, CCDFPoint[]>;

// Simulation data types
interface SimulationData {
  num_simulations?: number;
  black_project_model?: {
    years: number[];
    total_dark_compute: {
      median: number[];
      p25: number[];
      p75: number[];
      individual?: number[][];
    };
    h100_years_before_detection?: {
      median: number;
      p25: number;
      p75: number;
      individual?: number[];
    };
    // Multi-threshold CCDFs (keys are detection thresholds: "1", "2", "4")
    h100_years_ccdf?: MultiThresholdCCDF;
    time_to_detection?: {
      median: number;
    };
    time_to_detection_ccdf?: MultiThresholdCCDF;
    ai_rd_reduction?: {
      median: number;
    };
    ai_rd_reduction_ccdf?: MultiThresholdCCDF;
    chip_production_reduction_ccdf?: MultiThresholdCCDF;
    chips_produced?: {
      median: number;
    };
    posterior_prob_project?: {
      median: number[];
      p25: number[];
      p75: number[];
    };
  };
  black_fab?: {
    years: number[];
    is_operational?: {
      proportion: number[];
    };
    wafer_starts?: {
      median: number[];
      individual?: number[][];
    };
  };
  black_datacenters?: {
    capacity_ccdfs?: Record<string, CCDFPoint[]>;
  };
  initial_stock?: {
    lr_prc_accounting_samples?: number[];
  };
  [key: string]: unknown;
}

// Default parameters
const defaultParameters: Parameters = {
  // Key Parameters
  numYearsToSimulate: 10,
  numSimulations: 100,
  agreementYear: 2027,
  blackProjectStartYear: 2029,
  workersInCovertProject: 11300,
  meanDetectionTime100: 6.95,
  meanDetectionTime1000: 3.42,
  varianceDetectionTime: 3.88,
  proportionOfInitialChipStockToDivert: 0.05,
  intelligenceMedianError: 0.07,

  // Black Project Properties (from BlackProjectProperties class)
  totalLabor: 11300,
  fractionOfLaborDevotedToDatacenterConstruction: 0.885,
  fractionOfLaborDevotedToBlackFabConstruction: 0.022,
  fractionOfLaborDevotedToBlackFabOperation: 0.049,
  fractionOfLaborDevotedToAiResearch: 0.044,
  fractionOfDatacenterCapacityToDivert: 0.5,
  fractionOfLithographyScannersToDivert: 0.10,
  maxFractionOfTotalNationalEnergyConsumption: 0.05,
  buildCovertFab: true,
  blackFabMaxProcessNode: '28',

  // Detection Parameters (from BlackProjectPerceptionsParameters class)
  intelligenceMedianErrorInEstimateOfFabStock: 0.07,
  intelligenceMedianErrorInEnergyConsumptionEstimate: 0.07,
  intelligenceMedianErrorInSatelliteEstimate: 0.01,
  detectionThreshold: 100.0,

  // PRC Compute (from PRCComputeParameters class)
  totalPrcComputeTppH100eIn2025: 100000,
  annualGrowthRateOfPrcComputeStock: 2.2,
  prcArchitectureEfficiencyRelativeToStateOfTheArt: 1.0,
  proportionOfPrcChipStockProducedDomestically2026: 0.10,
  proportionOfPrcChipStockProducedDomestically2030: 0.40,
  prcLithographyScannersProducedInFirstYear: 20,
  prcAdditionalLithographyScannersProducedPerYear: 16,
  pLocalization28nm2030: 0.25,
  pLocalization14nm2030: 0.10,
  pLocalization7nm2030: 0.06,
  h100SizedChipsPerWafer: 28,
  wafersPerMonthPerLithographyScanner: 1000,
  constructionTimeFor5kWafersPerMonth: 1.4,
  constructionTimeFor100kWafersPerMonth: 2.41,
  fabWafersPerMonthPerOperatingWorker: 24.64,
  fabWafersPerMonthPerConstructionWorker: 14.1,

  // PRC Data Centers and Energy (from PRCDataCenterAndEnergyParameters class)
  energyEfficiencyOfComputeStockRelativeToStateOfTheArt: 0.20,
  totalPrcEnergyConsumptionGw: 1100,
  dataCenterMwPerYearPerConstructionWorker: 1.0,
  dataCenterMwPerOperatingWorker: 10.0,

  // US Compute (from USComputeParameters class)
  usFrontierProjectComputeTppH100eIn2025: 120325,
  usFrontierProjectComputeAnnualGrowthRate: 4.0,

  // Exogenous Compute Trends (from ExogenousComputeTrends class)
  transistorDensityScalingExponent: 1.49,
  stateOfTheArtArchitectureEfficiencyImprovementPerYear: 1.23,
  transistorDensityAtEndOfDennardScaling: 10.0,
  wattsTppDensityExponentBeforeDennard: -1.0,
  wattsTppDensityExponentAfterDennard: -0.33,
  stateOfTheArtEnergyEfficiencyImprovementPerYear: 1.26,
};

export function BlackProjectClient({ initialData, hideHeader = false }: BlackProjectClientProps) {
  const [data, setData] = useState<SimulationData | null>(initialData);
  const [isLoading, setIsLoading] = useState(!initialData);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('');
  const [parameters, setParameters] = useState<Parameters>(defaultParameters);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Load default data on mount
  useEffect(() => {
    if (initialData) return;

    const loadDefaultData = async () => {
      try {
        // Run a quick simulation with default parameters
        const response = await fetch('http://127.0.0.1:5329/api/run-black-project-simulation', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            parameters: {},
            num_simulations: 50,  // Fewer for initial load
            time_range: [
              defaultParameters.agreementYear,
              defaultParameters.agreementYear + defaultParameters.numYearsToSimulate,
            ],
          }),
        });
        if (response.ok) {
          const result = await response.json();
          if (result.success) {
            setData(result);
          } else {
            setError(result.error || 'Simulation failed');
            setStatus('Simulation failed');
          }
        } else {
          setError('Backend unavailable');
          setStatus('Backend unavailable');
        }
      } catch (err) {
        console.error('Failed to load default data:', err);
        setError('Backend unavailable - start the ai_futures_simulator backend on port 5329');
        setStatus('Backend unavailable');
      } finally {
        setIsLoading(false);
      }
    };

    loadDefaultData();
  }, [initialData]);

  const runSimulation = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setStatus('');

    try {
      // Use the new ai_futures_simulator backend API
      const response = await fetch('http://127.0.0.1:5329/api/run-black-project-simulation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          parameters: {
            // Black project properties
            black_project_start_year: parameters.blackProjectStartYear,
            total_labor: parameters.totalLabor,
            fraction_of_labor_devoted_to_datacenter_construction: parameters.fractionOfLaborDevotedToDatacenterConstruction,
            fraction_of_labor_devoted_to_black_fab_construction: parameters.fractionOfLaborDevotedToBlackFabConstruction,
            fraction_of_labor_devoted_to_black_fab_operation: parameters.fractionOfLaborDevotedToBlackFabOperation,
            fraction_of_labor_devoted_to_ai_research: parameters.fractionOfLaborDevotedToAiResearch,
            fraction_of_initial_compute_stock_to_divert_at_black_project_start: parameters.proportionOfInitialChipStockToDivert,
            fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start: parameters.fractionOfDatacenterCapacityToDivert,
            fraction_of_lithography_scanners_to_divert_at_black_project_start: parameters.fractionOfLithographyScannersToDivert,
            max_fraction_of_total_national_energy_consumption: parameters.maxFractionOfTotalNationalEnergyConsumption,
            build_a_black_fab: parameters.buildCovertFab,
            black_fab_max_process_node: parseFloat(parameters.blackFabMaxProcessNode),
            // Detection parameters
            intelligence_median_error_in_estimate_of_compute_stock: parameters.intelligenceMedianError,
            intelligence_median_error_in_estimate_of_fab_stock: parameters.intelligenceMedianErrorInEstimateOfFabStock,
            intelligence_median_error_in_energy_consumption_estimate_of_datacenter_capacity: parameters.intelligenceMedianErrorInEnergyConsumptionEstimate,
            intelligence_median_error_in_satellite_estimate_of_datacenter_capacity: parameters.intelligenceMedianErrorInSatelliteEstimate,
            mean_detection_time_for_100_workers: parameters.meanDetectionTime100,
            mean_detection_time_for_1000_workers: parameters.meanDetectionTime1000,
            variance_of_detection_time_given_num_workers: parameters.varianceDetectionTime,
            detection_threshold: parameters.detectionThreshold,
            // PRC compute parameters
            total_prc_compute_tpp_h100e_in_2025: parameters.totalPrcComputeTppH100eIn2025,
            annual_growth_rate_of_prc_compute_stock: parameters.annualGrowthRateOfPrcComputeStock,
            prc_architecture_efficiency_relative_to_state_of_the_art: parameters.prcArchitectureEfficiencyRelativeToStateOfTheArt,
            proportion_of_prc_chip_stock_produced_domestically_2026: parameters.proportionOfPrcChipStockProducedDomestically2026,
            proportion_of_prc_chip_stock_produced_domestically_2030: parameters.proportionOfPrcChipStockProducedDomestically2030,
            prc_lithography_scanners_produced_in_first_year: parameters.prcLithographyScannersProducedInFirstYear,
            prc_additional_lithography_scanners_produced_per_year: parameters.prcAdditionalLithographyScannersProducedPerYear,
            p_localization_28nm_2030: parameters.pLocalization28nm2030,
            p_localization_14nm_2030: parameters.pLocalization14nm2030,
            p_localization_7nm_2030: parameters.pLocalization7nm2030,
            h100_sized_chips_per_wafer: parameters.h100SizedChipsPerWafer,
            wafers_per_month_per_lithography_scanner: parameters.wafersPerMonthPerLithographyScanner,
            construction_time_for_5k_wafers_per_month: parameters.constructionTimeFor5kWafersPerMonth,
            construction_time_for_100k_wafers_per_month: parameters.constructionTimeFor100kWafersPerMonth,
            fab_wafers_per_month_per_operating_worker: parameters.fabWafersPerMonthPerOperatingWorker,
            fab_wafers_per_month_per_construction_worker_under_standard_timeline: parameters.fabWafersPerMonthPerConstructionWorker,
            // PRC data centers and energy parameters
            energy_efficiency_of_compute_stock_relative_to_state_of_the_art: parameters.energyEfficiencyOfComputeStockRelativeToStateOfTheArt,
            total_prc_energy_consumption_gw: parameters.totalPrcEnergyConsumptionGw,
            data_center_mw_per_year_per_construction_worker: parameters.dataCenterMwPerYearPerConstructionWorker,
            data_center_mw_per_operating_worker: parameters.dataCenterMwPerOperatingWorker,
            // US compute parameters
            us_frontier_project_compute_tpp_h100e_in_2025: parameters.usFrontierProjectComputeTppH100eIn2025,
            us_frontier_project_compute_annual_growth_rate: parameters.usFrontierProjectComputeAnnualGrowthRate,
            // Exogenous compute trends
            transistor_density_scaling_exponent: parameters.transistorDensityScalingExponent,
            state_of_the_art_architecture_efficiency_improvement_per_year: parameters.stateOfTheArtArchitectureEfficiencyImprovementPerYear,
            transistor_density_at_end_of_dennard_scaling_m_per_mm2: parameters.transistorDensityAtEndOfDennardScaling,
            watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended: parameters.wattsTppDensityExponentBeforeDennard,
            watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended: parameters.wattsTppDensityExponentAfterDennard,
            state_of_the_art_energy_efficiency_improvement_per_year: parameters.stateOfTheArtEnergyEfficiencyImprovementPerYear,
          },
          num_simulations: parameters.numSimulations,
          time_range: [
            parameters.agreementYear,
            parameters.agreementYear + parameters.numYearsToSimulate,
          ],
        }),
      });

      if (!response.ok) {
        throw new Error(`Simulation failed: ${response.statusText}`);
      }

      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error || 'Unknown simulation error');
      }
      setData(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      setStatus(`Error: ${message}`);
    } finally {
      setIsLoading(false);
    }
  }, [parameters]);

  // Auto-run simulation when parameters change (with debounce)
  useEffect(() => {
    // Skip initial render when we're loading default data
    if (isLoading && !data) return;

    const timeoutId = setTimeout(() => {
      runSimulation();
    }, 500); // 500ms debounce

    return () => clearTimeout(timeoutId);
  }, [parameters]); // eslint-disable-line react-hooks/exhaustive-deps

  // Dashboard values - calculate medians from individual simulation arrays
  const dashboardValues = useMemo(() => {
    if (!data?.black_project_model) {
      return {
        medianH100Years: '--',
        medianTimeToDetection: '--',
        aiRdReduction: '--',
        chipsProduced: '--',
      };
    }

    const model = data.black_project_model as {
      individual_project_h100_years_before_detection?: number[];
      individual_project_time_before_detection?: number[];
      individual_project_h100e_before_detection?: number[];
      [key: string]: unknown;
    };

    const formatNumber = (n: number) => {
      if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
      if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
      if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
      return n.toFixed(1);
    };

    // Helper to calculate median from array
    const getMedian = (arr: number[] | undefined): number | null => {
      if (!arr || arr.length === 0) return null;
      const sorted = [...arr].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };

    const h100YearsMedian = getMedian(model.individual_project_h100_years_before_detection);
    const timeMedian = getMedian(model.individual_project_time_before_detection);
    const h100eMedian = getMedian(model.individual_project_h100e_before_detection);

    // Get AI R&D reduction if available from backend, otherwise use placeholder
    const aiRdReductionData = model.ai_rd_reduction_median as number | undefined;

    return {
      medianH100Years: h100YearsMedian !== null
        ? `${formatNumber(h100YearsMedian)} H100-years`
        : '--',
      medianTimeToDetection: timeMedian !== null
        ? `${timeMedian.toFixed(1)} years`
        : '--',
      aiRdReduction: aiRdReductionData !== undefined
        ? `${(aiRdReductionData * 100).toFixed(1)}%`
        : '~5%', // Placeholder until backend provides this
      chipsProduced: h100eMedian !== null
        ? formatNumber(h100eMedian)
        : '--',
    };
  }, [data]);

  const updateParameter = <K extends keyof Parameters>(key: K, value: Parameters[K]) => {
    setParameters(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="min-h-screen bg-[#fffff8] flex flex-col">
      {/* Header - only shown when not embedded */}
      {!hideHeader && (
        <header className="fixed top-0 left-0 right-0 z-50 h-14 flex flex-row items-center justify-between px-6 border-b border-gray-200 bg-white">
          <nav className="flex flex-row">
            <Link
              href="/ai-timelines-and-takeoff"
              className="px-4 py-4 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50 border-b-2 border-transparent"
            >
              AI Timelines and Takeoff
            </Link>
            <Link
              href="/ai-black-projects"
              className="px-4 py-4 text-sm font-medium text-gray-900 border-b-2 border-blue-500"
            >
              Black Projects
            </Link>
          </nav>
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="lg:hidden p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </header>
      )}

      <div className={`flex flex-1 overflow-hidden ${hideHeader ? '' : 'mt-14'}`}>
        {/* Sidebar */}
        <aside
          className={`bp-sidebar ${sidebarOpen ? 'fixed inset-y-0 left-0 z-50' : 'hidden lg:block'}`}
          style={hideHeader || sidebarOpen ? { top: 0, height: '100vh' } : { top: '56px', height: 'calc(100vh - 56px)' }}
        >
          {/* Sidebar Header - fixed at top of sidebar */}
          <div className="bp-sidebar-header">
            <div className="sidebar-title">Black Project Parameters</div>
            <Link
              href="/ai-black-project-parameters"
              className="text-xs hover:underline"
              style={{ color: '#5E6FB8' }}
              target="_blank"
            >
              Open documentation →
            </Link>
          </div>

          {/* Sidebar Content - scrollable */}
          <div className="bp-sidebar-content">
            {/* Key parameters section */}
            <div className="mb-2">
              <div className="bp-section-header">
                <span>Key parameters</span>
                <div className="bp-section-line" />
              </div>
              <Slider
                label="Years to simulate"
                value={parameters.numYearsToSimulate}
                onChange={(v) => updateParameter('numYearsToSimulate', v)}
                min={1}
                max={20}
                step={0.5}
                formatValue={(v) => `${v} years`}
              />
              <Slider
                label="Number of simulations"
                value={parameters.numSimulations}
                onChange={(v) => updateParameter('numSimulations', v)}
                min={10}
                max={1000}
                step={10}
              />
              <Slider
                label="Slowdown agreement start year"
                value={parameters.agreementYear}
                onChange={(v) => updateParameter('agreementYear', v)}
                min={2026}
                max={2035}
                step={1}
              />
              <Slider
                label="Black project start year"
                value={parameters.blackProjectStartYear}
                onChange={(v) => updateParameter('blackProjectStartYear', v)}
                min={2024}
                max={2035}
                step={1}
              />
              <Slider
                label="Workers involved in covert project"
                value={parameters.workersInCovertProject}
                onChange={(v) => updateParameter('workersInCovertProject', v)}
                min={1000}
                max={100000}
                step={1000}
                formatValue={(v) => v.toLocaleString()}
              />
              <Slider
                label="100 workers mean detection time"
                value={parameters.meanDetectionTime100}
                onChange={(v) => updateParameter('meanDetectionTime100', v)}
                min={1}
                max={20}
                step={0.5}
                formatValue={(v) => `${v} years`}
              />
              <Slider
                label="1000 workers mean detection time"
                value={parameters.meanDetectionTime1000}
                onChange={(v) => updateParameter('meanDetectionTime1000', v)}
                min={1}
                max={20}
                step={0.5}
                formatValue={(v) => `${v} years`}
              />
              <Slider
                label="Detection time variance"
                value={parameters.varianceDetectionTime}
                onChange={(v) => updateParameter('varianceDetectionTime', v)}
                min={0.1}
                max={10}
                step={0.1}
              />
              <Slider
                label="Fraction of compute to divert"
                value={parameters.proportionOfInitialChipStockToDivert}
                onChange={(v) => updateParameter('proportionOfInitialChipStockToDivert', v)}
                min={0}
                max={0.5}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Median error in intelligence estimating compute stock"
                value={parameters.intelligenceMedianError}
                onChange={(v) => updateParameter('intelligenceMedianError', v)}
                min={0.01}
                max={0.5}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
            </div>

            {/* Black project properties */}
            <CollapsibleSection title="Black project properties">
              <Slider
                label="Total labor"
                value={parameters.totalLabor}
                onChange={(v) => updateParameter('totalLabor', v)}
                min={1000}
                max={100000}
                step={1000}
                formatValue={(v) => v.toLocaleString()}
              />
              <Slider
                label="Fraction of labor devoted to datacenter construction"
                value={parameters.fractionOfLaborDevotedToDatacenterConstruction}
                onChange={(v) => updateParameter('fractionOfLaborDevotedToDatacenterConstruction', v)}
                min={0}
                max={1}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Fraction of labor devoted to black fab construction"
                value={parameters.fractionOfLaborDevotedToBlackFabConstruction}
                onChange={(v) => updateParameter('fractionOfLaborDevotedToBlackFabConstruction', v)}
                min={0}
                max={1}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Fraction of labor devoted to black fab operation"
                value={parameters.fractionOfLaborDevotedToBlackFabOperation}
                onChange={(v) => updateParameter('fractionOfLaborDevotedToBlackFabOperation', v)}
                min={0}
                max={1}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Fraction of labor devoted to AI research"
                value={parameters.fractionOfLaborDevotedToAiResearch}
                onChange={(v) => updateParameter('fractionOfLaborDevotedToAiResearch', v)}
                min={0}
                max={1}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Fraction of datacenter capacity to divert"
                value={parameters.fractionOfDatacenterCapacityToDivert}
                onChange={(v) => updateParameter('fractionOfDatacenterCapacityToDivert', v)}
                min={0}
                max={1}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Fraction of lithography scanners to divert"
                value={parameters.fractionOfLithographyScannersToDivert}
                onChange={(v) => updateParameter('fractionOfLithographyScannersToDivert', v)}
                min={0}
                max={1}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Max fraction of total national energy consumption"
                value={parameters.maxFractionOfTotalNationalEnergyConsumption}
                onChange={(v) => updateParameter('maxFractionOfTotalNationalEnergyConsumption', v)}
                min={0}
                max={0.2}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <div className="bp-checkbox-group">
                <input
                  type="checkbox"
                  id="param-build-covert-fab"
                  checked={parameters.buildCovertFab}
                  onChange={(e) => updateParameter('buildCovertFab', e.target.checked)}
                />
                <label htmlFor="param-build-covert-fab">Build a black fab</label>
              </div>
              <div className="space-y-2 mb-2">
                <label className="block text-xs font-medium text-foreground">
                  Black fab max process node
                </label>
                <select
                  value={parameters.blackFabMaxProcessNode}
                  onChange={(e) => updateParameter('blackFabMaxProcessNode', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                >
                  <option value="130">130nm</option>
                  <option value="28">28nm</option>
                  <option value="14">14nm</option>
                  <option value="7">7nm</option>
                </select>
              </div>
            </CollapsibleSection>

            {/* Detection parameters */}
            <CollapsibleSection title="Detection parameters">
              <Slider
                label="Intelligence median error in estimate of fab stock"
                value={parameters.intelligenceMedianErrorInEstimateOfFabStock}
                onChange={(v) => updateParameter('intelligenceMedianErrorInEstimateOfFabStock', v)}
                min={0.01}
                max={0.5}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Intelligence median error in energy consumption estimate"
                value={parameters.intelligenceMedianErrorInEnergyConsumptionEstimate}
                onChange={(v) => updateParameter('intelligenceMedianErrorInEnergyConsumptionEstimate', v)}
                min={0.01}
                max={0.5}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Intelligence median error in satellite estimate"
                value={parameters.intelligenceMedianErrorInSatelliteEstimate}
                onChange={(v) => updateParameter('intelligenceMedianErrorInSatelliteEstimate', v)}
                min={0.001}
                max={0.1}
                step={0.001}
                formatValue={(v) => `${(v * 100).toFixed(1)}%`}
              />
              <Slider
                label="Detection threshold"
                value={parameters.detectionThreshold}
                onChange={(v) => updateParameter('detectionThreshold', v)}
                min={1}
                max={1000}
                step={10}
              />
            </CollapsibleSection>

            {/* PRC compute */}
            <CollapsibleSection title="PRC compute">
              <Slider
                label="Total PRC compute (TPP H100e) in 2025"
                value={parameters.totalPrcComputeTppH100eIn2025}
                onChange={(v) => updateParameter('totalPrcComputeTppH100eIn2025', v)}
                min={10000}
                max={500000}
                step={10000}
                formatValue={(v) => v.toLocaleString()}
              />
              <Slider
                label="Annual growth rate of PRC compute stock"
                value={parameters.annualGrowthRateOfPrcComputeStock}
                onChange={(v) => updateParameter('annualGrowthRateOfPrcComputeStock', v)}
                min={1}
                max={5}
                step={0.1}
                formatValue={(v) => `${v}x`}
              />
              <Slider
                label="PRC architecture efficiency relative to state of the art"
                value={parameters.prcArchitectureEfficiencyRelativeToStateOfTheArt}
                onChange={(v) => updateParameter('prcArchitectureEfficiencyRelativeToStateOfTheArt', v)}
                min={0.5}
                max={1.5}
                step={0.1}
              />
              <Slider
                label="Proportion of PRC chip stock produced domestically (2026)"
                value={parameters.proportionOfPrcChipStockProducedDomestically2026}
                onChange={(v) => updateParameter('proportionOfPrcChipStockProducedDomestically2026', v)}
                min={0}
                max={1}
                step={0.05}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Proportion of PRC chip stock produced domestically (2030)"
                value={parameters.proportionOfPrcChipStockProducedDomestically2030}
                onChange={(v) => updateParameter('proportionOfPrcChipStockProducedDomestically2030', v)}
                min={0}
                max={1}
                step={0.05}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="PRC lithography scanners produced in first year"
                value={parameters.prcLithographyScannersProducedInFirstYear}
                onChange={(v) => updateParameter('prcLithographyScannersProducedInFirstYear', v)}
                min={0}
                max={100}
                step={5}
              />
              <Slider
                label="PRC additional lithography scanners produced per year"
                value={parameters.prcAdditionalLithographyScannersProducedPerYear}
                onChange={(v) => updateParameter('prcAdditionalLithographyScannersProducedPerYear', v)}
                min={0}
                max={100}
                step={2}
              />
              <Slider
                label="Probability of 28nm localization by 2030"
                value={parameters.pLocalization28nm2030}
                onChange={(v) => updateParameter('pLocalization28nm2030', v)}
                min={0}
                max={1}
                step={0.05}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Probability of 14nm localization by 2030"
                value={parameters.pLocalization14nm2030}
                onChange={(v) => updateParameter('pLocalization14nm2030', v)}
                min={0}
                max={1}
                step={0.05}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Probability of 7nm localization by 2030"
                value={parameters.pLocalization7nm2030}
                onChange={(v) => updateParameter('pLocalization7nm2030', v)}
                min={0}
                max={1}
                step={0.05}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="H100-sized chips per wafer"
                value={parameters.h100SizedChipsPerWafer}
                onChange={(v) => updateParameter('h100SizedChipsPerWafer', v)}
                min={10}
                max={100}
                step={1}
              />
              <Slider
                label="Wafers per month per lithography scanner"
                value={parameters.wafersPerMonthPerLithographyScanner}
                onChange={(v) => updateParameter('wafersPerMonthPerLithographyScanner', v)}
                min={100}
                max={5000}
                step={100}
              />
              <Slider
                label="Construction time for 5k wafers per month"
                value={parameters.constructionTimeFor5kWafersPerMonth}
                onChange={(v) => updateParameter('constructionTimeFor5kWafersPerMonth', v)}
                min={0.5}
                max={5}
                step={0.1}
                formatValue={(v) => `${v} years`}
              />
              <Slider
                label="Construction time for 100k wafers per month"
                value={parameters.constructionTimeFor100kWafersPerMonth}
                onChange={(v) => updateParameter('constructionTimeFor100kWafersPerMonth', v)}
                min={0.5}
                max={5}
                step={0.1}
                formatValue={(v) => `${v} years`}
              />
              <Slider
                label="Fab wafers per month per operating worker"
                value={parameters.fabWafersPerMonthPerOperatingWorker}
                onChange={(v) => updateParameter('fabWafersPerMonthPerOperatingWorker', v)}
                min={1}
                max={100}
                step={1}
              />
              <Slider
                label="Fab wafers per month per construction worker"
                value={parameters.fabWafersPerMonthPerConstructionWorker}
                onChange={(v) => updateParameter('fabWafersPerMonthPerConstructionWorker', v)}
                min={1}
                max={50}
                step={1}
              />
            </CollapsibleSection>

            {/* PRC data centers and energy */}
            <CollapsibleSection title="PRC data centers and energy">
              <Slider
                label="Energy efficiency of compute stock relative to state of the art"
                value={parameters.energyEfficiencyOfComputeStockRelativeToStateOfTheArt}
                onChange={(v) => updateParameter('energyEfficiencyOfComputeStockRelativeToStateOfTheArt', v)}
                min={0.1}
                max={1}
                step={0.05}
              />
              <Slider
                label="Total PRC energy consumption (GW)"
                value={parameters.totalPrcEnergyConsumptionGw}
                onChange={(v) => updateParameter('totalPrcEnergyConsumptionGw', v)}
                min={500}
                max={2000}
                step={50}
                formatValue={(v) => `${v} GW`}
              />
              <Slider
                label="Data center MW per year per construction worker"
                value={parameters.dataCenterMwPerYearPerConstructionWorker}
                onChange={(v) => updateParameter('dataCenterMwPerYearPerConstructionWorker', v)}
                min={0.1}
                max={5}
                step={0.1}
                formatValue={(v) => `${v} MW`}
              />
              <Slider
                label="Data center MW per operating worker"
                value={parameters.dataCenterMwPerOperatingWorker}
                onChange={(v) => updateParameter('dataCenterMwPerOperatingWorker', v)}
                min={1}
                max={50}
                step={1}
                formatValue={(v) => `${v} MW`}
              />
            </CollapsibleSection>

            {/* US compute */}
            <CollapsibleSection title="US compute">
              <Slider
                label="US frontier project compute (TPP H100e) in 2025"
                value={parameters.usFrontierProjectComputeTppH100eIn2025}
                onChange={(v) => updateParameter('usFrontierProjectComputeTppH100eIn2025', v)}
                min={10000}
                max={500000}
                step={10000}
                formatValue={(v) => v.toLocaleString()}
              />
              <Slider
                label="US frontier project compute annual growth rate"
                value={parameters.usFrontierProjectComputeAnnualGrowthRate}
                onChange={(v) => updateParameter('usFrontierProjectComputeAnnualGrowthRate', v)}
                min={1}
                max={10}
                step={0.5}
                formatValue={(v) => `${v}x`}
              />
            </CollapsibleSection>

            {/* Exogenous compute trends */}
            <CollapsibleSection title="Exogenous compute trends">
              <Slider
                label="Transistor density scaling exponent"
                value={parameters.transistorDensityScalingExponent}
                onChange={(v) => updateParameter('transistorDensityScalingExponent', v)}
                min={1}
                max={2}
                step={0.01}
              />
              <Slider
                label="State of the art architecture efficiency improvement per year"
                value={parameters.stateOfTheArtArchitectureEfficiencyImprovementPerYear}
                onChange={(v) => updateParameter('stateOfTheArtArchitectureEfficiencyImprovementPerYear', v)}
                min={1}
                max={2}
                step={0.01}
                formatValue={(v) => `${v}x`}
              />
              <Slider
                label="Transistor density at end of Dennard scaling (M/mm²)"
                value={parameters.transistorDensityAtEndOfDennardScaling}
                onChange={(v) => updateParameter('transistorDensityAtEndOfDennardScaling', v)}
                min={1}
                max={50}
                step={1}
              />
              <Slider
                label="Watts/TPP vs transistor density exponent (before Dennard)"
                value={parameters.wattsTppDensityExponentBeforeDennard}
                onChange={(v) => updateParameter('wattsTppDensityExponentBeforeDennard', v)}
                min={-2}
                max={0}
                step={0.1}
              />
              <Slider
                label="Watts/TPP vs transistor density exponent (after Dennard)"
                value={parameters.wattsTppDensityExponentAfterDennard}
                onChange={(v) => updateParameter('wattsTppDensityExponentAfterDennard', v)}
                min={-1}
                max={0}
                step={0.05}
              />
              <Slider
                label="State of the art energy efficiency improvement per year"
                value={parameters.stateOfTheArtEnergyEfficiencyImprovementPerYear}
                onChange={(v) => updateParameter('stateOfTheArtEnergyEfficiencyImprovementPerYear', v)}
                min={1}
                max={2}
                step={0.01}
                formatValue={(v) => `${v}x`}
              />
            </CollapsibleSection>
          </div>
        </aside>

        {/* Mobile overlay */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main content */}
        <main className="flex-1 overflow-y-auto p-6 lg:ml-[260px]">
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">
              {error}
            </div>
          )}

          <div className="space-y-8">
            {/* Title and description */}
            <div>
              <h1 className="text-2xl md:text-3xl font-bold text-gray-800 mb-3">
                Dark compute model
              </h1>
              <p className="text-gray-600">
                This is a model of how much compute a covert AI project might operate if the PRC cheats an AI slowdown agreement.
                Predictions assume an agreement goes into force in{' '}
                <span className="font-semibold text-[#5E6FB8]">{parameters.agreementYear}</span>.
              </p>
            </div>

            {/* Dashboard and top plots - matching black_project_top_section.html layout */}
            <div className="flex flex-wrap gap-5 items-stretch" style={{ minHeight: '350px' }}>
              {/* Dashboard */}
              <div className="dashboard" style={{ flex: '0 0 auto', width: '240px', padding: '20px', display: 'flex', flexDirection: 'column', gap: 0, justifyContent: 'flex-start' }}>
                <div className="plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: '8px', margin: '0 0 20px 0' }}>
                  Median outcome
                </div>
                <div className="dashboard-item" style={{ margin: '0 0 20px 0' }}>
                  <div className="dashboard-value">{dashboardValues.medianH100Years}</div>
                  <div className="dashboard-label">Covert computation*</div>
                </div>
                <div className="dashboard-item" style={{ margin: '0 0 20px 0' }}>
                  <div className="dashboard-value">{dashboardValues.medianTimeToDetection}</div>
                  <div className="dashboard-label">Time to detection*</div>
                </div>
                <div className="dashboard-item" style={{ margin: '0 0 20px 0' }}>
                  <div className="dashboard-value">{dashboardValues.aiRdReduction}</div>
                  <div className="dashboard-label">Reduction in AI R&D computation of largest company*</div>
                </div>
                <div className="dashboard-item" style={{ margin: 0 }}>
                  <div className="dashboard-value">{dashboardValues.chipsProduced}</div>
                  <div className="dashboard-label">Chips covertly produced*</div>
                </div>
              </div>

              {/* Covert Compute CCDF */}
              <div className="plot-container" style={{ flex: '1 1 300px', minWidth: '300px', padding: '20px', display: 'flex', flexDirection: 'column' }}>
                <div className="plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: '8px', marginBottom: '10px', marginTop: 0 }}>
                  Covert compute
                </div>
                <div className="plot" style={{ flex: 1, minHeight: 0 }}>
                  <CCDFChart
                    data={data?.black_project_model?.h100_years_ccdf}
                    color={COLOR_PALETTE.chip_stock}
                    xLabel="H100-years"
                    yLabel="P(X > x)"
                    isLoading={isLoading}
                  />
                </div>
              </div>

              {/* Time to Detection CCDF */}
              <div className="plot-container" style={{ flex: '1 1 300px', minWidth: '300px', padding: '20px', display: 'flex', flexDirection: 'column' }}>
                <div className="plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: '8px', marginBottom: '10px', marginTop: 0 }}>
                  Time to detection (after which agreement ends)
                </div>
                <div className="plot" style={{ flex: 1, minHeight: 0 }}>
                  <CCDFChart
                    data={data?.black_project_model?.time_to_detection_ccdf}
                    color={COLOR_PALETTE.detection}
                    xLabel="Years"
                    yLabel="P(X > x)"
                    isLoading={isLoading}
                  />
                </div>
              </div>

              {/* Wrapper for last two plots so they wrap together (matching original) */}
              <div style={{ flex: '2 1 620px', minWidth: '620px', display: 'flex', gap: '20px' }}>
                {/* Chip Production CCDF */}
                <div className="plot-container" style={{ flex: '1 1 300px', minWidth: '300px', padding: '20px', display: 'flex', flexDirection: 'column' }}>
                  <div className="plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: '8px', marginBottom: '10px', marginTop: 0 }}>
                    Covert AI chip production relative to no slowdown*
                  </div>
                  <div className="plot" style={{ flex: 1, minHeight: 0 }}>
                    <CCDFChart
                      data={data?.black_project_model?.chip_production_reduction_ccdf}
                      color={COLOR_PALETTE.fab}
                      xLabel="Fraction"
                      yLabel="P(X > x)"
                      xAsPercent
                      showArea={false}
                      isLoading={isLoading}
                    />
                  </div>
                </div>

                {/* AI R&D Reduction CCDF */}
                <div className="plot-container" style={{ flex: '1 1 300px', minWidth: '300px', padding: '20px', display: 'flex', flexDirection: 'column' }}>
                  <div className="plot-title" style={{ textAlign: 'center', borderBottom: '1px solid #ddd', paddingBottom: '8px', marginBottom: '10px', marginTop: 0 }}>
                    Covert AI R&D computation relative to no slowdown*
                  </div>
                  <div className="plot" style={{ flex: 1, minHeight: 0 }}>
                    <CCDFChart
                      data={data?.black_project_model?.ai_rd_reduction_ccdf}
                      color={COLOR_PALETTE.datacenters_and_energy}
                      xLabel="Fraction"
                      yLabel="P(X > x)"
                      xAsPercent
                      showArea={false}
                      isLoading={isLoading}
                    />
                  </div>
                </div>
              </div>
            </div>

            <p className="text-xs text-gray-500 italic mt-4">
              *Unless otherwise specified, US intelligence &apos;detects&apos; a covert project after it receives &gt;4x update that the project exists, after which, USG exits the AI slowdown agreement.
            </p>

            {/* Divider */}
            <hr className="my-8 border-gray-200" />

            {/* How We Estimate Section (includes rate of computation and detection likelihood) */}
            <HowWeEstimateSection />

            {/* Divider */}
            <hr className="my-8 border-gray-200" />

            {/* Initial Stock Section - First among detailed breakdowns */}
            <div id="initialStockSection">
              <InitialStockSection
                data={data as unknown as BlackProjectData}
                isLoading={isLoading}
                diversionProportion={parameters.proportionOfInitialChipStockToDivert}
                agreementYear={parameters.agreementYear}
              />
            </div>

            {/* Divider */}
            <hr className="my-8 border-gray-200" />

            {/* Datacenter Section */}
            <div id="covertDataCentersSection">
              <DatacenterSection data={data as unknown as BlackProjectData} isLoading={isLoading} agreementYear={parameters.agreementYear} />
            </div>

            {/* Divider */}
            <hr className="my-8 border-gray-200" />

            {/* Covert Fab Section */}
            <div id="covertFabSection">
              <CovertFabSection data={data as unknown as BlackProjectData} isLoading={isLoading} />
            </div>

            {/* Debug data display */}
            {data && (
              <details className="mt-8">
                <summary className="cursor-pointer text-sm text-gray-500">View raw data</summary>
                <pre className="mt-2 p-4 bg-gray-50 rounded text-xs overflow-auto max-h-96">
                  {JSON.stringify(data, null, 2).slice(0, 5000)}...
                </pre>
              </details>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

