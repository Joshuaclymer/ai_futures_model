'use client';

import { useState, useCallback, useEffect, useMemo } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import './ai-black-projects.css';

// Import section components
import {
  DetectionSection,
  CovertFabSection,
  DatacenterSection,
  InitialStockSection,
} from '@/components/black-project';
import { BlackProjectData } from '@/types/blackProject';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

// Color palette matching the original
const COLORS = {
  chip_stock: '#5E6FB8',
  fab: '#E9A842',
  datacenters_and_energy: '#4AA896',
  detection: '#7BA3C4',
  survival_rate: '#E05A4F',
  gray: '#7F8C8D',
} as const;

export const COLOR_PALETTE = {
  ...COLORS,
  rgba: (colorName: keyof typeof COLORS, alpha: number): string => {
    const hex = COLORS[colorName] || '#000000';
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
};

// Slider component for the new sidebar design
function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  formatValue,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  formatValue?: (v: number) => string;
}) {
  const displayValue = formatValue ? formatValue(value) : value.toString();
  return (
    <div className="bp-slider-group">
      <label>{label}</label>
      <input
        type="range"
        className="bp-slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <div className="bp-slider-value">{displayValue}</div>
    </div>
  );
}

// Collapsible section component - matching timelines page styling
function CollapsibleSection({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="mb-3 pt-1 rounded-none !m-0 !px-0 border-t border-b-0 border-l-0 border-r-0 border-gray-300" open={defaultOpen}>
      <summary className="text-[11px] font-bold cursor-pointer py-0.5">{title}</summary>
      <div className="mt-2">{children}</div>
    </details>
  );
}

interface BlackProjectClientProps {
  initialData: SimulationData | null;
  hideHeader?: boolean;
}

// Parameter types
interface Parameters {
  agreementYear: number;
  numYearsToSimulate: number;
  timeStepYears: number;
  numSimulations: number;
  proportionOfInitialChipStockToDivert: number;
  datacenterConstructionLabor: number;
  yearsBeforeAgreementYearPRCStartsBuildingBlackDatacenters: number;
  buildCovertFab: boolean;
  operatingLabor: number;
  constructionLabor: number;
  processNode: string;
  scannerProportion: number;
  researcherHeadcount: number;
  pProjectExists: number;
  meanDetectionTime100: number;
  meanDetectionTime1000: number;
  varianceDetectionTime: number;
}

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
      ccdf?: { x: number; y: number }[];
    };
    time_to_detection?: {
      median: number;
      ccdf?: { x: number; y: number }[];
    };
    ai_rd_reduction?: {
      median: number;
      ccdf?: { x: number; y: number }[];
    };
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
    capacity_ccdfs?: Record<string, { x: number; y: number }[]>;
  };
  initial_stock?: {
    lr_prc_accounting_samples?: number[];
  };
  [key: string]: unknown;
}

// Default parameters
const defaultParameters: Parameters = {
  agreementYear: 2027,
  numYearsToSimulate: 10,
  timeStepYears: 0.25,
  numSimulations: 100,
  proportionOfInitialChipStockToDivert: 0.1,
  datacenterConstructionLabor: 10000,
  yearsBeforeAgreementYearPRCStartsBuildingBlackDatacenters: 2,
  buildCovertFab: true,
  operatingLabor: 5000,
  constructionLabor: 5000,
  processNode: 'best_indigenous',
  scannerProportion: 0.5,
  researcherHeadcount: 1000,
  pProjectExists: 0.3,
  meanDetectionTime100: 10,
  meanDetectionTime1000: 5,
  varianceDetectionTime: 0.5,
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
            // These will be used to configure the black project
            proportion_of_initial_compute_stock_to_divert: parameters.proportionOfInitialChipStockToDivert,
            datacenter_construction_labor: parameters.datacenterConstructionLabor,
            years_before_agreement_year_prc_starts_building_black_datacenters: parameters.yearsBeforeAgreementYearPRCStartsBuildingBlackDatacenters,
            build_a_black_fab: parameters.buildCovertFab,
            black_fab_operating_labor: parameters.operatingLabor,
            black_fab_construction_labor: parameters.constructionLabor,
            black_fab_process_node: parameters.processNode,
            black_fab_proportion_of_prc_lithography_scanners_devoted: parameters.scannerProportion,
            researcher_headcount: parameters.researcherHeadcount,
            p_project_exists: parameters.pProjectExists,
            mean_detection_time_for_100_workers: parameters.meanDetectionTime100,
            mean_detection_time_for_1000_workers: parameters.meanDetectionTime1000,
            variance_of_detection_time_given_num_workers: parameters.varianceDetectionTime,
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

  // Dashboard values
  const dashboardValues = useMemo(() => {
    if (!data?.black_project_model) {
      return {
        medianH100Years: '--',
        medianTimeToDetection: '--',
        aiRdReduction: '--',
        chipsProduced: '--',
      };
    }

    const model = data.black_project_model;

    const formatNumber = (n: number) => {
      if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
      if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
      if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
      return n.toFixed(1);
    };

    return {
      medianH100Years: model.h100_years_before_detection?.median
        ? `${formatNumber(model.h100_years_before_detection.median)} H100-years`
        : '--',
      medianTimeToDetection: model.time_to_detection?.median
        ? `${model.time_to_detection.median.toFixed(1)} years`
        : '--',
      aiRdReduction: model.ai_rd_reduction?.median
        ? `${(model.ai_rd_reduction.median * 100).toFixed(0)}%`
        : '--',
      chipsProduced: model.chips_produced?.median
        ? formatNumber(model.chips_produced.median)
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
        <header className="fixed top-0 left-0 right-0 z-50 flex flex-row items-center justify-between px-6 py-0 border-b border-gray-200 bg-white">
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

      <div className={`flex flex-1 overflow-hidden ${hideHeader ? '' : 'mt-[64px]'}`}>
        {/* Sidebar */}
        <aside
          className={`bp-sidebar ${sidebarOpen ? 'fixed inset-y-0 left-0 z-50' : 'hidden lg:block'}`}
          style={hideHeader ? { top: 0, height: '100vh' } : undefined}
        >
          {/* Sidebar Header */}
          <div className="bp-sidebar-header">
            <div className="flex items-center justify-between">
              <h1>Parameters</h1>
              <button
                onClick={() => setSidebarOpen(false)}
                className="lg:hidden p-1 text-gray-500 hover:text-gray-700"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Sidebar Content */}
          <div className="bp-sidebar-content">
            {/* Key Parameters Section */}
            <div className="mt-4 mb-2">
              <div className="bp-section-header">
                <span>Key Parameters</span>
                <div className="bp-section-line" />
              </div>
              <Slider
                label="Agreement Start Year"
                value={parameters.agreementYear}
                onChange={(v) => updateParameter('agreementYear', v)}
                min={2026}
                max={2035}
                step={1}
              />
              <Slider
                label="Number of Simulations"
                value={parameters.numSimulations}
                onChange={(v) => updateParameter('numSimulations', v)}
                min={10}
                max={1000}
                step={10}
              />
              <Slider
                label="Proportion of Chips to Divert"
                value={parameters.proportionOfInitialChipStockToDivert}
                onChange={(v) => updateParameter('proportionOfInitialChipStockToDivert', v)}
                min={0}
                max={0.5}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Slider
                label="Prior P(Project Exists)"
                value={parameters.pProjectExists}
                onChange={(v) => updateParameter('pProjectExists', v)}
                min={0}
                max={1}
                step={0.01}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
            </div>

            {/* Simulation Settings */}
            <CollapsibleSection title="Simulation Settings">
              <Slider
                label="Years to Simulate"
                value={parameters.numYearsToSimulate}
                onChange={(v) => updateParameter('numYearsToSimulate', v)}
                min={1}
                max={20}
                step={0.5}
                formatValue={(v) => `${v} years`}
              />
              <Slider
                label="Time Step"
                value={parameters.timeStepYears}
                onChange={(v) => updateParameter('timeStepYears', v)}
                min={0.1}
                max={1}
                step={0.05}
                formatValue={(v) => `${v} years`}
              />
            </CollapsibleSection>

            {/* Covert Datacenters */}
            <CollapsibleSection title="Covert Datacenters">
              <Slider
                label="Datacenter Construction Labor"
                value={parameters.datacenterConstructionLabor}
                onChange={(v) => updateParameter('datacenterConstructionLabor', v)}
                min={1000}
                max={50000}
                step={1000}
                formatValue={(v) => v.toLocaleString()}
              />
              <Slider
                label="Years Before Agreement to Start Building"
                value={parameters.yearsBeforeAgreementYearPRCStartsBuildingBlackDatacenters}
                onChange={(v) => updateParameter('yearsBeforeAgreementYearPRCStartsBuildingBlackDatacenters', v)}
                min={0}
                max={5}
                step={1}
                formatValue={(v) => `${v} years`}
              />
            </CollapsibleSection>

            {/* Covert Fab */}
            <CollapsibleSection title="Covert Fab">
              <div className="bp-checkbox-group">
                <input
                  type="checkbox"
                  id="buildCovertFab"
                  checked={parameters.buildCovertFab}
                  onChange={(e) => updateParameter('buildCovertFab', e.target.checked)}
                />
                <label htmlFor="buildCovertFab">Build covert fab</label>
              </div>
              <Slider
                label="Operating Labor"
                value={parameters.operatingLabor}
                onChange={(v) => updateParameter('operatingLabor', v)}
                min={1000}
                max={20000}
                step={500}
                formatValue={(v) => v.toLocaleString()}
              />
              <Slider
                label="Construction Labor"
                value={parameters.constructionLabor}
                onChange={(v) => updateParameter('constructionLabor', v)}
                min={1000}
                max={20000}
                step={500}
                formatValue={(v) => v.toLocaleString()}
              />
              <div className="space-y-2 mb-2">
                <label className="block text-xs font-medium text-foreground">
                  Process Node Strategy
                </label>
                <select
                  value={parameters.processNode}
                  onChange={(e) => updateParameter('processNode', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                >
                  <option value="best_indigenous">Best Indigenous</option>
                  <option value="best_indigenous_gte_28nm">Best Indigenous &gt;= 28nm</option>
                  <option value="best_indigenous_gte_14nm">Best Indigenous &gt;= 14nm</option>
                  <option value="best_indigenous_gte_7nm">Best Indigenous &gt;= 7nm</option>
                  <option value="nm130">130nm</option>
                  <option value="nm28">28nm</option>
                  <option value="nm14">14nm</option>
                  <option value="nm7">7nm</option>
                </select>
              </div>
              <Slider
                label="Scanner Proportion"
                value={parameters.scannerProportion}
                onChange={(v) => updateParameter('scannerProportion', v)}
                min={0}
                max={1}
                step={0.1}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
              />
            </CollapsibleSection>

            {/* AI Research */}
            <CollapsibleSection title="AI Research">
              <Slider
                label="Researcher Headcount"
                value={parameters.researcherHeadcount}
                onChange={(v) => updateParameter('researcherHeadcount', v)}
                min={0}
                max={10000}
                step={100}
                formatValue={(v) => v.toLocaleString()}
              />
            </CollapsibleSection>

            {/* Detection Parameters */}
            <CollapsibleSection title="Detection Parameters">
              <Slider
                label="Mean Detection Time (100 workers)"
                value={parameters.meanDetectionTime100}
                onChange={(v) => updateParameter('meanDetectionTime100', v)}
                min={1}
                max={20}
                step={0.5}
                formatValue={(v) => `${v} years`}
              />
              <Slider
                label="Mean Detection Time (1000 workers)"
                value={parameters.meanDetectionTime1000}
                onChange={(v) => updateParameter('meanDetectionTime1000', v)}
                min={1}
                max={20}
                step={0.5}
                formatValue={(v) => `${v} years`}
              />
              <Slider
                label="Detection Time Variance"
                value={parameters.varianceDetectionTime}
                onChange={(v) => updateParameter('varianceDetectionTime', v)}
                min={0.1}
                max={2}
                step={0.1}
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

            {/* Dashboard and top plots */}
            <div className="flex flex-wrap gap-5">
              {/* Dashboard */}
              <div className="bp-dashboard w-60 flex-shrink-0">
                <div className="bp-plot-title">Median outcome</div>
                <div className="space-y-4 mt-4">
                  <div className="bp-dashboard-item">
                    <div className="bp-dashboard-value">{dashboardValues.medianH100Years}</div>
                    <div className="bp-dashboard-label">Covert computation*</div>
                  </div>
                  <div className="bp-dashboard-item">
                    <div className="bp-dashboard-value">{dashboardValues.medianTimeToDetection}</div>
                    <div className="bp-dashboard-label">Time to detection*</div>
                  </div>
                  <div className="bp-dashboard-item">
                    <div className="bp-dashboard-value">{dashboardValues.aiRdReduction}</div>
                    <div className="bp-dashboard-label">Reduction in AI R&D computation of largest company*</div>
                  </div>
                  <div className="bp-dashboard-item">
                    <div className="bp-dashboard-value">{dashboardValues.chipsProduced}</div>
                    <div className="bp-dashboard-label">Chips covertly produced*</div>
                  </div>
                </div>
              </div>

              {/* Covert Compute CCDF */}
              <div className="bp-plot-container flex-1 min-w-[300px]">
                <div className="bp-plot-title">Covert compute</div>
                <CCDFPlot
                  data={data?.black_project_model?.h100_years_before_detection?.ccdf}
                  color={COLOR_PALETTE.chip_stock}
                  xLabel="H100-years"
                  yLabel="P(X > x)"
                  isLoading={isLoading}
                />
              </div>

              {/* Time to Detection CCDF */}
              <div className="bp-plot-container flex-1 min-w-[300px]">
                <div className="bp-plot-title">Time to detection (after which agreement ends)</div>
                <CCDFPlot
                  data={data?.black_project_model?.time_to_detection?.ccdf}
                  color={COLOR_PALETTE.detection}
                  xLabel="Years"
                  yLabel="P(X > x)"
                  isLoading={isLoading}
                />
              </div>
            </div>

            {/* Second row of plots */}
            <div className="flex flex-wrap gap-5">
              {/* AI R&D Reduction CCDF */}
              <div className="bp-plot-container flex-1 min-w-[300px]">
                <div className="bp-plot-title">Covert AI R&D computation relative to no slowdown*</div>
                <CCDFPlot
                  data={data?.black_project_model?.ai_rd_reduction?.ccdf}
                  color={COLOR_PALETTE.fab}
                  xLabel="Reduction"
                  yLabel="P(X > x)"
                  xAsPercent
                  isLoading={isLoading}
                />
              </div>

              {/* Dark Compute Over Time */}
              <div className="bp-plot-container flex-1 min-w-[300px]">
                <div className="bp-plot-title">Total dark compute over time</div>
                <TimeSeriesPlot
                  years={data?.black_project_model?.years}
                  median={data?.black_project_model?.total_dark_compute?.median}
                  p25={data?.black_project_model?.total_dark_compute?.p25}
                  p75={data?.black_project_model?.total_dark_compute?.p75}
                  color={COLOR_PALETTE.chip_stock}
                  yLabel="H100-equivalents"
                  isLoading={isLoading}
                />
              </div>
            </div>

            <p className="text-xs text-gray-500 italic mt-4">
              *Unless otherwise specified, US intelligence &apos;detects&apos; a covert project after it receives &gt;4x update that the project exists, after which, USG exits the AI slowdown agreement.
            </p>

            {/* Divider */}
            <hr className="my-8 border-gray-200" />

            {/* Detection Section */}
            <DetectionSection data={data as unknown as BlackProjectData} isLoading={isLoading} />

            {/* Divider */}
            <hr className="my-8 border-gray-200" />

            {/* Covert Fab Section */}
            <CovertFabSection data={data as unknown as BlackProjectData} isLoading={isLoading} />

            {/* Divider */}
            <hr className="my-8 border-gray-200" />

            {/* Datacenter Section */}
            <DatacenterSection data={data as unknown as BlackProjectData} isLoading={isLoading} />

            {/* Divider */}
            <hr className="my-8 border-gray-200" />

            {/* Initial Stock Section */}
            <InitialStockSection
              data={data as unknown as BlackProjectData}
              isLoading={isLoading}
              diversionProportion={parameters.proportionOfInitialChipStockToDivert}
            />

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

// CCDF Plot component
interface CCDFPlotProps {
  data?: { x: number; y: number }[];
  color: string;
  xLabel: string;
  yLabel: string;
  xAsPercent?: boolean;
  isLoading?: boolean;
}

function CCDFPlot({ data, color, xLabel, yLabel, xAsPercent, isLoading }: CCDFPlotProps) {
  if (isLoading) {
    return (
      <div className="bp-plot flex items-center justify-center">
        <div className="flex flex-col items-center gap-2">
          <div className="w-8 h-8 border-3 border-gray-200 border-t-blue-500 rounded-full animate-spin" />
          <span className="text-xs text-gray-400">Loading...</span>
        </div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="bp-plot flex items-center justify-center text-gray-400 text-sm">
        No data available
      </div>
    );
  }

  return (
    <div className="bp-plot">
      <Plot
        data={[
          {
            x: data.map(d => xAsPercent ? d.x * 100 : d.x),
            y: data.map(d => d.y),
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color, width: 2 },
            fill: 'tozeroy' as const,
            fillcolor: COLOR_PALETTE.rgba('chip_stock', 0.1),
          }
        ]}
        layout={{
          margin: { l: 50, r: 20, t: 10, b: 50 },
          xaxis: {
            title: { text: xLabel, font: { size: 11 } },
            tickfont: { size: 10 },
            ticksuffix: xAsPercent ? '%' : '',
          },
          yaxis: {
            title: { text: yLabel, font: { size: 11 } },
            tickfont: { size: 10 },
            range: [0, 1],
          },
          showlegend: false,
          hovermode: 'closest' as const,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}

// Time Series Plot component
interface TimeSeriesPlotProps {
  years?: number[];
  median?: number[];
  p25?: number[];
  p75?: number[];
  color: string;
  yLabel: string;
  isLoading?: boolean;
}

function TimeSeriesPlot({ years, median, p25, p75, color, yLabel, isLoading }: TimeSeriesPlotProps) {
  if (isLoading) {
    return (
      <div className="bp-plot flex items-center justify-center">
        <div className="flex flex-col items-center gap-2">
          <div className="w-8 h-8 border-3 border-gray-200 border-t-blue-500 rounded-full animate-spin" />
          <span className="text-xs text-gray-400">Loading...</span>
        </div>
      </div>
    );
  }

  if (!years || !median || years.length === 0) {
    return (
      <div className="bp-plot flex items-center justify-center text-gray-400 text-sm">
        No data available
      </div>
    );
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const traces: any[] = [];

  // Percentile band
  if (p25 && p75) {
    traces.push({
      x: [...years, ...years.slice().reverse()],
      y: [...p75, ...p25.slice().reverse()],
      type: 'scatter' as const,
      mode: 'lines' as const,
      fill: 'toself' as const,
      fillcolor: COLOR_PALETTE.rgba('chip_stock', 0.2),
      line: { color: 'transparent' },
      showlegend: false,
      hoverinfo: 'skip' as const,
    });
  }

  // Median line
  traces.push({
    x: years,
    y: median,
    type: 'scatter' as const,
    mode: 'lines' as const,
    line: { color, width: 2 },
    name: 'Median',
  });

  return (
    <div className="bp-plot">
      <Plot
        data={traces}
        layout={{
          margin: { l: 60, r: 20, t: 10, b: 50 },
          xaxis: {
            title: { text: 'Year', font: { size: 11 } },
            tickfont: { size: 10 },
          },
          yaxis: {
            title: { text: yLabel, font: { size: 11 } },
            tickfont: { size: 10 },
          },
          showlegend: false,
          hovermode: 'x unified' as const,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
