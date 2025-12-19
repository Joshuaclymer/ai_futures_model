'use client';

import { useState, useCallback, useEffect, useMemo } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import './black-project.css';

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

interface BlackProjectClientProps {
  initialData: SimulationData | null;
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

export function BlackProjectClient({ initialData }: BlackProjectClientProps) {
  const [data, setData] = useState<SimulationData | null>(initialData);
  const [isLoading, setIsLoading] = useState(!initialData);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>(
    initialData ? `Loaded ${initialData.num_simulations || 1000} simulations` : 'Loading...'
  );
  const [parameters, setParameters] = useState<Parameters>(defaultParameters);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Load default data on mount
  useEffect(() => {
    if (initialData) return;

    const loadDefaultData = async () => {
      try {
        setStatus('Loading cached results...');
        const response = await fetch('http://127.0.0.1:5001/get_default_results');
        if (response.ok) {
          const result = await response.json();
          setData(result);
          setStatus(`Loaded ${result.num_simulations || 1000} simulations`);
        } else {
          setStatus('No cache available - click Run Simulation');
        }
      } catch (err) {
        console.error('Failed to load default data:', err);
        setStatus('Backend unavailable - click Run Simulation');
      } finally {
        setIsLoading(false);
      }
    };

    loadDefaultData();
  }, [initialData]);

  const runSimulation = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setStatus('Running simulation...');

    try {
      const response = await fetch('http://127.0.0.1:5001/run_simulation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          'simulation_settings.agreement_start_year': parameters.agreementYear,
          'simulation_settings.num_years_to_simulate': parameters.numYearsToSimulate,
          'simulation_settings.time_step_years': parameters.timeStepYears,
          'simulation_settings.num_simulations': parameters.numSimulations,
          'black_project_properties.proportion_of_initial_compute_stock_to_divert': parameters.proportionOfInitialChipStockToDivert,
          'black_project_properties.datacenter_construction_labor': parameters.datacenterConstructionLabor,
          'black_project_properties.years_before_agreement_year_prc_starts_building_black_datacenters': parameters.yearsBeforeAgreementYearPRCStartsBuildingBlackDatacenters,
          'black_project_properties.build_a_black_fab': parameters.buildCovertFab,
          'black_project_properties.black_fab_operating_labor': parameters.operatingLabor,
          'black_project_properties.black_fab_construction_labor': parameters.constructionLabor,
          'black_project_properties.black_fab_process_node': parameters.processNode,
          'black_project_properties.black_fab_proportion_of_prc_lithography_scanners_devoted': parameters.scannerProportion,
          'black_project_properties.researcher_headcount': parameters.researcherHeadcount,
          'black_project_parameters.p_project_exists': parameters.pProjectExists,
          'black_project_parameters.detection_parameters.mean_detection_time_for_100_workers': parameters.meanDetectionTime100,
          'black_project_parameters.detection_parameters.mean_detection_time_for_1000_workers': parameters.meanDetectionTime1000,
          'black_project_parameters.detection_parameters.variance_of_detection_time_given_num_workers': parameters.varianceDetectionTime,
        }),
      });

      if (!response.ok) {
        throw new Error(`Simulation failed: ${response.statusText}`);
      }

      const result = await response.json();
      setData(result);
      setStatus(`Ran ${result.num_simulations || parameters.numSimulations} simulations`);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      setStatus(`Error: ${message}`);
    } finally {
      setIsLoading(false);
    }
  }, [parameters]);

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
      {/* Header */}
      <header className="flex flex-row items-center justify-between px-6 py-4 border-b border-gray-200 bg-white">
        <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
          <h1 className="text-lg sm:text-2xl font-bold font-et-book">
            <Link href="/">Covert Compute Production Model</Link>
          </h1>
          <nav className="flex flex-row gap-3 sm:gap-6 text-sm">
            <Link href="/" className="font-system-mono text-xs whitespace-nowrap text-gray-600 hover:text-gray-900">
              Home
            </Link>
            <Link href="/forecast" className="font-system-mono text-xs whitespace-nowrap text-gray-600 hover:text-gray-900">
              Forecast
            </Link>
          </nav>
        </div>
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="lg:hidden p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className={`bp-sidebar ${sidebarOpen ? 'fixed inset-y-0 left-0 z-50' : 'hidden lg:block'}`}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold">Parameters</h2>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-1 text-gray-500 hover:text-gray-700"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <button
            onClick={runSimulation}
            disabled={isLoading}
            className="w-full py-2 px-4 bg-[#5E6FB8] text-white rounded hover:bg-[#4B5A93] disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
          >
            {isLoading ? 'Running...' : 'Run Simulation'}
          </button>

          <div className={`bp-status ${error ? 'error' : 'success'} mt-2`}>
            {status}
          </div>

          <h2>Simulation Settings</h2>
          <div className="bp-param-group">
            <label>Agreement start year</label>
            <input
              type="number"
              value={parameters.agreementYear}
              onChange={(e) => updateParameter('agreementYear', parseInt(e.target.value))}
              min={2026}
              max={2035}
            />
          </div>
          <div className="bp-param-group">
            <label>Number of years to simulate</label>
            <input
              type="number"
              value={parameters.numYearsToSimulate}
              onChange={(e) => updateParameter('numYearsToSimulate', parseFloat(e.target.value))}
              min={1}
              step={0.5}
            />
          </div>
          <div className="bp-param-group">
            <label>Time increment (years)</label>
            <input
              type="number"
              value={parameters.timeStepYears}
              onChange={(e) => updateParameter('timeStepYears', parseFloat(e.target.value))}
              step={0.01}
            />
          </div>
          <div className="bp-param-group">
            <label>Number of simulations</label>
            <input
              type="number"
              value={parameters.numSimulations}
              onChange={(e) => updateParameter('numSimulations', parseInt(e.target.value))}
              min={1}
              max={10000}
            />
          </div>

          <h2>Covert Project Properties</h2>
          <div className="bp-param-group">
            <label>Proportion of initial chip stock to divert</label>
            <input
              type="number"
              value={parameters.proportionOfInitialChipStockToDivert}
              onChange={(e) => updateParameter('proportionOfInitialChipStockToDivert', parseFloat(e.target.value))}
              step={0.01}
              min={0}
              max={1}
            />
          </div>
          <div className="bp-param-group">
            <label>Datacenter construction labor</label>
            <input
              type="number"
              value={parameters.datacenterConstructionLabor}
              onChange={(e) => updateParameter('datacenterConstructionLabor', parseInt(e.target.value))}
            />
          </div>
          <div className="bp-param-group">
            <label>Years before agreement PRC starts building covert datacenters</label>
            <input
              type="number"
              value={parameters.yearsBeforeAgreementYearPRCStartsBuildingBlackDatacenters}
              onChange={(e) => updateParameter('yearsBeforeAgreementYearPRCStartsBuildingBlackDatacenters', parseInt(e.target.value))}
            />
          </div>
          <div className="bp-param-group">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={parameters.buildCovertFab}
                onChange={(e) => updateParameter('buildCovertFab', e.target.checked)}
                className="w-auto"
              />
              Build covert fab
            </label>
          </div>
          <div className="bp-param-group">
            <label>Number of workers to operate fab</label>
            <input
              type="number"
              value={parameters.operatingLabor}
              onChange={(e) => updateParameter('operatingLabor', parseInt(e.target.value))}
            />
          </div>
          <div className="bp-param-group">
            <label>Number of workers to construct fab</label>
            <input
              type="number"
              value={parameters.constructionLabor}
              onChange={(e) => updateParameter('constructionLabor', parseInt(e.target.value))}
            />
          </div>
          <div className="bp-param-group">
            <label>Process node strategy</label>
            <select
              value={parameters.processNode}
              onChange={(e) => updateParameter('processNode', e.target.value)}
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
          <div className="bp-param-group">
            <label>Number of AI researchers</label>
            <input
              type="number"
              value={parameters.researcherHeadcount}
              onChange={(e) => updateParameter('researcherHeadcount', parseInt(e.target.value))}
              min={0}
            />
          </div>

          <h2>Detection Parameters</h2>
          <div className="bp-param-group">
            <label>US prior probability that covert project exists</label>
            <input
              type="number"
              value={parameters.pProjectExists}
              onChange={(e) => updateParameter('pProjectExists', parseFloat(e.target.value))}
              step={0.01}
              min={0}
              max={1}
            />
          </div>
          <div className="bp-param-group">
            <label>Expected time to detect 100-person project (years)</label>
            <input
              type="number"
              value={parameters.meanDetectionTime100}
              onChange={(e) => updateParameter('meanDetectionTime100', parseFloat(e.target.value))}
              step={0.1}
            />
          </div>
          <div className="bp-param-group">
            <label>Expected time to detect 1000-person project (years)</label>
            <input
              type="number"
              value={parameters.meanDetectionTime1000}
              onChange={(e) => updateParameter('meanDetectionTime1000', parseFloat(e.target.value))}
              step={0.1}
            />
          </div>
          <div className="bp-param-group">
            <label>Detection time variance</label>
            <input
              type="number"
              value={parameters.varianceDetectionTime}
              onChange={(e) => updateParameter('varianceDetectionTime', parseFloat(e.target.value))}
              step={0.01}
            />
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
        <main className="flex-1 overflow-y-auto p-6">
          {isLoading && (
            <div className="flex items-center justify-center py-20">
              <div className="flex flex-col items-center gap-4">
                <div className="bp-loading-spinner" />
                <span className="text-sm text-gray-500">Running simulation...</span>
              </div>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">
              {error}
            </div>
          )}

          {!isLoading && data && (
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
                    data={data.black_project_model?.h100_years_before_detection?.ccdf}
                    color={COLOR_PALETTE.chip_stock}
                    xLabel="H100-years"
                    yLabel="P(X > x)"
                  />
                </div>

                {/* Time to Detection CCDF */}
                <div className="bp-plot-container flex-1 min-w-[300px]">
                  <div className="bp-plot-title">Time to detection (after which agreement ends)</div>
                  <CCDFPlot
                    data={data.black_project_model?.time_to_detection?.ccdf}
                    color={COLOR_PALETTE.detection}
                    xLabel="Years"
                    yLabel="P(X > x)"
                  />
                </div>
              </div>

              {/* Second row of plots */}
              <div className="flex flex-wrap gap-5">
                {/* AI R&D Reduction CCDF */}
                <div className="bp-plot-container flex-1 min-w-[300px]">
                  <div className="bp-plot-title">Covert AI R&D computation relative to no slowdown*</div>
                  <CCDFPlot
                    data={data.black_project_model?.ai_rd_reduction?.ccdf}
                    color={COLOR_PALETTE.fab}
                    xLabel="Reduction"
                    yLabel="P(X > x)"
                    xAsPercent
                  />
                </div>

                {/* Dark Compute Over Time */}
                {data.black_project_model?.years && data.black_project_model?.total_dark_compute && (
                  <div className="bp-plot-container flex-1 min-w-[300px]">
                    <div className="bp-plot-title">Total dark compute over time</div>
                    <TimeSeriesPlot
                      years={data.black_project_model.years}
                      median={data.black_project_model.total_dark_compute.median}
                      p25={data.black_project_model.total_dark_compute.p25}
                      p75={data.black_project_model.total_dark_compute.p75}
                      color={COLOR_PALETTE.chip_stock}
                      yLabel="H100-equivalents"
                    />
                  </div>
                )}
              </div>

              <p className="text-xs text-gray-500 italic mt-4">
                *Unless otherwise specified, US intelligence &apos;detects&apos; a covert project after it receives &gt;4x update that the project exists, after which, USG exits the AI slowdown agreement.
              </p>

              {/* Debug data display */}
              <details className="mt-8">
                <summary className="cursor-pointer text-sm text-gray-500">View raw data</summary>
                <pre className="mt-2 p-4 bg-gray-50 rounded text-xs overflow-auto max-h-96">
                  {JSON.stringify(data, null, 2).slice(0, 5000)}...
                </pre>
              </details>
            </div>
          )}

          {!isLoading && !data && !error && (
            <div className="text-center py-20">
              <p className="text-gray-500 mb-4">No simulation data available.</p>
              <button
                onClick={runSimulation}
                className="py-2 px-4 bg-[#5E6FB8] text-white rounded hover:bg-[#4B5A93]"
              >
                Run Simulation
              </button>
            </div>
          )}
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
}

function CCDFPlot({ data, color, xLabel, yLabel, xAsPercent }: CCDFPlotProps) {
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
  years: number[];
  median: number[];
  p25?: number[];
  p75?: number[];
  color: string;
  yLabel: string;
}

function TimeSeriesPlot({ years, median, p25, p75, color, yLabel }: TimeSeriesPlotProps) {
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
