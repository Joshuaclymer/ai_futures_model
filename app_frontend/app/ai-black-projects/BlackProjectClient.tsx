'use client';

import { useState, useMemo } from 'react';
import './ai-black-projects.css';

// Import components
import {
  Header,
  HEADER_HEIGHT,
  ParameterSidebar,
  TopChartsSection,
  CovertFabSection,
  DatacenterSection,
  InitialStockSection,
  HowWeEstimateSection,
} from './components';
import { RateOfComputationData, DetectionLikelihoodData, CovertFabApiData } from './components/sections';

// Import hooks and types
import { useSimulation } from './hooks/useSimulation';
import { SimulationData, Parameters } from './types';

// Re-export colors for backwards compatibility
import { COLOR_PALETTE as IMPORTED_COLORS } from './components/colors';
export const COLOR_PALETTE = IMPORTED_COLORS;

// Helper to map existing backend data to RateOfComputationData format
function mapToRateOfComputationData(data: SimulationData | null): RateOfComputationData | null {
  if (!data) return null;

  const years = data.black_project_model?.years || [];
  const bpm = data.black_project_model;
  const initialStock = data.initial_stock;
  const blackDatacenters = data.black_datacenters;

  return {
    years,
    // Initial chip stock samples from initial_stock
    initial_chip_stock_samples: initialStock?.initial_compute_stock_samples || [],
    // Acquired hardware = fab production (all sims)
    acquired_hardware: {
      years,
      median: bpm?.black_fab_flow_all_sims?.median || [],
      p25: bpm?.black_fab_flow_all_sims?.p25 || [],
      p75: bpm?.black_fab_flow_all_sims?.p75 || [],
    },
    // Surviving fraction
    surviving_fraction: {
      years,
      median: bpm?.survival_rate?.median || [],
      p25: bpm?.survival_rate?.p25 || [],
      p75: bpm?.survival_rate?.p75 || [],
    },
    // Covert chip stock (total surviving compute)
    covert_chip_stock: {
      years,
      median: bpm?.covert_chip_stock?.median || [],
      p25: bpm?.covert_chip_stock?.p25 || [],
      p75: bpm?.covert_chip_stock?.p75 || [],
    },
    // Datacenter capacity
    datacenter_capacity: {
      years,
      median: bpm?.datacenter_capacity?.median || blackDatacenters?.datacenter_capacity?.median || [],
      p25: bpm?.datacenter_capacity?.p25 || blackDatacenters?.datacenter_capacity?.p25 || [],
      p75: bpm?.datacenter_capacity?.p75 || blackDatacenters?.datacenter_capacity?.p75 || [],
    },
    // Energy usage - use datacenter capacity as proxy since total energy isn't directly available
    energy_usage: {
      years,
      median: bpm?.datacenter_capacity?.median || [],
      p25: bpm?.datacenter_capacity?.p25 || [],
      p75: bpm?.datacenter_capacity?.p75 || [],
    },
    // Energy stacked data
    energy_stacked_data: blackDatacenters?.energy_by_source || bpm?.black_project_energy || [],
    energy_source_labels: (blackDatacenters?.source_labels || bpm?.energy_source_labels || ['Initial Stock', 'Fab-Produced']) as [string, string],
    // Operating chips (in H100e, need to multiply by 1000 since black_datacenters uses K H100e)
    operating_chips: {
      years,
      median: (blackDatacenters?.operational_compute?.median || []).map((v: number) => v * 1000),
      p25: (blackDatacenters?.operational_compute?.p25 || []).map((v: number) => v * 1000),
      p75: (blackDatacenters?.operational_compute?.p75 || []).map((v: number) => v * 1000),
    },
    // Covert computation (cumulative H100-years)
    covert_computation: {
      years,
      median: bpm?.h100_years?.median || [],
      p25: bpm?.h100_years?.p25 || [],
      p75: bpm?.h100_years?.p75 || [],
    },
  };
}

// Helper to map existing backend data to DetectionLikelihoodData format
function mapToDetectionLikelihoodData(data: SimulationData | null): DetectionLikelihoodData | null {
  if (!data) return null;

  const years = data.black_project_model?.years || [];
  const bpm = data.black_project_model;
  const initialStock = data.initial_stock;

  return {
    years,
    // LR samples from initial_stock
    chip_evidence_samples: initialStock?.lr_prc_accounting_samples || [],
    sme_evidence_samples: initialStock?.lr_sme_inventory_samples || [],
    dc_evidence_samples: initialStock?.lr_satellite_datacenter_samples || [],
    // Energy evidence (time series)
    energy_evidence: {
      years,
      median: bpm?.lr_reported_energy?.median || [],
      p25: bpm?.lr_reported_energy?.p25 || [],
      p75: bpm?.lr_reported_energy?.p75 || [],
    },
    // Combined evidence from resource accounting
    combined_evidence: {
      years,
      median: bpm?.lr_combined_reported_assets?.median || [],
      p25: bpm?.lr_combined_reported_assets?.p25 || [],
      p75: bpm?.lr_combined_reported_assets?.p75 || [],
    },
    // Direct evidence (from other intel sources like HUMINT, SIGINT, etc.)
    direct_evidence: {
      years,
      median: bpm?.lr_other_intel?.median || [],
      p25: bpm?.lr_other_intel?.p25 || [],
      p75: bpm?.lr_other_intel?.p75 || [],
    },
    // Posterior probability
    posterior_prob: {
      years,
      median: bpm?.posterior_prob_project?.median || [],
      p25: bpm?.posterior_prob_project?.p25 || [],
      p75: bpm?.posterior_prob_project?.p75 || [],
    },
  };
}

// Helper to map existing backend data to CovertFabApiData format
function mapToCovertFabData(data: SimulationData | null): CovertFabApiData | null {
  if (!data) return null;

  const blackFab = data.black_fab;
  const bpm = data.black_project_model;
  if (!blackFab) return null;

  const years = blackFab.years || [];

  // Extract wafer starts samples from individual simulations (first value from each sim)
  const waferStartsSamples: number[] = [];
  if (blackFab.wafer_starts?.individual) {
    for (const simData of blackFab.wafer_starts.individual) {
      if (Array.isArray(simData) && simData.length > 0) {
        // Get the max value from each simulation (represents operational capacity)
        const maxVal = Math.max(...simData.filter((v: number) => v > 0));
        if (maxVal > 0) waferStartsSamples.push(maxVal);
      }
    }
  }

  // Build transistor density data with process nodes and watts per TPP
  const transistorDensityData: { node: string; density: number; probability?: number; wattsPerTpp?: number }[] = [];
  const processNodes = blackFab.individual_process_node || blackFab.process_node_by_sim || [];
  const nodeCounts: Record<string, { count: number; density: number; wattsPerTpp: number }> = {};

  if (blackFab.transistor_density?.individual && processNodes.length > 0) {
    for (let i = 0; i < processNodes.length; i++) {
      const node = processNodes[i];
      const densities = blackFab.transistor_density.individual[i];
      const wattsPerTppValues = blackFab.watts_per_tpp?.individual?.[i];
      if (densities && Array.isArray(densities)) {
        // Get final density and watts per TPP for this simulation
        const finalDensity = densities[densities.length - 1] || densities.reduce((a: number, b: number) => a + b, 0) / densities.length;
        const finalWattsPerTpp = wattsPerTppValues && Array.isArray(wattsPerTppValues)
          ? wattsPerTppValues[wattsPerTppValues.length - 1]
          : undefined;
        if (!nodeCounts[node]) {
          nodeCounts[node] = { count: 0, density: finalDensity, wattsPerTpp: finalWattsPerTpp || 1 };
        }
        nodeCounts[node].count++;
      }
    }
    // Convert to array with probabilities
    const totalSims = Object.values(nodeCounts).reduce((sum, v) => sum + v.count, 0);
    for (const [node, { count, density, wattsPerTpp }] of Object.entries(nodeCounts)) {
      transistorDensityData.push({
        node,
        density,
        probability: totalSims > 0 ? count / totalSims : 0,
        wattsPerTpp,
      });
    }
    // Sort by process node size descending (28nm before 14nm before 7nm)
    transistorDensityData.sort((a, b) => {
      const getNodeSize = (node: string) => {
        const match = node.match(/(\d+)/);
        return match ? parseInt(match[1], 10) : 0;
      };
      return getNodeSize(b.node) - getNodeSize(a.node);
    });
  }

  // H100 power consumption in watts (standard value)
  const H100_POWER_WATTS = 700;

  // Use black_fab_monthly_flow_all_sims for compute per month (monthly production rate, not cumulative)
  const computePerMonthMedian = bpm?.black_fab_monthly_flow_all_sims?.median || [];
  const computePerMonthP25 = bpm?.black_fab_monthly_flow_all_sims?.p25 || [];
  const computePerMonthP75 = bpm?.black_fab_monthly_flow_all_sims?.p75 || [];

  // Compute energy per month: compute_per_month * watts_per_tpp * H100_power / 1e9 (to GW)
  // This gives the energy required to run the chips produced each month
  const wattsPerTppMedian = blackFab.watts_per_tpp?.median || [];

  // Energy = compute (H100e) * watts_per_tpp_relative * H100_power / 1e9 (GW)
  const energyPerMonthMedian = computePerMonthMedian.map((compute: number, i: number) => {
    const wattsPerTpp = wattsPerTppMedian[i] || 1;
    return compute * wattsPerTpp * H100_POWER_WATTS / 1e9;
  });
  const energyPerMonthP25 = computePerMonthP25.map((compute: number, i: number) => {
    const wattsPerTpp = wattsPerTppMedian[i] || 1;
    return compute * wattsPerTpp * H100_POWER_WATTS / 1e9;
  });
  const energyPerMonthP75 = computePerMonthP75.map((compute: number, i: number) => {
    const wattsPerTpp = wattsPerTppMedian[i] || 1;
    return compute * wattsPerTpp * H100_POWER_WATTS / 1e9;
  });

  return {
    dashboard: blackFab.dashboard,
    compute_ccdf: blackFab.compute_ccdfs,
    time_series_data: {
      years,
      lr_combined: {
        years,
        median: blackFab.lr_combined?.median || [],
        p25: blackFab.lr_combined?.p25 || [],
        p75: blackFab.lr_combined?.p75 || [],
      },
      h100e_flow: {
        years,
        median: bpm?.black_fab_flow?.median || [],
        p25: bpm?.black_fab_flow?.p25 || [],
        p75: bpm?.black_fab_flow?.p75 || [],
      },
    },
    is_operational: {
      years,
      median: blackFab.is_operational?.proportion || [],
      p25: blackFab.is_operational?.proportion || [],
      p75: blackFab.is_operational?.proportion || [],
    },
    wafer_starts_samples: waferStartsSamples,
    chips_per_wafer: blackFab.chips_per_wafer?.median?.[0],
    architecture_efficiency: blackFab.architecture_efficiency_at_agreement,
    h100_power: H100_POWER_WATTS,
    transistor_density: transistorDensityData.length > 0 ? transistorDensityData : undefined,
    // Individual simulation points for scatter plot (density, wattsPerTpp pairs)
    individual_sim_points: (() => {
      const points: { density: number; wattsPerTpp: number }[] = [];
      const tdIndividual = blackFab.transistor_density?.individual || [];
      const wptIndividual = blackFab.watts_per_tpp?.individual || [];
      for (let i = 0; i < Math.min(tdIndividual.length, wptIndividual.length); i++) {
        const densities = tdIndividual[i];
        const wattsPerTpps = wptIndividual[i];
        if (Array.isArray(densities) && Array.isArray(wattsPerTpps) && densities.length > 0 && wattsPerTpps.length > 0) {
          points.push({
            density: densities[densities.length - 1],
            wattsPerTpp: wattsPerTpps[wattsPerTpps.length - 1],
          });
        }
      }
      return points;
    })(),
    compute_per_month: {
      years,
      median: computePerMonthMedian,
      p25: computePerMonthP25,
      p75: computePerMonthP75,
    },
    watts_per_tpp_curve: blackFab.watts_per_tpp_curve ? {
      densityRelative: blackFab.watts_per_tpp_curve.density_relative || [],
      wattsPerTppRelative: blackFab.watts_per_tpp_curve.watts_per_tpp_relative || [],
    } : undefined,
    energy_per_month: {
      years,
      median: energyPerMonthMedian,
      p25: energyPerMonthP25,
      p75: energyPerMonthP75,
    },
  };
}

interface BlackProjectClientProps {
  initialData: SimulationData | null;
  hideHeader?: boolean;
  externalSidebarOpen?: boolean;
  onExternalSidebarClose?: () => void;
}

export function BlackProjectClient({
  initialData,
  hideHeader = false,
  externalSidebarOpen,
  onExternalSidebarClose,
}: BlackProjectClientProps) {
  const [internalSidebarOpen, setInternalSidebarOpen] = useState(false);

  // Use external state if provided, otherwise use internal state
  const sidebarOpen = externalSidebarOpen !== undefined ? externalSidebarOpen : internalSidebarOpen;
  const setSidebarOpen = onExternalSidebarClose
    ? (open: boolean) => { if (!open) onExternalSidebarClose(); }
    : setInternalSidebarOpen;
  const { data, isLoading, error, parameters, updateParameter } = useSimulation(initialData);

  // Map existing backend data to section-specific formats
  const rateOfComputationData = useMemo(() => mapToRateOfComputationData(data), [data]);
  const detectionLikelihoodData = useMemo(() => mapToDetectionLikelihoodData(data), [data]);
  const covertFabData = useMemo(() => mapToCovertFabData(data), [data]);

  return (
    <div className="min-h-screen bg-[#fffff8]">
      {/* Header - fixed at top */}
      {!hideHeader && (
        <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
      )}

      {/* Content wrapper with margin-top equal to header height */}
      <div style={{ marginTop: hideHeader ? 0 : HEADER_HEIGHT }}>
        <div className="flex flex-1 overflow-hidden">
          {/* Parameter Sidebar */}
          <ParameterSidebar
            parameters={parameters}
            onParameterChange={updateParameter}
            isOpen={sidebarOpen}
            hideHeader={hideHeader}
          />

          {/* Mobile overlay - only render when NOT using external sidebar state (parent handles it) */}
          {sidebarOpen && externalSidebarOpen === undefined && (
            <div
              className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
              onClick={() => setSidebarOpen(false)}
            />
          )}

          {/* Main content */}
          <main className="flex-1 overflow-y-auto px-5 py-4 lg:ml-[260px]">
          {/* Error display */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">
              {error}
            </div>
          )}

          <div className="space-y-6">
            {/* Top section: Title, Dashboard, and CCDF Charts */}
            <TopChartsSection
              data={data}
              isLoading={isLoading}
              agreementYear={parameters.agreementYear}
            />

            <hr className="my-4 border-gray-200" />

            {/* How We Estimate Section */}
            <HowWeEstimateSection
              parameters={parameters}
              rateOfComputationData={rateOfComputationData}
              detectionLikelihoodData={detectionLikelihoodData}
            />

            <hr className="my-4 border-gray-200" />

            {/* Initial Stock Section */}
            <div id="initialStockSection">
              <InitialStockSection
                data={data}
                isLoading={isLoading}
                parameters={parameters}
              />
            </div>

            <hr className="my-4 border-gray-200" />

            {/* Datacenter Section */}
            <div id="covertDataCentersSection">
              <DatacenterSection
                data={data}
                isLoading={isLoading}
                parameters={parameters}
              />
            </div>

            <hr className="my-4 border-gray-200" />

            {/* Covert Fab Section */}
            <div id="covertFabSection">
              <CovertFabSection
                data={data}
                isLoading={isLoading}
                parameters={parameters}
                covertFabData={covertFabData}
              />
            </div>

          </div>
        </main>
        </div>
      </div>
    </div>
  );
}
