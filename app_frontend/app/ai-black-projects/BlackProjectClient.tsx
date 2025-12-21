'use client';

import { useState } from 'react';
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

// Import hooks and types
import { useSimulation } from './hooks/useSimulation';
import { SimulationData, Parameters } from './types';

// Re-export colors for backwards compatibility
import { COLOR_PALETTE as IMPORTED_COLORS } from './components/colors';
export const COLOR_PALETTE = IMPORTED_COLORS;

interface BlackProjectClientProps {
  initialData: SimulationData | null;
  hideHeader?: boolean;
}

export function BlackProjectClient({ initialData, hideHeader = false }: BlackProjectClientProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { data, isLoading, error, parameters, updateParameter } = useSimulation(initialData);

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

          {/* Mobile overlay */}
          {sidebarOpen && (
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
              rateOfComputationData={data?.rate_of_computation}
              detectionLikelihoodData={data?.detection_likelihood}
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
                covertFabData={data?.covert_fab}
              />
            </div>

          </div>
        </main>
        </div>
      </div>
    </div>
  );
}
