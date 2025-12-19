'use client';

import { useRouter } from 'next/navigation';
import { useCallback } from 'react';

export interface Simulation {
  id: string;
  label: string;
  forecaster: string;
  date: string;
  customIdentifier: string | null;
  dataPath: string;
  numSamples: number | null;
  modelVersion: string | null;
}

interface SimulationSelectorProps {
  simulations: Simulation[];
  currentSimulationId: string;
  className?: string;
}

export function SimulationSelector({
  simulations,
  currentSimulationId,
  className = '',
}: SimulationSelectorProps) {
  const router = useRouter();

  const handleChange = useCallback((event: React.ChangeEvent<HTMLSelectElement>) => {
    const newId = event.target.value;
    if (newId !== currentSimulationId) {
      router.push(`/forecast/${encodeURIComponent(newId)}`);
    }
  }, [router, currentSimulationId]);

  // Don't render if there's only one or no simulations
  if (simulations.length <= 1) {
    return null;
  }

  // Sort simulations by date (newest first)
  // Date format is MM-DD-YY, so we need to parse and compare
  const sortedSimulations = [...simulations].sort((a, b) => {
    // Parse date in MM-DD-YY format
    const parseDate = (dateStr: string): number => {
      const [month, day, year] = dateStr.split('-').map(Number);
      // Convert YY to full year (assume 2000s)
      const fullYear = year < 100 ? 2000 + year : year;
      return new Date(fullYear, month - 1, day).getTime();
    };

    const dateA = parseDate(a.date);
    const dateB = parseDate(b.date);

    // Sort descending (newest first)
    return dateB - dateA;
  });

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <label htmlFor="simulation-selector" className="text-sm font-medium text-gray-700">
        Forecast:
      </label>
      <select
        id="simulation-selector"
        value={currentSimulationId}
        onChange={handleChange}
        className="block rounded-md border-gray-300 bg-white py-1.5 pl-3 pr-10 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
      >
        {sortedSimulations.map((sim) => (
          <option key={sim.id} value={sim.id}>
            {sim.label}
          </option>
        ))}
      </select>
    </div>
  );
}

interface SimulationMetadataProps {
  simulation: Simulation;
  className?: string;
}

export function SimulationMetadata({
  simulation,
  className = '',
}: SimulationMetadataProps) {
  return (
    <div className={`text-xs text-gray-500 ${className}`}>
      <span>
        {simulation.forecaster} forecast
        {simulation.numSamples && ` (${simulation.numSamples.toLocaleString()} samples)`}
        {simulation.modelVersion && ` - Model ${simulation.modelVersion}`}
      </span>
    </div>
  );
}
