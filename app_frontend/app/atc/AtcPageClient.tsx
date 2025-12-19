'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Simulation } from '@/components/SimulationSelector';
import { AtcMilestoneSelector } from '@/components/AtcMilestoneSelector';
import { EditableCdfChart } from '@/components/EditableCdfChart';
import { PdfLineChart, DataPoint } from '@/components/PdfLineChart';
import { ChartSyncProvider } from '@/components/ChartSyncContext';
import { HeaderContent } from '@/components/HeaderContent';
import {
  ControlPoint,
  CdfPoint,
  initializeControlPoints,
  evaluateSplineAt,
  samplePdf,
  computePdfFromCdf,
} from '@/utils/cdfSplineUtils';
import { CdfDataResponse } from '@/app/api/atc-cdf/route';

// Color palette for milestones
const MILESTONE_COLORS: { [key: string]: string } = {
  'AC': '#2A623D',
  'AI2027-SC': '#af1e86ff',
  'SAR-level-experiment-selection-skill': '#000090',
  'SIAR-level-experiment-selection-skill': '#900000',
};

interface AtcPageClientProps {
  simulations: Simulation[];
  defaultSimulationId: string | null;
}

// Sort simulations by date (newest first)
function sortSimulationsByDate(simulations: Simulation[]): Simulation[] {
  return [...simulations].sort((a, b) => {
    const parseDate = (dateStr: string): number => {
      const [month, day, year] = dateStr.split('-').map(Number);
      const fullYear = year < 100 ? 2000 + year : year;
      return new Date(fullYear, month - 1, day).getTime();
    };
    return parseDate(b.date) - parseDate(a.date);
  });
}

export function AtcPageClient({ simulations, defaultSimulationId }: AtcPageClientProps) {
  const sortedSimulations = useMemo(() => sortSimulationsByDate(simulations), [simulations]);

  // State
  const [selectedSimulationId, setSelectedSimulationId] = useState<string>(
    defaultSimulationId ?? sortedSimulations[0]?.id ?? ''
  );
  const [selectedMilestone, setSelectedMilestone] = useState<string>('');
  const [cdfData, setCdfData] = useState<CdfDataResponse | null>(null);
  const [editedMilestones, setEditedMilestones] = useState<Map<string, ControlPoint[]>>(new Map());
  const [currentControlPoints, setCurrentControlPoints] = useState<ControlPoint[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load CDF data when simulation changes
  useEffect(() => {
    if (!selectedSimulationId) return;

    setIsLoading(true);
    setError(null);

    fetch(`/api/atc-cdf?simulationId=${encodeURIComponent(selectedSimulationId)}`)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to load data: ${res.status}`);
        return res.json();
      })
      .then((data: CdfDataResponse) => {
        setCdfData(data);
        // Reset edited milestones when simulation changes
        setEditedMilestones(new Map());
        // Select first milestone by default
        if (data.milestones.length > 0 && !selectedMilestone) {
          setSelectedMilestone(data.milestones[0]);
        }
      })
      .catch(err => {
        setError(err.message);
        setCdfData(null);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [selectedSimulationId]);

  // Initialize control points when milestone changes
  useEffect(() => {
    if (!cdfData || !selectedMilestone) {
      setCurrentControlPoints([]);
      return;
    }

    // Check if we have saved edits for this milestone
    const savedEdits = editedMilestones.get(selectedMilestone);
    if (savedEdits) {
      setCurrentControlPoints(savedEdits);
    } else {
      // Initialize from empirical CDF
      const milestoneData = cdfData.data[selectedMilestone];
      if (milestoneData) {
        const cdfPoints: CdfPoint[] = milestoneData.map(p => ({ year: p.year, cdf: p.cdf }));
        const initialPoints = initializeControlPoints(cdfPoints);
        setCurrentControlPoints(initialPoints);
      }
    }
  }, [selectedMilestone, cdfData, editedMilestones]);

  // Save control points when they change
  const handleControlPointsChange = useCallback((newPoints: ControlPoint[]) => {
    setCurrentControlPoints(newPoints);
    if (selectedMilestone) {
      setEditedMilestones(prev => {
        const next = new Map(prev);
        next.set(selectedMilestone, newPoints);
        return next;
      });
    }
  }, [selectedMilestone]);

  // Handle milestone change - save current edits first
  const handleMilestoneChange = useCallback((newMilestone: string) => {
    // Save current control points before switching
    if (selectedMilestone && currentControlPoints.length > 0) {
      setEditedMilestones(prev => {
        const next = new Map(prev);
        next.set(selectedMilestone, currentControlPoints);
        return next;
      });
    }
    setSelectedMilestone(newMilestone);
  }, [selectedMilestone, currentControlPoints]);

  // Reset current milestone to empirical
  const handleReset = useCallback(() => {
    if (!cdfData || !selectedMilestone) return;

    const milestoneData = cdfData.data[selectedMilestone];
    if (milestoneData) {
      const cdfPoints: CdfPoint[] = milestoneData.map(p => ({ year: p.year, cdf: p.cdf }));
      const initialPoints = initializeControlPoints(cdfPoints);
      setCurrentControlPoints(initialPoints);
      // Remove from edited milestones
      setEditedMilestones(prev => {
        const next = new Map(prev);
        next.delete(selectedMilestone);
        return next;
      });
    }
  }, [cdfData, selectedMilestone]);

  // Export all milestones to CSV
  const handleExport = useCallback(() => {
    if (!cdfData) return;

    // Use the original time points from the data
    const timePoints = cdfData.timePoints;

    // Build CSV rows
    const headers = ['time_decimal_year', ...cdfData.milestones];
    const rows: string[] = [headers.join(',')];

    for (const year of timePoints) {
      const values: string[] = [year.toFixed(6)];

      for (const milestone of cdfData.milestones) {
        const edited = editedMilestones.get(milestone);
        let cdfValue: number;

        if (edited && edited.length >= 2) {
          // Use the edited spline
          cdfValue = evaluateSplineAt(edited, year);
        } else {
          // Use the original empirical CDF
          const originalData = cdfData.data[milestone];
          const point = originalData.find(p => Math.abs(p.year - year) < 0.0001);
          cdfValue = point?.cdf ?? 0;
        }

        values.push(cdfValue.toFixed(10));
      }

      rows.push(values.join(','));
    }

    const csv = rows.join('\n');

    // Trigger download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `atc_${selectedSimulationId}_cdf.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [cdfData, editedMilestones, selectedSimulationId]);

  // Get empirical CDF for the current milestone
  const empiricalCdf = useMemo(() => {
    if (!cdfData || !selectedMilestone) return [];
    const data = cdfData.data[selectedMilestone];
    return data?.map(p => ({ year: p.year, cdf: p.cdf })) ?? [];
  }, [cdfData, selectedMilestone]);

  // Compute X domain for charts (shared between CDF and PDF)
  const xDomain = useMemo<[number, number]>(() => {
    if (empiricalCdf.length === 0) return [2025, 2050];
    const years = empiricalCdf.map(p => p.year);
    const min = Math.min(...years);
    const max = Math.min(Math.max(...years), 2055);
    const padding = (max - min) * 0.02;
    return [min - padding, max + padding];
  }, [empiricalCdf]);

  // Compute original PDF from empirical CDF
  const originalPdfSamples = useMemo(() => {
    if (empiricalCdf.length < 2) return [];
    return computePdfFromCdf(empiricalCdf, 300);
  }, [empiricalCdf]);

  // Compute edited PDF from control points
  const editedPdfSamples = useMemo(() => {
    if (currentControlPoints.length < 2) return [];
    return samplePdf(currentControlPoints, 300);
  }, [currentControlPoints]);

  // Merge both PDFs into a single data array for the chart
  const pdfData = useMemo<DataPoint[]>(() => {
    if (editedPdfSamples.length === 0 && originalPdfSamples.length === 0) return [];

    // Create a map of x values to data points
    const dataMap = new Map<number, DataPoint>();

    // Add original PDF data
    for (const sample of originalPdfSamples) {
      const rounded = Math.round(sample.x * 10000) / 10000; // Avoid floating point issues
      dataMap.set(rounded, { x: sample.x, original: sample.pdf });
    }

    // Add edited PDF data
    for (const sample of editedPdfSamples) {
      const rounded = Math.round(sample.x * 10000) / 10000;
      const existing = dataMap.get(rounded);
      if (existing) {
        existing.edited = sample.pdf;
      } else {
        dataMap.set(rounded, { x: sample.x, edited: sample.pdf });
      }
    }

    // Convert to array and sort by x
    return Array.from(dataMap.values()).sort((a, b) => a.x - b.x);
  }, [originalPdfSamples, editedPdfSamples]);

  // Compute Y domain for PDF chart (auto-scale based on both PDFs)
  const pdfYDomain = useMemo<[number, number]>(() => {
    if (pdfData.length === 0) return [0, 1];
    const maxOriginal = Math.max(...pdfData.map(d => (d.original as number) ?? 0));
    const maxEdited = Math.max(...pdfData.map(d => (d.edited as number) ?? 0));
    const maxPdf = Math.max(maxOriginal, maxEdited);
    return [0, maxPdf * 1.1 || 1];
  }, [pdfData]);

  // Get milestone color
  const milestoneColor = MILESTONE_COLORS[selectedMilestone] ?? '#2563eb';

  return (
    <div className="grid h-screen w-full grid-cols-[minmax(0,1fr)_auto] grid-rows-[minmax(0,1fr)] gap-0">
      <div className="relative col-start-1 row-start-1 flex min-h-0 flex-col">
        <div className="flex min-h-0 flex-col overflow-y-auto px-6 pb-10">
          <HeaderContent variant="inline" className="pt-6 pb-4" />
          <main className="mt-10 mx-auto max-w-5xl w-full px-6 pb-16">
            <h2 className="text-2xl font-semibold text-primary font-[family-name:var(--font-et-book)] mb-6">
              ATC Distribution Editor
            </h2>

            {/* Selectors */}
            <div className="flex flex-wrap gap-4 mb-6">
              {/* Forecast Selector */}
              <div className="flex items-center gap-2">
                <label htmlFor="forecast-selector" className="text-sm font-medium text-primary/70">
                  Forecast:
                </label>
                <select
                  id="forecast-selector"
                  value={selectedSimulationId}
                  onChange={(e) => setSelectedSimulationId(e.target.value)}
                  className="block rounded-md border-gray-300 bg-white py-1.5 pl-3 pr-10 text-sm shadow-sm focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
                >
                  {sortedSimulations.map((sim) => (
                    <option key={sim.id} value={sim.id}>
                      {sim.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Milestone Selector */}
              {cdfData && (
                <AtcMilestoneSelector
                  milestones={cdfData.milestones}
                  selectedMilestone={selectedMilestone}
                  editedMilestones={editedMilestones}
                  onMilestoneChange={handleMilestoneChange}
                />
              )}
            </div>

            {/* Loading/Error states */}
            {isLoading && (
              <div className="text-primary/50 py-8">Loading CDF data...</div>
            )}
            {error && (
              <div className="text-red-600 py-8">Error: {error}</div>
            )}

        {/* Chart */}
        {cdfData && selectedMilestone && !isLoading && (
          <ChartSyncProvider>
            <EditableCdfChart
              empiricalCdf={empiricalCdf}
              controlPoints={currentControlPoints}
              onControlPointsChange={handleControlPointsChange}
              milestoneName={selectedMilestone}
            />

            {/* PDF Chart */}
            {pdfData.length > 0 && (
              <div className="bg-white rounded-lg shadow p-4 mt-4">
                <h3 className="text-sm font-medium text-primary/70 mb-2">Implied PDF (Probability Density)</h3>
                <PdfLineChart
                  data={pdfData}
                  height={200}
                  xDomain={xDomain}
                  yDomain={pdfYDomain}
                  xTickFormatter={(v) => v.toFixed(0)}
                  xLabel="Year"
                  lines={[
                    {
                      dataKey: 'original',
                      stroke: '#9ca3af', // gray-400
                      strokeWidth: 1.5,
                      strokeOpacity: 0.7,
                      smooth: true,
                      name: 'Original',
                    },
                    {
                      dataKey: 'edited',
                      stroke: milestoneColor,
                      strokeWidth: 2,
                      smooth: true,
                      name: 'Edited',
                    },
                  ]}
                  tooltip={(point) => {
                    // Interpolate empirical CDF at hover point
                    const getEmpiricalCdfAt = (x: number): number => {
                      if (empiricalCdf.length === 0) return 0;
                      const sorted = [...empiricalCdf].sort((a, b) => a.year - b.year);
                      if (x <= sorted[0].year) return sorted[0].cdf;
                      if (x >= sorted[sorted.length - 1].year) return sorted[sorted.length - 1].cdf;
                      for (let i = 0; i < sorted.length - 1; i++) {
                        if (x >= sorted[i].year && x <= sorted[i + 1].year) {
                          const t = (x - sorted[i].year) / (sorted[i + 1].year - sorted[i].year);
                          return sorted[i].cdf + t * (sorted[i + 1].cdf - sorted[i].cdf);
                        }
                      }
                      return 0;
                    };
                    const empiricalCdfValue = getEmpiricalCdfAt(point.x);
                    const editedCdfValue = evaluateSplineAt(currentControlPoints, point.x);

                    return (
                      <div className="bg-white border border-gray-200 rounded px-2 py-1.5 shadow-md text-xs">
                        <div className="font-semibold text-primary">{point.x.toFixed(1)}</div>
                        <div className="text-primary/50">
                          Empirical: {(empiricalCdfValue * 100).toFixed(1)}%
                        </div>
                        <div style={{ color: milestoneColor }} className="font-medium">
                          ATC: {(editedCdfValue * 100).toFixed(1)}%
                        </div>
                      </div>
                    );
                  }}
                />
                <div className="flex items-center gap-4 mt-2">
                  <div className="flex items-center gap-1.5">
                    <div className="w-4 h-0.5 bg-gray-400 opacity-70"></div>
                    <span className="text-xs text-primary/50">Original (model forecast)</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-4 h-0.5" style={{ backgroundColor: milestoneColor }}></div>
                    <span className="text-xs text-primary/50">Edited</span>
                  </div>
                </div>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex gap-4 mt-6">
              <button
                onClick={handleReset}
                className="px-4 py-2 text-sm font-medium text-primary bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-accent"
              >
                Reset to Empirical
              </button>
              <button
                onClick={handleExport}
                className="px-4 py-2 text-sm font-medium text-white bg-accent border border-transparent rounded-md hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-accent"
              >
                Export ATC Forecast
              </button>
            </div>

            {/* Edited milestones indicator */}
            {editedMilestones.size > 0 && (
              <div className="mt-4 text-sm text-primary/50">
                Edited milestones: {Array.from(editedMilestones.keys()).join(', ')}
              </div>
            )}
          </ChartSyncProvider>
        )}
          </main>
        </div>
      </div>
    </div>
  );
}
