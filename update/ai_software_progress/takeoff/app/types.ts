export interface TooltipProps {
  active?: boolean;
  payload?: Array<{
      value: number;
      color: string;
      dataKey: string;
      name?: string;
  }>;
  label?: number;
}

export interface ChartDataPoint {
    year: number;
    horizonLength: number;
    horizonFormatted: string;
    effectiveCompute?: number | null;
    automationFraction?: number | null;
    trainingCompute?: number | null;
    experimentCapacity?: number | null;
    aiResearchTaste?: number | null;
    aiSoftwareProgressMultiplier?: number | null;
    aiSwProgressMultRefPresentDay?: number | null;
    serialCodingLaborMultiplier?: number | null;
    // New small-chart metrics
    humanLabor?: number | null;
    inferenceCompute?: number | null;
    experimentCompute?: number | null;
    researchEffort?: number | null;
    researchStock?: number | null;
    softwareProgressRate?: number | null;
    softwareEfficiency?: number | null;
    aiCodingLaborMultiplier?: number | null;
    aiCodingLaborMultRefPresentDay?: number | null;
    benchmarkHorizon?: number;
    benchmarkLabel?: string;
    // Monte Carlo trajectory data
    [key: string]: number | string | null | undefined; // Allow trajectory_0, trajectory_1, etc.
}

export interface ShapeProps {
    cx: number;
    cy: number;
    onMouseEnter?: (e: React.MouseEvent) => void;
    onMouseLeave?: () => void;
    style?: React.CSSProperties;
}
export interface BenchmarkPoint {
  year: number;
  horizonLength: number;
  label: string;
  model: string;
}
export interface TooltipProps {
    active?: boolean;
    payload?: Array<{
        value: number;
        color: string;
        dataKey: string;
        name?: string;
    }>;
    label?: number;
}
