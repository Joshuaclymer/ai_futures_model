/**
 * Shared chart configuration constants for consistent styling across all plots
 */

// Font sizes for Plotly charts
export const CHART_FONT_SIZES = {
  axisTitle: 11,      // Axis label text (e.g., "Year", "Probability")
  tickLabel: 10,      // Tick mark labels
  legend: 10,         // Legend text
  plotTitle: 12,      // Chart title (when shown inside plot)
} as const;

// Plot title shown below charts (in breakdown-label class)
export const BREAKDOWN_LABEL_SIZE = 13; // px

// Helper to create axis config with consistent font sizes
export function createAxisConfig(title: string, options?: {
  type?: 'linear' | 'log';
  tickformat?: string;
  range?: [number, number];
  automargin?: boolean;
  side?: 'left' | 'right';
  overlaying?: string;
  standoff?: number;
}) {
  return {
    title: {
      text: title,
      font: { size: CHART_FONT_SIZES.axisTitle },
      ...(options?.standoff ? { standoff: options.standoff } : {}),
    },
    tickfont: { size: CHART_FONT_SIZES.tickLabel },
    ...(options?.type ? { type: options.type } : {}),
    ...(options?.tickformat ? { tickformat: options.tickformat } : {}),
    ...(options?.range ? { range: options.range } : {}),
    ...(options?.automargin ? { automargin: options.automargin } : {}),
    ...(options?.side ? { side: options.side } : {}),
    ...(options?.overlaying ? { overlaying: options.overlaying } : {}),
  };
}

// Helper to create legend config with consistent styling
export function createLegendConfig(options?: {
  x?: number;
  y?: number;
  xanchor?: 'left' | 'right' | 'center';
  yanchor?: 'top' | 'bottom' | 'middle';
  orientation?: 'v' | 'h';
}) {
  return {
    x: options?.x ?? 0.98,
    y: options?.y ?? 0.98,
    xanchor: options?.xanchor ?? 'right',
    yanchor: options?.yanchor ?? 'top',
    bgcolor: 'rgba(255,255,248,0.9)',
    borderwidth: 0,
    font: { size: CHART_FONT_SIZES.legend },
    ...(options?.orientation ? { orientation: options.orientation } : {}),
  };
}
