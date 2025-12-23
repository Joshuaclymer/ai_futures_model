/**
 * Shared chart configuration constants for consistent styling across all plots
 *
 * This file centralizes "magic numbers" to make the codebase more maintainable.
 * When adjusting chart styling, update values here rather than in individual components.
 */

// Font sizes for Plotly charts (in pixels)
export const CHART_FONT_SIZES = {
  axisTitle: 11,      // Axis label text (e.g., "Year", "Probability")
  tickLabel: 10,      // Tick mark labels
  legend: 10,         // Legend text
  plotTitle: 12,      // Chart title (when shown inside plot)
} as const;

// Standard chart heights (in pixels)
export const CHART_HEIGHTS = {
  breakdown: 240,     // Height of breakdown/equation charts
  dashboard: 240,     // Height of dashboard charts
  historical: 240,    // Height of historical comparison charts
  topSection: 280,    // Height of main top section charts
} as const;

// Standard chart margins for Plotly (in pixels)
export const CHART_MARGINS = {
  // Default margins for most charts
  default: { l: 50, r: 20, t: 10, b: 50 },
  // Compact margins for smaller charts (PDF, stacked area)
  compact: { l: 45, r: 10, t: 10, b: 35 },
  // Margins for charts with dual y-axes
  dualAxis: { l: 55, r: 55, t: 10, b: 50 },
  // Margins for time series with title
  withTitle: { l: 60, r: 20, t: 30, b: 50 },
  // Margins for time series without title
  noTitle: { l: 60, r: 20, t: 10, b: 50 },
} as const;

// UI spacing constants (in pixels)
export const UI_SPACING = {
  sectionGap: 20,           // Gap between major sections
  chartTitleMargin: 10,     // Margin below chart titles
  dashboardItemGap: 20,     // Gap between dashboard items
  breakdownGap: 5,          // Gap between breakdown items
} as const;

// Tooltip/popup dimensions
export const TOOLTIP_DIMENSIONS = {
  defaultWidth: 400,
  minUsableWidth: 200,
  minHeight: 200,
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
