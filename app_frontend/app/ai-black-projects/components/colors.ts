/**
 * Color palette for black project visualizations
 * Matches the original Flask implementation exactly
 */
export const COLOR_PALETTE = {
  // Primary colors for data series (named by usage)
  chip_stock: '#5E6FB8',           // Initial chip stock, PRC chip stock evidence (Indigo)
  fab: '#E9A842',                  // Covert fab production, SME stock evidence (Marigold)
  datacenters_and_energy: '#4AA896', // Satellite/datacenter evidence (Viridian)
  detection: '#7BA3C4',            // Detection/LR plots, energy consumption (Pewter Blue)

  // Secondary/accent colors
  survival_rate: '#E05A4F',        // Survival rate (Vermillion)
  gray: '#7F8C8D',                 // Neutral/disabled
} as const;

// Helper function to get rgba version with alpha (for palette colors)
export function rgba(colorName: keyof typeof COLOR_PALETTE, alpha: number): string {
  return hexToRgba(COLOR_PALETTE[colorName], alpha);
}

// Helper function to convert any hex color to rgba
export function hexToRgba(hex: string, alpha: number): string {
  // Handle undefined/null hex values
  if (!hex) {
    return `rgba(128, 128, 128, ${alpha})`; // fallback gray
  }
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Helper to darken a color by a factor (0-1, where 0.8 = 20% darker)
export function darken(colorName: keyof typeof COLOR_PALETTE, factor: number): string {
  const hex = COLOR_PALETTE[colorName];
  const r = Math.round(parseInt(hex.slice(1, 3), 16) * factor);
  const g = Math.round(parseInt(hex.slice(3, 5), 16) * factor);
  const b = Math.round(parseInt(hex.slice(5, 7), 16) * factor);
  return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
}

export type ColorName = keyof typeof COLOR_PALETTE;
