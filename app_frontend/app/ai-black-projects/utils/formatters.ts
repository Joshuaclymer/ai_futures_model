/**
 * Formatting utilities for dashboard values.
 * All formatters round to 2 significant figures for display.
 */

/**
 * Round a number to specified significant figures.
 */
export function toSigFigs(n: number, sigFigs: number = 2): number {
  if (n === 0) return 0;
  const magnitude = Math.floor(Math.log10(Math.abs(n)));
  const multiplier = Math.pow(10, sigFigs - 1 - magnitude);
  return Math.round(n * multiplier) / multiplier;
}

/**
 * Format a number to 2 significant figures as a string.
 */
export function formatSigFigs(n: number, sigFigs: number = 2): string {
  if (n === 0) return '0';
  const rounded = toSigFigs(n, sigFigs);
  const magnitude = Math.floor(Math.log10(Math.abs(rounded)));

  // Determine decimal places needed
  if (magnitude >= sigFigs - 1) {
    return rounded.toLocaleString('en-US', { maximumFractionDigits: 0 });
  } else {
    const decimals = Math.max(0, sigFigs - 1 - magnitude);
    return rounded.toFixed(decimals);
  }
}

/**
 * Format large numbers with K/M/B suffixes, rounded to 2 sig figs.
 */
export function formatNumber(n: number): string {
  if (n >= 1e9) return `${formatSigFigs(n / 1e9)}B`;
  if (n >= 1e6) return `${formatSigFigs(n / 1e6)}M`;
  if (n >= 1e3) return `${formatSigFigs(n / 1e3)}K`;
  return formatSigFigs(n);
}

/**
 * Format H100 equivalent values with appropriate units, rounded to 2 sig figs.
 */
export function formatH100e(value: number): string {
  if (value >= 1_000_000) {
    return `${formatSigFigs(value / 1_000_000)}M H100e`;
  } else if (value >= 1_000) {
    return `${formatSigFigs(value / 1_000)}K H100e`;
  }
  return `${formatSigFigs(value)} H100e`;
}

/**
 * Format energy values (GW input), rounded to 2 sig figs.
 */
export function formatEnergy(energyGW: number): string {
  if (energyGW >= 1) {
    return `${formatSigFigs(energyGW)} GW`;
  } else if (energyGW >= 0.001) {
    return `${formatSigFigs(energyGW * 1000)} MW`;
  }
  return `${formatSigFigs(energyGW * 1000)} MW`;
}

/**
 * Format capacity values (GW), rounded to 2 sig figs.
 */
export function formatCapacity(gw: number): string {
  if (gw >= 1) {
    return `${formatSigFigs(gw)} GW`;
  } else if (gw >= 0.001) {
    return `${formatSigFigs(gw * 1000)} MW`;
  }
  return `${formatSigFigs(gw * 1000)} MW`;
}

/**
 * Format percentage values (0-1 input), rounded to 2 sig figs.
 */
export function formatPercent(value: number): string {
  return `${formatSigFigs(value * 100)}%`;
}

/**
 * Format years, rounded to 2 sig figs.
 */
export function formatYears(years: number): string {
  return `${formatSigFigs(years)} years`;
}
