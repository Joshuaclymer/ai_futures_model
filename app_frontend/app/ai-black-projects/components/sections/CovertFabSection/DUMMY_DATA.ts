/**
 * DUMMY DATA - FOR VISUALIZATION TESTING ONLY
 *
 * This file contains fake/dummy data used to test chart visualizations.
 * DO NOT use this data for any real analysis or production purposes.
 * Replace with actual API data when available.
 */

// Generate years array from agreement year
export function generateYears(agreementYear: number = 2030, numYears: number = 7): number[] {
  return Array.from({ length: numYears * 4 + 1 }, (_, i) => agreementYear + i * 0.25);
}

export interface DummyChartData {
  years: number[];
  median: number[];
  p25: number[];
  p75: number[];
  individual?: number[][];
}

export interface CCDFPoint {
  x: number;
  y: number;
}

// Dashboard data for median outcome
export interface DashboardData {
  production: string;      // e.g., "800K H100e"
  energy: string;          // e.g., "2.8 GW"
  probFabBuilt: string;    // e.g., "56.2%"
  yearsOperational: string; // e.g., "0.8"
  processNode: string;     // e.g., "28nm"
}

export function getDummyDashboard(): DashboardData {
  return {
    production: '800K H100e',
    energy: '2.8 GW',
    probFabBuilt: '56.2%',
    yearsOperational: '0.8',
    processNode: '28nm',
  };
}

// CCDF data for covert compute produced before detection
export function getDummyComputeCCDF(): Record<number, CCDFPoint[]> {
  const generateCCDF = (shift: number = 0): CCDFPoint[] => {
    const points: CCDFPoint[] = [];
    for (let i = 0; i <= 100; i++) {
      const x = Math.pow(10, 2 + i * 0.05 + shift); // 100 to 10M range
      const y = Math.max(0, 1 - (i / 100) ** 0.5);
      points.push({ x, y });
    }
    return points;
  };

  return {
    1: generateCCDF(0.3),  // >1x update threshold
    2: generateCCDF(0.1),  // >2x update threshold
    4: generateCCDF(0),    // >4x update threshold (most conservative)
  };
}

// Time series data for simulation runs (LR and H100e)
export function getDummyTimeSeriesData(agreementYear: number = 2030): {
  years: number[];
  lrCombined: DummyChartData;
  h100eFlow: DummyChartData;
} {
  const years = generateYears(agreementYear);

  // LR grows exponentially
  const lrMedian = years.map((_, i) => Math.pow(1.5, i / 4));
  const lrP25 = lrMedian.map(v => v * 0.7);
  const lrP75 = lrMedian.map(v => v * 1.4);

  // H100e production grows then plateaus
  const h100eMedian = years.map((_, i) => {
    const t = i / 4;
    return t < 1 ? 0 : Math.min(500000, 100000 * (t - 1));
  });
  const h100eP25 = h100eMedian.map(v => v * 0.6);
  const h100eP75 = h100eMedian.map(v => v * 1.4);

  return {
    years,
    lrCombined: { years, median: lrMedian, p25: lrP25, p75: lrP75 },
    h100eFlow: { years, median: h100eMedian, p25: h100eP25, p75: h100eP75 },
  };
}

// Breaking down fab production

// Probability construction has finished (S-curve over time)
export function getDummyIsOperational(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  // S-curve: starts at 0, approaches 1
  const median = years.map((_, i) => {
    const t = i / 4; // years from start
    return 1 / (1 + Math.exp(-2 * (t - 2))); // midpoint at 2 years
  });
  const p25 = median;
  const p75 = median;
  return { years, median, p25, p75 };
}

// Production capacity (wafer starts per month) - distribution of final values
export function getDummyWaferStarts(): number[] {
  // Generate 100 simulated final values
  const values: number[] = [];
  for (let i = 0; i < 100; i++) {
    // Log-normal-ish distribution centered around 5000
    const value = 5000 * (0.5 + Math.random() * 1.5);
    values.push(value);
  }
  return values.sort((a, b) => a - b);
}

// Working H100-sized chips per wafer (constant value)
export function getDummyChipsPerWafer(): number {
  return 28;
}

// Transistor density relative to H100 by process node
export function getDummyTransistorDensity(): { node: string; density: number }[] {
  return [
    { node: '28nm', density: 0.14 },
    { node: '14nm', density: 0.5 },
    { node: '7nm', density: 1.0 },
  ];
}

// Architecture efficiency relative to H100 (constant at agreement year)
export function getDummyArchitectureEfficiency(): number {
  return 0.85;
}

// Compute produced per month (time series)
export function getDummyComputePerMonth(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  const isOp = getDummyIsOperational(agreementYear);

  // Compute = is_operational * wafer_starts * chips_per_wafer * transistor_density * arch_efficiency
  const baseValue = 5000 * 28 * 0.14 * 0.85;
  const median = years.map((_, i) => isOp.median[i] * baseValue);
  const p25 = median.map(v => v * 0.6);
  const p75 = median.map(v => v * 1.5);

  return { years, median, p25, p75 };
}

// Energy efficiency section

// Watts per performance relative to H100 (curve: density vs watts/TPP)
export function getDummyWattsPerTppCurve(): { densityRelative: number[]; wattsPerTppRelative: number[] } {
  const densityRelative: number[] = [];
  const wattsPerTppRelative: number[] = [];

  // Power law: W/TPP ~ density^exponent
  // Before Dennard: exponent ~ -0.5, After Dennard: exponent ~ -0.15
  for (let i = 0; i <= 50; i++) {
    const density = Math.pow(10, -3 + i * 0.1); // 0.001 to 100
    densityRelative.push(density);

    // Dennard ended around density = 0.02 (relative to H100)
    const dennardThreshold = 0.02;
    let wattsPerTpp;
    if (density < dennardThreshold) {
      wattsPerTpp = Math.pow(density, -0.5);
    } else {
      const wattsAtThreshold = Math.pow(dennardThreshold, -0.5);
      wattsPerTpp = wattsAtThreshold * Math.pow(density / dennardThreshold, -0.15);
    }
    wattsPerTppRelative.push(wattsPerTpp);
  }

  return { densityRelative, wattsPerTppRelative };
}

// H100 energy requirements (constant)
export function getDummyH100Power(): number {
  return 700; // Watts
}

// Energy requirements per month (time series)
export function getDummyEnergyPerMonth(agreementYear: number = 2030): DummyChartData {
  const compute = getDummyComputePerMonth(agreementYear);
  const h100Power = getDummyH100Power();
  const wattsPerTppRelative = 3; // For 28nm node

  // Energy (GW) = H100e * watts_per_tpp_rel * H100_power / 1e9
  const median = compute.median.map(v => (v * wattsPerTppRelative * h100Power) / 1e9);
  const p25 = median.map(v => v * 0.6);
  const p75 = median.map(v => v * 1.5);

  return { years: compute.years, median, p25, p75 };
}
