/**
 * ⚠️ DUMMY DATA - FOR VISUALIZATION TESTING ONLY ⚠️
 *
 * This file contains fake/dummy data used to test chart visualizations.
 * DO NOT use this data for any real analysis or production purposes.
 * Replace with actual API data when available.
 */

// Generate years array from agreement year
export function generateYears(agreementYear: number = 2030, numYears: number = 7): number[] {
  return Array.from({ length: numYears * 4 + 1 }, (_, i) => agreementYear + i * 0.25);
}

// Generate exponential growth data (for chip stock, datacenter capacity, etc.)
function generateExponentialGrowth(
  years: number[],
  initialValue: number,
  growthRate: number,
  noise: number = 0.1
): { median: number[]; p25: number[]; p75: number[] } {
  const median = years.map((_, i) => initialValue * Math.pow(1 + growthRate, i / 4));
  const p25 = median.map(v => v * (1 - noise));
  const p75 = median.map(v => v * (1 + noise));
  return { median, p25, p75 };
}

// Generate decay data (for surviving fraction)
function generateDecay(
  years: number[],
  halfLife: number = 5
): { median: number[]; p25: number[]; p75: number[] } {
  const agreementYear = years[0];
  const median = years.map(y => Math.pow(0.5, (y - agreementYear) / halfLife));
  const p25 = median.map(v => Math.max(0, v - 0.05));
  const p75 = median.map(v => Math.min(1, v + 0.05));
  return { median, p25, p75 };
}

// Generate linear growth data (for datacenter capacity)
function generateLinearGrowth(
  years: number[],
  initialValue: number,
  ratePerYear: number,
  noise: number = 0.1
): { median: number[]; p25: number[]; p75: number[] } {
  const agreementYear = years[0];
  const median = years.map(y => initialValue + ratePerYear * (y - agreementYear));
  const p25 = median.map(v => v * (1 - noise));
  const p75 = median.map(v => v * (1 + noise));
  return { median, p25, p75 };
}

// Generate S-curve data (for posterior probability)
function generateSCurve(
  years: number[],
  midpoint: number = 3,
  steepness: number = 2
): { median: number[]; p25: number[]; p75: number[] } {
  const agreementYear = years[0];
  const median = years.map(y => {
    const t = y - agreementYear;
    return 1 / (1 + Math.exp(-steepness * (t - midpoint)));
  });
  const p25 = median.map(v => Math.max(0, v - 0.1));
  const p75 = median.map(v => Math.min(1, v + 0.1));
  return { median, p25, p75 };
}

// Generate step function data (for fab operational)
function generateStepFunction(
  years: number[],
  stepYear: number = 2
): { median: number[]; p25: number[]; p75: number[] } {
  const agreementYear = years[0];
  const median = years.map(y => (y - agreementYear >= stepYear ? 1 : 0));
  return { median, p25: median, p75: median };
}

export interface DummyChartData {
  years: number[];
  median: number[];
  p25: number[];
  p75: number[];
}

/**
 * ⚠️ DUMMY DATA GENERATORS ⚠️
 * All data below is FAKE and for testing only
 */

export function getDummyInitialChipStock(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  // Initial stock is flat (it's what was diverted at the start)
  const initialValue = 50000; // 50K H100 equivalents
  const median = years.map(() => initialValue);
  const p25 = median.map(v => v * 0.8);
  const p75 = median.map(v => v * 1.2);
  return { years, median, p25, p75 };
}

export function getDummyAcquiredHardware(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  // Grows over time as covert fab produces chips
  const data = generateExponentialGrowth(years, 1000, 0.3, 0.2);
  return { years, ...data };
}

export function getDummySurvivingFraction(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  const data = generateDecay(years, 6);
  return { years, ...data };
}

export function getDummyCovertChipStock(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  // Combination of initial + acquired, with decay
  const initial = getDummyInitialChipStock(agreementYear);
  const acquired = getDummyAcquiredHardware(agreementYear);
  const surviving = getDummySurvivingFraction(agreementYear);

  const median = years.map((_, i) =>
    (initial.median[i] + acquired.median[i]) * surviving.median[i]
  );
  const p25 = median.map(v => v * 0.7);
  const p75 = median.map(v => v * 1.3);

  return { years, median, p25, p75 };
}

export function getDummyDatacenterCapacity(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  // Linear growth in GW
  const data = generateLinearGrowth(years, 5, 8, 0.15); // Start at 5 GW, add 8 GW/year
  return { years, ...data };
}

export function getDummyEnergyUsage(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  // Energy required by chip stock (grows with chip stock)
  const chipStock = getDummyCovertChipStock(agreementYear);
  const energyPerChip = 0.0005; // GW per chip
  const median = chipStock.median.map(v => v * energyPerChip);
  const p25 = median.map(v => v * 0.8);
  const p75 = median.map(v => v * 1.2);
  return { years, median, p25, p75 };
}

export function getDummyOperatingChips(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  // Min of chip stock and datacenter capacity (energy limited)
  const chipStock = getDummyCovertChipStock(agreementYear);
  const datacenter = getDummyDatacenterCapacity(agreementYear);
  const energyPerChip = 0.0005;

  const median = years.map((_, i) => {
    const chipsFromEnergy = datacenter.median[i] / energyPerChip;
    return Math.min(chipStock.median[i], chipsFromEnergy);
  });
  const p25 = median.map(v => v * 0.7);
  const p75 = median.map(v => v * 1.3);

  return { years, median, p25, p75 };
}

export function getDummyCovertComputation(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  // Cumulative H100-years
  const operating = getDummyOperatingChips(agreementYear);
  let cumulative = 0;
  const median = operating.median.map(v => {
    cumulative += v * 0.25; // 0.25 years per time step
    return cumulative;
  });
  const p25 = median.map(v => v * 0.6);
  const p75 = median.map(v => v * 1.4);

  return { years, median, p25, p75 };
}

// Detection-related dummy data

// Generate log-normal samples for likelihood ratio distributions
function generateLogNormalSamples(
  numSamples: number,
  medianLR: number,
  spread: number = 0.5
): number[] {
  // Generate samples with a log-normal distribution
  const samples: number[] = [];
  const logMedian = Math.log(medianLR);
  for (let i = 0; i < numSamples; i++) {
    // Box-Muller transform for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const logValue = logMedian + spread * z;
    samples.push(Math.exp(logValue));
  }
  return samples;
}

// Sample-based evidence data (for PDF charts - static distributions)
export function getDummyChipAccountingEvidenceSamples(numSamples: number = 100): number[] {
  // Likelihood ratios around 1.2 median, with spread
  return generateLogNormalSamples(numSamples, 1.3, 0.6);
}

export function getDummySMEAccountingEvidenceSamples(numSamples: number = 100): number[] {
  // Likelihood ratios around 1.15 median
  return generateLogNormalSamples(numSamples, 1.2, 0.5);
}

export function getDummyDatacenterAccountingEvidenceSamples(numSamples: number = 100): number[] {
  // Likelihood ratios around 1.1 median
  return generateLogNormalSamples(numSamples, 1.1, 0.4);
}

export function getDummyEnergyAccountingEvidenceSamples(numSamples: number = 100): number[] {
  // Likelihood ratios around 1.05 median (energy is harder to trace)
  return generateLogNormalSamples(numSamples, 1.05, 0.3);
}

export function getDummyCombinedAccountingEvidenceSamples(numSamples: number = 100): number[] {
  // Combined evidence - product of all sources, so higher median and wider spread
  return generateLogNormalSamples(numSamples, 1.8, 0.8);
}

export function getDummyDirectEvidenceSamples(numSamples: number = 100): number[] {
  // Direct evidence from HUMINT, SIGINT, etc.
  return generateLogNormalSamples(numSamples, 2.0, 0.9);
}

// Time series versions (kept for backwards compatibility and for charts that do vary over time)
export function getDummyChipAccountingEvidence(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  // Likelihood ratio from chip accounting discrepancies
  const data = generateExponentialGrowth(years, 1.0, 0.1, 0.3);
  return { years, ...data };
}

export function getDummySMEAccountingEvidence(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  const data = generateExponentialGrowth(years, 1.0, 0.08, 0.25);
  return { years, ...data };
}

export function getDummyDatacenterAccountingEvidence(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  const data = generateExponentialGrowth(years, 1.0, 0.05, 0.2);
  return { years, ...data };
}

export function getDummyEnergyAccountingEvidence(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  const data = generateExponentialGrowth(years, 1.0, 0.03, 0.15);
  return { years, ...data };
}

export function getDummyCombinedAccountingEvidence(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  const chip = getDummyChipAccountingEvidence(agreementYear);
  const sme = getDummySMEAccountingEvidence(agreementYear);
  const dc = getDummyDatacenterAccountingEvidence(agreementYear);
  const energy = getDummyEnergyAccountingEvidence(agreementYear);

  // Product of all evidence
  const median = years.map((_, i) =>
    chip.median[i] * sme.median[i] * dc.median[i] * energy.median[i]
  );
  const p25 = median.map(v => v * 0.5);
  const p75 = median.map(v => v * 1.5);

  return { years, median, p25, p75 };
}

export function getDummyDirectEvidence(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  // Evidence from HUMINT, SIGINT, etc. - grows over time
  const data = generateExponentialGrowth(years, 1.0, 0.15, 0.4);
  return { years, ...data };
}

export function getDummyPosteriorProbability(agreementYear: number = 2030): DummyChartData {
  const years = generateYears(agreementYear);
  // S-curve approaching 1 as evidence accumulates
  const data = generateSCurve(years, 3, 1.5);
  return { years, ...data };
}

// Historical data (static, not time series)
export const DUMMY_HISTORICAL_ACCURACY = {
  categories: ['Soviet Nuclear', 'Soviet Missiles', 'Iraq WMD', 'China Naval', 'Iran Nuclear'],
  estimates: [0.7, 0.6, 0.3, 0.8, 0.65],
  actual: [1.0, 1.0, 0.0, 1.0, 0.8],
};

export const DUMMY_DETECTION_LATENCY = {
  programs: ['Soviet Bomb', 'Pakistani Bomb', 'Iraqi Program', 'Syrian Reactor', 'NK Program'],
  yearsToDetect: [4, 6, 8, 2, 5],
  programSize: [10000, 5000, 3000, 500, 8000], // workers
};
