/**
 * DUMMY DATA - FOR VISUALIZATION TESTING ONLY
 *
 * This file contains fake/dummy data used to test chart visualizations.
 * Replace with actual API data when available.
 */

// Generate log-normal samples for histogram display
function generateLogNormalSamples(
  median: number,
  sigma: number = 0.5,
  count: number = 1000
): number[] {
  const samples: number[] = [];
  const mu = Math.log(median);

  for (let i = 0; i < count; i++) {
    // Box-Muller transform for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const logValue = mu + sigma * z;
    samples.push(Math.exp(logValue));
  }

  return samples;
}

// Generate normal samples
function generateNormalSamples(
  mean: number,
  stdDev: number,
  count: number = 1000
): number[] {
  const samples: number[] = [];

  for (let i = 0; i < count; i++) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    samples.push(mean + stdDev * z);
  }

  return samples;
}

export interface TimeSeriesPercentiles {
  p25: number[];
  median: number[];
  p75: number[];
}

export interface InitialStockDummyData {
  // Histogram samples
  initial_prc_stock_samples: number[];
  initial_compute_stock_samples: number[];
  diversion_proportion: number;
  lr_prc_accounting_samples: number[];
  initial_black_project_detection_probs: Record<string, number>;

  // Time series data for PRC compute over time
  prc_compute_years: number[];
  prc_compute_over_time: TimeSeriesPercentiles;
  prc_domestic_compute_over_time: { median: number[] };
  proportion_domestic_by_year: number[];
  largest_company_compute_over_time?: number[];
  state_of_the_art_energy_efficiency_relative_to_h100: number;
}

export function generateInitialStockDummyData(
  diversionProportion: number = 0.1,
  agreementYear: number = 2027
): InitialStockDummyData {
  // Generate years from 2025 to agreement year
  const startYear = 2025;
  const years: number[] = [];
  for (let y = startYear; y <= agreementYear; y++) {
    years.push(y);
  }

  // PRC compute stock growth (assuming ~1.5x annual growth from 1M H100e in 2025)
  const baseStock2025 = 1_000_000; // 1M H100e in 2025
  const annualGrowthRate = 1.5;

  const computeMedian: number[] = [];
  const computeP25: number[] = [];
  const computeP75: number[] = [];
  const domesticMedian: number[] = [];
  const proportionDomestic: number[] = [];
  const largestCompany: number[] = [];

  for (let i = 0; i < years.length; i++) {
    const yearsFromStart = years[i] - startYear;
    const median = baseStock2025 * Math.pow(annualGrowthRate, yearsFromStart);
    computeMedian.push(median);
    computeP25.push(median * 0.6); // 25th percentile ~60% of median
    computeP75.push(median * 1.5); // 75th percentile ~150% of median

    // Domestic production starts small and grows
    const domesticProp = Math.min(0.5, 0.1 + 0.1 * yearsFromStart);
    proportionDomestic.push(domesticProp);
    domesticMedian.push(median * domesticProp);

    // Largest AI company (e.g., Google/OpenAI) - grows faster
    largestCompany.push(baseStock2025 * 0.3 * Math.pow(1.8, yearsFromStart));
  }

  // PRC stock at agreement start (log-normal, based on projected growth)
  const finalYearIndex = years.length - 1;
  const prcStockMedian = computeMedian[finalYearIndex];
  const initial_prc_stock_samples = generateLogNormalSamples(prcStockMedian, 0.4, 1000);

  // Initial compute stock = PRC stock x diversion proportion
  const initial_compute_stock_samples = initial_prc_stock_samples.map(
    s => s * diversionProportion
  );

  // LR from PRC accounting (roughly normal around 1.5)
  const lr_prc_accounting_samples = generateNormalSamples(1.5, 0.4, 1000)
    .map(v => Math.max(1, v)); // LR >= 1

  // Detection probabilities by threshold
  const initial_black_project_detection_probs: Record<string, number> = {
    '1x': 0.85,
    '2x': 0.45,
    '4x': 0.12,
  };

  // State of the art energy efficiency relative to H100
  // Assuming ~2x improvement per year from H100 baseline
  const yearsFromH100 = agreementYear - 2023; // H100 released ~2023
  const state_of_the_art_energy_efficiency_relative_to_h100 = Math.pow(1.35, yearsFromH100);

  return {
    initial_prc_stock_samples,
    initial_compute_stock_samples,
    diversion_proportion: diversionProportion,
    lr_prc_accounting_samples,
    initial_black_project_detection_probs,
    prc_compute_years: years,
    prc_compute_over_time: {
      p25: computeP25,
      median: computeMedian,
      p75: computeP75,
    },
    prc_domestic_compute_over_time: {
      median: domesticMedian,
    },
    proportion_domestic_by_year: proportionDomestic,
    largest_company_compute_over_time: largestCompany,
    state_of_the_art_energy_efficiency_relative_to_h100,
  };
}

// Pre-generated dummy data for default display
export const DEFAULT_DUMMY_DATA = generateInitialStockDummyData(0.1, 2027);
