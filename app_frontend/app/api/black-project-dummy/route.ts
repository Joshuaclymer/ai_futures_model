import { NextRequest, NextResponse } from 'next/server';

// Dummy data for the black project sections
// This endpoint provides realistic test data while the real backend is being developed

export async function GET(request: NextRequest): Promise<NextResponse> {
  const agreementYear = parseInt(request.nextUrl.searchParams.get('agreement_year') || '2027');
  const diversionProportion = parseFloat(request.nextUrl.searchParams.get('diversion_proportion') || '0.05');

  // Generate initial stock data
  const initialStockData = generateInitialStockData(diversionProportion, agreementYear);

  const dummyData = {
    success: true,
    num_simulations: 100,

    // Initial stock section data
    initial_stock: initialStockData,

    // Datacenter section data
    black_datacenters: {
      // Individual simulation results for dashboard calculations
      individual_capacity_before_detection: generateRandomArray(100, 30, 70),
      individual_time_before_detection: generateRandomArray(100, 0.5, 2.0),

      // Time series data
      years: generateYears(agreementYear, agreementYear + 8),
      datacenter_capacity: {
        median: [0, 5, 15, 30, 45, 55, 55, 55, 55],
        p25: [0, 3, 10, 22, 35, 45, 45, 45, 45],
        p75: [0, 7, 20, 38, 55, 65, 65, 65, 65],
      },
      lr_datacenters: {
        median: [1, 1.2, 1.5, 2, 3, 5, 8, 12, 20],
        p25: [1, 1.1, 1.3, 1.6, 2.2, 3.5, 5.5, 8, 14],
        p75: [1, 1.3, 1.8, 2.5, 4, 7, 12, 18, 30],
      },

      // CCDF data for capacity built before detection
      capacity_ccdfs: {
        '1': [
          { x: 5, y: 1.0 }, { x: 10, y: 0.98 }, { x: 15, y: 0.95 },
          { x: 20, y: 0.90 }, { x: 25, y: 0.82 }, { x: 30, y: 0.72 },
          { x: 35, y: 0.60 }, { x: 40, y: 0.48 }, { x: 45, y: 0.36 },
          { x: 50, y: 0.25 }, { x: 55, y: 0.16 }, { x: 60, y: 0.09 },
          { x: 65, y: 0.04 }, { x: 70, y: 0.02 }, { x: 80, y: 0.01 },
        ],
        '4': [
          { x: 5, y: 1.0 }, { x: 10, y: 0.96 }, { x: 15, y: 0.90 },
          { x: 20, y: 0.82 }, { x: 25, y: 0.70 }, { x: 30, y: 0.58 },
          { x: 35, y: 0.45 }, { x: 40, y: 0.34 }, { x: 45, y: 0.24 },
          { x: 50, y: 0.16 }, { x: 55, y: 0.10 }, { x: 60, y: 0.05 },
          { x: 65, y: 0.02 }, { x: 70, y: 0.01 }, { x: 80, y: 0.005 },
        ],
      },

      // PRC capacity data
      prc_capacity_years: generateYears(2020, agreementYear),
      prc_capacity_gw: generateGrowthData(2020, agreementYear, 50, 1.15),
      prc_capacity_at_agreement_year_gw: 132,

      // Parameters
      fraction_diverted: 0.01,
      total_prc_energy_gw: 1000,
      max_proportion_energy: 0.01,
      construction_workers: 10000,
      mw_per_worker_per_year: 0.2,
      datacenter_start_year: agreementYear - 2,
    },

    // Fab section data
    black_fab: {
      years: generateYears(agreementYear, agreementYear + 8),
      // Individual simulation results for dashboard
      individual_energy_before_detection: generateRandomArray(100, 1.5, 4.5),
      individual_production_before_detection: generateRandomArray(100, 50000, 200000),

      // Time series for fab production
      wafer_starts: {
        median: [0, 0, 100, 500, 1200, 2000, 2500, 2800, 3000],
        p25: [0, 0, 50, 300, 800, 1400, 1800, 2000, 2200],
        p75: [0, 0, 150, 700, 1600, 2600, 3200, 3600, 3800],
      },
      lr_fab: {
        median: [1, 1.1, 1.3, 1.8, 2.5, 4, 6, 10, 15],
        p25: [1, 1.05, 1.15, 1.4, 1.8, 2.5, 3.5, 6, 9],
        p75: [1, 1.15, 1.5, 2.2, 3.5, 6, 10, 16, 25],
      },

      // CCDF data for fab metrics
      production_ccdfs: {
        '1': generateCcdfData(50000, 250000, 15),
        '4': generateCcdfData(30000, 200000, 15),
      },
      energy_ccdfs: {
        '1': generateCcdfData(1, 6, 15),
        '4': generateCcdfData(0.5, 5, 15),
      },

      // Fab parameters
      fab_construction_time: 2.5,
      architecture_efficiency: 0.8,
      wafers_per_scanner: 1000,
    },

    // Main project model data
    black_project_model: {
      years: generateYears(agreementYear, agreementYear + 8),
      // Overall project metrics
      individual_project_h100_years_before_detection: generateRandomArray(100, 5000, 25000),
      individual_project_time_before_detection: generateRandomArray(100, 0.8, 3.0),
      individual_project_h100e_before_detection: generateRandomArray(100, 8000, 20000),

      // CCDFs for main dashboard
      h100_years_ccdf: {
        '2': generateCcdfData(1000, 50000, 20),
        '4': generateCcdfData(500, 40000, 20),
      },
      time_to_detection_ccdf: {
        '2': generateCcdfData(0.5, 5, 15),
        '4': generateCcdfData(0.3, 4, 15),
      },
      // CCDFs for chip production and AI R&D reduction
      chip_production_reduction_ccdf: {
        '2': [
          { x: 0.001, y: 1.0 }, { x: 0.005, y: 0.95 }, { x: 0.01, y: 0.88 },
          { x: 0.02, y: 0.78 }, { x: 0.05, y: 0.60 }, { x: 0.1, y: 0.42 },
          { x: 0.15, y: 0.28 }, { x: 0.2, y: 0.18 }, { x: 0.3, y: 0.08 },
        ],
        '4': [
          { x: 0.001, y: 1.0 }, { x: 0.005, y: 0.92 }, { x: 0.01, y: 0.82 },
          { x: 0.02, y: 0.68 }, { x: 0.05, y: 0.48 }, { x: 0.1, y: 0.30 },
          { x: 0.15, y: 0.18 }, { x: 0.2, y: 0.10 }, { x: 0.3, y: 0.04 },
        ],
      },
      ai_rd_reduction_ccdf: {
        '2': [
          { x: 0.001, y: 1.0 }, { x: 0.005, y: 0.96 }, { x: 0.01, y: 0.90 },
          { x: 0.02, y: 0.82 }, { x: 0.05, y: 0.65 }, { x: 0.1, y: 0.48 },
          { x: 0.2, y: 0.28 }, { x: 0.3, y: 0.15 }, { x: 0.4, y: 0.06 },
        ],
        '4': [
          { x: 0.001, y: 1.0 }, { x: 0.005, y: 0.94 }, { x: 0.01, y: 0.85 },
          { x: 0.02, y: 0.72 }, { x: 0.05, y: 0.52 }, { x: 0.1, y: 0.35 },
          { x: 0.2, y: 0.18 }, { x: 0.3, y: 0.08 }, { x: 0.4, y: 0.03 },
        ],
      },
      ai_rd_reduction_median: 0.05,
    },
  };

  return NextResponse.json(dummyData);
}

// Helper functions
function generateYears(start: number, end: number): number[] {
  const years: number[] = [];
  for (let y = start; y <= end; y++) {
    years.push(y);
  }
  return years;
}

function generateRandomArray(length: number, min: number, max: number): number[] {
  const arr: number[] = [];
  for (let i = 0; i < length; i++) {
    arr.push(min + Math.random() * (max - min));
  }
  return arr.sort((a, b) => a - b);
}

function generateGrowthData(startYear: number, endYear: number, startValue: number, growthRate: number) {
  const years = endYear - startYear + 1;
  const median: number[] = [];
  const p25: number[] = [];
  const p75: number[] = [];

  for (let i = 0; i < years; i++) {
    const baseValue = startValue * Math.pow(growthRate, i);
    median.push(baseValue);
    p25.push(baseValue * 0.8);
    p75.push(baseValue * 1.2);
  }

  return { median, p25, p75 };
}

function generateCcdfData(minX: number, maxX: number, numPoints: number): { x: number; y: number }[] {
  const data: { x: number; y: number }[] = [];
  const step = (maxX - minX) / (numPoints - 1);

  for (let i = 0; i < numPoints; i++) {
    const x = minX + step * i;
    // CCDF decreases from 1 to near 0, using exponential decay
    const y = Math.exp(-3 * (i / (numPoints - 1)));
    data.push({ x: Math.round(x * 100) / 100, y: Math.round(y * 1000) / 1000 });
  }

  return data;
}

// Generate log-normal samples
function generateLogNormalSamples(median: number, sigma: number = 0.5, count: number = 1000): number[] {
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
function generateNormalSamples(mean: number, stdDev: number, count: number = 1000): number[] {
  const samples: number[] = [];

  for (let i = 0; i < count; i++) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    samples.push(mean + stdDev * z);
  }

  return samples;
}

// Generate initial stock section data
function generateInitialStockData(diversionProportion: number, agreementYear: number) {
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
    computeP25.push(median * 0.6);
    computeP75.push(median * 1.5);

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
    .map(v => Math.max(1, v));

  // Detection probabilities by threshold
  const initial_black_project_detection_probs: Record<string, number> = {
    '1x': 0.85,
    '2x': 0.45,
    '4x': 0.12,
  };

  // State of the art energy efficiency relative to H100
  const yearsFromH100 = agreementYear - 2023;
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
