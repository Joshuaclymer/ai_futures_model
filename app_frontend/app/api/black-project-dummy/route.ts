import { NextRequest, NextResponse } from 'next/server';

// Dummy data for the black project sections
// This endpoint provides realistic test data while the real backend is being developed

// Default parameters (matching frontend defaults)
const defaultParams = {
  agreementYear: 2027,
  proportionOfInitialChipStockToDivert: 0.05,
  workersInCovertProject: 11300,
  fractionOfLaborDevotedToDatacenterConstruction: 0.885,
  fractionOfDatacenterCapacityToDivert: 0.5,
  maxFractionOfTotalNationalEnergyConsumption: 0.05,
  totalPrcEnergyConsumptionGw: 1100,
  priorOddsOfCovertProject: 0.3,
};

// Type for parameters passed from frontend
interface SimulationParameters {
  agreementYear?: number;
  proportionOfInitialChipStockToDivert?: number;
  workersInCovertProject?: number;
  fractionOfLaborDevotedToDatacenterConstruction?: number;
  fractionOfDatacenterCapacityToDivert?: number;
  maxFractionOfTotalNationalEnergyConsumption?: number;
  totalPrcEnergyConsumptionGw?: number;
  priorOddsOfCovertProject?: number;
  [key: string]: unknown;
}

// Handle POST requests with full parameters
export async function POST(request: NextRequest): Promise<NextResponse> {
  const params: SimulationParameters = await request.json();
  return generateResponse(params);
}

// Handle GET requests for backwards compatibility
export async function GET(request: NextRequest): Promise<NextResponse> {
  const params: SimulationParameters = {
    agreementYear: parseInt(request.nextUrl.searchParams.get('agreement_year') || '2027'),
    proportionOfInitialChipStockToDivert: parseFloat(request.nextUrl.searchParams.get('diversion_proportion') || '0.05'),
  };
  return generateResponse(params);
}

function generateResponse(params: SimulationParameters): NextResponse {
  // Merge with defaults
  const agreementYear = params.agreementYear ?? defaultParams.agreementYear;
  const diversionProportion = params.proportionOfInitialChipStockToDivert ?? defaultParams.proportionOfInitialChipStockToDivert;

  // Generate initial stock data
  const initialStockData = generateInitialStockData(diversionProportion, agreementYear);

  // Generate rate of computation data (for RateOfComputationSection)
  const rateOfComputationData = generateRateOfComputationData(agreementYear);

  // Generate covert fab section data
  const covertFabData = generateCovertFabData(agreementYear);

  // Generate detection likelihood section data
  const detectionLikelihoodData = generateDetectionLikelihoodData(agreementYear);

  const dummyData = {
    success: true,
    num_simulations: 100,

    // Initial stock section data
    initial_stock: initialStockData,

    // Rate of computation section data (HowWeEstimate)
    rate_of_computation: rateOfComputationData,

    // Covert fab section data
    covert_fab: covertFabData,

    // Detection likelihood section data
    detection_likelihood: detectionLikelihoodData,

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

      // CCDFs for main dashboard - includes 1x, 2x, 4x detection thresholds
      h100_years_ccdf: {
        '1': generateCcdfData(5000, 1000000, 20),   // 1x threshold - highest values
        '2': generateCcdfData(1000, 500000, 20),    // 2x threshold - medium values
        '4': generateCcdfData(100, 100000, 20),     // 4x threshold - lowest values (detected earlier)
      },
      time_to_detection_ccdf: {
        '1': generateCcdfData(1.0, 7, 15),   // 1x threshold - longer time to detection
        '2': generateCcdfData(0.5, 5, 15),   // 2x threshold
        '4': generateCcdfData(0.2, 3, 15),   // 4x threshold - shorter time (detected earlier)
      },
      // CCDFs for chip production relative to no-slowdown scenarios
      // P(Ratio < x) - probability covert production is less than x times no-slowdown production
      // Keys: 'global' = relative to global production, 'prc' = relative to PRC production
      chip_production_reduction_ccdf: {
        'global': [
          { x: 1, y: 1.0 }, { x: 0.1, y: 0.995 }, { x: 0.01, y: 0.98 },
          { x: 0.001, y: 0.95 }, { x: 0.0001, y: 0.88 },
        ],
        'prc': [
          { x: 1, y: 1.0 }, { x: 0.1, y: 0.98 }, { x: 0.01, y: 0.92 },
          { x: 0.001, y: 0.82 }, { x: 0.0001, y: 0.65 },
        ],
      },
      // CCDFs for AI R&D computation relative to no-slowdown scenarios
      // Keys: 'largest_ai_company' = relative to largest AI company, 'prc' = relative to PRC
      ai_rd_reduction_ccdf: {
        'largest_ai_company': [
          { x: 1, y: 1.0 }, { x: 0.1, y: 0.995 }, { x: 0.01, y: 0.95 },
          { x: 0.001, y: 0.75 }, { x: 0.0001, y: 0.22 },
        ],
        'prc': [
          { x: 1, y: 1.0 }, { x: 0.1, y: 0.98 }, { x: 0.01, y: 0.88 },
          { x: 0.001, y: 0.65 }, { x: 0.0001, y: 0.20 },
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

// Generate years array with quarterly intervals
function generateQuarterlyYears(agreementYear: number, numYears: number = 7): number[] {
  return Array.from({ length: numYears * 4 + 1 }, (_, i) => agreementYear + i * 0.25);
}

// Generate exponential growth data
function generateExponentialGrowthData(
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
function generateDecayData(
  years: number[],
  halfLife: number = 5
): { median: number[]; p25: number[]; p75: number[] } {
  const agreementYear = years[0];
  const median = years.map(y => Math.pow(0.5, (y - agreementYear) / halfLife));
  const p25 = median.map(v => Math.max(0, v - 0.05));
  const p75 = median.map(v => Math.min(1, v + 0.05));
  return { median, p25, p75 };
}

// Generate linear growth data
function generateLinearGrowthData(
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

// Generate S-curve data with confidence intervals
function generateSCurveData(
  years: number[],
  midpoint: number = 2,
  steepness: number = 2,
  noise: number = 0.15
): { median: number[]; p25: number[]; p75: number[] } {
  const agreementYear = years[0];
  const median = years.map(y => {
    const t = y - agreementYear;
    return 1 / (1 + Math.exp(-steepness * (t - midpoint)));
  });
  // For probabilities, shift the midpoint to create confidence intervals
  const p25 = years.map(y => {
    const t = y - agreementYear;
    return 1 / (1 + Math.exp(-steepness * (t - midpoint - noise * 2)));
  });
  const p75 = years.map(y => {
    const t = y - agreementYear;
    return 1 / (1 + Math.exp(-steepness * (t - midpoint + noise * 2)));
  });
  return { median, p25, p75 };
}

// Generate rate of computation section data (for RateOfComputationSection)
function generateRateOfComputationData(agreementYear: number) {
  const years = generateQuarterlyYears(agreementYear);

  // Initial chip stock samples (log-normal distribution)
  const initialChipStockSamples = generateLogNormalSamples(50000, 0.3, 100);

  // Acquired hardware (exponential growth)
  const acquiredHardware = generateExponentialGrowthData(years, 1000, 0.3, 0.2);

  // Surviving fraction (decay)
  const survivingFraction = generateDecayData(years, 6);

  // Covert chip stock (combination)
  const initialValue = 50000;
  const covertChipStock = {
    median: years.map((_, i) =>
      (initialValue + acquiredHardware.median[i]) * survivingFraction.median[i]
    ),
    p25: [] as number[],
    p75: [] as number[],
  };
  covertChipStock.p25 = covertChipStock.median.map(v => v * 0.7);
  covertChipStock.p75 = covertChipStock.median.map(v => v * 1.3);

  // Datacenter capacity (linear growth)
  const datacenterCapacity = generateLinearGrowthData(years, 5, 8, 0.15);

  // Energy per chip (GW per H100e)
  const energyPerChip = 0.0005; // GW per chip

  // Stacked energy data: [initialStockEnergy, fabProducedEnergy] per year
  // Initial stock energy decays with surviving fraction
  // Fab-produced energy comes from acquired hardware
  const energyStackedData: [number, number][] = years.map((_, i) => {
    const initialStockEnergy = initialValue * survivingFraction.median[i] * energyPerChip;
    const fabProducedEnergy = acquiredHardware.median[i] * survivingFraction.median[i] * energyPerChip;
    return [initialStockEnergy, fabProducedEnergy];
  });

  // Time series for backward compatibility
  const energyUsage = {
    median: covertChipStock.median.map(v => v * energyPerChip),
    p25: [] as number[],
    p75: [] as number[],
  };
  energyUsage.p25 = energyUsage.median.map(v => v * 0.8);
  energyUsage.p75 = energyUsage.median.map(v => v * 1.2);

  // Operating chips (min of chip stock and datacenter capacity)
  const operatingChips = {
    median: years.map((_, i) => {
      const chipsFromEnergy = datacenterCapacity.median[i] / energyPerChip;
      return Math.min(covertChipStock.median[i], chipsFromEnergy);
    }),
    p25: [] as number[],
    p75: [] as number[],
  };
  operatingChips.p25 = operatingChips.median.map(v => v * 0.7);
  operatingChips.p75 = operatingChips.median.map(v => v * 1.3);

  // Covert computation (cumulative H100-years)
  let cumulative = 0;
  const covertComputation = {
    median: operatingChips.median.map(v => {
      cumulative += v * 0.25;
      return cumulative;
    }),
    p25: [] as number[],
    p75: [] as number[],
  };
  covertComputation.p25 = covertComputation.median.map(v => v * 0.6);
  covertComputation.p75 = covertComputation.median.map(v => v * 1.4);

  return {
    years,
    initial_chip_stock_samples: initialChipStockSamples,
    acquired_hardware: { years, ...acquiredHardware },
    surviving_fraction: { years, ...survivingFraction },
    covert_chip_stock: { years, ...covertChipStock },
    datacenter_capacity: { years, ...datacenterCapacity },
    energy_usage: { years, ...energyUsage },
    energy_stacked_data: energyStackedData,
    energy_source_labels: ['Initial Stock', 'Fab-Produced'] as [string, string],
    operating_chips: { years, ...operatingChips },
    covert_computation: { years, ...covertComputation },
  };
}

// Generate covert fab section data
function generateCovertFabData(agreementYear: number) {
  const years = generateQuarterlyYears(agreementYear);

  // Dashboard data
  const dashboard = {
    production: '800K H100e',
    energy: '2.8 GW',
    probFabBuilt: '56.2%',
    yearsOperational: '0.8',
    processNode: '28nm',
  };

  // CCDF data for covert compute produced before detection
  const computeCCDF: Record<number, { x: number; y: number }[]> = {};
  for (const threshold of [1, 2, 4]) {
    const shift = threshold === 4 ? 0 : threshold === 2 ? 0.1 : 0.3;
    const points: { x: number; y: number }[] = [];
    for (let i = 0; i <= 100; i++) {
      const x = Math.pow(10, 2 + i * 0.05 + shift);
      const y = Math.max(0, 1 - Math.pow(i / 100, 0.5));
      points.push({ x, y });
    }
    computeCCDF[threshold] = points;
  }

  // Time series data for simulation runs
  const lrMedian = years.map((_, i) => Math.pow(1.5, i / 4));
  const lrP25 = lrMedian.map(v => v * 0.7);
  const lrP75 = lrMedian.map(v => v * 1.4);

  const h100eMedian = years.map((_, i) => {
    const t = i / 4;
    return t < 1 ? 0 : Math.min(500000, 100000 * (t - 1));
  });
  const h100eP25 = h100eMedian.map(v => v * 0.6);
  const h100eP75 = h100eMedian.map(v => v * 1.4);

  const timeSeriesData = {
    years,
    lr_combined: { years, median: lrMedian, p25: lrP25, p75: lrP75 },
    h100e_flow: { years, median: h100eMedian, p25: h100eP25, p75: h100eP75 },
  };

  // Is operational (S-curve)
  const isOperational = generateSCurveData(years, 2, 2);

  // Wafer starts (distribution samples)
  const waferStartsSamples: number[] = [];
  for (let i = 0; i < 100; i++) {
    waferStartsSamples.push(5000 * (0.5 + Math.random() * 1.5));
  }
  waferStartsSamples.sort((a, b) => a - b);

  // Constants
  const chipsPerWafer = 28;
  const architectureEfficiency = 0.85;
  const h100Power = 700;

  // Transistor density by process node with pre-computed watts per TPP
  // Watts per TPP formula: density^-0.5 before Dennard threshold (0.02),
  // then wattsAtThreshold * (density/threshold)^-0.15 after
  const dennardThreshold = 0.02;
  const computeWattsPerTpp = (density: number): number => {
    if (density < dennardThreshold) {
      return Math.pow(density, -0.5);
    } else {
      const wattsAtThreshold = Math.pow(dennardThreshold, -0.5);
      return wattsAtThreshold * Math.pow(density / dennardThreshold, -0.15);
    }
  };

  const transistorDensity = [
    { node: '28nm', density: 0.14, wattsPerTpp: computeWattsPerTpp(0.14) },
    { node: '14nm', density: 0.5, wattsPerTpp: computeWattsPerTpp(0.5) },
    { node: '7nm', density: 1.0, wattsPerTpp: computeWattsPerTpp(1.0) },
  ];

  // Compute per month
  const baseValue = 5000 * 28 * 0.14 * 0.85;
  const computePerMonth = {
    years,
    median: years.map((_, i) => isOperational.median[i] * baseValue),
    p25: [] as number[],
    p75: [] as number[],
  };
  computePerMonth.p25 = computePerMonth.median.map(v => v * 0.6);
  computePerMonth.p75 = computePerMonth.median.map(v => v * 1.5);

  // Watts per TPP curve (uses the same computeWattsPerTpp function defined above)
  const wattsPerTppCurve: { densityRelative: number[]; wattsPerTppRelative: number[] } = {
    densityRelative: [],
    wattsPerTppRelative: [],
  };
  for (let i = 0; i <= 50; i++) {
    const density = Math.pow(10, -3 + i * 0.1);
    wattsPerTppCurve.densityRelative.push(density);
    wattsPerTppCurve.wattsPerTppRelative.push(computeWattsPerTpp(density));
  }

  // Energy per month
  const wattsPerTppRelative = 3;
  const energyPerMonth = {
    years,
    median: computePerMonth.median.map(v => (v * wattsPerTppRelative * h100Power) / 1e9),
    p25: [] as number[],
    p75: [] as number[],
  };
  energyPerMonth.p25 = energyPerMonth.median.map(v => v * 0.6);
  energyPerMonth.p75 = energyPerMonth.median.map(v => v * 1.5);

  return {
    dashboard,
    compute_ccdf: computeCCDF,
    time_series_data: timeSeriesData,
    is_operational: { years, ...isOperational },
    wafer_starts_samples: waferStartsSamples,
    chips_per_wafer: chipsPerWafer,
    architecture_efficiency: architectureEfficiency,
    h100_power: h100Power,
    transistor_density: transistorDensity,
    compute_per_month: computePerMonth,
    watts_per_tpp_curve: wattsPerTppCurve,
    energy_per_month: energyPerMonth,
  };
}

// Generate detection likelihood section data
function generateDetectionLikelihoodData(agreementYear: number) {
  const years = generateQuarterlyYears(agreementYear);

  // PDF samples for static evidence sources
  const chipEvidenceSamples = generateLogNormalSamples(1.3, 0.6, 100);
  const smeEvidenceSamples = generateLogNormalSamples(1.2, 0.5, 100);
  const dcEvidenceSamples = generateLogNormalSamples(1.1, 0.4, 100);

  // Energy evidence (time series - grows over time)
  const energyEvidence = generateExponentialGrowthData(years, 1.0, 0.03, 0.15);

  // Combined accounting evidence (product of all sources)
  const combinedEvidence = generateExponentialGrowthData(years, 1.0, 0.15, 0.4);

  // Direct evidence (grows over time from HUMINT, SIGINT, etc.)
  const directEvidence = generateExponentialGrowthData(years, 1.0, 0.15, 0.4);

  // Posterior probability (S-curve approaching 1)
  const posteriorProb = generateSCurveData(years, 3, 1.5);

  return {
    years,
    chip_evidence_samples: chipEvidenceSamples,
    sme_evidence_samples: smeEvidenceSamples,
    dc_evidence_samples: dcEvidenceSamples,
    energy_evidence: { years, ...energyEvidence },
    combined_evidence: { years, ...combinedEvidence },
    direct_evidence: { years, ...directEvidence },
    posterior_prob: { years, ...posteriorProb },
  };
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

  // Calculate energy requirements samples from compute stock samples
  // Energy = (chips * H100_power_watts / efficiency) / 1e9 (GW)
  const H100_POWER_WATTS = 700;
  const prcEfficiencyRelativeToSOTA = 0.20; // Default PRC efficiency
  const combinedEfficiency = state_of_the_art_energy_efficiency_relative_to_h100 * prcEfficiencyRelativeToSOTA;
  const initial_energy_samples = initial_compute_stock_samples.map(
    h100e => (h100e * H100_POWER_WATTS) / combinedEfficiency / 1e9
  );

  return {
    initial_prc_stock_samples,
    initial_compute_stock_samples,
    initial_energy_samples,
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
