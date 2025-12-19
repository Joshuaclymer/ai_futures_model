import path from "node:path";
import { readFile } from "node:fs/promises";
import { Suspense } from "react";
import { cacheLife, cacheTag } from "next/cache";
import yaml from "js-yaml";
import { MilestoneDistributionPoint, MilestoneDistributionData, MilestoneCdfPoint } from "@/components/MilestoneDistributionChart";
import { ConditionalTimingPoint, ConditionalTimingData, ConditionalCdfPoint } from "@/components/ConditionalMilestoneTimingChart";
import { ForecastChartsSection } from "@/components/ForecastChartsSection";
import { HeaderContent } from "@/components/HeaderContent";
import { SimulationSelector, SimulationMetadata, Simulation } from "@/components/SimulationSelector";
import { SamplingDistributionsButton } from "@/components/SamplingDistributionsButton";
import type { SamplingConfig } from "@/types/samplingConfig";

function parseCsvLine(line: string): string[] {
  const fields: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === "\"") {
      inQuotes = !inQuotes;
      continue;
    }

    if (char === "," && !inQuotes) {
      fields.push(current.trim());
      current = "";
      continue;
    }

    current += char;
  }

  fields.push(current.trim());

  return fields.map(value => value.replace(/^"|"$/g, ""));
}

interface MilestoneStatistics {
  [key: string]: {
    achievementRate: number;
    mode: number;
    p10: number;
    p50: number;
    p90: number;
  };
}

// Interpolate CDF value at a specific year from CDF data points
function interpolateCdfAtYear(cdfPoints: { year: number; cdf: number }[], targetYear: number): number {
  if (cdfPoints.length === 0) return 0;

  // Sort by year
  const sorted = [...cdfPoints].sort((a, b) => a.year - b.year);

  // If target is before first point, return 0
  if (targetYear <= sorted[0].year) return sorted[0].cdf;

  // If target is after last point, return last CDF value
  if (targetYear >= sorted[sorted.length - 1].year) return sorted[sorted.length - 1].cdf;

  // Find surrounding points and interpolate
  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i].year <= targetYear && sorted[i + 1].year >= targetYear) {
      const t = (targetYear - sorted[i].year) / (sorted[i + 1].year - sorted[i].year);
      return sorted[i].cdf + t * (sorted[i + 1].cdf - sorted[i].cdf);
    }
  }

  return 0;
}

// Interpolate CDF value at a specific time (years from AC) from conditional CDF data
function interpolateCdfAtTime(cdfPoints: { timeFromAC: number; cdf: number }[], targetTime: number): number {
  if (cdfPoints.length === 0) return 0;

  // Sort by time
  const sorted = [...cdfPoints].sort((a, b) => a.timeFromAC - b.timeFromAC);

  // If target is before first point, return 0
  if (targetTime <= sorted[0].timeFromAC) return sorted[0].cdf;

  // If target is after last point, return last CDF value
  if (targetTime >= sorted[sorted.length - 1].timeFromAC) return sorted[sorted.length - 1].cdf;

  // Find surrounding points and interpolate
  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i].timeFromAC <= targetTime && sorted[i + 1].timeFromAC >= targetTime) {
      const t = (targetTime - sorted[i].timeFromAC) / (sorted[i + 1].timeFromAC - sorted[i].timeFromAC);
      return sorted[i].cdf + t * (sorted[i + 1].cdf - sorted[i].cdf);
    }
  }

  return 0;
}

// Format probability as percentage string
function formatProbability(prob: number): string {
  return `${(prob * 100).toFixed(1)}%`;
}

export interface ArrivalProbabilities {
  milestones: {
    name: string;
    displayName: string;
    probabilities: { [year: string]: number };
  }[];
  years: number[];
}

export interface TakeoffProbabilities {
  conditions: {
    label: string;
    probabilities: { [metric: string]: number };
  }[];
  metrics: { key: string; label: string }[];
}

interface SimulationRegistry {
  version: number;
  defaultSimulation: string | null;
  simulations: Simulation[];
}

async function loadSimulationRegistry(): Promise<SimulationRegistry> {
  const registryPath = path.join(process.cwd(), "app/forecast/data/simulations/simulations.json");
  try {
    const raw = await readFile(registryPath, "utf8");
    return JSON.parse(raw) as SimulationRegistry;
  } catch {
    // Return empty registry if file doesn't exist
    return {
      version: 1,
      defaultSimulation: null,
      simulations: [],
    };
  }
}

interface SamplingConfigWithRaw {
  config: SamplingConfig;
  rawYaml: string;
}

async function loadSamplingConfig(basePath: string): Promise<SamplingConfigWithRaw | null> {
  const configPath = path.join(process.cwd(), basePath, "sampling_config.yaml");
  try {
    const rawYaml = await readFile(configPath, "utf8");
    const config = yaml.load(rawYaml) as SamplingConfig;
    return { config, rawYaml };
  } catch {
    // Return null if file doesn't exist (for older simulations)
    return null;
  }
}

async function loadMilestoneDistributions(basePath: string): Promise<MilestoneDistributionData> {
  // Load the overlay distributions file
  const filePath = path.join(process.cwd(), basePath, "milestone_pdfs_overlay_distributions.csv");
  const raw = await readFile(filePath, "utf8");

  const lines = raw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [header, ...dataLines] = lines;
  const headers = header.split(",");
  const milestoneNames = headers.slice(1); // Skip "time_decimal_year"

  // Load statistics
  const statsPath = path.join(process.cwd(), basePath, "milestone_pdfs_overlay_statistics.csv");
  const statsRaw = await readFile(statsPath, "utf8");

  const statsLines = statsRaw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [statsHeader, ...statsDataLines] = statsLines;
  const statsHeaders = parseCsvLine(statsHeader);

  const milestoneIndex = statsHeaders.indexOf("milestone_name");
  const achievementRateIndex = statsHeaders.indexOf("achievement_rate_pct");
  const modeIndex = statsHeaders.indexOf("mode");
  const p10Index = statsHeaders.indexOf("p10");
  const p50Index = statsHeaders.indexOf("p50");
  const p90Index = statsHeaders.indexOf("p90");

  const statistics: MilestoneStatistics = {};

  for (const line of statsDataLines) {
    const fields = parseCsvLine(line);
    const milestoneName = fields[milestoneIndex];

    if (milestoneName) {
      statistics[milestoneName] = {
        achievementRate: Number.parseFloat(fields[achievementRateIndex]) / 100,
        mode: Number.parseFloat(fields[modeIndex]),
        p10: Number.parseFloat(fields[p10Index]),
        p50: Number.parseFloat(fields[p50Index]),
        p90: Number.parseFloat(fields[p90Index]),
      };
    }
  }

  // Parse distribution data for each milestone
  const milestones: { [key: string]: MilestoneDistributionPoint[] } = {};

  for (const milestoneName of milestoneNames) {
    milestones[milestoneName] = [];
  }

  for (const line of dataLines) {
    const values = line.split(",");
    const year = Number.parseFloat(values[0]);

    if (!Number.isFinite(year)) continue;

    for (let i = 0; i < milestoneNames.length; i += 1) {
      const density = Number.parseFloat(values[i + 1]);

      if (Number.isFinite(density)) {
        milestones[milestoneNames[i]].push({
          year,
          probabilityDensity: density,
        });
      }
    }
  }

  // Load empirical CDF data if available
  const cdf: { [key: string]: MilestoneCdfPoint[] } = {};
  const cdfPath = path.join(process.cwd(), basePath, "milestone_pdfs_overlay_cdf.csv");
  try {
    const cdfRaw = await readFile(cdfPath, "utf8");
    const cdfLines = cdfRaw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line.length > 0);

    const [cdfHeader, ...cdfDataLines] = cdfLines;
    const cdfHeaders = cdfHeader.split(",");
    const cdfMilestoneNames = cdfHeaders.slice(1);

    for (const milestoneName of cdfMilestoneNames) {
      cdf[milestoneName] = [];
    }

    for (const line of cdfDataLines) {
      const values = line.split(",");
      const year = Number.parseFloat(values[0]);

      if (!Number.isFinite(year)) continue;

      for (let i = 0; i < cdfMilestoneNames.length; i += 1) {
        const cdfValue = Number.parseFloat(values[i + 1]);

        if (Number.isFinite(cdfValue)) {
          cdf[cdfMilestoneNames[i]].push({
            year,
            cdf: cdfValue,
          });
        }
      }
    }
  } catch {
    // CDF file doesn't exist - fallback to PDF-derived CDF
  }

  // Normalize each milestone distribution by its achievement rate
  for (const milestoneName of milestoneNames) {
    const points = milestones[milestoneName];
    const stats = statistics[milestoneName];

    if (!stats) continue;

    const sorted = points.sort((a, b) => a.year - b.year);
    const totalDensity = sorted.reduce((acc, point) => acc + point.probabilityDensity, 0);

    if (totalDensity > 0) {
      milestones[milestoneName] = sorted.map(point => ({
        ...point,
        probabilityDensity: (point.probabilityDensity / totalDensity) * stats.achievementRate,
      }));
    }
  }

  // Filter to only show specific milestones
  // Remove milestones from this array to omit them from the chart
  const milestonesToShow = [
    'AC',
    // 'AI2027-SC',
    'SAR-level-experiment-selection-skill',
    // 'SIAR-level-experiment-selection-skill',
  ];

  const filteredMilestones: { [key: string]: MilestoneDistributionPoint[] } = {};
  const filteredStatistics: MilestoneStatistics = {};
  const filteredCdf: { [key: string]: MilestoneCdfPoint[] } = {};

  for (const milestoneName of milestonesToShow) {
    if (milestones[milestoneName]) {
      filteredMilestones[milestoneName] = milestones[milestoneName];
    }
    if (statistics[milestoneName]) {
      filteredStatistics[milestoneName] = statistics[milestoneName];
    }
    if (cdf[milestoneName]) {
      filteredCdf[milestoneName] = cdf[milestoneName];
    }
  }

  return {
    milestones: filteredMilestones,
    statistics: filteredStatistics,
    cdf: Object.keys(filteredCdf).length > 0 ? filteredCdf : undefined,
  };
}

interface ConditionalTimingStatistics {
  [key: string]: {
    achievementRate: number;
    mode: number;
    p10: number;
    p50: number;
    p90: number;
  };
}

async function loadConditionalTimingDistributions(basePath: string, year: number): Promise<ConditionalTimingData> {
  // Load the AC conditional timing distributions for the specified year
  const filePath = path.join(process.cwd(), basePath, `ac_${year}_time_until_distributions.csv`);

  let raw: string;
  try {
    raw = await readFile(filePath, "utf8");
  } catch {
    // File doesn't exist - return empty data
    return {
      milestones: {},
      statistics: {},
      conditionDescription: `Time until milestone achievement, conditional on achieving AC (Automated Coder) in ${year} (no data available)`,
    };
  }

  const lines = raw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [header, ...dataLines] = lines;
  const headers = header.split(",");
  const milestoneNames = headers.slice(1); // Skip "time_decimal_year"

  // Load statistics
  const statsPath = path.join(process.cwd(), basePath, `ac_${year}_time_until_statistics.csv`);
  let statsRaw: string;
  try {
    statsRaw = await readFile(statsPath, "utf8");
  } catch {
    // Stats file doesn't exist - return data without statistics
    statsRaw = "";
  }

  const statsLines = statsRaw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [statsHeader, ...statsDataLines] = statsLines;
  const statsHeaders = parseCsvLine(statsHeader);

  const milestoneIndex = statsHeaders.indexOf("milestone_name");
  const achievementRateIndex = statsHeaders.indexOf("achievement_rate_pct");
  const modeIndex = statsHeaders.indexOf("mode");
  const p10Index = statsHeaders.indexOf("p10");
  const p50Index = statsHeaders.indexOf("p50");
  const p90Index = statsHeaders.indexOf("p90");

  const statistics: ConditionalTimingStatistics = {};

  for (const line of statsDataLines) {
    const fields = parseCsvLine(line);
    const milestoneName = fields[milestoneIndex];

    if (milestoneName) {
      statistics[milestoneName] = {
        achievementRate: Number.parseFloat(fields[achievementRateIndex]) / 100,
        mode: Number.parseFloat(fields[modeIndex]),
        p10: Number.parseFloat(fields[p10Index]),
        p50: Number.parseFloat(fields[p50Index]),
        p90: Number.parseFloat(fields[p90Index]),
      };
    }
  }

  // Parse distribution data for each milestone
  const milestones: { [key: string]: ConditionalTimingPoint[] } = {};

  for (const milestoneName of milestoneNames) {
    milestones[milestoneName] = [];
  }

  for (const line of dataLines) {
    const values = line.split(",");
    const timeFromAC = Number.parseFloat(values[0]);

    if (!Number.isFinite(timeFromAC)) continue;

    for (let i = 0; i < milestoneNames.length; i += 1) {
      const density = Number.parseFloat(values[i + 1]);

      if (Number.isFinite(density) && density > 0) {
        milestones[milestoneNames[i]].push({
          timeFromAC,
          probabilityDensity: density,
        });
      }
    }
  }

  // Load empirical CDF data if available
  const cdf: { [key: string]: ConditionalCdfPoint[] } = {};
  const cdfPath = path.join(process.cwd(), basePath, `ac_${year}_time_until_cdf.csv`);
  try {
    const cdfRaw = await readFile(cdfPath, "utf8");
    const cdfLines = cdfRaw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line.length > 0);

    const [cdfHeader, ...cdfDataLines] = cdfLines;
    const cdfHeaders = cdfHeader.split(",");
    const cdfMilestoneNames = cdfHeaders.slice(1);

    for (const milestoneName of cdfMilestoneNames) {
      cdf[milestoneName] = [];
    }

    for (const line of cdfDataLines) {
      const values = line.split(",");
      const timeFromAC = Number.parseFloat(values[0]);

      if (!Number.isFinite(timeFromAC)) continue;

      for (let i = 0; i < cdfMilestoneNames.length; i += 1) {
        const cdfValue = Number.parseFloat(values[i + 1]);

        if (Number.isFinite(cdfValue)) {
          cdf[cdfMilestoneNames[i]].push({
            timeFromAC,
            cdf: cdfValue,
          });
        }
      }
    }
  } catch {
    // CDF file doesn't exist - fallback to PDF-derived CDF
  }

  // Filter to only show specific milestones
  // Remove milestones from this array to omit them from the chart
  const milestonesToShow = [
    // 'AI2027-SC',
    // 'AIR-5x',
    // 'AIR-25x',
    // 'AIR-250x',
    // 'AIR-2000x',
    // 'AIR-10000x',
    'SAR-level-experiment-selection-skill',
    'SIAR-level-experiment-selection-skill',
    // 'STRAT-AI',
    'TED-AI',
    'ASI',
  ];

  const filteredMilestones: { [key: string]: ConditionalTimingPoint[] } = {};
  const filteredStatistics: ConditionalTimingStatistics = {};
  const filteredCdf: { [key: string]: ConditionalCdfPoint[] } = {};

  for (const milestoneName of milestonesToShow) {
    if (milestones[milestoneName]) {
      filteredMilestones[milestoneName] = milestones[milestoneName];
    }
    if (statistics[milestoneName]) {
      filteredStatistics[milestoneName] = statistics[milestoneName];
    }
    if (cdf[milestoneName]) {
      filteredCdf[milestoneName] = cdf[milestoneName];
    }
  }

  return {
    milestones: filteredMilestones,
    statistics: filteredStatistics,
    conditionDescription: `Time until milestone achievement, conditional on achieving AC (Automated Coder) in ${year}`,
    cdf: Object.keys(filteredCdf).length > 0 ? filteredCdf : undefined,
  };
}

async function loadUnconditionalTimingDistributions(basePath: string): Promise<ConditionalTimingData> {
  // Load the AC unconditional timing distributions
  const filePath = path.join(process.cwd(), basePath, "ac_unconditional_time_until_distributions.csv");

  let raw: string;
  try {
    raw = await readFile(filePath, "utf8");
  } catch {
    // File doesn't exist - return empty data
    return {
      milestones: {},
      statistics: {},
      conditionDescription: "Time until milestone achievement, unconditional (all trajectories achieving AC)",
    };
  }

  const lines = raw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [header, ...dataLines] = lines;
  const headers = header.split(",");
  const milestoneNames = headers.slice(1); // Skip "time_decimal_year"

  // Load statistics
  const statsPath = path.join(process.cwd(), basePath, "ac_unconditional_time_until_statistics.csv");
  let statsRaw: string;
  try {
    statsRaw = await readFile(statsPath, "utf8");
  } catch {
    // Stats file doesn't exist - return data without statistics
    statsRaw = "";
  }

  const statsLines = statsRaw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [statsHeader, ...statsDataLines] = statsLines;
  const statsHeaders = parseCsvLine(statsHeader);

  const milestoneIndex = statsHeaders.indexOf("milestone_name");
  const achievementRateIndex = statsHeaders.indexOf("achievement_rate_pct");
  const modeIndex = statsHeaders.indexOf("mode");
  const p10Index = statsHeaders.indexOf("p10");
  const p50Index = statsHeaders.indexOf("p50");
  const p90Index = statsHeaders.indexOf("p90");

  const statistics: ConditionalTimingStatistics = {};

  for (const line of statsDataLines) {
    const fields = parseCsvLine(line);
    const milestoneName = fields[milestoneIndex];

    if (milestoneName) {
      statistics[milestoneName] = {
        achievementRate: Number.parseFloat(fields[achievementRateIndex]) / 100,
        mode: Number.parseFloat(fields[modeIndex]),
        p10: Number.parseFloat(fields[p10Index]),
        p50: Number.parseFloat(fields[p50Index]),
        p90: Number.parseFloat(fields[p90Index]),
      };
    }
  }

  // Parse distribution data for each milestone
  const milestones: { [key: string]: ConditionalTimingPoint[] } = {};

  for (const milestoneName of milestoneNames) {
    milestones[milestoneName] = [];
  }

  for (const line of dataLines) {
    const values = line.split(",");
    const timeFromAC = Number.parseFloat(values[0]);

    if (!Number.isFinite(timeFromAC)) continue;

    for (let i = 0; i < milestoneNames.length; i += 1) {
      const density = Number.parseFloat(values[i + 1]);

      if (Number.isFinite(density) && density > 0) {
        milestones[milestoneNames[i]].push({
          timeFromAC,
          probabilityDensity: density,
        });
      }
    }
  }

  // Load empirical CDF data if available
  const cdf: { [key: string]: ConditionalCdfPoint[] } = {};
  const cdfPath = path.join(process.cwd(), basePath, "ac_unconditional_time_until_cdf.csv");
  try {
    const cdfRaw = await readFile(cdfPath, "utf8");
    const cdfLines = cdfRaw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line.length > 0);

    const [cdfHeader, ...cdfDataLines] = cdfLines;
    const cdfHeaders = cdfHeader.split(",");
    const cdfMilestoneNames = cdfHeaders.slice(1);

    for (const milestoneName of cdfMilestoneNames) {
      cdf[milestoneName] = [];
    }

    for (const line of cdfDataLines) {
      const values = line.split(",");
      const timeFromAC = Number.parseFloat(values[0]);

      if (!Number.isFinite(timeFromAC)) continue;

      for (let i = 0; i < cdfMilestoneNames.length; i += 1) {
        const cdfValue = Number.parseFloat(values[i + 1]);

        if (Number.isFinite(cdfValue)) {
          cdf[cdfMilestoneNames[i]].push({
            timeFromAC,
            cdf: cdfValue,
          });
        }
      }
    }
  } catch {
    // CDF file doesn't exist - fallback to PDF-derived CDF
  }

  // Filter to only show specific milestones
  const milestonesToShow = [
    'SAR-level-experiment-selection-skill',
    'SIAR-level-experiment-selection-skill',
    'TED-AI',
    'ASI',
  ];

  const filteredMilestones: { [key: string]: ConditionalTimingPoint[] } = {};
  const filteredStatistics: ConditionalTimingStatistics = {};
  const filteredCdf: { [key: string]: ConditionalCdfPoint[] } = {};

  for (const milestoneName of milestonesToShow) {
    if (milestones[milestoneName]) {
      filteredMilestones[milestoneName] = milestones[milestoneName];
    }
    if (statistics[milestoneName]) {
      filteredStatistics[milestoneName] = statistics[milestoneName];
    }
    if (cdf[milestoneName]) {
      filteredCdf[milestoneName] = cdf[milestoneName];
    }
  }

  return {
    milestones: filteredMilestones,
    statistics: filteredStatistics,
    conditionDescription: "Time until milestone achievement, unconditional (all trajectories achieving AC)",
    cdf: Object.keys(filteredCdf).length > 0 ? filteredCdf : undefined,
  };
}

// Cached content component that loads and displays forecast data
async function CachedForecastContent({
  simulationId,
  registry,
}: {
  simulationId: string | null;
  registry: SimulationRegistry;
}) {
  'use cache';
  cacheLife('hours');
  cacheTag(`forecast-${simulationId ?? 'default'}`);

  // Determine which simulation to use
  let currentSimulation: Simulation | null = null;
  let dataBasePath: string;

  if (registry.simulations.length > 0) {
    // Find requested simulation or use default
    if (simulationId) {
      currentSimulation = registry.simulations.find(s => s.id === simulationId) ?? null;
    }
    if (!currentSimulation && registry.defaultSimulation) {
      currentSimulation = registry.simulations.find(s => s.id === registry.defaultSimulation) ?? null;
    }
    if (!currentSimulation) {
      currentSimulation = registry.simulations[0];
    }
  }

  // Determine data path
  if (currentSimulation) {
    dataBasePath = `app/forecast/data/${currentSimulation.dataPath}`;
  } else {
    // Fallback to legacy flat data structure for backward compatibility
    dataBasePath = "app/forecast/data";
  }

  const distributionData = await loadMilestoneDistributions(dataBasePath);
  const conditionalTimingData2027 = await loadConditionalTimingDistributions(dataBasePath, 2027);
  const conditionalTimingData2030 = await loadConditionalTimingDistributions(dataBasePath, 2030);
  const conditionalTimingData2035 = await loadConditionalTimingDistributions(dataBasePath, 2035);
  const unconditionalTimingData = await loadUnconditionalTimingDistributions(dataBasePath);
  const samplingConfigData = await loadSamplingConfig(dataBasePath);

  // Calculate shared y-domain across all conditional charts using scaled densities
  const allDatasets = [conditionalTimingData2027, conditionalTimingData2030, conditionalTimingData2035];
  let maxDensity = 0;

  for (const dataset of allDatasets) {
    for (const milestoneName of Object.keys(dataset.milestones)) {
      const stats = dataset.statistics[milestoneName];
      const points = [...dataset.milestones[milestoneName]].sort((a, b) => a.timeFromAC - b.timeFromAC);

      if (points.length === 0) continue;

      const totalDensity = points.reduce((sum, point) => sum + point.probabilityDensity, 0);
      if (!Number.isFinite(totalDensity) || totalDensity <= 0) continue;

      // Find the empirical median from the raw PDF
      let running = 0;
      let empiricalMedianTime = points[0].timeFromAC;
      for (const point of points) {
        running += point.probabilityDensity / totalDensity;
        if (running >= 0.5) {
          empiricalMedianTime = point.timeFromAC;
          break;
        }
      }

      // Calculate scaling factor to shift the median
      const targetMedian = stats?.p50;
      const hasMedian = typeof targetMedian === 'number' && Number.isFinite(targetMedian);

      if (!hasMedian || empiricalMedianTime === 0) {
        // No scaling, just check max normalized density
        for (const point of points) {
          const density = point.probabilityDensity / totalDensity;
          if (Number.isFinite(density) && density > maxDensity) {
            maxDensity = density;
          }
        }
        continue;
      }

      // Scale the time axis so empirical median becomes target median
      const timeScaleFactor = targetMedian / empiricalMedianTime;

      // Check max scaled density
      for (const point of points) {
        const scaledDensity = (point.probabilityDensity / totalDensity) / timeScaleFactor;
        if (Number.isFinite(scaledDensity) && scaledDensity > maxDensity) {
          maxDensity = scaledDensity;
        }
      }
    }
  }

  const headroom = maxDensity === 0 ? 0.1 : maxDensity * 0.1;
  const sharedYDomain: [number, number] = [0, maxDensity + headroom];

  // Compute arrival probabilities from CDF data
  const arrivalYears = [2027, 2030, 2035];
  const arrivalMilestones = [
    { name: 'AC', displayName: 'AC (Automated Coder)' },
    { name: 'SAR-level-experiment-selection-skill', displayName: 'SAR (Superhuman AI Researcher)' },
  ];

  const arrivalProbabilities: ArrivalProbabilities = {
    milestones: arrivalMilestones.map(milestone => ({
      name: milestone.name,
      displayName: milestone.displayName,
      probabilities: Object.fromEntries(
        arrivalYears.map(year => {
          const cdfPoints = distributionData.cdf?.[milestone.name] ?? [];
          // Use end of year (Dec 31) as target
          const prob = interpolateCdfAtYear(cdfPoints, year + 1);
          return [year.toString(), prob];
        })
      ),
    })),
    years: arrivalYears,
  };

  // Compute takeoff probabilities from conditional timing CDF data
  const takeoffMetrics = [
    { key: 'TED-AI', label: 'AC\u2192TED-AI \u22649mo', threshold: 0.75 },
    { key: 'ASI', label: 'AC\u2192ASI \u22641yr', threshold: 1.0 },
  ];

  const conditionalDatasets = [
    { year: 2027, data: conditionalTimingData2027, label: 'AC in 2027' },
    { year: 2030, data: conditionalTimingData2030, label: 'AC in 2030' },
    { year: 2035, data: conditionalTimingData2035, label: 'AC in 2035' },
  ];

  // Compute unconditional takeoff probabilities
  const unconditionalCondition = {
    label: 'Unconditional',
    probabilities: Object.fromEntries(
      takeoffMetrics.map(metric => {
        const cdfPoints = unconditionalTimingData.cdf?.[metric.key] ?? [];
        const prob = interpolateCdfAtTime(cdfPoints, metric.threshold);
        return [metric.key, prob];
      })
    ),
  };

  const takeoffProbabilities: TakeoffProbabilities = {
    conditions: [
      unconditionalCondition,
      ...conditionalDatasets.map(({ label, data }) => ({
        label,
        probabilities: Object.fromEntries(
          takeoffMetrics.map(metric => {
            const cdfPoints = data.cdf?.[metric.key] ?? [];
            const prob = interpolateCdfAtTime(cdfPoints, metric.threshold);
            return [metric.key, prob];
          })
        ),
      })),
    ],
    metrics: takeoffMetrics.map(m => ({ key: m.key, label: m.label })),
  };

  return (
    <div className="grid h-screen w-full grid-cols-[minmax(0,1fr)_auto] grid-rows-[minmax(0,1fr)] gap-0">
      <div className="relative col-start-1 row-start-1 flex min-h-0 flex-col">
        <div className="flex min-h-0 flex-col overflow-y-auto px-6 pb-10">
          <HeaderContent variant="inline" className="pt-6 pb-4" />
          <main className="mt-10 mx-auto max-w-5xl px-6 pb-16">
            <section className="space-y-8">
              {/* Simulation Selector and Controls */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  {registry.simulations.length > 1 && currentSimulation && (
                    <SimulationSelector
                      simulations={registry.simulations}
                      currentSimulationId={currentSimulation.id}
                    />
                  )}
                  {samplingConfigData && currentSimulation && (
                    <SamplingDistributionsButton
                      config={samplingConfigData.config}
                      rawYaml={samplingConfigData.rawYaml}
                      simulationId={currentSimulation.id}
                    />
                  )}
                </div>
                {currentSimulation && (
                  <SimulationMetadata simulation={currentSimulation} />
                )}
              </div>
            </section>
            <ForecastChartsSection
              distributionData={distributionData}
              conditionalTimingData2027={conditionalTimingData2027}
              conditionalTimingData2030={conditionalTimingData2030}
              conditionalTimingData2035={conditionalTimingData2035}
              unconditionalTimingData={unconditionalTimingData}
              sharedYDomain={sharedYDomain}
              numSamples={currentSimulation?.numSamples ?? null}
              arrivalProbabilities={arrivalProbabilities}
              takeoffProbabilities={takeoffProbabilities}
            />
          </main>
        </div>
      </div>
    </div>
  );
}

// Content component that resolves params and passes to cached component
async function ForecastPageContent({
  params,
}: {
  params: Promise<{ simulationId?: string[] }>;
}) {
  const resolvedParams = await params;
  const simulationId = resolvedParams.simulationId?.[0] ?? null;

  // Load registry outside of cache to determine simulation
  const registry = await loadSimulationRegistry();

  return <CachedForecastContent simulationId={simulationId} registry={registry} />;
}

// Main page component that wraps content in Suspense
export default function ForecastPage({
  params,
}: {
  params: Promise<{ simulationId?: string[] }>;
}) {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center text-sm text-slate-500">
        Loading forecast...
      </div>
    }>
      <ForecastPageContent params={params} />
    </Suspense>
  );
}
