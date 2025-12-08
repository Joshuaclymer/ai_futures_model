import path from "node:path";
import { readFile } from "node:fs/promises";
import { cacheLife } from "next/cache";
import MilestoneDistributionChart, { MilestoneDistributionPoint, MilestoneDistributionData } from "@/components/MilestoneDistributionChart";
import ConditionalMilestoneTimingChart, { ConditionalTimingPoint, ConditionalTimingData } from "@/components/ConditionalMilestoneTimingChart";
import { HeaderContent } from "@/components/HeaderContent";

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

async function loadMilestoneDistributions(): Promise<MilestoneDistributionData> {
  // Load the overlay distributions file
  const filePath = path.join(process.cwd(), "app/forecast/data/milestone_pdfs_overlay_distributions.csv");
  const raw = await readFile(filePath, "utf8");

  const lines = raw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [header, ...dataLines] = lines;
  const headers = header.split(",");
  const milestoneNames = headers.slice(1); // Skip "time_decimal_year"

  // Load statistics
  const statsPath = path.join(process.cwd(), "app/forecast/data/milestone_pdfs_overlay_statistics.csv");
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

  for (const milestoneName of milestonesToShow) {
    if (milestones[milestoneName]) {
      filteredMilestones[milestoneName] = milestones[milestoneName];
    }
    if (statistics[milestoneName]) {
      filteredStatistics[milestoneName] = statistics[milestoneName];
    }
  }

  return {
    milestones: filteredMilestones,
    statistics: filteredStatistics,
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

async function loadConditionalTimingDistributions(year: number): Promise<ConditionalTimingData> {
  // Load the AC conditional timing distributions for the specified year
  const filePath = path.join(process.cwd(), `app/forecast/data/ac_${year}_time_until_distributions.csv`);
  const raw = await readFile(filePath, "utf8");

  const lines = raw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const [header, ...dataLines] = lines;
  const headers = header.split(",");
  const milestoneNames = headers.slice(1); // Skip "time_decimal_year"

  // Load statistics
  const statsPath = path.join(process.cwd(), `app/forecast/data/ac_${year}_time_until_statistics.csv`);
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

  for (const milestoneName of milestonesToShow) {
    if (milestones[milestoneName]) {
      filteredMilestones[milestoneName] = milestones[milestoneName];
    }
    if (statistics[milestoneName]) {
      filteredStatistics[milestoneName] = statistics[milestoneName];
    }
  }

  return {
    milestones: filteredMilestones,
    statistics: filteredStatistics,
    conditionDescription: `Time until milestone achievement, conditional on achieving AC (Automated Coder) in ${year}`,
  };
}

export default async function ForecastPage() {
  'use cache';
  cacheLife('hours');

  const distributionData = await loadMilestoneDistributions();
  const conditionalTimingData2027 = await loadConditionalTimingDistributions(2027);
  const conditionalTimingData2030 = await loadConditionalTimingDistributions(2030);
  const conditionalTimingData2035 = await loadConditionalTimingDistributions(2035);

  // Calculate shared y-domain across all three conditional charts using scaled densities
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

  return <div className="grid h-screen w-full grid-cols-[minmax(0,1fr)_auto] grid-rows-[minmax(0,1fr)] gap-0">
    <div className="relative col-start-1 row-start-1 flex min-h-0 flex-col">
      <div className="flex min-h-0 flex-col overflow-y-auto px-6 pb-10">
        <HeaderContent variant="inline" className="pt-6 pb-4" />
        <main className="mt-10 mx-auto max-w-5xl px-6 pb-16">
          <section className="space-y-8">
            {/* <h2 className="text-2xl font-semibold text-gray-900">AI Capabilities Forecast</h2> */}
            <p className="text-base leading-relaxed text-gray-600">
              Eli Lifland, a co-author of this model, estimated each parameter in the model, quantified his uncertainty as a probability distribution over each one, then simulated 10,000 trajectories. Here are the results.
            </p>
            <p className="text-sm text-gray-600">
              Learn more about our assumptions and reasoning <a href="https://docs.google.com/document/d/1wsS2U4IG6k3C3wOzbNvzsljRuoaX_MqRG2NNZQEmER4/edit?tab=t.0#heading=h.o93tyjmgjiun" className="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener noreferrer">here</a>.
            </p>
            <div className="space-y-4">
            </div>
          </section>
          <section className="space-y-8">
            <h2 className="text-xl font-semibold text-gray-900">Timelines to Automated AI R&D</h2>
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                This forecast sketches when AI systems might become capable of automating AI R&D. Each colored line represents the probability of reaching a different milestone in each month. Taller stretches correspond to months that
                capture more of the overall chance.
              </p>
              <MilestoneDistributionChart
                data={distributionData}
                title="Probability of Reaching AI Milestones"
              />
              <p className="text-xs text-gray-500">
                Probability densities are estimated based on 10,000 simulated trajectories. Our model is still under development, and forecasts may change.
              </p>
            </div>
          </section>

          <section className="space-y-8 mt-16">
            <h2 className="text-xl font-semibold text-gray-900">Takeoff Speeds</h2>
            <p className="text-sm text-gray-600">
              The charts below show how long it might take to reach various milestones after achieving AC (Automated Coder),
              conditional on achieving AC within different years. The x-axis represents years from AC achievement, and the curves show
              the probability density for when each subsequent milestone might be reached.
            </p>

            {/* Shared Legend */}
            <div className="flex flex-wrap gap-4 text-xs justify-center">
              <div className="flex items-center gap-2">
                <div className="w-8 h-0.5" style={{ backgroundColor: '#000090' }} />
                <span className="font-medium" style={{ color: '#000090' }}>
                  SAR (Superhuman AI Researcher)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-0.5" style={{ backgroundColor: '#900000' }} />
                <span className="font-medium" style={{ color: '#900000' }}>
                  SIAR (Superintelligent AI Researcher)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-0.5" style={{ backgroundColor: '#2A623D' }} />
                <span className="font-medium" style={{ color: '#2A623D' }}>
                  TED-AI (Top Expert Dominating AI)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-0.5" style={{ backgroundColor: '#af1e86ff' }} />
                <span className="font-medium" style={{ color: '#af1e86ff' }}>
                  ASI (Artifical Superintelligence)
                </span>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-800">Given AC in 2027</h3>
                <ConditionalMilestoneTimingChart
                  data={conditionalTimingData2027}
                  title="Time Until Milestones (Given AC in 2027)"
                  maxTimeYears={7}
                  width={420}
                  sharedYDomain={sharedYDomain}
                  showLegend={false}
                />
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-800">Given AC in 2030</h3>
                <ConditionalMilestoneTimingChart
                  data={conditionalTimingData2030}
                  title="Time Until Milestones (Given AC in 2030)"
                  maxTimeYears={7}
                  width={420}
                  sharedYDomain={sharedYDomain}
                  showLegend={false}
                />
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-800">Given AC in 2035</h3>
                <ConditionalMilestoneTimingChart
                  data={conditionalTimingData2035}
                  title="Time Until Milestones (Given AC in 2035)"
                  maxTimeYears={7}
                  width={420}
                  sharedYDomain={sharedYDomain}
                  showLegend={false}
                />
              </div>
            </div>
            <p className="text-base leading-relaxed text-gray-600">
              We find that shorter timelines to the Automated Coder milestone correlate with faster takeoff to superintelligence. We are substantially more uncertain about takeoff speeds farther into the future. One reason is because our model relies on exogenous compute forecasts which are increasingly speculative in the late 2030s and beyond.
            </p>
            <p className="text-base leading-relaxed text-gray-600">
              Our model does not simulate the effects of hardware R&D automation, which we expect to be relevant in scenarios where takeoff is longer than 1 to 2 years. We also do not model the possibility of a robotics-driven industrial explosion. The effect is to bias the median takeoff lengths to be longer than we expect overall. For these reasons, we view our model as best equipped to evaluate the probability of fast takeoffs (driven primarily by software), rather than providing a median estimate that considers all possible mechanisms.
            </p>
          </section>
        </main>
      </div>
    </div>
  </div>;
}
