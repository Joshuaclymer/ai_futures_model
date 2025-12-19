import path from "node:path";
import { readFile, readdir } from "node:fs/promises";
import { Suspense } from "react";
import { cacheLife, cacheTag } from "next/cache";
import { HeaderContent } from "@/components/HeaderContent";
import Image from "next/image";
import TimeHorizonNewvsOld from "@/components/TimeHorizonNewvsOld";

// Types
interface Run {
  id: string;
  label: string;
  forecaster: "eli" | "daniel";
  date: string;
  numSamples: number;
  dataPath: string;
}

interface RunRegistry {
  version: number;
  defaultRun: string | null;
  runs: Run[];
}

interface ArrivalProbabilities {
  milestone: string;
  byDec2027: string;
  byDec2030: string;
  byDec2035: string;
}

interface FastTakeoffProbabilities {
  condition: string;
  values: { [key: string]: string };
}

interface FigureManifest {
  shortTimelines: string[];
  fastTakeoff: string[];
  timelineCorrelation: string[];
  horizonTrajectories: string[];
  mOverBeta: string | null;
  continuousPlots: {
    doublingDifficultyGrowthFactor: string | null;
    presentDoublingTime: string | null;
    aiResearchTasteSlope: string | null;
    medianToTopTasteMultiplier: string | null;
  };
  noCorrelation: {
    probAi2027TakeoffVsSarArrival: string | null;
    parameterSensitivityScatterAc: string | null;
  };
}

// CSV parsing utility
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

// Data loading functions
async function loadRunRegistry(): Promise<RunRegistry> {
  const registryPath = path.join(process.cwd(), "app/results/data/runs/runs.json");
  try {
    const raw = await readFile(registryPath, "utf8");
    return JSON.parse(raw) as RunRegistry;
  } catch {
    return {
      version: 1,
      defaultRun: null,
      runs: [],
    };
  }
}

async function loadArrivalProbabilities(basePath: string): Promise<ArrivalProbabilities[]> {
  const filePath = path.join(process.cwd(), basePath, "short_timelines_outputs", "arrival_probabilities.csv");
  try {
    const raw = await readFile(filePath, "utf8");
    const lines = raw.split(/\r?\n/).map(line => line.trim()).filter(line => line.length > 0);
    const [_header, ...dataLines] = lines;

    return dataLines.map(line => {
      const fields = parseCsvLine(line);
      return {
        milestone: fields[0],
        byDec2027: fields[1],
        byDec2030: fields[2],
        byDec2035: fields[3],
      };
    });
  } catch {
    return [];
  }
}

async function loadFastTakeoffProbabilities(basePath: string, source: "sar" | "ac"): Promise<FastTakeoffProbabilities[]> {
  const filePath = path.join(process.cwd(), basePath, "fast_takeoff_outputs", `conditional_fast_takeoff_probs_${source}.csv`);
  try {
    const raw = await readFile(filePath, "utf8");
    const lines = raw.split(/\r?\n/).map(line => line.trim()).filter(line => line.length > 0);
    const [header, ...dataLines] = lines;
    const headers = parseCsvLine(header);

    return dataLines.map(line => {
      const fields = parseCsvLine(line);
      const values: { [key: string]: string } = {};
      for (let i = 1; i < headers.length; i++) {
        values[headers[i]] = fields[i];
      }
      return {
        condition: fields[0],
        values,
      };
    });
  } catch {
    return [];
  }
}

async function loadFigureManifest(runId: string, forecaster: "eli" | "daniel"): Promise<FigureManifest> {
  const manifest: FigureManifest = {
    shortTimelines: [],
    fastTakeoff: [],
    timelineCorrelation: [],
    horizonTrajectories: [],
    mOverBeta: null,
    continuousPlots: {
      doublingDifficultyGrowthFactor: null,
      presentDoublingTime: null,
      aiResearchTasteSlope: null,
      medianToTopTasteMultiplier: null,
    },
    noCorrelation: {
      probAi2027TakeoffVsSarArrival: null,
      parameterSensitivityScatterAc: null,
    },
  };

  // Images are stored in public/results/<run-id>/
  const baseDir = path.join(process.cwd(), "public", "results", runId);

  // Short timelines figures
  try {
    const shortTimelineDir = path.join(baseDir, "short_timelines_outputs");
    const files = await readdir(shortTimelineDir);
    manifest.shortTimelines = files.filter(f => f.endsWith(".png"));
  } catch { /* directory doesn't exist */ }

  // Fast takeoff figures
  try {
    const fastTakeoffDir = path.join(baseDir, "fast_takeoff_outputs");
    const files = await readdir(fastTakeoffDir);
    manifest.fastTakeoff = files.filter(f => f.endsWith(".png"));
  } catch { /* directory doesn't exist */ }

  // Timeline correlation figures
  try {
    const timelineCorrelationDir = path.join(baseDir, "timeline_takeoff_correlation");
    const files = await readdir(timelineCorrelationDir);
    manifest.timelineCorrelation = files.filter(f => f.endsWith(".png"));
  } catch { /* directory doesn't exist */ }

  // Horizon trajectories
  try {
    const horizonDir = path.join(baseDir, "horizon_trajectories");
    const files = await readdir(horizonDir);
    manifest.horizonTrajectories = files.filter(f => f.endsWith(".png"));
  } catch { /* directory doesn't exist */ }

  // m/beta histogram
  try {
    const files = await readdir(baseDir);
    const mOverBetaFile = files.find(f => f === "m_over_beta_hist.png");
    if (mOverBetaFile) {
      manifest.mOverBeta = mOverBetaFile;
    }
  } catch { /* directory doesn't exist */ }

  // Continuous plots
  const continuousPlotsConfig = [
    { key: "doublingDifficultyGrowthFactor" as const, dir: "doubling-difficulty-growth-factor", file: "aa-time_vs_doubling-difficulty-growth-factor.png" },
    { key: "presentDoublingTime" as const, dir: "present-doubling-time", file: "aa-time_vs_present-doubling-time.png" },
    { key: "aiResearchTasteSlope" as const, dir: "ai-research-taste-slope", file: "p-1yr-takeoff-AC_vs_ai-research-taste-slope.png" },
    { key: "medianToTopTasteMultiplier" as const, dir: "median-to-top-taste-multiplier", file: "p-1yr-takeoff-AC_vs_median-to-top-taste-multiplier.png" },
  ];

  for (const { key, dir, file } of continuousPlotsConfig) {
    try {
      const plotDir = path.join(baseDir, "continuous_plots", dir);
      const files = await readdir(plotDir);
      if (files.includes(file)) {
        manifest.continuousPlots[key] = `continuous_plots/${dir}/${file}`;
      }
    } catch { /* directory doesn't exist */ }
  }

  // No-correlation comparison figures - forecaster specific
  // Stored in public/results/<forecaster>_no_correlation/
  const noCorrelationDir = path.join(process.cwd(), "public", "results", `${forecaster}_no_correlation`);
  const noCorrelationConfig = [
    { key: "probAi2027TakeoffVsSarArrival" as const, path: "fast_takeoff_outputs/prob_ai2027_takeoff_vs_sar_arrival.png" },
    { key: "parameterSensitivityScatterAc" as const, path: "timeline_takeoff_correlation/parameter_sensitivity_scatter_ac.png" },
  ];

  for (const { key, path: filePath } of noCorrelationConfig) {
    try {
      const fullPath = path.join(noCorrelationDir, filePath);
      await readFile(fullPath);
      manifest.noCorrelation[key] = `/results/${forecaster}_no_correlation/${filePath}`;
    } catch { /* file doesn't exist */ }
  }

  return manifest;
}

// Components
function RunSelector({ runs, currentRunId }: { runs: Run[]; currentRunId: string }) {
  if (runs.length <= 1) return null;

  return (
    <div className="mb-8 flex items-center gap-4">
      <label className="text-sm font-medium text-gray-700">Parameter Set:</label>
      <div className="flex gap-2">
        {runs.map(run => (
          <a
            key={run.id}
            href={`/results/${run.id}`}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              run.id === currentRunId
                ? "bg-[#2A623D] text-white"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {run.label}
          </a>
        ))}
      </div>
    </div>
  );
}

function ResultsSection({
  title,
  id,
  children,
}: {
  title: string;
  id?: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="space-y-6 mt-16 first:mt-0">
      <h2 className="text-2xl font-semibold text-gray-900 font-[family-name:var(--font-et-book)]">
        {title}
      </h2>
      {children}
    </section>
  );
}

function ProbabilityTable({
  title,
  headers,
  rows,
}: {
  title?: string;
  headers: string[];
  rows: { label: string; values: string[] }[];
}) {
  return (
    <div className="my-6">
      {title && <h3 className="text-lg font-medium text-gray-800 mb-3">{title}</h3>}
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse border border-gray-300">
          <thead>
            <tr className="bg-gray-50">
              <th className="border border-gray-300 px-4 py-2 text-left text-sm font-semibold text-gray-900">
                {headers[0]}
              </th>
              {headers.slice(1).map((header, i) => (
                <th key={i} className="border border-gray-300 px-4 py-2 text-right text-sm font-semibold text-gray-900">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                <td className="border border-gray-300 px-4 py-2 text-sm font-medium text-gray-900">
                  {row.label}
                </td>
                {row.values.map((value, j) => (
                  <td key={j} className="border border-gray-300 px-4 py-2 text-right text-sm text-gray-700">
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FigurePlaceholder({
  alt,
  caption,
  className = "",
}: {
  alt: string;
  caption?: string;
  className?: string;
}) {
  return (
    <figure className={`my-8 ${className}`}>
      <div className="relative w-full bg-gray-100 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center" style={{ minHeight: "400px" }}>
        <div className="text-center p-8">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <p className="mt-4 text-sm text-gray-500">Figure not yet available</p>
          <p className="mt-1 text-xs text-gray-400">{alt}</p>
        </div>
      </div>
      {caption && (
        <figcaption className="mt-2 text-center text-sm text-gray-600 italic">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

function FigureDisplay({
  src,
  alt,
  caption,
  className = "",
}: {
  src: string;
  alt: string;
  caption?: string;
  className?: string;
}) {
  return (
    <figure className={`my-8 ${className}`}>
      <div className="relative w-full">
        <Image
          src={src}
          alt={alt}
          width={1200}
          height={800}
          className="w-full h-auto rounded-lg"
          style={{ maxWidth: "100%", height: "auto" }}
        />
      </div>
      {caption && (
        <figcaption className="mt-2 text-center text-sm text-gray-600 italic">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

// Main cached content component
async function CachedResultsContent({
  runId,
  registry,
}: {
  runId: string | null;
  registry: RunRegistry;
}) {
  "use cache";
  cacheLife("hours");
  cacheTag(`results-${runId ?? "default"}`);

  // Determine which run to use
  let currentRun: Run | null = null;
  let dataBasePath: string;

  if (registry.runs.length > 0) {
    if (runId) {
      currentRun = registry.runs.find(r => r.id === runId) ?? null;
    }
    if (!currentRun && registry.defaultRun) {
      currentRun = registry.runs.find(r => r.id === registry.defaultRun) ?? null;
    }
    if (!currentRun) {
      currentRun = registry.runs[0];
    }
  }

  if (!currentRun) {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-slate-500">
        No results data available. Please configure a results run.
      </div>
    );
  }

  dataBasePath = `app/results/data/${currentRun.dataPath}`;
  const publicFiguresPath = `/results/${currentRun.id}`;

  // Load data
  const [arrivalProbs, fastTakeoffSAR, fastTakeoffAC, figures] = await Promise.all([
    loadArrivalProbabilities(dataBasePath),
    loadFastTakeoffProbabilities(dataBasePath, "sar"),
    loadFastTakeoffProbabilities(dataBasePath, "ac"),
    loadFigureManifest(currentRun.id, currentRun.forecaster),
  ]);

  return (
    <div className="grid h-screen w-full grid-cols-[minmax(0,1fr)_auto] grid-rows-[minmax(0,1fr)] gap-0">
      <div className="relative col-start-1 row-start-1 flex min-h-0 flex-col">
        <div className="flex min-h-0 flex-col overflow-y-auto px-6 pb-10">
          <HeaderContent variant="inline" className="pt-6 pb-4" />
          <main className="mt-10 mx-auto max-w-5xl px-6 pb-16">
            {/* Run Selector */}
            <RunSelector runs={registry.runs} currentRunId={currentRun.id} />

            {/* Introduction */}
            <ResultsSection title="Results" id="introduction">
              <p className="text-base leading-relaxed text-gray-600">
                We focus primarily on analyzing the following outcomes, chosen based on a combination of
                action relevance and which parts of our model behavior we are relatively more confident about:
              </p>
              <ol className="list-decimal list-inside space-y-2 text-base text-gray-600 ml-4">
                <li>
                  <strong>Probability of short timelines to AI R&D automation:</strong> p(AC) and p(SAR) by 2027, 2030, and 2035.
                </li>
                <li>
                  <strong>Probability of fast takeoff post-AC+SAR:</strong> Probability that takeoff is at least as fast as AI 2027, and that it&apos;s &lt;= 1 year. Where takeoff ends at either TED-AI or ASI.
                </li>
              </ol>
              <p className="text-base leading-relaxed text-gray-600">
                We then analyze the correlation in our model between timelines and takeoff, followed by various outcomes.
              </p>
              <p className="text-base leading-relaxed text-gray-600">
                Next, we discuss the possibility of a software intelligence explosion, including conditions for it in our model and how it affects our model behavior.
              </p>
              <p className="text-sm text-gray-500">
                Results computed from {currentRun.numSamples.toLocaleString()} Monte Carlo samples using {currentRun.label}.
              </p>
            </ResultsSection>

            {/* Probability of Short Timelines */}
            <ResultsSection title="Probability of Short Timelines to AI R&D Automation" id="short-timelines">
              <p className="text-base leading-relaxed text-gray-600">
                Since we aren&apos;t modeling experiment selection labor, we technically cannot measure a SAR, we can only see when the experiment selection skill reaches top human level. But we think this is a good proxy for a SAR, so for the sake of space we will usually abbreviate SAR-level-experiment-selection as SAR.
              </p>

              {arrivalProbs.length > 0 && (
                <ProbabilityTable
                  title="Chance of AC and SAR by the end of 2027, 2030, and 2035"
                  headers={["Milestone", "By Dec 2027", "By Dec 2030", "By Dec 2035"]}
                  rows={arrivalProbs.map(p => ({
                    label: p.milestone,
                    values: [p.byDec2027, p.byDec2030, p.byDec2035],
                  }))}
                />
              )}

              <p className="text-base leading-relaxed text-gray-600">
                Essentially, there&apos;s a significant but clearly &lt;50% chance of timelines as short as AI 2027 (i.e. SAR by 2027), and a &gt;50% chance of full AI R&D automation (SAR) within a decade.
              </p>

              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 my-6">
                <h4 className="font-semibold text-amber-800 mb-2">Limitations</h4>
                <p className="text-sm text-amber-900 mb-2">The following limitations our model may be substantially affecting these results:</p>
                <ol className="list-decimal list-inside space-y-2 text-sm text-amber-900">
                  <li>We aren&apos;t modeling data as an input to AI progress, which could either lengthen timelines if we get bottlenecked by data, or shorten timelines if data creation turns out to be particularly automatable. We&apos;ve attempted to account for the former possibility by putting some weight on exponential and subexponential-in-effective-compute time horizon progressions.</li>
                  <li>The shape of the super/sub-exponentials created by our methodology is different from what we expect. We think the true behavior will likely be more aggressive than the model predicts at later points in time, and less aggressive at earlier points in time. It&apos;s unclear how this would affect the models&apos; predictions.</li>
                  <li>We aren&apos;t modeling how software improvements change experiment compute requirements. This likely would push toward shorter timelines.</li>
                  <li>We aren&apos;t modeling the automation of hardware R&D or hardware production. This shouldn&apos;t affect AC timelines, but might affect SAR timelines a little.</li>
                </ol>
              </div>

              {figures.shortTimelines.includes("ac_sar_pdfs_overlay.png") && (
                <FigureDisplay
                  src={`${publicFiguresPath}/short_timelines_outputs/ac_sar_pdfs_overlay.png`}
                  alt="AC and SAR Arrival Time Distributions"
                  caption="AC and SAR arrival time probability distributions. The 90th percentiles show long tails, especially for SAR."
                />
              )}

              {figures.shortTimelines.includes("ac_to_sar_duration_pdf.png") && (
                <>
                  <h3 className="text-lg font-medium text-gray-800 mt-8">Transition Duration: AC → SAR</h3>
                  <p className="text-base leading-relaxed text-gray-600">
                    There is a substantial probability of a very fast AC→SAR transition, but also a long tail and a median of 1.2 years.
                  </p>
                  <FigureDisplay
                    src={`${publicFiguresPath}/short_timelines_outputs/ac_to_sar_duration_pdf.png`}
                    alt="Transition Duration from AC to SAR"
                    caption="Distribution of transition times from AC to SAR (including censored trajectories)."
                  />
                </>
              )}
            </ResultsSection>

            {/* Fast Takeoff */}
            <ResultsSection title="Probability of Fast Takeoff Post-AC+SAR" id="fast-takeoff">
              <p className="text-base leading-relaxed text-gray-600">
                We define 2 primary outcomes of interest:
              </p>
              <ol className="list-decimal list-inside space-y-3 text-base text-gray-600 ml-4">
                <li>
                  <strong>Takeoff at least as fast as AI 2027:</strong> In accordance with AI 2027&apos;s capability progression, there are several possible indicators of this:
                  <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                    <li>p(SAR→TED-AI) ≤ 3 months</li>
                    <li>p(SAR→SIAR) ≤ 4 months</li>
                    <li>p(SAR→ASI) ≤ 5 months</li>
                    <li>p(AC→TED-AI) ≤ 9 months</li>
                    <li>p(AC→ASI) ≤ 1 year</li>
                  </ul>
                </li>
                <li>
                  <strong>One-year takeoff, beginning from SAR:</strong> Some possible indicators of this.
                  <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                    <li>p(SAR→TED-AI) ≤ 1 year</li>
                    <li>p(SAR→SIAR) ≤ 1 year</li>
                    <li>p(SAR→ASI) ≤ 1 year</li>
                  </ul>
                </li>
              </ol>

              {fastTakeoffSAR.length > 0 && (
                <ProbabilityTable
                  title="Fast Takeoff Probabilities (from SAR)"
                  headers={["Condition", ...Object.keys(fastTakeoffSAR[0]?.values || {})]}
                  rows={fastTakeoffSAR.map(p => ({
                    label: p.condition,
                    values: Object.values(p.values),
                  }))}
                />
              )}

              <p className="text-base leading-relaxed text-gray-600">
                We see a strong correlation between timelines and takeoff, with takeoff as fast as AI 2027 being more likely than not if SAR arrives in 2027, but overall 30% likely. There&apos;s a very high chance of a one-year takeoff conditional on SAR in 2027, but unconditionally a one-year takeoff is 46%.
              </p>

              {/* P(AI 2027 Takeoff) vs SAR Arrival Year */}
              {figures.fastTakeoff.includes("prob_ai2027_takeoff_vs_sar_arrival.png") && (
                <FigureDisplay
                  src={`${publicFiguresPath}/fast_takeoff_outputs/prob_ai2027_takeoff_vs_sar_arrival.png`}
                  alt="P(AI 2027 Takeoff or Faster) vs SAR Arrival Year"
                  caption="Probability of AI-2027-speed takeoff vs SAR arrival year."
                />
              )}

              {/* No-correlation comparison */}
              <h3 className="text-lg font-medium text-gray-800 mt-8">Comparison: No Parameter Correlations</h3>
              <p className="text-base leading-relaxed text-gray-600">
                Here is the same plot with no correlations between parameter values. We still see a correlation between timelines and takeoff, but a substantially smaller one.
              </p>
              {figures.noCorrelation.probAi2027TakeoffVsSarArrival ? (
                <FigureDisplay
                  src={figures.noCorrelation.probAi2027TakeoffVsSarArrival}
                  alt="P(AI 2027 Takeoff or Faster) vs SAR Arrival Year - No Correlation"
                  caption="Same analysis but with parameter correlations removed. The timeline-takeoff relationship is weaker but still present."
                />
              ) : (
                <FigurePlaceholder
                  alt="P(AI 2027 Takeoff or Faster) vs SAR Arrival Year - No Correlation"
                  caption="No-correlation run not yet available for this forecaster."
                />
              )}

              {/* P(≤1 Year Takeoff) vs SAR Arrival Year */}
              {figures.fastTakeoff.includes("prob_1yr_takeoff_vs_sar_arrival.png") && (
                <FigureDisplay
                  src={`${publicFiguresPath}/fast_takeoff_outputs/prob_1yr_takeoff_vs_sar_arrival.png`}
                  alt="P(≤1 Year Takeoff) vs SAR Arrival Year"
                  caption="Probability of ≤1 year takeoff vs SAR arrival year."
                />
              )}
            </ResultsSection>

            {/* Timeline-Takeoff Correlation */}
            <ResultsSection title="Correlation Between Timelines and Takeoff" id="correlation">
              <p className="text-base leading-relaxed text-gray-600">
                Let&apos;s look at the correlation between timelines to AC and takeoff after AC. The scatter plot below shows each parameter&apos;s Spearman correlation with AC arrival time (x-axis) and takeoff speed from AC to ASI (y-axis).
              </p>

              {/* P(AI 2027 Takeoff) vs AC Arrival Year */}
              {figures.fastTakeoff.includes("prob_ai2027_takeoff_vs_ac_arrival.png") && (
                <FigureDisplay
                  src={`${publicFiguresPath}/fast_takeoff_outputs/prob_ai2027_takeoff_vs_ac_arrival.png`}
                  alt="P(AI 2027 Takeoff or Faster) vs AC Arrival Year"
                  caption="Probability of AI-2027-speed takeoff vs AC arrival year."
                />
              )}

              {/* Parameter Effects Scatter */}
              {figures.timelineCorrelation.includes("parameter_sensitivity_scatter_ac.png") && (
                <>
                  <h3 className="text-lg font-medium text-gray-800 mt-8">Parameter Effects (AC)</h3>
                  <p className="text-base leading-relaxed text-gray-600">
                    We can see that the parameters with the largest associations with AC timelines are those that most affect the time horizon progression:
                  </p>
                  <ol className="list-decimal list-inside space-y-2 text-base text-gray-600 ml-4 mt-2">
                    <li><strong>doubling_difficulty_growth_factor:</strong> The factor by which the effective compute requirement grows/decays from one time horizon doubling to the next (conditional on any growth/decay, as opposed to constant doubling time i.e. exponential; 0.95 would mean that the requirement gets 5% lower each doubling)</li>
                    <li><strong>present_doubling_time:</strong> The doubling time of the time horizon trend in the present day.</li>
                    <li><strong>gap_years:</strong> Whether a post-time-horizon-requirement gap is added before achievement of the coding automation anchor (AC/AC) and the magnitude of the gap. The importance of the magnitude of the gap would be larger if you filtered for only simulations that included a gap.</li>
                  </ol>
                  <FigureDisplay
                    src={`${publicFiguresPath}/timeline_takeoff_correlation/parameter_sensitivity_scatter_ac.png`}
                    alt="Parameter Effects: AC Timeline vs Takeoff Speed"
                    caption="Each point is a parameter. X-axis: correlation with AC arrival time. Y-axis: correlation with takeoff speed (AC→ASI)."
                  />
                </>
              )}

              {/* No-correlation comparison for parameter effects */}
              <h3 className="text-lg font-medium text-gray-800 mt-8">Parameter Effects (AC) - No Correlations</h3>
              <p className="text-base leading-relaxed text-gray-600">
                If we instead plot a run with no correlations between parameters, we see that the present doubling time no longer is correlated with faster takeoff and the doubling difficulty growth factor and AI research slope both have smaller associations with takeoff and timeline lengths, respectively. So keep in mind that with correlations between parameters included, associations with takeoff or timelines length aren&apos;t always causal.
              </p>
              {figures.noCorrelation.parameterSensitivityScatterAc ? (
                <FigureDisplay
                  src={figures.noCorrelation.parameterSensitivityScatterAc}
                  alt="Parameter Effects: AC Timeline vs Takeoff Speed - No Correlation"
                  caption="Same analysis but with parameter correlations removed. Note how some parameter-outcome relationships change."
                />
              ) : (
                <FigurePlaceholder
                  alt="Parameter Effects: AC Timeline vs Takeoff Speed - No Correlation"
                  caption="No-correlation run not yet available for this forecaster."
                />
              )}

              {figures.timelineCorrelation.includes("parameter_sensitivity_scatter_sar.png") && (
                <>
                  <h3 className="text-lg font-medium text-gray-800 mt-8">Parameter Effects (SAR)</h3>
                  <p className="text-base leading-relaxed text-gray-600">
                    Similar analysis for SAR timelines shows which parameters most influence both the arrival time of full AI R&D automation and subsequent takeoff speed.
                  </p>
                  <FigureDisplay
                    src={`${publicFiguresPath}/timeline_takeoff_correlation/parameter_sensitivity_scatter_sar.png`}
                    alt="Parameter Effects: SAR Timeline vs Takeoff Speed"
                    caption="Each point is a parameter. X-axis: correlation with SAR arrival time. Y-axis: correlation with takeoff speed (SAR→ASI)."
                  />
                </>
              )}
            </ResultsSection>

            {/* Time Horizon Trajectories */}
            <ResultsSection title="Time Horizon Trajectories" id="trajectories">
              <p className="text-base leading-relaxed text-gray-600">
                Here are all the time horizon trajectories, shaded by how well they backcast the METR points according to mean squared error (MSE). We view this sort of backcasting as an important exercise but not the end-all be-all, given that the data we have are so limited. Thanks to titotal&apos;s review of our previous model for prompting us to do more of this.
              </p>

              {figures.horizonTrajectories.includes("horizon_trajectories_all_shaded.png") ? (
                <FigureDisplay
                  src={`${publicFiguresPath}/horizon_trajectories/horizon_trajectories_all_shaded.png`}
                  alt="Complete Time Horizon Trajectories (All)"
                  caption="All time horizon trajectories, colored by MSE: green (low) → yellow → red (high)."
                />
              ) : (
                <FigurePlaceholder
                  alt="Complete Time Horizon Trajectories (All)"
                  caption="All time horizon trajectories, colored by MSE: green (low) → yellow → red (high)."
                />
              )}

              <h3 className="text-lg font-medium text-gray-800 mt-8">Filtered Trajectories (MSE ≤ 1.0)</h3>
              <p className="text-base leading-relaxed text-gray-600">
                Here are approximately the top 1/3 of trajectories in terms of how well they backcast. Trajectories that backcast better tend to reach higher time horizons earlier.
              </p>
              {figures.horizonTrajectories.includes("horizon_trajectories_mse_1.0.png") ? (
                <FigureDisplay
                  src={`${publicFiguresPath}/horizon_trajectories/horizon_trajectories_mse_1.0.png`}
                  alt="Complete Time Horizon Trajectories (MSE ≤ 1.0)"
                  caption="Time horizon trajectories filtered to MSE ≤ 1.0."
                />
              ) : (
                <FigurePlaceholder
                  alt="Complete Time Horizon Trajectories (MSE ≤ 1.0)"
                  caption="Time horizon trajectories filtered to MSE ≤ 1.0."
                />
              )}
            </ResultsSection>

            {/* New vs Old Model Comparison */}
            <ResultsSection title="New vs Old Model Comparison" id="model-comparison">
              <p className="text-base leading-relaxed text-gray-600">
                This figure compares our OLD timelines model (Apr 2025) with the NEW model (Dec 2025). Each set of three
                curves represents central trajectories filtered for achieving superhuman coder (SC) in Mar 2027, Jun 2028, and Jun 2030 respectively.
              </p>
              <TimeHorizonNewvsOld />
            </ResultsSection>

            {/* m/β Distribution */}
            {figures.mOverBeta && (
              <ResultsSection title="Distribution of m/β" id="m-over-beta">
                <p className="text-base leading-relaxed text-gray-600">
                  The ratio m/β determines whether a software intelligence explosion (SIE) occurs. When m/β &gt; 1, the model exhibits superexponential growth in effective compute under fixed hardware constraints.
                </p>
                <FigureDisplay
                  src={`${publicFiguresPath}/${figures.mOverBeta}`}
                  alt="Distribution of m/β"
                  caption="Distribution of m/β across Monte Carlo samples. Values > 1 indicate conditions for a software intelligence explosion."
                />
              </ResultsSection>
            )}

            {/* Parameter Sensitivity: Continuous Plots */}
            {(figures.continuousPlots.doublingDifficultyGrowthFactor ||
              figures.continuousPlots.presentDoublingTime ||
              figures.continuousPlots.aiResearchTasteSlope ||
              figures.continuousPlots.medianToTopTasteMultiplier) && (
              <ResultsSection title="Parameter Sensitivity Analysis" id="parameter-sensitivity">
                <p className="text-base leading-relaxed text-gray-600">
                  These plots show how individual parameters affect key outcomes. The x-axis shows parameter values,
                  and the y-axis shows probabilities or outcome values. Shaded regions indicate confidence intervals.
                </p>

                {figures.continuousPlots.doublingDifficultyGrowthFactor && (
                  <>
                    <h3 className="text-lg font-medium text-gray-800 mt-8">P(AC by Date) vs Doubling Difficulty Growth Factor</h3>
                    <FigureDisplay
                      src={`${publicFiguresPath}/${figures.continuousPlots.doublingDifficultyGrowthFactor}`}
                      alt="P(AC by Date) vs Doubling Difficulty Growth Factor"
                      caption="How the doubling difficulty growth factor affects probability of reaching AC by various dates."
                    />
                  </>
                )}

                {figures.continuousPlots.presentDoublingTime && (
                  <>
                    <h3 className="text-lg font-medium text-gray-800 mt-8">P(AC by Date) vs Present Doubling Time</h3>
                    <FigureDisplay
                      src={`${publicFiguresPath}/${figures.continuousPlots.presentDoublingTime}`}
                      alt="P(AC by Date) vs Present Doubling Time"
                      caption="How the present doubling time affects probability of reaching AC by various dates."
                    />
                  </>
                )}

                {figures.continuousPlots.aiResearchTasteSlope && (
                  <>
                    <h3 className="text-lg font-medium text-gray-800 mt-8">P(AI 2027 Speed Takeoff) vs AI Research Taste Slope</h3>
                    <FigureDisplay
                      src={`${publicFiguresPath}/${figures.continuousPlots.aiResearchTasteSlope}`}
                      alt="P(AI 2027 Speed Takeoff) vs AI Research Taste Slope"
                      caption="How AI research taste slope affects the probability of AI-2027-speed takeoff."
                    />
                  </>
                )}

                {figures.continuousPlots.medianToTopTasteMultiplier && (
                  <>
                    <h3 className="text-lg font-medium text-gray-800 mt-8">P(AI 2027 Speed Takeoff) vs Median To Top Taste Multiplier</h3>
                    <FigureDisplay
                      src={`${publicFiguresPath}/${figures.continuousPlots.medianToTopTasteMultiplier}`}
                      alt="P(AI 2027 Speed Takeoff) vs Median To Top Taste Multiplier"
                      caption="How the median-to-top taste multiplier affects the probability of AI-2027-speed takeoff."
                    />
                  </>
                )}
              </ResultsSection>
            )}

            {/* Footer */}
            <div className="mt-16 pt-8 border-t border-gray-200">
              <p className="text-sm text-gray-500">
                Generated from Monte Carlo simulation with {currentRun.numSamples.toLocaleString()} samples.
                See the <a href="https://docs.google.com/document/d/1wsS2U4IG6k3C3wOzbNvzsljRuoaX_MqRG2NNZQEmER4" className="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener noreferrer">full methodology documentation</a> for details.
              </p>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}

// Content component that resolves params
async function ResultsPageContent({
  params,
}: {
  params: Promise<{ runId?: string[] }>;
}) {
  const resolvedParams = await params;
  const runId = resolvedParams.runId?.[0] ?? null;

  const registry = await loadRunRegistry();

  return <CachedResultsContent runId={runId} registry={registry} />;
}

// Main page component
export default function ResultsPage({
  params,
}: {
  params: Promise<{ runId?: string[] }>;
}) {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center text-sm text-slate-500">
          Loading results...
        </div>
      }
    >
      <ResultsPageContent params={params} />
    </Suspense>
  );
}
