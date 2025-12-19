'use client';

import { useState } from 'react';
import MilestoneDistributionChart, { MilestoneDistributionData } from './MilestoneDistributionChart';
import ConditionalMilestoneTimingChart, { ConditionalTimingData } from './ConditionalMilestoneTimingChart';
import { ProbabilityTable } from './ProbabilityTable';
import { ArrivalProbabilities, TakeoffProbabilities } from '@/app/forecast/[[...simulationId]]/page';
import { MILESTONE_EXPLANATIONS } from '@/constants/chartExplanations';

interface ForecastChartsSectionProps {
  distributionData: MilestoneDistributionData;
  conditionalTimingData2027: ConditionalTimingData;
  conditionalTimingData2030: ConditionalTimingData;
  conditionalTimingData2035: ConditionalTimingData;
  unconditionalTimingData?: ConditionalTimingData;
  sharedYDomain: [number, number];
  numSamples: number | null;
  arrivalProbabilities: ArrivalProbabilities;
  takeoffProbabilities: TakeoffProbabilities;
}

// Format probability as percentage string (rounded to nearest percent)
function formatProbability(prob: number): string {
  return `${Math.round(prob * 100)}%`;
}

// Map milestone names to their short codes for looking up explanations
function getMilestoneDescription(milestoneName: string): string | undefined {
  // Map full milestone names to their short codes used in MILESTONE_EXPLANATIONS
  const nameToCode: Record<string, string> = {
    'AC': 'AC',
    'SAR-level-experiment-selection-skill': 'SAR',
    'SIAR-level-experiment-selection-skill': 'SIAR',
    'TED-AI': 'TED-AI',
    'ASI': 'ASI',
  };
  const code = nameToCode[milestoneName];
  return code ? MILESTONE_EXPLANATIONS[code] : undefined;
}

export function ForecastChartsSection({
  distributionData,
  conditionalTimingData2027,
  conditionalTimingData2030,
  conditionalTimingData2035,
  unconditionalTimingData,
  sharedYDomain,
  numSamples,
  arrivalProbabilities,
  takeoffProbabilities,
}: ForecastChartsSectionProps) {
  const [showCdf, setShowCdf] = useState(false);

  // Build arrival probabilities table data
  const arrivalHeaders = ['Milestone', ...arrivalProbabilities.years.map(y => `By Dec ${y}`)];
  const arrivalRows = arrivalProbabilities.milestones.map(m => ({
    label: m.displayName,
    values: arrivalProbabilities.years.map(y => formatProbability(m.probabilities[y.toString()])),
    description: getMilestoneDescription(m.name),
  }));

  // Build takeoff probabilities table data
  const takeoffHeaders = ['Condition', ...takeoffProbabilities.metrics.map(m => m.label)];
  const takeoffRows = takeoffProbabilities.conditions.map(c => ({
    label: c.label,
    values: takeoffProbabilities.metrics.map(m => formatProbability(c.probabilities[m.key])),
  }));

  return (
    <>
      {/* Executive Summary Tables */}
      <section className="space-y-8 mb-12 mt-12">
        <h2 className="text-xl font-semibold text-gray-900">Summary</h2>
        <p className="text-base leading-relaxed text-gray-600">
          We estimated each parameter in the model, quantified our uncertainty as a probability distribution over each one, then simulated thousands of trajectories. Here are the results.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <ProbabilityTable
            title="How likely are short timelines?"
            headers={arrivalHeaders}
            rows={arrivalRows}
          />
          <ProbabilityTable
            title="How likely are fast takeoffs?"
            headers={takeoffHeaders}
            rows={takeoffRows}
          />
        </div>
        <p className="text-sm text-gray-600">
          Learn more about our parameter estimates and reasoning <a href="https://docs.google.com/document/d/1ru6Okbxb6XuH18Cz8439sdQJazMV39hNxsWDokh97r0/edit?tab=t.0#heading=h.pt3lm912lfru" className="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener noreferrer">here</a>.
        </p>
      </section>

      {/* Main Milestone Distribution Chart */}
      <section className="space-y-8">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900">Timelines to Automated AI R&D</h2>
          {/* PDF/CDF Toggle */}
          <div className="flex items-center gap-2 text-sm">
            <span className="text-gray-600">View:</span>
            <button
              onClick={() => setShowCdf(false)}
              className={`px-3 py-1 rounded transition-colors ${!showCdf
                ? 'bg-gray-800 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
            >
              PDF
            </button>
            <button
              onClick={() => setShowCdf(true)}
              className={`px-3 py-1 rounded transition-colors ${showCdf
                ? 'bg-gray-800 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
            >
              CDF
            </button>
          </div>
        </div>
        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            This forecast sketches when AI systems might become capable of automating AI R&D. Each colored line represents the probability of reaching a different milestone in each month. Taller stretches correspond to months that
            capture more of the overall chance.
          </p>
          <MilestoneDistributionChart
            data={distributionData}
            title="Probability of Reaching AI Milestones"
            showCdf={showCdf}
          />
          <p className="text-xs text-gray-500">
            Probability densities are estimated based on {numSamples?.toLocaleString() ?? '10,000'} simulated trajectories. Our model is still under development, and forecasts may change.
          </p>
        </div>
      </section>

      {/* Conditional Timing Charts */}
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
              showCdf={showCdf}
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
              showCdf={showCdf}
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
              showCdf={showCdf}
            />
          </div>
        </div>

        {/* Unconditional Takeoff Speeds Chart */}
        {unconditionalTimingData && Object.keys(unconditionalTimingData.milestones).length > 0 && (
          <div className="mt-8 space-y-4">
            <h3 className="text-lg font-semibold text-gray-800">Unconditional (all trajectories achieving AC)</h3>
            <p className="text-sm text-gray-600">
              This chart shows the distribution of time from AC to subsequent milestones across all trajectories that achieve AC, regardless of when AC is reached.
            </p>
            <div className="max-w-2xl mx-auto">
              <ConditionalMilestoneTimingChart
                data={unconditionalTimingData}
                title="Time Until Milestones (Unconditional)"
                maxTimeYears={10}
                width={700}
                sharedYDomain={sharedYDomain}
                showLegend={false}
                showCdf={showCdf}
              />
            </div>
          </div>
        )}

        <p className="text-base leading-relaxed text-gray-600">
          We find that shorter timelines to the Automated Coder milestone correlate with faster takeoff to superintelligence. We are substantially more uncertain about takeoff speeds farther into the future. One reason is because our model relies on exogenous (i.e. not affected by the rest of the model) compute forecasts which are increasingly speculative in the late 2030s and beyond.
        </p>
        <p className="text-base leading-relaxed text-gray-600">
          Our model does not simulate the effects of hardware R&D automation, which we expect to be relevant in scenarios where takeoff is longer than 1 to 2 years. We also do not model the possibility of a robotics-driven industrial explosion. This leads us to expect takeoff to be somewhat faster than what our model predicts. For these reasons, we view our model as best equipped to evaluate the probability of fast takeoffs (driven primarily by software), rather than providing a median estimate that considers all possible mechanisms.
        </p>
      </section>
    </>
  );
}
