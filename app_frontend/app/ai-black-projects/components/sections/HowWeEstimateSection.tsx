'use client';

import { IntroSection, RateOfComputationSection, DetectionLikelihoodSection } from './HowWeEstimate';
import { Parameters } from '../../types';
import { RateOfComputationData } from './HowWeEstimate/RateOfComputationSection';
import { DetectionLikelihoodData } from './HowWeEstimate/DetectionLikelihoodSection';

interface HowWeEstimateSectionProps {
  parameters: Parameters;
  rateOfComputationData?: RateOfComputationData | null;
  detectionLikelihoodData?: DetectionLikelihoodData | null;
}

export function HowWeEstimateSection({ parameters, rateOfComputationData, detectionLikelihoodData }: HowWeEstimateSectionProps) {
  const agreementYear = parameters.agreementYear;

  return (
    <div>
      <section
        className="rounded-lg p-6 md:p-8"
        style={{
          backgroundColor: '#f9f9f9',
          border: '1px solid #e8e8e8',
        }}
      >
        {/* Intro with Figure 1 */}
        <IntroSection />

        {/* Rate of computation subsection */}
        <RateOfComputationSection agreementYear={agreementYear} data={rateOfComputationData} parameters={parameters} />

        {/* Detection likelihood subsection */}
        <DetectionLikelihoodSection agreementYear={agreementYear} data={detectionLikelihoodData} parameters={parameters} />
      </section>

      {/* Closing note - outside the grey box */}
      <p className="text-sm text-[#666] mt-4">
        <span className="hidden md:inline">
          The next sections break down these components in greater detail.
        </span>
        <span className="md:hidden">
          View the page on desktop for a more detailed breakdown.
        </span>
      </p>
    </div>
  );
}
