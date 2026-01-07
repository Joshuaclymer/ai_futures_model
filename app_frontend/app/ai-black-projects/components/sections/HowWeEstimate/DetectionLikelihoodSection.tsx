'use client';

import { TimeSeriesChart, PDFChart } from '../../charts';
import { COLOR_PALETTE, darken } from '../../colors';
import { ParamLink, ParamValue } from '../../ui';
import { DetectionLatencyChart, IntelligenceAccuracyChart } from './HistoricalCharts';
import { Parameters } from '../../../types';

// Types for detection likelihood data from API
interface TimeSeriesData {
  years: number[];
  median: number[];
  p25: number[];
  p75: number[];
}

export interface DetectionLikelihoodData {
  years: number[];
  chip_evidence_samples: number[];
  sme_evidence_samples: number[];
  dc_evidence_samples: number[];
  energy_evidence: TimeSeriesData;
  combined_evidence: TimeSeriesData;
  direct_evidence: TimeSeriesData;
  posterior_prob: TimeSeriesData;
}

interface SubsectionItemProps {
  children: React.ReactNode;
}

function SubsectionItem({ children }: SubsectionItemProps) {
  return (
    <div className="relative">
      <span className="absolute -left-[26px] top-1/2 -translate-y-1/2 text-gray-300 text-[22px] leading-none">
        &mdash;
      </span>
      {children}
    </div>
  );
}

interface BreakdownChartProps {
  title: string;
  description?: string;
  descriptionNode?: React.ReactNode;
  data: { years: number[]; median: number[]; p25: number[]; p75: number[] };
  color: string;
  yLabel?: string;
  yRange?: [number, number];
}

function BreakdownChart({ title, description, descriptionNode, data, color, yLabel, yRange }: BreakdownChartProps) {
  return (
    <div className="breakdown-item">
      <div className="breakdown-plot">
        <TimeSeriesChart
          years={data.years}
          median={data.median}
          p25={data.p25}
          p75={data.p75}
          color={color}
          yLabel={yLabel}
          showBand={true}
          bandAlpha={0.15}
          yRange={yRange}
        />
      </div>
      <div className="breakdown-label">{title}</div>
      {(description || descriptionNode) && (
        <div className="breakdown-description" onClick={(e) => e.stopPropagation()}>
          {descriptionNode || description}
        </div>
      )}
    </div>
  );
}

interface PDFBreakdownChartProps {
  title: string;
  description?: string;
  descriptionNode?: React.ReactNode;
  samples: number[];
  color: string;
  xLabel?: string;
}

function PDFBreakdownChart({ title, description, descriptionNode, samples, color, xLabel = 'LR' }: PDFBreakdownChartProps) {
  return (
    <div className="breakdown-item">
      <div className="breakdown-plot">
        <PDFChart
          samples={samples}
          color={color}
          xLabel={xLabel}
          yLabel="Prob"
          logScale={true}
          numBins={10}
          showMedianLine={false}
        />
      </div>
      <div className="breakdown-label">{title}</div>
      {(description || descriptionNode) && (
        <div className="breakdown-description" onClick={(e) => e.stopPropagation()}>
          {descriptionNode || description}
        </div>
      )}
    </div>
  );
}

function Operator({ children }: { children: React.ReactNode }) {
  return <div className="breakdown-operator">{children}</div>;
}


interface DetectionLikelihoodSectionProps {
  agreementYear?: number;
  data?: DetectionLikelihoodData | null;
  parameters?: Parameters;
}

export function DetectionLikelihoodSection({ agreementYear = 2030, data, parameters }: DetectionLikelihoodSectionProps) {
  // Use data from API with empty fallbacks
  const chipEvidenceSamples = data?.chip_evidence_samples || [];
  const smeEvidenceSamples = data?.sme_evidence_samples || [];
  const dcEvidenceSamples = data?.dc_evidence_samples || [];
  const energyEvidence = data?.energy_evidence || { years: [], median: [], p25: [], p75: [] };
  const combinedEvidence = data?.combined_evidence || { years: [], median: [], p25: [], p75: [] };
  const directEvidence = data?.direct_evidence || { years: [], median: [], p25: [], p75: [] };
  const posteriorProb = data?.posterior_prob || { years: [], median: [], p25: [], p75: [] };

  return (
    <div className="pt-5">
      <h3 className="text-xl font-semibold text-[#333] mb-4">
        Predicting the likelihood of detection
      </h3>
      <p className="text-[14px] text-[#666] leading-[1.6] max-w-[900px] mb-3">
        We model when &quot;detection&quot; occurs by predicting how much <em>confidence</em> US
        intelligence has that a covert PRC AI project exists. Once this confidence exceeds a
        certain threshold (we chose a 4x increase in odds), the project is &quot;detected,&quot;
        and the US exits the agreement.
      </p>
      <p className="text-[14px] text-[#666] leading-[1.6] max-w-[900px] mb-5">
        US intelligence becomes more confident in the existence of the covert project because
        of two sources of evidence: <strong>accounting for key resources</strong> and noticing
        discrepancies and <strong>direct observations of covert operations</strong> (via IMGINT,
        HUMINT, SIGINT, etc).
      </p>

      {/* Bordered subsections with vertical line */}
      <div className="border-l-2 border-gray-300 pl-6 ml-1 space-y-5">
        {/* Resource accounting subsection */}
        <SubsectionItem>
          <p className="text-[14px] text-[#666] leading-[1.6] max-w-[900px]">
            US intelligence accounts for resources used in AI projects (such as chips, SME, energy, etc).
            If the amount of resources that the PRC reports differs from the amount that the US estimates
            the PRC has, this provides evidence some of these resources were diverted to a covert effort.
          </p>
        </SubsectionItem>

        {/* Equation visualization - resource accounting */}
        <div className="breakdown-equation-row">
          <PDFBreakdownChart
            title="Evidence from chip accounting"
            descriptionNode={
              <>
                Assuming a <ParamLink paramId="param-fraction-compute-divert">{parameters ? `${(parameters.proportionOfInitialChipStockToDivert * 100).toFixed(0)}%` : '5%'}</ParamLink> chip discrepancy and{' '}
                <ParamLink paramId="param-median-error-chip-stock">{parameters ? `${(parameters.intelligenceMedianErrorInEstimateOfFabStock * 100).toFixed(0)}%` : '7%'}</ParamLink> median error in US estimates.
              </>
            }
            samples={chipEvidenceSamples}
            color={COLOR_PALETTE.chip_stock}
          />
          <Operator>&times;</Operator>
          <PDFBreakdownChart
            title="Evidence from SME accounting"
            descriptionNode={
              <>
                Assuming a <ParamLink paramId="param-fraction-scanners-divert">{parameters ? `${(parameters.fractionOfLithographyScannersToDivert * 100).toFixed(0)}%` : '10%'}</ParamLink> SME discrepancy and{' '}
                <ParamLink paramId="param-median-error-chip-stock">{parameters ? `${(parameters.intelligenceMedianErrorInEstimateOfFabStock * 100).toFixed(0)}%` : '7%'}</ParamLink> median error in US estimates.
              </>
            }
            samples={smeEvidenceSamples}
            color={COLOR_PALETTE.fab}
          />
          <Operator>&times;</Operator>
          <PDFBreakdownChart
            title="Evidence from datacenter accounting"
            descriptionNode={
              <>
                Assuming that the PRC diverts{' '}
                <ParamLink paramId="param-fraction-datacenter-divert">{parameters ? `${(parameters.fractionOfDatacenterCapacityToDivert * 100).toFixed(0)}%` : '50%'}</ParamLink>{' '}
                of datacenter capacity not initially built for stealth, and the median error in the United States estimate of this capacity based on satellite imagery is{' '}
                <ParamLink paramId="param-median-error-satellite">{parameters ? `${(parameters.intelligenceMedianErrorInSatelliteEstimate * 100).toFixed(1)}%` : '1%'}</ParamLink>.
              </>
            }
            samples={dcEvidenceSamples}
            color={COLOR_PALETTE.datacenters_and_energy}
          />
          <Operator>&times;</Operator>
          <BreakdownChart
            title="Evidence from energy accounting"
            descriptionNode={
              <>
                Assuming that the fraction of PRC energy consumption that covert datacenters account for in 2037 is{' '}
                <ParamLink paramId="param-max-energy-fraction">{parameters ? `${(parameters.maxFractionOfTotalNationalEnergyConsumption * 100).toFixed(1)}%` : '5%'}</ParamLink>, and US median error in accounting for PRC energy consumption is{' '}
                <ParamLink paramId="param-median-error-energy">{parameters ? `${(parameters.intelligenceMedianErrorInEnergyConsumptionEstimate * 100).toFixed(0)}%` : '7%'}</ParamLink>.
              </>
            }
            data={energyEvidence}
            color={darken('datacenters_and_energy', 0.8)}
            yLabel="LR"
            yRange={[0.5, 2]}
          />
          <Operator>=</Operator>
          <BreakdownChart
            title="Combined evidence from resource accounting"
            description="Product of all accounting evidence, assuming conditional independence."
            data={combinedEvidence}
            color={COLOR_PALETTE.detection}
            yLabel="LR"
          />
        </div>

        {/* Direct observations subsection */}
        <SubsectionItem>
          <p className="text-[14px] text-[#666] leading-[1.6] max-w-[900px]">
            The US also receives <strong>direct evidence of covert operations</strong> from spies,
            satellite imagery, compromised devices, etc. Multiplying updates from{' '}
            <strong>discrepancies</strong> with updates from <strong>direct observations</strong>{' '}
            predicts the level of belief of US intelligence that a covert project exists.
          </p>
        </SubsectionItem>

        {/* Final equation visualization */}
        <div className="breakdown-equation-row">
          <div className="breakdown-box-item">
            <div className="breakdown-box">
              <div className="breakdown-box-inner">
                {parameters ? (
                  <ParamValue paramKey="priorOddsOfCovertProject" parameters={parameters} />
                ) : '30%'}
              </div>
            </div>
            <div className="breakdown-label">Prior odds of<br/>covert project</div>
          </div>
          <Operator>&times;</Operator>
          <BreakdownChart
            title="Evidence from resource accounting"
            description="Combined LR from chip, SME, datacenter, and energy accounting."
            data={combinedEvidence}
            color={COLOR_PALETTE.detection}
            yLabel="LR"
          />
          <Operator>&times;</Operator>
          <BreakdownChart
            title="Direct evidence of covert operations"
            description="Evidence from HUMINT, SIGINT, IMGINT, etc."
            data={directEvidence}
            color={COLOR_PALETTE.detection}
            yLabel="LR"
          />
          <div className="breakdown-limits-arrow">
            <span className="breakdown-limits-label" style={{ fontSize: '10px' }}>Convert to<br/>probability</span>
            <span className="breakdown-limits-symbol">&rarr;</span>
          </div>
          <BreakdownChart
            title="Posterior probability of covert project"
            description="Probability that a covert project exists given all evidence."
            data={posteriorProb}
            color={COLOR_PALETTE.detection}
            yLabel="P"
          />
        </div>

        {/* Historical data note */}
        <div className="mt-8">
          <p className="text-[14px] text-[#666] leading-[1.6] max-w-[900px]">
            How do we predict the speed of intelligence gathering if much of the relevant information
            is classified? We rely on two main pieces of historical data: (1) how well intelligence
            agencies estimated the sizes of weapon stockpiles, and (2) how quickly intelligence
            agencies detected covert nuclear operations.
          </p>
        </div>

        {/* Historical data visualization */}
        <div className="flex gap-6 flex-wrap justify-center my-8">
          <div className="flex-1 min-w-[300px] max-w-[450px]">
            <div className="h-[260px] bg-white border border-gray-200 rounded overflow-hidden">
              <IntelligenceAccuracyChart />
            </div>
            <p className="text-center mt-3 text-[13px] font-bold text-[#555]" style={{ fontFamily: 'et-book, Georgia, serif' }}>
              Historical accuracy of intelligence estimates
            </p>
            <p className="text-center text-[9px] text-[#777] italic">
              Accuracy of estimates of military asset stockpiles (median error ~15%).
            </p>
          </div>
          <div className="flex-1 min-w-[300px] max-w-[450px]">
            <div className="h-[260px] bg-white border border-gray-200 rounded overflow-hidden">
              <DetectionLatencyChart />
            </div>
            <p className="text-center mt-3 text-[13px] font-bold text-[#555]" style={{ fontFamily: 'et-book, Georgia, serif' }}>
              Historical detection speed
            </p>
            <p className="text-center text-[9px] text-[#777] italic">
              Detection speed of covert nuclear operations.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
