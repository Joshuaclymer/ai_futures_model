/**
 * Explanatory text for chart elements and concepts
 */

export const HORIZON_LENGTH_EXPLANATION = 
  "The coding time horizon is the maximum length of coding tasks frontier AI systems can complete with a success rate of 80%, with the length defined as the time taken by typical AI company employees who do similar tasks."

export const COMPUTE_CHART_EXPLANATION = 
  "Effective compute represents the amount of compute that would be needed given early 2025 algorithms in order to achieve the frontier AI capability level at a given time. " +
  "We choose early 2025 as the reference point to line up with Grok 3's training compute.";

export const AI_R_D_PROGRESS_MULTIPLIER_EXPLANATION =
  "The AI Software R&D Uplift is the speedup in software progress (i.e., the software efficiency growth rate) " +
  "that would be achieved if the frontier AI systems at a given time were deployed within today's leading AI company.";

export const MILESTONE_FULL_NAMES: Record<string, string> = {
  'AC': 'Automated Coder',
  'SAR': 'Superhuman AI Researcher',
  'SIAR': 'SuperIntelligent AI Researcher',
  'STRAT-AI': 'Strategic AI',
  'TED-AI': 'Top-Expert-Dominating AI',
  'ASI': 'Artificial SuperIntelligence',
};

export const MILESTONE_EXPLANATIONS: Record<string, string> = {
  'AC': 'An AC can (if dropped into present day) autonomously replace the AGI project\'s coding staff without human help, given 5% of the project\'s compute.',
  'SAR': 'A SAR has experiment selection skill matching the top AGI project researcher.',
  'SIAR': 'A SIAR is 2 times as far above the top AGI project researcher as the top AGI project researcher is above the median. (That is, 2x more SDs as sampled from the human distribution, not necessarily a full 2x larger difference in experiment selection skill values.)', // todo improve explanatino
  'STRAT-AI': 'AI directs large-scale R&D piortfolios, making strategic decisions that compound progress across teams.',
  'TED-AI': 'A TED-AI is are at least as good as top human experts in virtually all cognitive tasks',
  'ASI': 'An ASI is 2 times as far above the top human as the top human is above the median. (That is, 2x more SDs within the human distribution, not necessarily a full 2x larger difference in absolute performance metrics.)',
};

export const SMALL_CHART_EXPLANATIONS = {
  // Coding Automation Fractions
  automationFraction: 'The fraction of coding work involved in frontier AI research that can be efficiently automated.',
  // AI Parallel Coding Labor Multiplier
  aiCodingLaborMultiplier: 'A multiplier of 2x would mean that AIs were increasing coding productivity by as much if they had an extra copy of all of their employees.',
  // AI Experiment Selection Skill
  aiResearchTaste: 'If AIs were to select experiments, this would be the value per experiment, where 1 is the mean value per experiment for human researcers.',
  // Research Effort
  researchEffort: 'Measures the amount of quality-adjusted research happening at a given time.',
  // Cumulative Research Effort
  researchStock: 'The accumluation of software research effort over time. Cumulative research effort is to research effort as distance traveled is to speed.',
  // Software Progress Rate
  softwareProgressRate: 'By how many orders of magnitude software efficiency grows per year.',
  // Software Efficiency
  softwareEfficiency: 'Software efficiency measures how efficiently the training process at a given time can convert training compute into performance. It is multiplied by training compute to get effective compute.',
  // Experiment Throughput
  experimentCapacity: 'The number of experiments the leading AI company can run per unit time, where experiments are weighted by how much compute they use and how labor-intensive they are to code.',
  // Inference Compute for Coding Automation
  inferenceCompute: 'The amount of compute used by the leading AI company for automating coding, in H100-equivalents.',
  // Experiment Compute
  experimentCompute: 'The amount of compute used by the leading AI company for running experiments, in H100-equivalents.',
  // Human Coding Labor
  humanLabor: 'The amount of coding staff at the leading AI company.',
};
