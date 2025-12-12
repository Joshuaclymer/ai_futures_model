'use client';

import { createContext, useContext, useState, type ReactNode } from 'react';

type ParameterName = string | null;

interface ParameterHoverContextValue {
  hoveredParameter: ParameterName;
  setHoveredParameter: (param: ParameterName) => void;
}

const ParameterHoverContext = createContext<ParameterHoverContextValue | undefined>(undefined);

export function useParameterHover() {
  const context = useContext(ParameterHoverContext);
  if (!context) {
    throw new Error('useParameterHover must be used within a ParameterHoverProvider');
  }
  return context;
}

export function ParameterHoverProvider({ children }: { children: ReactNode }) {
  const [hoveredParameter, setHoveredParameter] = useState<ParameterName>(null);

  return (
    <ParameterHoverContext.Provider value={{ hoveredParameter, setHoveredParameter }}>
      {children}
    </ParameterHoverContext.Provider>
  );
}

// Mapping from parameter names to SVG text node labels that should be highlighted
export const PARAMETER_TO_SVG_NODES: Record<string, string[]> = {
  // Main sliders
  present_doubling_time: ['Effective compute', 'Automated coder'],
  ac_time_horizon_minutes: ['Effective compute','Automated coder'],
  doubling_difficulty_growth_factor: ['Effective compute','Automated coder'],
  
  // Coding automation parameters
  swe_multiplier_at_present_day: ['Effective compute', 'Coding automation fraction and efficiency'],
  coding_automation_efficiency_slope: ['Effective compute', 'Coding automation fraction and efficiency', 'Aggregate Labor', 'Aggregate coding labor'],
  rho_coding_labor: ['Human coding labor', 'Automation compute', 'Coding automation fraction and efficiency', 'Aggregate coding labor'],
  max_serial_coding_labor_multiplier: ['Human coding labor', 'Automation compute', 'Coding automation fraction and efficiency', 'Aggregate coding labor'],
  
  // Experiment throughput parameters
  rho_experiment_capacity: ['Aggregate coding labor', 'Experiment compute', 'Experiment throughput'],
  alpha_experiment_capacity: ['Aggregate coding labor', 'Experiment compute', 'Experiment throughput'],
  experiment_compute_exponent: ['Experiment compute', 'Experiment throughput'],
  coding_labor_exponent: ['Aggregate coding labor', 'Experiment throughput'],
  inf_labor_asymptote: ['Aggregate coding labor', 'Experiment throughput'],
  inf_compute_asymptote: ['Experiment compute', 'Experiment throughput'],
  inv_compute_anchor_exp_cap: ['Experiment compute', 'Experiment throughput'],
  
  // AI research taste parameters
  ai_research_taste_at_coding_automation_anchor_sd: ['Automated coder', 'Automated experiment selection skill'],
  ai_research_taste_slope: ['Effective compute', 'Automated experiment selection skill'],
  median_to_top_taste_multiplier: ['Effective compute', 'Automated experiment selection skill', 'Human experiment selection skill'],
  
  // Software progress parameters
  software_progress_rate_at_reference_year: ['Software research effort', 'Software efficiency'],
  
  // Gap/horizon parameters
  saturation_horizon_minutes: ['Effective compute', 'Automated coder'],
  gap_years: ['Effective compute', 'Automated coder'],
  
  // Extra parameters
  present_day: ['Inputs'], // TODO think more about what this should highlight
  present_horizon: ['Effective compute', 'Automated coder'],
  automation_fraction_at_coding_automation_anchor: ['Effective compute','Coding automation fraction and efficiency', 'Automated coder'],
  
  // Additional parameters not currently highlighted in diagram
  taste_limit: [],
  taste_limit_smoothing: [],
  ted_ai_m2b: [],
  optimal_ces_eta_init: [],
  top_percentile: [],
};

// Rationales for parameter default values
export const PARAMETER_RATIONALES: Record<string, string> = {
  // Input parameters
  constant_training_compute_growth_rate: 'Recent compute scaling trends.',
  slowdown_year: 'Investment and fab capacity constraints.',
  post_slowdown_training_compute_growth_rate: 'Investment and fab capacity constraints.',

  // Main sliders
  present_doubling_time: 'In between the doubling time of the past long-term trend and the potential recent speedup.',
  ac_time_horizon_minutes: 'Considering what tasks are required to automate coding in AGI projects, then adjusting up due to requiring >80% reliability and the gap between the time horizon benchmark and real-world task.',
  doubling_difficulty_growth_factor: 'Based on intuitions about how the difficulty of time horizon doubling changes over time, also informed by past data.',
  ai_research_taste_slope: 'Based on data regarding how quickly AIs have moved through the human range for a variety of tasks.',
  median_to_top_taste_multiplier: 'Based on surveys of frontier AI researchers and AI experts.',

  // Coding time horizon parameters
  saturation_horizon_minutes: 'Estimating at which point the time horizon trend might "saturate," i.e. further improvements would not be very helpful for real-world tasks.',
  gap_years: 'Intuitions regarding how large the gap between time horizon saturation and Automated Coder (AC) might be, relative to the typical effective compute needed to get to AC from today',

  // Coding automation parameters
  swe_multiplier_at_present_day: 'Surveys regarding how much coding at AI companies is currently being sped up.',
  coding_automation_efficiency_slope: 'Data regarding how quickly coding automation efficiency has increased over time.',
  rho_coding_labor: 'Intuition regaridng the level of substitutability between coding tasks.',
  max_serial_coding_labor_multiplier: 'Intuitive estimates of the maximum thinking speed and actions increase, and the efficienncy of this thinking+actions.',

  // Experiment throughput parameters
  rho_experiment_capacity: 'Set via the experiment throughput increase from infinite coding labor and the increase from experiment compute.',
  alpha_experiment_capacity: 'Set via the experiment throughput constraints',
  experiment_compute_exponent: 'Set via the experiment throughput constraint',
  coding_labor_exponent: 'Survyes of frontier AI researchers and AI experts as well as our intuitive estimates.',
  inf_labor_asymptote: 'Estimates of what extent extremely fast coding could speed things up despite compute bottlenecks.',
  inf_compute_asymptote: 'Estimating how quickly AGI projects could fully utilize infinite compute, and sanity checking against more marginal constraints that we aren\'t explicitly modeling',
  inv_compute_anchor_exp_cap: 'Surveys of frontier AI researchers and AI experts.',

  // AI research taste parameters
  ai_research_taste_at_coding_automation_anchor_sd: 'Surveys of frontier AI researchers and AI experts, plus a minor intuitive adjustment.',
  taste_limit: 'Data on how superhuman Chess and Go AIs are, plus an intuitive adjustment based on how experiment selection differs from these cases.',
  taste_limit_smoothing: 'Intuition and sanity checking that it seems to give reasonable results.',

  // General capabilities parameters
  ted_ai_m2b: 'Intuitions regarding the spjikiness of AI capability profiles (ASI is always 2 M2Bs above TED-AI).',

  // Effective compute parameters
  software_progress_rate_at_reference_year: 'Analysis of data from the Epoch Capabiltiles Index.',

  // Extra parameters
  present_day: 'The date GPT-5 was released.',
  present_horizon: 'GPT-5\'s time horizon.',
  automation_fraction_at_coding_automation_anchor: 'With our current modelig decisions, 100% automation is required for Automated Coder.',
  optimal_ces_eta_init: 'The value that, if it were the same for all tasks, would be required to completely replace humans in present day conditions.',
  top_percentile: 'A rough estimate of the amount of people who do experiment seleciton at current AGI projects.',
};

