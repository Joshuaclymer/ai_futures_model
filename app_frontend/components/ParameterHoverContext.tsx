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
  'software_r_and_d.present_doubling_time': ['Effective compute', 'Automated coder'],
  'software_r_and_d.ac_time_horizon_minutes': ['Effective compute', 'Automated coder'],
  'software_r_and_d.doubling_difficulty_growth_factor': ['Effective compute', 'Automated coder'],

  // Coding automation parameters
  'software_r_and_d.swe_multiplier_at_present_day': ['Effective compute', 'Coding automation fraction and efficiency'],
  'software_r_and_d.coding_automation_efficiency_slope': ['Effective compute', 'Coding automation fraction and efficiency', 'Aggregate Labor', 'Aggregate coding labor'],
  'software_r_and_d.rho_coding_labor': ['Human coding labor', 'Automation compute', 'Coding automation fraction and efficiency', 'Aggregate coding labor'],
  'software_r_and_d.max_serial_coding_labor_multiplier': ['Human coding labor', 'Automation compute', 'Coding automation fraction and efficiency', 'Aggregate coding labor'],

  // Experiment throughput parameters
  'software_r_and_d.rho_experiment_capacity': ['Aggregate coding labor', 'Experiment compute', 'Experiment throughput'],
  'software_r_and_d.alpha_experiment_capacity': ['Aggregate coding labor', 'Experiment compute', 'Experiment throughput'],
  'software_r_and_d.experiment_compute_exponent': ['Experiment compute', 'Experiment throughput'],
  'software_r_and_d.coding_labor_exponent': ['Aggregate coding labor', 'Experiment throughput'],
  'software_r_and_d.inf_labor_asymptote': ['Aggregate coding labor', 'Experiment throughput'],
  'software_r_and_d.inf_compute_asymptote': ['Experiment compute', 'Experiment throughput'],
  'software_r_and_d.inv_compute_anchor_exp_cap': ['Experiment compute', 'Experiment throughput'],

  // AI research taste parameters
  'software_r_and_d.ai_research_taste_at_coding_automation_anchor_sd': ['Automated coder', 'Automated experiment selection skill'],
  'software_r_and_d.ai_research_taste_slope': ['Effective compute', 'Automated experiment selection skill'],
  'software_r_and_d.median_to_top_taste_multiplier': ['Effective compute', 'Automated experiment selection skill', 'Human experiment selection skill'],

  // Software progress parameters
  'software_r_and_d.software_progress_rate_at_reference_year': ['Software research effort', 'Software efficiency'],

  // Gap/horizon parameters
  'software_r_and_d.saturation_horizon_minutes': ['Effective compute', 'Automated coder'],
  'software_r_and_d.gap_years': ['Effective compute', 'Automated coder'],

  // Extra parameters
  'software_r_and_d.present_day': ['Inputs'], // TODO think more about what this should highlight
  'software_r_and_d.present_horizon': ['Effective compute', 'Automated coder'],
  'software_r_and_d.automation_fraction_at_coding_automation_anchor': ['Effective compute', 'Coding automation fraction and efficiency', 'Automated coder'],

  // Additional parameters not currently highlighted in diagram
  'software_r_and_d.taste_limit': [],
  'software_r_and_d.taste_limit_smoothing': [],
  'software_r_and_d.ted_ai_m2b': [],
  'software_r_and_d.optimal_ces_eta_init': [],
  'software_r_and_d.top_percentile': [],
};

// Rationales for parameter default values
export const PARAMETER_RATIONALES: Record<string, string> = {
  // Input parameters
  'compute.USComputeParameters.total_us_compute_annual_growth_rate': 'Recent compute scaling trends.',

  // Main sliders
  'software_r_and_d.present_doubling_time': 'In between the doubling time of the past long-term trend and the potential recent speedup.',
  'software_r_and_d.ac_time_horizon_minutes': 'We considered what tasks are required to automate coding in AGI projects, then adjusted our estimate upward to make up for only requiring >80% reliability and also to account for a gap between the time horizon benchmark and real-world tasks.',
  'software_r_and_d.doubling_difficulty_growth_factor': 'Based on intuitions about how the difficulty of time horizon doubling changes over time, also informed by past data.',
  'software_r_and_d.ai_research_taste_slope': 'Based on data regarding how quickly AIs have moved through the human range for a variety of tasks.',
  'software_r_and_d.median_to_top_taste_multiplier': 'Based on surveys of frontier AI researchers and AI experts.',

  // Coding time horizon parameters
  'software_r_and_d.saturation_horizon_minutes': 'Estimating at which point the time horizon trend might "saturate," i.e. further improvements would not be very helpful for real-world tasks.',
  'software_r_and_d.gap_years': 'Intuitions regarding how large the gap between time horizon saturation and Automated Coder (AC) might be, relative to the typical effective compute needed to get to AC from today',

  // Coding automation parameters
  'software_r_and_d.swe_multiplier_at_present_day': 'Surveys regarding how much coding at AI companies is currently being sped up.',
  'software_r_and_d.coding_automation_efficiency_slope': 'Data regarding how quickly coding automation efficiency has increased over time.',
  'software_r_and_d.rho_coding_labor': 'Intuition about the level of substitutability between coding tasks.',
  'software_r_and_d.max_serial_coding_labor_multiplier': 'Intuitive estimates of the maximum thinking speed and actions increase, and the efficiency of this thinking+actions.',
  'software_r_and_d.automation_logistic_asymptote': 'Allows the logistic curve to overshoot 100% before being clipped, enabling smoother transitions in automation fraction.',

  // Experiment throughput parameters
  'software_r_and_d.rho_experiment_capacity': 'Set via the experiment throughput increase from infinite coding labor and the increase from experiment compute.',
  'software_r_and_d.alpha_experiment_capacity': 'Set via the experiment throughput constraints',
  'software_r_and_d.experiment_compute_exponent': 'Set via the experiment throughput constraint',
  'software_r_and_d.coding_labor_exponent': 'Survyes of frontier AI researchers and AI experts as well as our intuitive estimates.',
  'software_r_and_d.inf_labor_asymptote': 'Estimates of what extent extremely fast coding could speed things up despite compute bottlenecks.',
  'software_r_and_d.inf_compute_asymptote': 'Estimating how quickly AGI projects could fully utilize infinite compute, and sanity checking against more marginal constraints that we aren\'t explicitly modeling',
  'software_r_and_d.inv_compute_anchor_exp_cap': 'Surveys of frontier AI researchers and AI experts.',

  // AI research taste parameters
  'software_r_and_d.ai_research_taste_at_coding_automation_anchor_sd': 'Surveys of frontier AI researchers and AI experts, plus a minor intuitive adjustment.',
  'software_r_and_d.taste_limit': 'Data on how superhuman Chess and Go AIs are, plus an intuitive adjustment based on how experiment selection differs from these cases.',
  'software_r_and_d.taste_limit_smoothing': 'Intuition and sanity checking that it seems to give reasonable results.',

  // General capabilities parameters
  'software_r_and_d.ted_ai_m2b': 'Intuitions regarding the spikiness of AI capability profiles (ASI is always 2 M2Bs above TED-AI).',

  // Effective compute parameters
  'software_r_and_d.software_progress_rate_at_reference_year': 'Analysis of data from the Epoch Capabiltiles Index.',

  // Extra parameters
  'software_r_and_d.present_day': 'The date GPT-5 was released.',
  'software_r_and_d.present_horizon': 'GPT-5\'s time horizon.',
  'software_r_and_d.automation_fraction_at_coding_automation_anchor': 'With our current modelig decisions, 100% automation is required for Automated Coder.',
  'software_r_and_d.optimal_ces_eta_init': 'The value that, if it were the same for all tasks, would be required to completely replace humans in present day conditions.',
  'software_r_and_d.top_percentile': 'A rough estimate of the amount of people who do experiment seleciton at current AGI projects.',
};
