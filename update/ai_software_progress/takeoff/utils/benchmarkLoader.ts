import { benchmarkData } from './benchmarkData';

interface BenchmarkPoint {
  year: number;
  horizonLength: number;
  label: string;
  model: string;
}

// Convert date string to decimal year
function dateToDecimalYear(dateString: string): number {
  const date = new Date(dateString);
  const year = date.getFullYear();
  const startOfYear = new Date(year, 0, 1);
  const dayOfYear = Math.floor((date.getTime() - startOfYear.getTime()) / (24 * 60 * 60 * 1000)) + 1;
  const daysInYear = new Date(year, 11, 31).getDate() === 31 && new Date(year, 1, 29).getDate() === 29 ? 366 : 365;
  return year + (dayOfYear - 1) / daysInYear;
}

// Convert model key to display name
function modelKeyToDisplayName(key: string): string {
  const displayNames: { [key: string]: string } = {
    'davinci_002': 'Davinci-002',
    'gpt_3_5_turbo_instruct': 'GPT-3.5 Turbo Instruct',
    'gpt_4': 'GPT-4',
    'gpt_4_0125': 'GPT-4 (Jan 2024)',
    'gpt_4_1106': 'GPT-4 Turbo (Nov 2023)',
    'gpt_4_turbo': 'GPT-4 Turbo',
    'gpt_4o': 'GPT-4o',
    'gpt2': 'GPT-2',
    'claude_3_opus': 'Claude 3 Opus',
    'claude_3_5_sonnet': 'Claude 3.5 Sonnet',
    'claude_3_5_sonnet_20241022': 'Claude 3.5 Sonnet (Oct)',
    'claude_3_7_sonnet': 'Claude 3.7 Sonnet',
    'claude_4_opus': 'Claude 4 Opus',
    'claude_4_sonnet': 'Claude 4 Sonnet',
    'gemini_2_5_pro_preview': 'Gemini 2.5 Pro',
    'deepseek_v3': 'DeepSeek V3',
    'deepseek_v3_0324': 'DeepSeek V3 (Mar 2025)',
    'deepseek_r1': 'DeepSeek R1',
    'deepseek_r1_0528': 'DeepSeek R1 (May 2025)',
    'o1_preview': 'o1-preview',
    'o1_elicited': 'o1 (Elicited)',
    'o3': 'o3',
    'o4-mini': 'o4-mini',
    'qwen_2_72b': 'Qwen 2 72B',
    'qwen_2_5_72b': 'Qwen 2.5 72B',
    'grok_4': 'Grok 4',
    'gpt_5': 'GPT-5'
  };
  return displayNames[key] || key;
}

// Determine model company from key
function getModelCompany(key: string): string {
  if (key.startsWith('gpt') || key.startsWith('davinci') || key.startsWith('o1') || key.startsWith('o3') || key.startsWith('o4')) {
    return 'openai';
  } else if (key.startsWith('claude')) {
    return 'claude';
  } else if (key.startsWith('gemini')) {
    return 'google';
  } else if (key.startsWith('deepseek')) {
    return 'deepseek';
  } else if (key.startsWith('qwen')) {
    return 'alibaba';
  } else if (key.startsWith('grok')) {
    return 'xai';
  }
  return 'other';
}

export function loadBenchmarkData(): BenchmarkPoint[] {
  const benchmarkPoints: BenchmarkPoint[] = [];
  
  Object.entries(benchmarkData.results).forEach(([modelKey, modelData]) => {
    // Only include SOTA (state-of-the-art) models
    if (!modelData.is_sota) {
      return;
    }
    
    // Get the p80 horizon length from the first available agent
    const firstAgent = Object.values(modelData.agents)[0];
    if (firstAgent && firstAgent.p80_horizon_length) {
      const horizonLength = firstAgent.p80_horizon_length.estimate;
      const year = dateToDecimalYear(modelData.release_date);
      const label = modelKeyToDisplayName(modelKey);
      const model = getModelCompany(modelKey);
      
      benchmarkPoints.push({
        year,
        horizonLength,
        label,
        model
      });
    }
  });
  
  // Sort by year for consistency
  benchmarkPoints.sort((a, b) => a.year - b.year);
  
  return benchmarkPoints;
}