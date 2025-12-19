'use client';

import type { MilestoneMap } from '@/types/milestones';
import { MILESTONE_FULL_NAMES, MILESTONE_EXPLANATIONS } from '@/constants/chartExplanations';

export interface StaticMilestoneRowProps {
  milestones: MilestoneMap | null;
  simulationEndYear?: number;
  width: number;
}

const COLORS = {
  foreground: '#0D0D0D',
  muted: '#64748b', // slate-500
};

const FONTS = {
  label: 'Menlo, Consolas, monospace',
  date: 'et-book, Georgia, serif',
};

// Key milestones to display (using full keys from data)
const KEY_MILESTONES = ['AC', 'SAR-level-experiment-selection-skill', 'ASI'] as const;

// Display keys for milestones
const DISPLAY_KEYS: Record<string, string> = {
  'AC': 'AC',
  'SAR-level-experiment-selection-skill': 'SAR',
  'ASI': 'ASI',
};

// Full names for milestones
const FULL_NAMES: Record<string, string> = {
  'AC': 'Automated Coder',
  'SAR-level-experiment-selection-skill': 'Superhuman AI Researcher',
  'ASI': 'Artificial Superintelligence',
};

interface MilestoneDisplay {
  monthNumber: number;
  year: number;
}

function getMilestoneDisplay(decimalYear: number | null, simulationEndYear: number): MilestoneDisplay | null {
  if (decimalYear == null || !Number.isFinite(decimalYear)) {
    return null;
  }
  if (decimalYear > simulationEndYear) {
    return null; // Will show ">2045"
  }

  const year = Math.floor(decimalYear);
  const fraction = decimalYear - year;
  const monthIndexRaw = Math.round(fraction * 12);
  const monthIndex = monthIndexRaw % 12;
  const adjustedYear = monthIndexRaw === 12 ? year + 1 : year;

  return {
    monthNumber: monthIndex + 1,
    year: adjustedYear,
  };
}

function formatDate(milestone: MilestoneDisplay | null, simulationEndYear: number): string {
  if (!milestone) {
    return `>${simulationEndYear}`;
  }
  const month = String(milestone.monthNumber).padStart(2, '0');
  return `${month}/${milestone.year}`;
}

interface MilestoneRowData {
  key: string;
  fullName: string;
  dateString: string;
  explanation?: string;
}

/**
 * Static milestone rows for PNG export, styled like KeyMilestonePanel
 */
export function StaticMilestoneRow({
  milestones,
  simulationEndYear = 2045,
  width,
}: StaticMilestoneRowProps) {
  // Build milestone data
  const milestoneRows: MilestoneRowData[] = KEY_MILESTONES.map((key) => {
    const milestone = milestones?.[key];
    const time = milestone?.time as number | null | undefined;
    const display = getMilestoneDisplay(time ?? null, simulationEndYear);
    const dateString = formatDate(display, simulationEndYear);
    const displayKey = DISPLAY_KEYS[key] || key;
    const fullName = FULL_NAMES[key] || MILESTONE_FULL_NAMES[key] || key;
    // Use displayKey for explanation lookup since MILESTONE_EXPLANATIONS uses short keys
    const explanation = MILESTONE_EXPLANATIONS[displayKey];

    return {
      key: displayKey,
      fullName,
      dateString,
      explanation,
    };
  });

  const rowHeight = 50;
  const totalHeight = milestoneRows.length * rowHeight + 20; // +20 for padding
  const leftPadding = 20;

  return (
    <svg
      width={width}
      height={totalHeight}
      viewBox={`0 0 ${width} ${totalHeight}`}
    >
      {milestoneRows.map((row, index) => {
        const y = 30 + index * rowHeight;
        const labelText = `${row.key} (${row.fullName}):`;

        return (
          <g key={row.key}>
            {/* Label */}
            <text
              x={leftPadding}
              y={y}
              fontSize="13"
              fontWeight="600"
              fontFamily={FONTS.label}
              fill={COLORS.foreground}
            >
              {labelText}
            </text>

            {/* Date - positioned after label with character spacing */}
            <text
              x={leftPadding + 280}
              y={y}
              fontSize="24"
              fontWeight="bold"
              fontFamily={FONTS.date}
              fill={COLORS.foreground}
              letterSpacing="4"
            >
              {row.dateString}
            </text>

            {/* Brief explanation */}
            {row.explanation && (
              <text
                x={leftPadding + 420}
                y={y}
                fontSize="11"
                fontFamily={FONTS.label}
                fill={COLORS.muted}
              >
                — {row.explanation.length > 50 ? row.explanation.substring(0, 50) + '...' : row.explanation}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/**
 * Get milestone row data for use in other components
 */
export function getMilestoneRowData(
  milestones: MilestoneMap | null,
  simulationEndYear = 2045
): MilestoneRowData[] {
  return KEY_MILESTONES.map((key) => {
    const milestone = milestones?.[key];
    const time = milestone?.time as number | null | undefined;
    const display = getMilestoneDisplay(time ?? null, simulationEndYear);
    const dateString = formatDate(display, simulationEndYear);
    const displayKey = DISPLAY_KEYS[key] || key;
    const fullName = FULL_NAMES[key] || MILESTONE_FULL_NAMES[key] || key;
    const explanation = MILESTONE_EXPLANATIONS[displayKey];

    return {
      key: displayKey,
      fullName,
      dateString,
      explanation,
    };
  });
}
