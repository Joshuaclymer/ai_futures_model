export interface MilestoneInfo {
  time?: number | null;
  [key: string]: number | string | boolean | null | undefined;
}

export type MilestoneMap = Record<string, MilestoneInfo>;


