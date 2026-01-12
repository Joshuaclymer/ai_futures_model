/**
 * Type guards for validating API response data structures.
 *
 * These functions provide runtime type checking to safely validate
 * data from the API before using it in components.
 */

/**
 * Checks if a value is a non-null object
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Checks if a value is an array of numbers
 */
function isNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every(item => typeof item === 'number');
}

/**
 * Checks if a value is an optional array of numbers (can be undefined)
 */
function isOptionalNumberArray(value: unknown): value is number[] | undefined {
  return value === undefined || isNumberArray(value);
}

/**
 * Type guard for TimeSeriesPercentiles structure
 */
export interface TimeSeriesPercentiles {
  p25: number[];
  median: number[];
  p75: number[];
}

export function isTimeSeriesPercentiles(value: unknown): value is TimeSeriesPercentiles {
  if (!isObject(value)) return false;
  return (
    isNumberArray(value.p25) &&
    isNumberArray(value.median) &&
    isNumberArray(value.p75)
  );
}

/**
 * Type guard for InitialStockData from API
 */
export interface InitialStockData {
  initial_prc_stock_samples: number[];
  initial_compute_stock_samples: number[];
  initial_energy_samples: number[];
  diversion_proportion: number;
  lr_compute_accounting_samples: number[];
  initial_black_project_detection_probs: Record<string, number>;
  prc_compute_years: number[];
  prc_compute_over_time: TimeSeriesPercentiles;
  prc_domestic_compute_over_time: { median: number[] };
  proportion_domestic_by_year: number[];
  largest_company_compute_over_time?: number[];
  state_of_the_art_energy_efficiency_relative_to_h100: number;
}

export function isInitialStockData(value: unknown): value is InitialStockData {
  if (!isObject(value)) return false;

  // Check required array fields exist (don't need to validate every element for performance)
  const hasRequiredArrays = (
    Array.isArray(value.initial_prc_stock_samples) &&
    Array.isArray(value.initial_compute_stock_samples) &&
    Array.isArray(value.initial_energy_samples) &&
    Array.isArray(value.prc_compute_years)
  );

  if (!hasRequiredArrays) return false;

  // Check required object fields
  const hasRequiredObjects = (
    isObject(value.initial_black_project_detection_probs) &&
    isObject(value.prc_compute_over_time) &&
    isObject(value.prc_domestic_compute_over_time)
  );

  if (!hasRequiredObjects) return false;

  // Check prc_compute_over_time has expected structure
  const prcCompute = value.prc_compute_over_time as Record<string, unknown>;
  if (!Array.isArray(prcCompute.median)) return false;

  // Check prc_domestic_compute_over_time has median array
  const domestic = value.prc_domestic_compute_over_time as Record<string, unknown>;
  if (!Array.isArray(domestic.median)) return false;

  // Check required numeric fields
  if (typeof value.diversion_proportion !== 'number') return false;
  if (typeof value.state_of_the_art_energy_efficiency_relative_to_h100 !== 'number') return false;

  return true;
}

/**
 * Safely parse InitialStockData from API response.
 * Returns null if validation fails.
 */
export function parseInitialStockData(value: unknown): InitialStockData | null {
  if (isInitialStockData(value)) {
    return value;
  }
  console.warn('[typeGuards] Invalid InitialStockData structure:', value);
  return null;
}

/**
 * Type guard for CCDF data point
 */
export interface CCDFPoint {
  x: number;
  y: number;
}

export function isCCDFPoint(value: unknown): value is CCDFPoint {
  if (!isObject(value)) return false;
  return typeof value.x === 'number' && typeof value.y === 'number';
}

export function isCCDFPointArray(value: unknown): value is CCDFPoint[] {
  return Array.isArray(value) && value.every(isCCDFPoint);
}

/**
 * Type guard for time series data with years and percentiles
 */
export interface TimeSeriesData {
  years: number[];
  median: number[];
  p25: number[];
  p75: number[];
}

export function isTimeSeriesData(value: unknown): value is TimeSeriesData {
  if (!isObject(value)) return false;
  return (
    isNumberArray(value.years) &&
    isNumberArray(value.median) &&
    isNumberArray(value.p25) &&
    isNumberArray(value.p75)
  );
}
