// CDF Spline utilities for the ATC Distribution Editor

import { ScaleConfig } from './chartUtils';

export interface ControlPoint {
  id: string;
  x: number;        // decimal year
  y: number;        // CDF value (0-1)
  isLocked: boolean; // endpoints locked vertically
}

export interface CdfPoint {
  year: number;
  cdf: number;
}

/**
 * Create an inverse scale function that maps pixel values back to domain values
 */
export function createInverseScale(config: ScaleConfig): (pixel: number) => number {
  const { domain, range, type } = config;
  const [domainMin, domainMax] = domain;
  const [rangeMin, rangeMax] = range;

  if (type === 'log') {
    const logDomainMin = Math.log10(domainMin);
    const logDomainMax = Math.log10(domainMax);
    return (pixel: number) => {
      const t = (pixel - rangeMin) / (rangeMax - rangeMin);
      return Math.pow(10, logDomainMin + t * (logDomainMax - logDomainMin));
    };
  }

  // Linear inverse scale
  return (pixel: number) => {
    const t = (pixel - rangeMin) / (rangeMax - rangeMin);
    return domainMin + t * (domainMax - domainMin);
  };
}

/**
 * Find the CDF point closest to a target CDF value
 */
function findClosestCdfPoint(cdfData: CdfPoint[], targetCdf: number): CdfPoint {
  let closest = cdfData[0];
  let minDiff = Math.abs(cdfData[0].cdf - targetCdf);

  for (const point of cdfData) {
    const diff = Math.abs(point.cdf - targetCdf);
    if (diff < minDiff) {
      minDiff = diff;
      closest = point;
    }
  }

  return closest;
}

/**
 * Initialize control points from empirical CDF data
 * Samples at quantiles within a visible year range
 */
export function initializeControlPoints(
  cdfData: CdfPoint[],
  numPoints: number = 7,
  maxYear: number = 2055
): ControlPoint[] {
  if (cdfData.length === 0) {
    return [];
  }

  // Sort by year and filter to visible range
  const sorted = [...cdfData].sort((a, b) => a.year - b.year);
  const visibleData = sorted.filter(p => p.year <= maxYear);

  if (visibleData.length === 0) {
    return [];
  }

  // Get the CDF range within the visible data
  const minCdf = visibleData[0].cdf;
  const maxCdf = visibleData[visibleData.length - 1].cdf;

  // Define quantiles for initial control points, scaled to the visible CDF range
  const baseQuantiles = numPoints === 7
    ? [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    : Array.from({ length: numPoints }, (_, i) => i / (numPoints - 1));

  // Scale quantiles to the actual CDF range in visible data
  const scaledQuantiles = baseQuantiles.map(q => minCdf + q * (maxCdf - minCdf));

  return scaledQuantiles.map((targetCdf, i) => {
    const closest = findClosestCdfPoint(visibleData, targetCdf);
    return {
      id: `cp-${i}-${Date.now()}`,
      x: closest.year,
      y: closest.cdf,
      isLocked: i === 0, // Only lock the first point at 0%
    };
  });
}

/**
 * Enforce monotonicity constraint on control points
 * Ensures CDF is always non-decreasing
 */
export function enforceMonotonicity(points: ControlPoint[]): ControlPoint[] {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const result: ControlPoint[] = [];

  let minY = 0;
  for (const point of sorted) {
    const constrainedY = Math.max(point.y, minY);
    result.push({ ...point, y: Math.min(constrainedY, 1) });
    minY = constrainedY;
  }

  return result;
}

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Constrain a control point's X position between its neighbors
 */
export function constrainXBetweenNeighbors(
  points: ControlPoint[],
  pointId: string,
  newX: number,
  minSpacing: number = 0.1
): number {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const index = sorted.findIndex(p => p.id === pointId);

  if (index === -1) return newX;

  const prevX = index > 0 ? sorted[index - 1].x + minSpacing : -Infinity;
  const nextX = index < sorted.length - 1 ? sorted[index + 1].x - minSpacing : Infinity;

  return clamp(newX, prevX, nextX);
}

// ============================================================================
// I-Spline Implementation for Monotone C² Continuous CDF Interpolation
// ============================================================================
//
// I-splines are integrated M-splines (normalized B-splines). They are:
// - Monotonically non-decreasing by construction
// - C² continuous for cubic (order 4) splines
// - Perfect for CDF fitting
//
// The CDF is represented as: F(x) = a + b * sum(w_i * I_i(x))
// where w_i >= 0 ensures monotonicity, and a, b are scaling parameters.

/**
 * Evaluate a single B-spline basis function using Cox-de Boor recursion
 * @param knots - knot vector
 * @param i - basis function index
 * @param k - order (degree + 1)
 * @param x - evaluation point
 */
function bsplineBasis(knots: number[], i: number, k: number, x: number): number {
  if (k === 1) {
    // Order 1 (degree 0): step function
    return (x >= knots[i] && x < knots[i + 1]) ? 1 : 0;
  }

  const denom1 = knots[i + k - 1] - knots[i];
  const denom2 = knots[i + k] - knots[i + 1];

  let term1 = 0;
  let term2 = 0;

  if (denom1 > 1e-10) {
    term1 = ((x - knots[i]) / denom1) * bsplineBasis(knots, i, k - 1, x);
  }
  if (denom2 > 1e-10) {
    term2 = ((knots[i + k] - x) / denom2) * bsplineBasis(knots, i + 1, k - 1, x);
  }

  return term1 + term2;
}

/**
 * Evaluate M-spline basis function (normalized B-spline)
 * M_i,k(x) = k * B_i,k(x) / (knots[i+k] - knots[i])
 */
function msplineBasis(knots: number[], i: number, k: number, x: number): number {
  const denom = knots[i + k] - knots[i];
  if (denom < 1e-10) return 0;
  return (k / denom) * bsplineBasis(knots, i, k, x);
}

/**
 * Numerical integration for I-spline basis
 */
function isplineBasisNumerical(knots: number[], i: number, k: number, x: number): number {
  if (x <= knots[i]) return 0;
  if (x >= knots[i + k]) return 1;

  // Integrate M-spline from knots[i] to x using Simpson's rule
  const a = knots[i];
  const b = x;
  const n = 50; // number of intervals (must be even for Simpson's)
  const h = (b - a) / n;

  let sum = msplineBasis(knots, i, k, a) + msplineBasis(knots, i, k, b);

  for (let j = 1; j < n; j++) {
    const xj = a + j * h;
    const coef = (j % 2 === 0) ? 2 : 4;
    sum += coef * msplineBasis(knots, i, k, xj);
  }

  return (h / 3) * sum;
}

/**
 * Create knot vector for cubic I-splines
 * For n data points, we create interior knots between them
 * Plus boundary knots with appropriate multiplicity
 */
function createKnotVector(xValues: number[], order: number = 4): number[] {
  const n = xValues.length;
  const xMin = xValues[0];
  const xMax = xValues[n - 1];

  // For cubic splines (order 4), we need order-1 = 3 repeated knots at boundaries
  // Interior knots can be at the data points or between them

  const knots: number[] = [];

  // Repeated boundary knots at start
  for (let i = 0; i < order; i++) {
    knots.push(xMin);
  }

  // Interior knots: place between data points for smoother fit
  for (let i = 1; i < n - 1; i++) {
    knots.push(xValues[i]);
  }

  // Repeated boundary knots at end
  for (let i = 0; i < order; i++) {
    knots.push(xMax);
  }

  return knots;
}

/**
 * Compute number of I-spline basis functions for given knots and order
 */
function numBasisFunctions(knots: number[], order: number): number {
  return knots.length - order;
}

/**
 * Fit I-spline to data points using non-negative least squares
 * Returns coefficients w_i >= 0
 */
function fitIspline(
  xData: number[],
  yData: number[],
  knots: number[],
  order: number
): number[] {
  const nBasis = numBasisFunctions(knots, order);
  const nData = xData.length;

  // Build design matrix: A[i][j] = I_j(x_i)
  const A: number[][] = [];
  for (let i = 0; i < nData; i++) {
    const row: number[] = [];
    for (let j = 0; j < nBasis; j++) {
      row.push(isplineBasisNumerical(knots, j, order, xData[i]));
    }
    A.push(row);
  }

  // We want to solve: y = a + (b-a) * sum(w_j * I_j(x))
  // where a = y[0], b = y[n-1] (boundary values)
  // So: (y - a) / (b - a) = sum(w_j * I_j(x))
  // Let z = (y - a) / (b - a), solve: z = A * w, w >= 0, sum(w) = 1

  const a = yData[0];
  const b = yData[nData - 1];
  const range = b - a;

  if (range < 1e-10) {
    // Flat CDF, return uniform weights
    return new Array(nBasis).fill(1 / nBasis);
  }

  // Normalize target values
  const z: number[] = yData.map(y => (y - a) / range);

  // Use iterative non-negative least squares (simplified version)
  // Start with uniform weights
  let w = new Array(nBasis).fill(1 / nBasis);

  // Iterative refinement
  const maxIter = 100;
  const tol = 1e-8;

  for (let iter = 0; iter < maxIter; iter++) {
    // Compute residuals
    const residuals: number[] = [];
    for (let i = 0; i < nData; i++) {
      let pred = 0;
      for (let j = 0; j < nBasis; j++) {
        pred += w[j] * A[i][j];
      }
      residuals.push(z[i] - pred);
    }

    // Compute gradient for each weight
    const grad: number[] = new Array(nBasis).fill(0);
    for (let j = 0; j < nBasis; j++) {
      for (let i = 0; i < nData; i++) {
        grad[j] += residuals[i] * A[i][j];
      }
    }

    // Update weights with projection to non-negative
    const stepSize = 0.1;
    let maxChange = 0;
    for (let j = 0; j < nBasis; j++) {
      const newW = Math.max(0, w[j] + stepSize * grad[j]);
      maxChange = Math.max(maxChange, Math.abs(newW - w[j]));
      w[j] = newW;
    }

    // Normalize weights to sum to 1
    const sumW = w.reduce((s, v) => s + v, 0);
    if (sumW > 1e-10) {
      w = w.map(v => v / sumW);
    }

    if (maxChange < tol) break;
  }

  return w;
}

/**
 * I-spline model that encapsulates knots, weights, and scaling
 */
interface ISplineModel {
  knots: number[];
  order: number;
  weights: number[];
  yMin: number;
  yMax: number;
}

/**
 * Build I-spline model from control points
 */
function buildISplineModel(sorted: ControlPoint[]): ISplineModel {
  const n = sorted.length;
  const order = 4; // cubic

  if (n < 2) {
    return {
      knots: [0, 1],
      order: 4,
      weights: [1],
      yMin: 0,
      yMax: 1,
    };
  }

  const xData = sorted.map(p => p.x);
  const yData = sorted.map(p => p.y);

  const knots = createKnotVector(xData, order);
  const weights = fitIspline(xData, yData, knots, order);

  return {
    knots,
    order,
    weights,
    yMin: yData[0],
    yMax: yData[n - 1],
  };
}

/**
 * Evaluate I-spline model at a given x
 */
function evaluateISplineModel(model: ISplineModel, x: number): number {
  const { knots, order, weights, yMin, yMax } = model;
  const nBasis = weights.length;

  // Clamp x to the domain
  const xMin = knots[order - 1];
  const xMax = knots[knots.length - order];

  if (x <= xMin) return yMin;
  if (x >= xMax) return yMax;

  // Compute weighted sum of I-spline basis functions
  let sum = 0;
  for (let i = 0; i < nBasis; i++) {
    sum += weights[i] * isplineBasisNumerical(knots, i, order, x);
  }

  // Scale to [yMin, yMax]
  return yMin + (yMax - yMin) * sum;
}

/**
 * Evaluate I-spline model derivative (PDF) at a given x
 */
function evaluateISplineModelDerivative(model: ISplineModel, x: number): number {
  const { knots, order, weights, yMin, yMax } = model;
  const nBasis = weights.length;

  // Clamp x to the domain
  const xMin = knots[order - 1];
  const xMax = knots[knots.length - order];

  if (x <= xMin || x >= xMax) return 0;

  // Compute weighted sum of M-spline (I-spline derivative) basis functions
  let sum = 0;
  for (let i = 0; i < nBasis; i++) {
    sum += weights[i] * msplineBasis(knots, i, order, x);
  }

  // Scale by the range
  return (yMax - yMin) * sum;
}

/**
 * Monotone C² continuous CDF interpolation using I-splines.
 *
 * I-splines (Integrated M-splines) are:
 * - Monotonically non-decreasing by construction (when weights are non-negative)
 * - C² continuous for cubic (order 4) splines
 * - Perfect for CDF fitting
 */
export function evaluateSplineAt(points: ControlPoint[], x: number): number {
  const sorted = [...points].sort((a, b) => a.x - b.x);

  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0].y;

  // Handle x outside the range
  if (x <= sorted[0].x) return sorted[0].y;
  if (x >= sorted[sorted.length - 1].x) return sorted[sorted.length - 1].y;

  // Build and evaluate I-spline model
  const model = buildISplineModel(sorted);
  const y = evaluateISplineModel(model, x);

  // Clamp to valid CDF range
  return clamp(y, 0, 1);
}

/**
 * Evaluate the PDF (derivative of the CDF) at a given x value.
 * This computes the derivative of the I-spline, which is a linear combination of M-splines.
 * M-splines are non-negative, so the PDF is guaranteed non-negative when weights are non-negative.
 */
export function evaluatePdfAt(points: ControlPoint[], x: number): number {
  const sorted = [...points].sort((a, b) => a.x - b.x);

  if (sorted.length < 2) return 0;

  // Handle x outside the range - PDF is 0 outside
  if (x < sorted[0].x || x > sorted[sorted.length - 1].x) return 0;

  // Build and evaluate I-spline model derivative
  const model = buildISplineModel(sorted);
  const pdf = evaluateISplineModelDerivative(model, x);

  // PDF should be non-negative (guaranteed by I-spline construction, but clamp for safety)
  return Math.max(0, pdf);
}

/**
 * Sample the PDF at regular intervals for visualization
 */
export function samplePdf(
  controlPoints: ControlPoint[],
  numSamples: number = 200
): { x: number; pdf: number }[] {
  if (controlPoints.length < 2) return [];

  const sorted = [...controlPoints].sort((a, b) => a.x - b.x);
  const minX = sorted[0].x;
  const maxX = sorted[sorted.length - 1].x;

  const samples: { x: number; pdf: number }[] = [];

  for (let i = 0; i <= numSamples; i++) {
    const x = minX + (maxX - minX) * (i / numSamples);
    samples.push({
      x,
      pdf: evaluatePdfAt(sorted, x),
    });
  }

  return samples;
}

/**
 * Sample the spline at regular intervals for export
 */
export function sampleSplineForExport(
  controlPoints: ControlPoint[],
  timePoints: number[]
): CdfPoint[] {
  return timePoints.map(year => ({
    year,
    cdf: evaluateSplineAt(controlPoints, year),
  }));
}

/**
 * Compute PDF from empirical CDF data using numerical differentiation.
 * Uses centered differences for interior points and forward/backward differences at boundaries.
 */
export function computePdfFromCdf(
  cdfData: CdfPoint[],
  numSamples: number = 200
): { x: number; pdf: number }[] {
  if (cdfData.length < 2) return [];

  // Sort by year
  const sorted = [...cdfData].sort((a, b) => a.year - b.year);
  const minX = sorted[0].year;
  const maxX = sorted[sorted.length - 1].year;

  // Helper to interpolate CDF value at any x
  function interpolateCdf(x: number): number {
    if (x <= sorted[0].year) return sorted[0].cdf;
    if (x >= sorted[sorted.length - 1].year) return sorted[sorted.length - 1].cdf;

    // Binary search for the interval
    let lo = 0;
    let hi = sorted.length - 1;
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2);
      if (sorted[mid].year <= x) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    // Linear interpolation
    const x0 = sorted[lo].year;
    const x1 = sorted[hi].year;
    const y0 = sorted[lo].cdf;
    const y1 = sorted[hi].cdf;
    const t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
  }

  const samples: { x: number; pdf: number }[] = [];
  const h = (maxX - minX) / numSamples;

  for (let i = 0; i <= numSamples; i++) {
    const x = minX + (maxX - minX) * (i / numSamples);

    // Compute derivative using centered differences
    const epsilon = h * 0.5;
    const cdfPlus = interpolateCdf(x + epsilon);
    const cdfMinus = interpolateCdf(x - epsilon);
    const pdf = Math.max(0, (cdfPlus - cdfMinus) / (2 * epsilon));

    samples.push({ x, pdf });
  }

  return samples;
}

/**
 * Add a new control point at the midpoint of the largest gap
 */
export function addControlPoint(points: ControlPoint[]): ControlPoint[] {
  if (points.length < 2) return points;

  const sorted = [...points].sort((a, b) => a.x - b.x);

  // Find the largest gap
  let maxGap = 0;
  let gapIndex = 0;

  for (let i = 0; i < sorted.length - 1; i++) {
    const gap = sorted[i + 1].x - sorted[i].x;
    if (gap > maxGap) {
      maxGap = gap;
      gapIndex = i;
    }
  }

  // Create new point at midpoint
  const midX = (sorted[gapIndex].x + sorted[gapIndex + 1].x) / 2;
  const midY = evaluateSplineAt(sorted, midX);

  const newPoint: ControlPoint = {
    id: `cp-new-${Date.now()}`,
    x: midX,
    y: midY,
    isLocked: false,
  };

  return [...points, newPoint];
}

/**
 * Remove a control point by ID (cannot remove locked points)
 */
export function removeControlPoint(points: ControlPoint[], pointId: string): ControlPoint[] {
  const point = points.find(p => p.id === pointId);
  if (!point || point.isLocked) return points;

  return points.filter(p => p.id !== pointId);
}

/**
 * Generate SVG path string for a spline through control points
 */
export function generateSplinePath(
  controlPoints: ControlPoint[],
  xScale: (value: number) => number,
  yScale: (value: number) => number,
  numSamples: number = 100
): string {
  if (controlPoints.length < 2) return '';

  const sorted = [...controlPoints].sort((a, b) => a.x - b.x);
  const minX = sorted[0].x;
  const maxX = sorted[sorted.length - 1].x;

  const pathParts: string[] = [];

  for (let i = 0; i <= numSamples; i++) {
    const x = minX + (maxX - minX) * (i / numSamples);
    const y = evaluateSplineAt(sorted, x);
    const px = xScale(x);
    const py = yScale(y);

    if (i === 0) {
      pathParts.push(`M ${px} ${py}`);
    } else {
      pathParts.push(`L ${px} ${py}`);
    }
  }

  return pathParts.join(' ');
}
