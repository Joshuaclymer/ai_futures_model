"""
Helper functions for computing percentiles and CCDFs.

These are used throughout the response building process to aggregate
simulation data into statistical summaries.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Callable

logger = logging.getLogger(__name__)


def compute_ccdf(values: List[float]) -> List[Dict[str, float]]:
    """
    Compute Complementary Cumulative Distribution Function (CCDF).

    Matches reference model's calculate_ccdf which computes P(X > x) not P(X >= x).
    NOTE: Reference does NOT filter out zeros - they are valid data points.
    """
    # Only filter out infinities and NaNs, but keep zeros and negative values
    values = [v for v in values if v < float('inf') and not np.isnan(v)]
    if not values:
        return []
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    ccdf = []
    prev_val = None
    for i, val in enumerate(sorted_vals):
        if val != prev_val:
            # P(X > x) = (count of values strictly greater than x) / total
            num_greater = n - (i + 1)
            ccdf.append({"x": float(val), "y": float(num_greater / n)})
            prev_val = val
        else:
            # For duplicate values, update the last y (will be lower)
            num_greater = n - (i + 1)
            ccdf[-1]["y"] = float(num_greater / n)
    return ccdf


def get_percentiles_with_individual(
    all_data: List[Dict],
    extractor: Callable[[Dict], List[float]]
) -> Dict[str, Any]:
    """
    Compute percentiles (median, p25, p75) with individual simulation data.

    Args:
        all_data: List of simulation data dicts
        extractor: Function to extract values from each simulation

    Returns:
        Dict with 'individual', 'median', 'p25', 'p75' keys
    """
    try:
        values = [extractor(d) for d in all_data]
        arr = np.array(values)
        if arr.size == 0:
            return {"individual": [], "median": [], "p25": [], "p75": []}
        return {
            "individual": [list(v) for v in values],
            "median": np.percentile(arr, 50, axis=0).tolist(),
            "p25": np.percentile(arr, 25, axis=0).tolist(),
            "p75": np.percentile(arr, 75, axis=0).tolist(),
        }
    except Exception as e:
        logger.warning(f"Error computing percentiles: {e}")
        return {"individual": [], "median": [], "p25": [], "p75": []}


def get_percentiles(
    all_data: List[Dict],
    extractor: Callable[[Dict], List[float]]
) -> Dict[str, List[float]]:
    """
    Compute percentiles without individual data (just median, p25, p75).
    """
    result = get_percentiles_with_individual(all_data, extractor)
    return {"median": result["median"], "p25": result["p25"], "p75": result["p75"]}


def get_fab_percentiles_with_individual(
    fab_built_sims: List[Dict],
    extractor: Callable[[Dict], List[float]]
) -> Dict[str, Any]:
    """
    Compute percentiles only over fab-built simulations (for black_fab section).
    """
    try:
        if not fab_built_sims:
            return {"individual": [], "median": [], "p25": [], "p75": []}
        values = [extractor(d) for d in fab_built_sims]
        arr = np.array(values)
        if arr.size == 0:
            return {"individual": [], "median": [], "p25": [], "p75": []}
        return {
            "individual": [list(v) for v in values],
            "median": np.percentile(arr, 50, axis=0).tolist(),
            "p25": np.percentile(arr, 25, axis=0).tolist(),
            "p75": np.percentile(arr, 75, axis=0).tolist(),
        }
    except Exception as e:
        logger.warning(f"Error computing fab percentiles: {e}")
        return {"individual": [], "median": [], "p25": [], "p75": []}
