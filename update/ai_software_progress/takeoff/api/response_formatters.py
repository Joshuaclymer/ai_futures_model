"""
Response formatting utilities for API endpoints.
Converts internal model results to frontend-compatible formats.
"""

from typing import Dict, List, Any, Optional, cast
from importlib import import_module

np = cast(Any, import_module('numpy'))


def _maybe_float(value: Any) -> Optional[float]:
    """Convert value to float if valid, otherwise return None."""
    if value is None:
        return None
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    return cast if np.isfinite(cast) else None


def build_time_series_payload(results: Dict[str, Any]) -> List[Dict[str, Optional[float]]]:
    """
    Build frontend-compatible time series payload from model results.

    Converts model.results dictionary (snake_case arrays) to array of objects
    with camelCase field names expected by the frontend.

    Args:
        results: Dictionary containing model computation results with keys:
            - 'times': array of time points
            - 'progress': cumulative progress values
            - 'effective_compute': effective compute in OOMs
            - 'horizon_lengths': time horizon lengths
            - 'research_stock': research stock values
            - 'automation_fraction': automation fraction at each time
            - 'training_compute': training compute in OOMs
            - 'experiment_capacity': experiment capacity values
            - 'ai_research_taste': AI research taste values
            - 'ai_software_progress_multipliers': software progress multipliers
            - 'serial_coding_labor_multipliers': serial coding labor multipliers vs human-only

    Returns:
        List of dictionaries with camelCase keys for frontend consumption
    """
    times = np.asarray(results.get('times', []), dtype=float)
    progress = np.asarray(results.get('progress', []), dtype=float)
    effective_compute = np.asarray(results.get('effective_compute', []), dtype=float)
    horizon_lengths = np.asarray(results.get('horizon_lengths', []), dtype=float)
    research_stock = np.asarray(results.get('research_stock', []), dtype=float)
    automation_fraction = np.asarray(results.get('automation_fraction', []), dtype=float)
    training_compute = np.asarray(results.get('training_compute', []), dtype=float)
    experiment_capacity = np.asarray(results.get('experiment_capacity', []), dtype=float)
    ai_research_taste = np.asarray(results.get('ai_research_taste', []), dtype=float)
    ai_software_progress_multipliers = np.asarray(results.get('ai_software_progress_multipliers', []), dtype=float)
    ai_sw_progress_mult_ref_present_day = np.asarray(results.get('ai_sw_progress_mult_ref_present_day', []), dtype=float)
    serial_coding_labor_multipliers = np.asarray(results.get('serial_coding_labor_multipliers', []), dtype=float)
    ai_coding_labor_multipliers = np.asarray(results.get('ai_coding_labor_multipliers', []), dtype=float)
    ai_coding_labor_mult_ref_present_day = np.asarray(results.get('ai_coding_labor_mult_ref_present_day', []), dtype=float)

    # Additional metrics
    research_efforts = np.asarray(results.get('research_efforts', []), dtype=float)
    software_progress_rates = np.asarray(results.get('software_progress_rates', []), dtype=float)
    software_efficiency = np.asarray(results.get('software_efficiency', []), dtype=float)

    # Input time series (may be at different sampling) -> interpolate to model times
    input_ts = results.get('input_time_series', {}) or {}
    input_time = np.asarray(input_ts.get('time', []), dtype=float)
    input_human_labor = np.asarray(input_ts.get('L_HUMAN', []), dtype=float)
    input_inference_compute = np.asarray(input_ts.get('inference_compute', []), dtype=float)
    input_experiment_compute = np.asarray(input_ts.get('experiment_compute', []), dtype=float)

    def _interp_to_times(src_x: Any, src_y: Any, dst_x: Any) -> Any:
        if src_x.size > 0 and src_y.size > 0 and dst_x.size > 0:
            try:
                finite_y = np.asarray(src_y, dtype=float)

                if np.all(np.isfinite(finite_y)) and np.all(finite_y > 0):
                    # Mirror the model solver: interpolate exponentially when series stays positive.
                    log_y = np.log(finite_y)
                    left_log = float(log_y[0])
                    right_log = float(log_y[-1])
                    interp_log = np.interp(dst_x, src_x, log_y, left=left_log, right=right_log)
                    return np.exp(interp_log)

                # Clamp to bounds by extending edge values (linear fallback)
                left_val = float(finite_y[0]) if np.isfinite(finite_y[0]) else 0.0
                right_val = float(finite_y[-1]) if np.isfinite(finite_y[-1]) else left_val
                return np.interp(dst_x, src_x, finite_y, left=left_val, right=right_val)
            except Exception:
                pass
        return np.full_like(dst_x, np.nan, dtype=float)

    human_labor_series = _interp_to_times(input_time, input_human_labor, times)
    inference_compute_series = _interp_to_times(input_time, input_inference_compute, times)
    experiment_compute_series = _interp_to_times(input_time, input_experiment_compute, times)

    payload: List[Dict[str, Optional[float]]] = []

    for idx, year in enumerate(times):
        payload.append({
            'year': float(year),
            'progress': _maybe_float(progress[idx]) if idx < progress.size else None,
            'effectiveCompute': _maybe_float(effective_compute[idx]) if idx < effective_compute.size else None,
            'horizonLength': _maybe_float(horizon_lengths[idx]) if idx < horizon_lengths.size else None,
            'researchStock': _maybe_float(research_stock[idx]) if idx < research_stock.size else None,
            'automationFraction': _maybe_float(automation_fraction[idx]) if idx < automation_fraction.size else None,
            'trainingCompute': _maybe_float(training_compute[idx]) if idx < training_compute.size else None,
            'experimentCapacity': _maybe_float(experiment_capacity[idx]) if idx < experiment_capacity.size else None,
            'aiResearchTaste': _maybe_float(ai_research_taste[idx]) if idx < ai_research_taste.size else None,
            'aiSoftwareProgressMultiplier': _maybe_float(ai_software_progress_multipliers[idx]) if idx < ai_software_progress_multipliers.size else None,
            'aiSwProgressMultRefPresentDay': _maybe_float(ai_sw_progress_mult_ref_present_day[idx]) if idx < ai_sw_progress_mult_ref_present_day.size else None,
            'serialCodingLaborMultiplier': _maybe_float(serial_coding_labor_multipliers[idx]) if idx < serial_coding_labor_multipliers.size else None,
            'aiCodingLaborMultiplier': _maybe_float(ai_coding_labor_multipliers[idx]) if idx < ai_coding_labor_multipliers.size else None,
            'aiCodingLaborMultRefPresentDay': _maybe_float(ai_coding_labor_mult_ref_present_day[idx]) if idx < ai_coding_labor_mult_ref_present_day.size else None,
            # New metrics
            'researchEffort': _maybe_float(research_efforts[idx]) if idx < research_efforts.size else None,
            'softwareProgressRate': _maybe_float(software_progress_rates[idx]) if idx < software_progress_rates.size else None,
            'softwareEfficiency': _maybe_float(software_efficiency[idx]) if idx < software_efficiency.size else None,
            'humanLabor': _maybe_float(human_labor_series[idx]) if idx < human_labor_series.size else None,
            'inferenceCompute': _maybe_float(inference_compute_series[idx]) if idx < inference_compute_series.size else None,
            'experimentCompute': _maybe_float(experiment_compute_series[idx]) if idx < experiment_compute_series.size else None,
        })

    return payload
