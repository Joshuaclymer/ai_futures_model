"""
Automatic comparison of shared keys between local and reference APIs.

Automatically finds matching keys between API responses and computes:
- For scalars: percent difference
- For arrays: median percent difference
- For time series (with percentiles): average percent difference across all time points
- For CCDFs: interpolate to common x-values and compare y-values (probabilities)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class KeyComparisonResult:
    """Result of comparing a single key between local and reference APIs."""
    key_path: str  # Full path like "black_project_model.survival_rate.median"
    data_type: str  # 'scalar', 'array', 'time_series', 'ccdf', 'object', 'unknown'
    local_value: Any  # Summary value (median for arrays/time series)
    reference_value: Any  # Summary value
    avg_pct_diff: float  # Average percent difference
    max_pct_diff: float  # Maximum percent difference (for time series)
    num_points: int  # Number of data points compared
    status: str  # 'PASS', 'WARN', 'FAIL', 'SKIP'
    note: str = ""


@dataclass
class ComparisonSummary:
    """Summary of all key comparisons."""
    total_keys: int
    compared_keys: int
    skipped_keys: int
    passed: int
    warned: int
    failed: int
    results: List[KeyComparisonResult] = field(default_factory=list)


def get_nested_value(data: Dict, path: str) -> Any:
    """Get a value from nested dict using dot-separated path."""
    keys = path.split('.')
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def set_nested_keys(data: Dict, prefix: str = "") -> List[str]:
    """Recursively get all leaf key paths from a nested dictionary."""
    paths = []
    for key, value in data.items():
        full_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            # Check if it's a time series structure (has median, p25, p75)
            if 'median' in value or 'p25' in value or 'p75' in value:
                paths.append(full_path)
            # Check if it's a CCDF structure (has keys like '1', '2', '4')
            elif all(k.isdigit() or k in ['1x', '2x', '4x'] for k in value.keys() if isinstance(k, str)):
                paths.append(full_path)
            else:
                paths.extend(set_nested_keys(value, full_path))
        elif isinstance(value, list):
            paths.append(full_path)
        else:
            paths.append(full_path)
    return paths


def classify_value(value: Any) -> str:
    """Classify the type of a value for comparison purposes."""
    if value is None:
        return 'unknown'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 'scalar'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        if len(value) == 0:
            return 'empty_array'
        if all(isinstance(x, (int, float)) for x in value):
            return 'array'
        if all(isinstance(x, dict) and 'x' in x and 'y' in x for x in value):
            return 'ccdf'
        if all(isinstance(x, list) for x in value):
            return 'array_2d'
        return 'mixed_array'
    if isinstance(value, dict):
        if 'median' in value:
            return 'time_series'
        if all(k.isdigit() or k in ['1', '2', '4', '1x', '2x', '4x'] for k in value.keys()):
            return 'ccdf_dict'
        return 'object'
    return 'unknown'


def compute_array_diff(local: List[float], ref: List[float], sort_arrays: bool = True) -> Tuple[float, float, int, str]:
    """
    Compute average and max percent difference between two arrays.

    For Monte Carlo samples, we sort both arrays first to compare distribution
    percentiles rather than arbitrary element ordering.

    Args:
        local: Local array
        ref: Reference array
        sort_arrays: If True, sort both arrays before comparing (for MC samples)

    Returns:
        (avg_pct_diff, max_pct_diff, num_points, warning_note)
    """
    warning_note = ""

    if not local or not ref:
        return 0.0, 0.0, 0, ""

    # Check for length mismatch
    if len(local) != len(ref):
        warning_note = f"Length mismatch: local={len(local)}, ref={len(ref)}"

    # Use the shorter length
    n = min(len(local), len(ref))
    if n == 0:
        return 0.0, 0.0, 0, warning_note

    local_arr = np.array(local[:n], dtype=float)
    ref_arr = np.array(ref[:n], dtype=float)

    # Sort arrays to compare distribution percentiles (important for MC samples)
    if sort_arrays:
        local_arr = np.sort(local_arr)
        ref_arr = np.sort(ref_arr)

    # Compute percent differences with proper zero handling
    pct_diffs = np.zeros(n)
    for i in range(n):
        local_val = local_arr[i]
        ref_val = ref_arr[i]

        # Both near zero - consider it a match
        if abs(ref_val) < 1e-10 and abs(local_val) < 1e-10:
            pct_diffs[i] = 0.0
        # Reference near zero but local is not - use absolute difference threshold
        elif abs(ref_val) < 1e-10:
            # If local is also small (< 0.01), consider it close enough
            pct_diffs[i] = 0.0 if abs(local_val) < 0.01 else 100.0
        else:
            pct_diffs[i] = abs(local_val - ref_val) / abs(ref_val) * 100

    # Cap extreme values
    pct_diffs = np.clip(pct_diffs, 0, 1000)

    avg_diff = float(np.mean(pct_diffs))
    max_diff = float(np.max(pct_diffs))

    return avg_diff, max_diff, n, warning_note


def compute_ccdf_diff(local_ccdf: List[dict], ref_ccdf: List[dict]) -> Tuple[float, float, int, str]:
    """
    Compare two CCDFs by interpolating to common x-values and comparing y-values.

    CCDFs are lists of {"x": value, "y": probability} points.
    We interpolate both to common x-values and compute the average absolute difference
    in probabilities.

    Returns:
        (avg_diff, max_diff, num_points, warning_note)
        where diff is in percentage points (0-100 scale for probabilities)
    """
    warning_note = ""

    if not local_ccdf or not ref_ccdf:
        return 0.0, 0.0, 0, "Empty CCDF"

    # Extract x and y values
    local_x = np.array([p['x'] for p in local_ccdf])
    local_y = np.array([p['y'] for p in local_ccdf])
    ref_x = np.array([p['x'] for p in ref_ccdf])
    ref_y = np.array([p['y'] for p in ref_ccdf])

    # Sort by x
    local_order = np.argsort(local_x)
    local_x, local_y = local_x[local_order], local_y[local_order]
    ref_order = np.argsort(ref_x)
    ref_x, ref_y = ref_x[ref_order], ref_y[ref_order]

    # Create common x-values spanning both ranges
    x_min = min(local_x.min(), ref_x.min())
    x_max = max(local_x.max(), ref_x.max())

    # Use 50 evenly spaced points for comparison
    num_points = 50
    common_x = np.linspace(x_min, x_max, num_points)

    # Interpolate both CCDFs to common x-values
    # For CCDF, values outside the range should be 1.0 (left) or 0.0 (right)
    local_interp = np.interp(common_x, local_x, local_y, left=1.0, right=0.0)
    ref_interp = np.interp(common_x, ref_x, ref_y, left=1.0, right=0.0)

    # Compute absolute differences in probability (as percentage points)
    # Multiply by 100 to convert from 0-1 to 0-100 scale
    prob_diffs = np.abs(local_interp - ref_interp) * 100

    avg_diff = float(np.mean(prob_diffs))
    max_diff = float(np.max(prob_diffs))

    return avg_diff, max_diff, num_points, warning_note


def compute_ccdf_dict_diff(local_dict: dict, ref_dict: dict) -> Tuple[float, float, int, str]:
    """
    Compare CCDF dicts (keyed by threshold like '1', '2', '4').

    Returns average diff across all thresholds.
    """
    # Find common threshold keys
    local_keys = set(local_dict.keys())
    ref_keys = set(ref_dict.keys())
    common_keys = local_keys & ref_keys

    if not common_keys:
        return 0.0, 0.0, 0, "No common threshold keys"

    all_diffs = []
    max_diffs = []
    total_points = 0

    for key in sorted(common_keys):
        local_ccdf = local_dict[key]
        ref_ccdf = ref_dict[key]

        if isinstance(local_ccdf, list) and isinstance(ref_ccdf, list):
            avg, max_d, n, _ = compute_ccdf_diff(local_ccdf, ref_ccdf)
            if n > 0:
                all_diffs.append(avg)
                max_diffs.append(max_d)
                total_points += n

    if not all_diffs:
        return 0.0, 0.0, 0, "No comparable CCDFs"

    # Average across all thresholds
    avg_diff = float(np.mean(all_diffs))
    max_diff = float(np.max(max_diffs))

    note = f"Compared {len(all_diffs)} thresholds: {sorted(common_keys)}"
    return avg_diff, max_diff, total_points, note


def compare_values(
    key_path: str,
    local_value: Any,
    ref_value: Any,
) -> KeyComparisonResult:
    """
    Compare local and reference values for a single key.

    For time series, computes average percent difference across all time points.
    """
    local_type = classify_value(local_value)
    ref_type = classify_value(ref_value)

    # Skip if types don't match or are non-comparable
    skip_types = {'unknown', 'empty_array', 'string', 'boolean', 'mixed_array', 'object'}
    if local_type in skip_types or ref_type in skip_types:
        return KeyComparisonResult(
            key_path=key_path,
            data_type=local_type,
            local_value=None,
            reference_value=None,
            avg_pct_diff=0.0,
            max_pct_diff=0.0,
            num_points=0,
            status='SKIP',
            note=f"Non-comparable type: {local_type}/{ref_type}"
        )

    # CCDF comparison (list of {x, y} points)
    if local_type == 'ccdf' and ref_type == 'ccdf':
        avg_diff, max_diff, n, note = compute_ccdf_diff(local_value, ref_value)

        # For CCDFs, diff is in percentage points (0-100 scale)
        # Use same thresholds as other metrics
        status = 'PASS' if avg_diff <= 5 else ('WARN' if avg_diff <= 25 else 'FAIL')

        return KeyComparisonResult(
            key_path=key_path,
            data_type='ccdf',
            local_value=f"{len(local_value)} pts",
            reference_value=f"{len(ref_value)} pts",
            avg_pct_diff=avg_diff,
            max_pct_diff=max_diff,
            num_points=n,
            status=status,
            note=note + " (diff in probability %pts)" if note else "diff in probability %pts",
        )

    # CCDF dict comparison (dict keyed by threshold like '1', '2', '4')
    if local_type == 'ccdf_dict' and ref_type == 'ccdf_dict':
        avg_diff, max_diff, n, note = compute_ccdf_dict_diff(local_value, ref_value)

        status = 'PASS' if avg_diff <= 5 else ('WARN' if avg_diff <= 25 else 'FAIL')

        return KeyComparisonResult(
            key_path=key_path,
            data_type='ccdf_dict',
            local_value=f"{len(local_value)} thresholds",
            reference_value=f"{len(ref_value)} thresholds",
            avg_pct_diff=avg_diff,
            max_pct_diff=max_diff,
            num_points=n,
            status=status,
            note=note + " (diff in probability %pts)" if note else "diff in probability %pts",
        )

    # Scalar comparison
    if local_type == 'scalar' and ref_type == 'scalar':
        if abs(ref_value) < 1e-10:
            pct_diff = 0.0 if abs(local_value) < 1e-10 else 100.0
        else:
            pct_diff = abs(local_value - ref_value) / abs(ref_value) * 100

        status = 'PASS' if pct_diff <= 5 else ('WARN' if pct_diff <= 25 else 'FAIL')

        return KeyComparisonResult(
            key_path=key_path,
            data_type='scalar',
            local_value=local_value,
            reference_value=ref_value,
            avg_pct_diff=pct_diff,
            max_pct_diff=pct_diff,
            num_points=1,
            status=status,
        )

    # Array comparison
    if local_type == 'array' and ref_type == 'array':
        avg_diff, max_diff, n, note = compute_array_diff(local_value, ref_value)

        # Use median values as summary
        local_median = float(np.median(local_value)) if local_value else 0
        ref_median = float(np.median(ref_value)) if ref_value else 0

        status = 'PASS' if avg_diff <= 5 else ('WARN' if avg_diff <= 25 else 'FAIL')

        return KeyComparisonResult(
            key_path=key_path,
            data_type='array',
            local_value=local_median,
            reference_value=ref_median,
            avg_pct_diff=avg_diff,
            max_pct_diff=max_diff,
            num_points=n,
            status=status,
            note=note,
        )

    # Time series comparison (has median key)
    if local_type == 'time_series' and ref_type == 'time_series':
        local_median = local_value.get('median', [])
        ref_median = ref_value.get('median', [])

        avg_diff, max_diff, n, note = compute_array_diff(local_median, ref_median, sort_arrays=False)

        # Use overall median as summary value
        local_summary = float(np.median(local_median)) if local_median else 0
        ref_summary = float(np.median(ref_median)) if ref_median else 0

        status = 'PASS' if avg_diff <= 5 else ('WARN' if avg_diff <= 25 else 'FAIL')

        return KeyComparisonResult(
            key_path=key_path,
            data_type='time_series',
            local_value=local_summary,
            reference_value=ref_summary,
            avg_pct_diff=avg_diff,
            max_pct_diff=max_diff,
            num_points=n,
            status=status,
            note=note,
        )

    # 2D array comparison (e.g., individual simulation data)
    if local_type == 'array_2d' and ref_type == 'array_2d':
        # Compare medians across simulations at each time point
        try:
            local_arr = np.array(local_value)
            ref_arr = np.array(ref_value)
            local_median = np.median(local_arr, axis=0).tolist()
            ref_median = np.median(ref_arr, axis=0).tolist()
            # Don't sort - these are time series (median at each time point)
            avg_diff, max_diff, n, note = compute_array_diff(local_median, ref_median, sort_arrays=False)

            local_summary = float(np.median(local_median)) if local_median else 0
            ref_summary = float(np.median(ref_median)) if ref_median else 0

            status = 'PASS' if avg_diff <= 5 else ('WARN' if avg_diff <= 25 else 'FAIL')

            return KeyComparisonResult(
                key_path=key_path,
                data_type='array_2d',
                local_value=local_summary,
                reference_value=ref_summary,
                avg_pct_diff=avg_diff,
                max_pct_diff=max_diff,
                num_points=n,
                status=status,
                note=note,
            )
        except Exception as e:
            return KeyComparisonResult(
                key_path=key_path,
                data_type='array_2d',
                local_value=None,
                reference_value=None,
                avg_pct_diff=0.0,
                max_pct_diff=0.0,
                num_points=0,
                status='SKIP',
                note=f"Error comparing 2D arrays: {e}"
            )

    # Mixed type - skip
    return KeyComparisonResult(
        key_path=key_path,
        data_type=f"{local_type}/{ref_type}",
        local_value=None,
        reference_value=None,
        avg_pct_diff=0.0,
        max_pct_diff=0.0,
        num_points=0,
        status='SKIP',
        note=f"Type mismatch: {local_type} vs {ref_type}"
    )


def find_shared_keys(local_data: Dict, ref_data: Dict) -> List[str]:
    """Find all keys that exist in both local and reference data.

    Excludes percentile keys (p10, p25, p75, p90) - we only compare medians.
    """
    local_keys = set(set_nested_keys(local_data))
    ref_keys = set(set_nested_keys(ref_data))
    shared = local_keys & ref_keys

    # Filter out percentile keys - only compare medians
    percentile_patterns = ['.p10', '.p25', '.p75', '.p90', '_p10', '_p25', '_p75', '_p90']
    filtered = [k for k in shared if not any(p in k for p in percentile_patterns)]

    return sorted(filtered)


def compare_apis(
    local_data: Dict,
    ref_data: Dict,
    verbose: bool = True,
) -> ComparisonSummary:
    """
    Automatically compare all shared keys between local and reference APIs.

    Args:
        local_data: Local API response
        ref_data: Reference API response
        verbose: Whether to print progress

    Returns:
        ComparisonSummary with all results
    """
    # Find shared keys
    shared_keys = find_shared_keys(local_data, ref_data)

    if verbose:
        print(f"  Found {len(shared_keys)} shared keys to compare")

    results = []
    passed = warned = failed = skipped = 0

    for key_path in shared_keys:
        local_value = get_nested_value(local_data, key_path)
        ref_value = get_nested_value(ref_data, key_path)

        result = compare_values(key_path, local_value, ref_value)
        results.append(result)

        if result.status == 'PASS':
            passed += 1
        elif result.status == 'WARN':
            warned += 1
        elif result.status == 'FAIL':
            failed += 1
        else:
            skipped += 1

    return ComparisonSummary(
        total_keys=len(shared_keys),
        compared_keys=passed + warned + failed,
        skipped_keys=skipped,
        passed=passed,
        warned=warned,
        failed=failed,
        results=results,
    )


def print_comparison_results(summary: ComparisonSummary, show_all: bool = True):
    """Print comparison results in a readable format."""
    print("\n" + "=" * 80)
    print("AUTOMATIC KEY COMPARISON RESULTS")
    print("=" * 80)

    print(f"\nSummary:")
    print(f"  Total shared keys: {summary.total_keys}")
    print(f"  Compared: {summary.compared_keys}")
    print(f"  Skipped: {summary.skipped_keys}")
    print(f"  ✓ PASS: {summary.passed}")
    print(f"  ⚠ WARN: {summary.warned}")
    print(f"  ✗ FAIL: {summary.failed}")

    # Group results by status
    fails = [r for r in summary.results if r.status == 'FAIL']
    warns = [r for r in summary.results if r.status == 'WARN']
    passes = [r for r in summary.results if r.status == 'PASS']
    skips = [r for r in summary.results if r.status == 'SKIP']

    def format_value(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            if abs(v) >= 1e6:
                return f"{v:.2e}"
            elif abs(v) >= 100:
                return f"{v:.1f}"
            elif abs(v) >= 1:
                return f"{v:.2f}"
            else:
                return f"{v:.4f}"
        return str(v)

    if fails:
        print(f"\n{'─' * 80}")
        print("FAILURES (avg diff > 25%):")
        print(f"{'─' * 80}")
        for r in sorted(fails, key=lambda x: -x.avg_pct_diff):
            print(f"  ✗ {r.key_path}")
            print(f"    Type: {r.data_type}, Points: {r.num_points}")
            print(f"    Local: {format_value(r.local_value)}, Ref: {format_value(r.reference_value)}")
            print(f"    Avg diff: {r.avg_pct_diff:.1f}%, Max diff: {r.max_pct_diff:.1f}%")
            if r.note:
                print(f"    Note: {r.note}")

    if warns:
        print(f"\n{'─' * 80}")
        print("WARNINGS (avg diff 5-25%):")
        print(f"{'─' * 80}")
        for r in sorted(warns, key=lambda x: -x.avg_pct_diff):
            print(f"  ⚠ {r.key_path}")
            print(f"    Type: {r.data_type}, Points: {r.num_points}")
            print(f"    Local: {format_value(r.local_value)}, Ref: {format_value(r.reference_value)}")
            print(f"    Avg diff: {r.avg_pct_diff:.1f}%, Max diff: {r.max_pct_diff:.1f}%")

    if passes:
        print(f"\n{'─' * 80}")
        print("PASSES (avg diff <= 5%):")
        print(f"{'─' * 80}")
        for r in sorted(passes, key=lambda x: -x.avg_pct_diff):
            print(f"  ✓ {r.key_path}: {r.avg_pct_diff:.1f}% (local: {format_value(r.local_value)}, ref: {format_value(r.reference_value)})")

    print(f"\n{'=' * 80}")
    if summary.failed > 0:
        print(f"RESULT: {summary.failed} FAILURES")
    elif summary.warned > 0:
        print(f"RESULT: {summary.warned} WARNINGS (no failures)")
    else:
        print("RESULT: ALL PASSED")
    print("=" * 80)
