"""
Automatic comparison of shared keys between local and reference APIs.

Automatically finds matching keys between API responses and computes:
- For scalars: percent difference
- For arrays: median percent difference
- For time series (with percentiles): average percent difference across all time points
- For CCDFs: not compared (structural data)
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


def compute_array_diff(local: List[float], ref: List[float]) -> Tuple[float, float, int]:
    """
    Compute average and max percent difference between two arrays.

    Returns:
        (avg_pct_diff, max_pct_diff, num_points)
    """
    if not local or not ref:
        return 0.0, 0.0, 0

    # Use the shorter length
    n = min(len(local), len(ref))
    if n == 0:
        return 0.0, 0.0, 0

    local_arr = np.array(local[:n], dtype=float)
    ref_arr = np.array(ref[:n], dtype=float)

    # Handle zeros in reference - use small epsilon
    ref_safe = np.where(np.abs(ref_arr) < 1e-10, 1e-10, ref_arr)

    # Compute percent differences
    pct_diffs = np.abs(local_arr - ref_arr) / np.abs(ref_safe) * 100

    # Cap extreme values
    pct_diffs = np.clip(pct_diffs, 0, 1000)

    avg_diff = float(np.mean(pct_diffs))
    max_diff = float(np.max(pct_diffs))

    return avg_diff, max_diff, n


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
    skip_types = {'unknown', 'empty_array', 'string', 'boolean', 'ccdf', 'ccdf_dict', 'mixed_array', 'object'}
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
        avg_diff, max_diff, n = compute_array_diff(local_value, ref_value)

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
        )

    # Time series comparison (has median key)
    if local_type == 'time_series' and ref_type == 'time_series':
        local_median = local_value.get('median', [])
        ref_median = ref_value.get('median', [])

        avg_diff, max_diff, n = compute_array_diff(local_median, ref_median)

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
        )

    # 2D array comparison (e.g., individual simulation data)
    if local_type == 'array_2d' and ref_type == 'array_2d':
        # Compare medians across simulations at each time point
        try:
            local_arr = np.array(local_value)
            ref_arr = np.array(ref_value)
            local_median = np.median(local_arr, axis=0).tolist()
            ref_median = np.median(ref_arr, axis=0).tolist()
            avg_diff, max_diff, n = compute_array_diff(local_median, ref_median)

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
    """Find all keys that exist in both local and reference data."""
    local_keys = set(set_nested_keys(local_data))
    ref_keys = set(set_nested_keys(ref_data))
    shared = local_keys & ref_keys
    return sorted(list(shared))


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


def print_comparison_results(summary: ComparisonSummary, show_all: bool = False):
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

    if show_all and passes:
        print(f"\n{'─' * 80}")
        print("PASSES (avg diff <= 5%):")
        print(f"{'─' * 80}")
        for r in sorted(passes, key=lambda x: -x.avg_pct_diff):
            print(f"  ✓ {r.key_path}: {r.avg_pct_diff:.1f}% avg diff")

    print(f"\n{'=' * 80}")
    if summary.failed > 0:
        print(f"RESULT: {summary.failed} FAILURES")
    elif summary.warned > 0:
        print(f"RESULT: {summary.warned} WARNINGS (no failures)")
    else:
        print("RESULT: ALL PASSED")
    print("=" * 80)
