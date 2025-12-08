#!/usr/bin/env python3
"""
Extract all numeric data from rollouts.cache.json into a pandas DataFrame.

This module provides efficient extraction of parameters, milestone times, and
results from the cache file. The resulting DataFrame has one row per rollout
with columns for all numeric fields.

Performance optimizations:
- Single pass through the JSON data
- Row-based dict collection (most efficient for sparse data)
- Leverages pandas' optimized C/Cython internals for DataFrame construction
- Automatic handling of missing values as NaN
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def _is_numeric(value: Any) -> bool:
    """Check if a value is numeric (int, float, or numpy numeric type)."""
    return isinstance(value, (int, float, np.integer, np.floating))


def _flatten_dict(
    d: Dict[str, Any],
    prefix: str = "",
    separator: str = "."
) -> Dict[str, float]:
    """
    Flatten a nested dictionary, extracting only numeric values.

    Args:
        d: Dictionary to flatten
        prefix: Prefix for keys (used in recursion)
        separator: Separator between nested keys

    Returns:
        Flattened dictionary with only numeric values
    """
    result = {}
    for key, value in d.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively flatten nested dicts
            result.update(_flatten_dict(value, new_key, separator))
        elif _is_numeric(value):
            result[new_key] = float(value)
        # Skip non-numeric values (lists, strings, etc.)

    return result


def extract_dataframe(cache_path: Path) -> pd.DataFrame:
    """
    Extract all numeric data from rollouts.cache.json into a DataFrame.

    The DataFrame will have columns for:
    - sample_id (index)
    - has_error (1 if rollout failed, 0 if successful)
    - All parameters (prefixed with 'param.')
    - All time_series_parameters (prefixed with 'ts_param.')
    - All numeric results fields
    - All milestone times (prefixed with 'milestone.')
    - All milestone metrics (prefixed with 'milestone.<name>.')

    Args:
        cache_path: Path to rollouts.cache.json

    Returns:
        DataFrame with one row per rollout and columns for all numeric data
    """
    print(f"Loading cache file: {cache_path}")

    with open(cache_path, 'r') as f:
        data = json.load(f)

    rollouts = data.get('rollouts', [])
    print(f"Processing {len(rollouts)} rollouts...")

    # Collect data as one dict per rollout (correct and most performant)
    rollout_dicts = []
    sample_ids = []

    for i, rollout in enumerate(rollouts):
        if i > 0 and i % 1000 == 0:
            print(f"  Processed {i}/{len(rollouts)} rollouts...")

        sample_id = rollout.get('sample_id')
        sample_ids.append(sample_id)

        # Create a dict for this rollout's data
        row_data = {}

        # Check if rollout has an error (1 = error, 0 = no error)
        row_data['has_error'] = 1 if 'error' in rollout else 0

        # Extract parameters
        params = rollout.get('parameters', {})
        row_data.update(_flatten_dict(params, prefix='param'))

        # Extract time series parameters
        ts_params = rollout.get('time_series_parameters', {})
        row_data.update(_flatten_dict(ts_params, prefix='ts_param'))

        # Extract results (excluding non-numeric fields)
        results = rollout.get('results', {})

        # Handle milestones separately
        milestones = results.get('milestones', {})
        for milestone_name, milestone_info in milestones.items():
            if not isinstance(milestone_info, dict):
                continue

            # Extract milestone time
            if 'time' in milestone_info and _is_numeric(milestone_info['time']):
                key = f'milestone.{milestone_name}.time'
                row_data[key] = float(milestone_info['time'])

            # Extract other numeric milestone fields
            for field_name, field_value in milestone_info.items():
                if field_name != 'time' and _is_numeric(field_value):
                    key = f'milestone.{milestone_name}.{field_name}'
                    row_data[key] = float(field_value)

        # Extract other numeric results (excluding input_time_series, exp_capacity_params, milestones)
        for key, value in results.items():
            if key in ('input_time_series', 'milestones'):
                # Skip time series data (too large) and milestones (already handled)
                continue
            elif key == 'exp_capacity_params':
                # Flatten exp_capacity_params
                exp_data = _flatten_dict(value, prefix='exp_capacity')
                row_data.update(exp_data)
            elif key == 'takeoff_progress_multipliers':
                # Skip lists for now (could add length as a feature)
                continue
            elif _is_numeric(value):
                row_data[key] = float(value)

        rollout_dicts.append(row_data)

    print(f"Processed {len(rollouts)} rollouts")

    # Convert to DataFrame (pandas automatically handles missing values as NaN)
    print(f"Creating DataFrame...")
    df = pd.DataFrame(rollout_dicts, index=sample_ids)
    df.index.name = 'sample_id'

    print(f"\nDataFrame shape: {df.shape}")
    print(f"  Rows (rollouts): {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    return df


def save_dataframe(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save DataFrame as pickle file.

    Args:
        df: DataFrame to save
        output_dir: Directory to save in
    """
    output_dir = Path(output_dir)
    pickle_path = output_dir / "rollouts_dataframe.pkl"

    print(f"\nSaving pickle to: {pickle_path}")
    df.to_pickle(pickle_path)

    print(f"\nTo load in Python:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_pickle('{pickle_path}')")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Extract all numeric data from rollouts.cache.json into a pandas DataFrame"
    )
    parser.add_argument(
        "cache_path",
        type=Path,
        help="Path to rollouts.cache.json file (e.g., outputs/20251113-eli-10k/rollouts.cache.json)"
    )

    args = parser.parse_args()
    cache_path = args.cache_path

    if not cache_path.exists():
        print(f"Error: Cache file not found: {cache_path}")
        return

    # Extract data
    df = extract_dataframe(cache_path)

    # Print summary
    print("\n" + "="*60)
    print("DATAFRAME SUMMARY")
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"\nColumn categories:")

    # Count columns by prefix
    prefixes = defaultdict(int)
    for col in df.columns:
        if '.' in col:
            prefix = col.split('.')[0]
            prefixes[prefix] += 1
        else:
            prefixes['results'] += 1

    for prefix, count in sorted(prefixes.items()):
        print(f"  {prefix}: {count} columns")

    # Save
    save_dataframe(df, cache_path.parent)

    print("="*60)


if __name__ == "__main__":
    main()
