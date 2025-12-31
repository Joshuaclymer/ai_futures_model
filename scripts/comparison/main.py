#!/usr/bin/env python3
"""
Main entry point for comparing local API with reference discrete model.

Automatically compares all shared keys between:
- Local API: /api/get-data-for-ai-black-projects-page (continuous ODE model)
- Reference API: https://dark-compute.onrender.com/run_simulation (discrete model)

For time series data, computes the *average* percent difference across all time points.

Usage:
    python -m scripts.comparison.main [--samples N] [--no-cache] [--show-all]
"""

import argparse
import sys
from pathlib import Path

from .config import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_START_YEAR,
    DEFAULT_AGREEMENT_YEAR,
    DEFAULT_NUM_YEARS,
    DEFAULT_TOTAL_LABOR,
)
from .reference_api import fetch_reference_api, clear_cache as clear_reference_cache
from .local_simulator import fetch_local_api, clear_local_cache
from .auto_compare import compare_apis, print_comparison_results


def run_comparison(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    use_cache: bool = True,
    show_all: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Run full comparison between local API and reference API.

    Automatically compares all shared keys between the two APIs.
    For time series, computes average percent difference.

    Args:
        num_samples: Number of Monte Carlo samples
        use_cache: Whether to use cached API responses
        show_all: Whether to show passing comparisons too
        verbose: Whether to print progress messages

    Returns:
        True if no failures, False otherwise
    """
    print("=" * 60)
    print("Local API vs Reference API Comparison")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Samples: {num_samples}")
    print(f"  Agreement Year: {DEFAULT_AGREEMENT_YEAR}")
    print(f"  Simulation Years: {DEFAULT_NUM_YEARS}")
    print(f"  Reference Start Year: {DEFAULT_START_YEAR}")
    print(f"  Total Labor: {DEFAULT_TOTAL_LABOR}")
    print(f"  Use Cache: {use_cache}")
    print()

    # Step 1: Fetch reference data
    print("Step 1: Fetching reference API data")
    ref_response = fetch_reference_api(
        num_samples=num_samples,
        use_cache=use_cache,
        verbose=verbose,
    )

    if ref_response is None:
        print("ERROR: Failed to fetch reference data")
        return False

    # Step 2: Fetch local API data
    print("\nStep 2: Fetching local API data")
    local_response = fetch_local_api(
        num_simulations=num_samples,
        agreement_year=DEFAULT_AGREEMENT_YEAR,
        num_years=DEFAULT_NUM_YEARS,
        use_cache=use_cache,
        verbose=verbose,
    )

    if local_response is None:
        print("ERROR: Failed to fetch local API data")
        print("Make sure the backend is running at http://localhost:5329")
        return False

    # Step 3: Automatically compare all shared keys
    print("\nStep 3: Comparing shared keys (computing average % diff for time series)")
    summary = compare_apis(
        local_data=local_response,
        ref_data=ref_response,
        verbose=verbose,
    )

    # Step 4: Print results
    print_comparison_results(summary, show_all=show_all)

    # Return success status
    return summary.failed == 0


def clear_cache():
    """Clear all cached API responses."""
    print("Clearing reference API cache...")
    clear_reference_cache()
    print("Clearing local API cache...")
    clear_local_cache()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Compare local API with reference discrete model'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f'Number of Monte Carlo samples (default: {DEFAULT_NUM_SAMPLES})'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable API response caching'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cached API responses and exit'
    )
    parser.add_argument(
        '--show-all', '-a',
        action='store_true',
        help='Show all comparisons including passes'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    if args.clear_cache:
        print("Clearing cache...")
        clear_cache()
        print("Done.")
        return 0

    success = run_comparison(
        num_samples=args.samples,
        use_cache=not args.no_cache,
        show_all=args.show_all,
        verbose=not args.quiet,
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
