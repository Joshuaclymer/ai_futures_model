#!/usr/bin/env python3
"""
Main entry point for comparing local API with reference discrete model.

Automatically compares all shared keys between:
- ai_futures_simulator: continuous ODE model (runs in-process)
- Reference API: http://127.0.0.1:5001/run_simulation (discrete model)

For time series data, computes the *average* percent difference across all time points.

Usage:
    python -m scripts.comparison.main [--samples N] [--no-cache] [--show-all]
"""

import argparse
import sys

from .config import (
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_AI_SLOWDOWN_START_YEAR,
    DEFAULT_END_YEAR,
)
from .reference_api import fetch_reference_api, clear_cache as clear_reference_cache
from .ai_futures_simulator_api import fetch_ai_futures_simulator
from .auto_compare import compare_apis, print_comparison_results


def run_comparison(
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    ai_slowdown_start_year: int = DEFAULT_AI_SLOWDOWN_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    use_cache: bool = True,
    show_all: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Run full comparison between local API and reference API.

    Automatically compares all shared keys between the two APIs.
    For time series, computes average percent difference.

    Args:
        num_simulations: Number of Monte Carlo simulations
        ai_slowdown_start_year: Year when AI slowdown agreement takes effect
        end_year: Simulation end year
        use_cache: Whether to use cached reference API responses
        show_all: Whether to show passing comparisons too
        verbose: Whether to print progress messages

    Returns:
        True if no failures, False otherwise
    """
    print("=" * 60)
    print("ai_futures_simulator vs Reference API Comparison")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Simulations: {num_simulations}")
    print(f"  Agreement year: {ai_slowdown_start_year}, End year: {end_year}")
    print(f"  Use Cache (reference only): {use_cache}")
    print()

    # Step 1: Fetch reference data
    print("Step 1: Fetching reference API data")
    ref_response = fetch_reference_api(
        num_simulations=num_simulations,
        start_year=ai_slowdown_start_year,  # Reference API still uses start_year
        end_year=end_year,
        use_cache=use_cache,
        verbose=verbose,
    )

    if ref_response is None:
        print("ERROR: Failed to fetch reference data")
        return False

    # Step 2: Fetch ai_futures_simulator data (never cached - always run fresh)
    print("\nStep 2: Fetching ai_futures_simulator data")
    afs_response = fetch_ai_futures_simulator(
        num_simulations=num_simulations,
        ai_slowdown_start_year=ai_slowdown_start_year,
        end_year=end_year,
        use_cache=False,  # Never cache ai_futures_simulator - always run fresh
        verbose=verbose,
    )

    if afs_response is None:
        print("ERROR: Failed to run ai_futures_simulator")
        return False

    # Step 3: Automatically compare all shared keys
    print("\nStep 3: Comparing shared keys (computing average % diff for time series)")
    summary = compare_apis(
        local_data=afs_response,
        ref_data=ref_response,
        verbose=verbose,
    )

    # Step 4: Print results
    print_comparison_results(summary, show_all=show_all)

    # Return success status
    return summary.failed == 0


def clear_cache():
    """Clear cached reference API responses (local simulator is never cached)."""
    print("Clearing reference API cache...")
    clear_reference_cache()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Compare local API with reference discrete model'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=DEFAULT_NUM_SIMULATIONS,
        help=f'Number of Monte Carlo simulations (default: {DEFAULT_NUM_SIMULATIONS})'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable reference API response caching'
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
        num_simulations=args.samples,
        use_cache=not args.no_cache,
        show_all=args.show_all,
        verbose=not args.quiet,
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
