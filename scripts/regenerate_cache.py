#!/usr/bin/env python3
"""
Regenerate the black project cache with fresh simulations.

Usage:
    python scripts/regenerate_cache.py [--num-simulations N]

Default is 1000 simulations.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add paths for imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "ai_futures_simulator"))
sys.path.insert(0, str(repo_root / "app_backend"))

from endpoints.black_project import run_black_project_simulations, extract_black_project_plot_data


def main():
    parser = argparse.ArgumentParser(description="Regenerate black project cache")
    parser.add_argument(
        "--num-simulations", "-n",
        type=int,
        default=1000,
        help="Number of simulations to run (default: 1000)"
    )
    parser.add_argument(
        "--agreement-year",
        type=int,
        default=2030,
        help="Year when AI slowdown agreement takes effect (default: 2030)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2037,
        help="Year when simulation ends (default: 2037)"
    )
    args = parser.parse_args()

    cache_path = repo_root / "app_backend" / "cache" / "black_project_default.json"

    print(f"Regenerating black project cache...")
    print(f"  Simulations: {args.num_simulations}")
    print(f"  Agreement year: {args.agreement_year}")
    print(f"  End year: {args.end_year}")
    print()

    start_time = time.time()

    # Run simulations
    result = run_black_project_simulations(
        frontend_params={},
        num_simulations=args.num_simulations,
        agreement_year=args.agreement_year,
        end_year=args.end_year,
    )

    # Extract plot data
    response = extract_black_project_plot_data(result, {})

    # Write to cache file
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(response, f)

    elapsed = time.time() - start_time
    print(f"Cache regenerated in {elapsed:.1f}s")
    print(f"  Output: {cache_path}")


if __name__ == "__main__":
    main()
