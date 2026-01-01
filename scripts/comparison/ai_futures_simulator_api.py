"""
ai_futures_simulator API client for fetching data from the continuous model.

Runs simulations directly in-process (no HTTP server needed).
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

from .config import (
    CACHE_DIR,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_START_YEAR,
    DEFAULT_END_YEAR,
)


def fetch_ai_futures_simulator(
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    use_cache: bool = True,
    verbose: bool = True,
) -> Optional[Dict]:
    """Run ai_futures_simulator simulations directly in-process."""
    cache_file = CACHE_DIR / f"afs_{num_simulations}_{start_year}_{end_year}.json"

    if use_cache and cache_file.exists():
        if verbose:
            print(f"  Loading cached data from {cache_file.name}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    if verbose:
        print(f"  Running {num_simulations} ai_futures_simulator simulations...")

    try:
        import sys
        repo_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(repo_root))
        sys.path.insert(0, str(repo_root / "app_backend"))
        sys.path.insert(0, str(repo_root / "ai_futures_simulator"))
        from api_utils.black_project_simulation import run_black_project_simulations, extract_black_project_plot_data

        start = time.time()
        frontend_params = {}
        result = run_black_project_simulations(
            frontend_params=frontend_params,
            num_simulations=num_simulations,
            time_range=[start_year, end_year],
        )
        response = extract_black_project_plot_data(result, frontend_params)
        elapsed = time.time() - start

        if verbose:
            print(f"  Simulations completed in {elapsed:.1f}s")

        if use_cache:
            CACHE_DIR.mkdir(exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(response, f)
            if verbose:
                print(f"  Cached to {cache_file.name}")

        return response

    except Exception as e:
        if verbose:
            print(f"  ERROR running simulations: {e}")
            import traceback
            traceback.print_exc()
        return None
