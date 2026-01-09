"""
Reference API client for fetching data from the discrete model.
"""

import json
import time
import urllib.request
from typing import Dict, Optional

from .config import (
    REFERENCE_API_URL,
    CACHE_DIR,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_AGREEMENT_YEAR,
    DEFAULT_END_YEAR,
)


def fetch_reference_api(
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    start_year: int = DEFAULT_AGREEMENT_YEAR,  # Reference API uses start_year naming
    end_year: int = DEFAULT_END_YEAR,
    timeout: int = 180,
    use_cache: bool = True,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Fetch simulation results from the reference API.

    Args:
        num_simulations: Number of Monte Carlo simulations to run
        start_year: Agreement year (reference API uses start_year naming)
        end_year: Simulation end year
        timeout: API request timeout in seconds
        use_cache: Whether to use cached responses
        verbose: Whether to print progress messages

    Returns:
        API response as a dictionary, or None if request failed
    """
    cache_file = CACHE_DIR / f"reference_{num_simulations}_{start_year}_{end_year}.json"

    # Try cache first
    if use_cache and cache_file.exists():
        if verbose:
            print(f"  Loading cached reference data from {cache_file.name}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    if verbose:
        print(f"  Calling reference API ({num_simulations} simulations)...")

    num_years = end_year - start_year
    request_data = {
        'simulation_settings.num_simulations': num_simulations,
        'simulation_settings.agreement_start_year': start_year,
        'simulation_settings.num_years_to_simulate': num_years,
    }
    # When local cache is disabled, also disable backend cache to ensure fresh results
    if not use_cache:
        request_data['use_cache'] = False
    data = json.dumps(request_data).encode()

    req = urllib.request.Request(
        REFERENCE_API_URL,
        data=data,
        headers={'Content-Type': 'application/json'}
    )

    try:
        start = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode())
        elapsed = time.time() - start

        if verbose:
            print(f"  Reference API responded in {elapsed:.1f}s")

        # Cache the result
        if use_cache:
            CACHE_DIR.mkdir(exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            if verbose:
                print(f"  Cached to {cache_file.name}")

        return result

    except Exception as e:
        if verbose:
            print(f"  ERROR calling reference API: {e}")
        return None


def clear_cache():
    """Clear all cached reference API responses."""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob("reference_*.json"):
            cache_file.unlink()
            print(f"  Deleted {cache_file.name}")
