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
    DEFAULT_NUM_SAMPLES,
    DEFAULT_START_YEAR,
    DEFAULT_TOTAL_LABOR,
)


def fetch_reference_api(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    start_year: int = DEFAULT_START_YEAR,
    total_labor: int = DEFAULT_TOTAL_LABOR,
    timeout: int = 180,
    use_cache: bool = True,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Fetch simulation results from the reference API.

    Args:
        num_samples: Number of Monte Carlo samples to run
        start_year: Black project start year
        total_labor: Total labor allocated to black project
        timeout: API request timeout in seconds
        use_cache: Whether to use cached responses
        verbose: Whether to print progress messages

    Returns:
        API response as a dictionary, or None if request failed
    """
    cache_file = CACHE_DIR / f"reference_{num_samples}_{start_year}_{total_labor}.json"

    # Try cache first
    if use_cache and cache_file.exists():
        if verbose:
            print(f"  Loading cached reference data from {cache_file.name}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    if verbose:
        print(f"  Calling reference API ({num_samples} samples)...")

    data = json.dumps({
        'num_samples': num_samples,
        'start_year': start_year,
        'total_labor': total_labor,
    }).encode()

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
