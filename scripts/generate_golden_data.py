"""
Fetch golden data from aifuturesmodel.com production API.

This script fetches trajectory data from the production model at aifuturesmodel.com
and saves it as golden data for testing the PyTorch implementation.

Usage:
    python -m scripts.generate_golden_data [--regenerate]

Options:
    --regenerate    Regenerate all golden data files, even if they exist
"""

import argparse
import hashlib
import http.client
import json
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DATA_DIR = PROJECT_ROOT / "tests" / "golden_data"

# Production API configuration
PRODUCTION_API_URL = "https://aifuturesmodel.com/api/compute"
DEFAULT_TIME_RANGE = [2015, 2050]
DEFAULT_TIMEOUT = 60
MAX_REDIRECTS = 5


def fetch_trajectory(
    parameters: Dict[str, Any],
    time_range: Optional[List[int]] = None,
    initial_progress: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Fetch a single trajectory from the production API.

    Args:
        parameters: Model parameters to use. Empty dict uses defaults.
        time_range: [start_year, end_year]. Defaults to [2015, 2050].
        initial_progress: Initial progress value.
        timeout: Request timeout in seconds.

    Returns:
        API response containing time_series, summary, and milestones.
    """
    if time_range is None:
        time_range = DEFAULT_TIME_RANGE

    request_data = {
        "parameters": parameters,
        "time_range": time_range,
        "initial_progress": initial_progress,
    }

    data = json.dumps(request_data).encode("utf-8")

    # Use http.client directly to handle POST redirects properly
    url = PRODUCTION_API_URL
    redirects = 0

    while redirects < MAX_REDIRECTS:
        parsed = urllib.parse.urlparse(url)

        if parsed.scheme == "https":
            conn = http.client.HTTPSConnection(
                parsed.netloc,
                timeout=timeout,
                context=ssl.create_default_context(),
            )
        else:
            conn = http.client.HTTPConnection(parsed.netloc, timeout=timeout)

        try:
            path = parsed.path
            if parsed.query:
                path = f"{path}?{parsed.query}"

            conn.request(
                "POST",
                path,
                body=data,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(data)),
                },
            )
            response = conn.getresponse()

            # Handle redirects
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise RuntimeError(f"Redirect {response.status} without Location header")

                # Handle relative URLs
                if location.startswith("/"):
                    location = f"{parsed.scheme}://{parsed.netloc}{location}"

                url = location
                redirects += 1
                conn.close()
                continue

            if response.status != 200:
                body = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"API returned status {response.status}: {body[:500]}")

            result = json.loads(response.read().decode("utf-8"))

            if not result.get("success"):
                raise RuntimeError(f"API returned error: {result}")

            return result

        finally:
            conn.close()

    raise RuntimeError(f"Too many redirects (>{MAX_REDIRECTS})")


def save_golden_data(
    filename: str,
    response: Dict[str, Any],
    parameters: Dict[str, Any],
    time_range: List[int],
) -> Path:
    """
    Save golden data to a file.

    Adds metadata about generation parameters.
    """
    GOLDEN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Add generation metadata
    golden_data = {
        "_metadata": {
            "source": "aifuturesmodel.com/api/compute",
            "parameters": parameters,
            "time_range": time_range,
        },
        **response,
    }

    filepath = GOLDEN_DATA_DIR / filename
    with open(filepath, "w") as f:
        json.dump(golden_data, f, indent=2)

    print(f"  Saved: {filepath.relative_to(PROJECT_ROOT)}")
    return filepath


def generate_default_golden(regenerate: bool = False) -> Optional[Path]:
    """Generate golden data with default parameters."""
    filename = "default_trajectory.json"
    filepath = GOLDEN_DATA_DIR / filename

    if not regenerate and filepath.exists():
        print(f"  Skipping {filename} (already exists)")
        return filepath

    print(f"  Fetching default trajectory...")
    response = fetch_trajectory(parameters={}, time_range=DEFAULT_TIME_RANGE)
    return save_golden_data(filename, response, {}, DEFAULT_TIME_RANGE)


# Representative parameter sets to test - based on production model's common scenarios
SCENARIOS = [
    {
        "name": "fast_progress",
        "description": "Fast AI progress scenario",
        "params": {
            "present_doubling_time": 0.3,
        },
    },
    {
        "name": "slow_progress",
        "description": "Slow AI progress scenario",
        "params": {
            "present_doubling_time": 0.8,
        },
    },
    {
        "name": "high_automation",
        "description": "High automation potential",
        "params": {
            "rho_coding_labor": -1,
        },
    },
    {
        "name": "low_automation",
        "description": "Low automation potential",
        "params": {
            "rho_coding_labor": -5,
        },
    },
]


def generate_scenario_golden(scenario: Dict[str, Any], regenerate: bool = False) -> Optional[Path]:
    """Generate golden data for a specific scenario."""
    name = scenario["name"]
    params = scenario["params"]
    filename = f"trajectory_{name}.json"
    filepath = GOLDEN_DATA_DIR / filename

    if not regenerate and filepath.exists():
        print(f"  Skipping {filename} (already exists)")
        return filepath

    print(f"  Fetching {scenario['description']}...")
    response = fetch_trajectory(parameters=params, time_range=DEFAULT_TIME_RANGE)
    return save_golden_data(filename, response, params, DEFAULT_TIME_RANGE)


def generate_all_golden_data(regenerate: bool = False) -> List[Path]:
    """Generate all golden data files."""
    print("Generating golden data from aifuturesmodel.com production API")
    print("=" * 60)

    generated = []

    # Default trajectory
    print("\nDefault trajectory:")
    path = generate_default_golden(regenerate=regenerate)
    if path:
        generated.append(path)

    # Scenario trajectories
    print("\nScenario trajectories:")
    for scenario in SCENARIOS:
        path = generate_scenario_golden(scenario, regenerate=regenerate)
        if path:
            generated.append(path)

    print(f"\nGenerated {len(generated)} golden data files")
    return generated


def verify_input_data_match() -> bool:
    """
    Verify that local input_data.csv matches production.

    Returns True if they match, False otherwise.
    """
    local_input_data = PROJECT_ROOT / "ai_futures_simulator" / "parameters" / "input_data.csv"

    if not local_input_data.exists():
        print("ERROR: Local input_data.csv not found")
        return False

    with open(local_input_data, "rb") as f:
        local_hash = hashlib.sha256(f.read()).hexdigest()

    # Store expected hash in golden data for verification
    expected_hash_file = GOLDEN_DATA_DIR / "input_data_hash.txt"

    if expected_hash_file.exists():
        with open(expected_hash_file, "r") as f:
            expected_hash = f.read().strip()

        if local_hash != expected_hash:
            print(f"WARNING: input_data.csv hash mismatch!")
            print(f"  Local:    {local_hash}")
            print(f"  Expected: {expected_hash}")
            return False

        print("  input_data.csv hash verified")
        return True
    else:
        # First time - save the hash
        GOLDEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(expected_hash_file, "w") as f:
            f.write(local_hash)
        print(f"  Saved input_data.csv hash: {local_hash[:16]}...")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Fetch golden data from aifuturesmodel.com production API"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate all golden data files, even if they exist",
    )
    parser.add_argument(
        "--verify-input-data",
        action="store_true",
        help="Only verify input_data.csv hash matches",
    )
    args = parser.parse_args()

    if args.verify_input_data:
        print("\nVerifying input_data.csv...")
        if verify_input_data_match():
            print("OK")
            sys.exit(0)
        else:
            print("FAILED")
            sys.exit(1)

    # Verify input data first
    print("\nVerifying input_data.csv...")
    if not verify_input_data_match():
        print("WARNING: Continuing anyway, but tests may fail")

    # Generate golden data
    generate_all_golden_data(regenerate=args.regenerate)


if __name__ == "__main__":
    main()
