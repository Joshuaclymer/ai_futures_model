"""
Local API client for fetching data from the continuous model backend.

Fetches from the /api/get-data-for-ai-black-projects-page endpoint
which serves the same data displayed on the frontend.
"""

import json
import time
import urllib.request
from typing import Dict, Optional

from .config import (
    LOCAL_API_URL,
    CACHE_DIR,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_AGREEMENT_YEAR,
)


def fetch_local_api(
    num_simulations: int = DEFAULT_NUM_SAMPLES,
    agreement_year: int = DEFAULT_AGREEMENT_YEAR,
    num_years: int = 10,
    timeout: int = 600,
    use_cache: bool = True,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Fetch simulation results from the local API.

    Args:
        num_simulations: Number of Monte Carlo simulations to run
        agreement_year: Agreement/slowdown start year
        num_years: Number of years to simulate
        timeout: API request timeout in seconds
        use_cache: Whether to use cached responses
        verbose: Whether to print progress messages

    Returns:
        API response as a dictionary, or None if request failed
    """
    cache_file = CACHE_DIR / f"local_{num_simulations}_{agreement_year}_{num_years}.json"

    # Try cache first
    if use_cache and cache_file.exists():
        if verbose:
            print(f"  Loading cached local data from {cache_file.name}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    if verbose:
        print(f"  Calling local API ({num_simulations} simulations)...")

    data = json.dumps({
        'parameters': {},
        'num_simulations': num_simulations,
        'time_range': [agreement_year, agreement_year + num_years],
    }).encode()

    req = urllib.request.Request(
        LOCAL_API_URL,
        data=data,
        headers={'Content-Type': 'application/json'}
    )

    try:
        start = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode())
        elapsed = time.time() - start

        if verbose:
            print(f"  Local API responded in {elapsed:.1f}s")

        # Check for error (new format doesn't have success wrapper)
        if result.get('error'):
            if verbose:
                print(f"  ERROR: Local API returned error: {result.get('error')}")
            return None

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
            print(f"  ERROR calling local API: {e}")
        return None


def extract_local_metrics(response: Dict) -> Dict[str, Dict]:
    """
    Extract metrics from local API response into comparison format.

    Maps the local API response structure to the format expected by
    the comparison functions.

    Args:
        response: Raw local API response

    Returns:
        Dictionary of metrics with statistics (median, p10, p90)
    """
    metrics = {}

    # Extract years
    black_project_model = response.get('black_project_model', {})
    rate_of_computation = response.get('rate_of_computation', {})
    detection_likelihood = response.get('detection_likelihood', {})
    black_datacenters = response.get('black_datacenters', {})
    covert_fab = response.get('covert_fab', {})

    # Years from black_project_model
    metrics['years'] = black_project_model.get('years', [])

    def convert_percentiles(data: Dict) -> Dict:
        """Convert p25/p75 to p10/p90 format expected by comparison code."""
        if not data:
            return {}
        return {
            'median': data.get('median', []),
            'p10': data.get('p25', data.get('p10', data.get('median', []))),
            'p90': data.get('p75', data.get('p90', data.get('median', []))),
        }

    # Survival rate: rate_of_computation.surviving_fraction
    if 'surviving_fraction' in rate_of_computation:
        metrics['survival_rate'] = convert_percentiles(rate_of_computation['surviving_fraction'])

    # Cumulative LR: black_datacenters.lr_datacenters (combined LR over time)
    if 'lr_datacenters' in black_datacenters:
        metrics['cumulative_lr'] = convert_percentiles(black_datacenters['lr_datacenters'])

    # Posterior probability: detection_likelihood.posterior_prob
    if 'posterior_prob' in detection_likelihood:
        metrics['posterior_prob'] = convert_percentiles(detection_likelihood['posterior_prob'])

    # LR other intel (direct evidence): detection_likelihood.direct_evidence
    if 'direct_evidence' in detection_likelihood:
        metrics['lr_other_intel'] = convert_percentiles(detection_likelihood['direct_evidence'])

    # LR reported energy: detection_likelihood.energy_evidence
    if 'energy_evidence' in detection_likelihood:
        metrics['lr_reported_energy'] = convert_percentiles(detection_likelihood['energy_evidence'])

    # Combined evidence LR: detection_likelihood.combined_evidence
    if 'combined_evidence' in detection_likelihood:
        metrics['combined_evidence'] = convert_percentiles(detection_likelihood['combined_evidence'])

    # LR PRC accounting: from initial_stock samples (static value, replicate across time)
    initial_stock = response.get('initial_stock', {})
    lr_prc_samples = initial_stock.get('lr_prc_accounting_samples', [])
    if lr_prc_samples and metrics.get('years'):
        import numpy as np
        median_val = float(np.median(lr_prc_samples))
        p10_val = float(np.percentile(lr_prc_samples, 10))
        p90_val = float(np.percentile(lr_prc_samples, 90))
        num_years = len(metrics['years'])
        metrics['lr_prc_accounting'] = {
            'median': [median_val] * num_years,
            'p10': [p10_val] * num_years,
            'p90': [p90_val] * num_years,
        }

    # Operating compute: black_project_model.operational_compute
    if 'operational_compute' in black_project_model:
        metrics['operating_compute'] = convert_percentiles(black_project_model['operational_compute'])

    # Covert chip stock: black_project_model.covert_chip_stock
    if 'covert_chip_stock' in black_project_model:
        metrics['covert_chip_stock'] = convert_percentiles(black_project_model['covert_chip_stock'])

    # Total black project: same as covert_chip_stock (maps to reference total_black_project)
    if 'covert_chip_stock' in black_project_model:
        metrics['total_black_project'] = convert_percentiles(black_project_model['covert_chip_stock'])

    # Datacenter capacity: black_project_model.datacenter_capacity or black_datacenters.datacenter_capacity
    if 'datacenter_capacity' in black_project_model:
        metrics['datacenter_capacity'] = convert_percentiles(black_project_model['datacenter_capacity'])
    elif 'datacenter_capacity' in black_datacenters:
        metrics['datacenter_capacity'] = convert_percentiles(black_datacenters['datacenter_capacity'])

    # Additional metrics that may be useful
    # Fab monthly production
    time_series = covert_fab.get('time_series_data', {})
    if 'h100e_flow' in time_series:
        metrics['fab_cumulative_production'] = convert_percentiles(time_series['h100e_flow'])

    # Fab combined LR: covert_fab.time_series_data.lr_combined
    if 'lr_combined' in time_series:
        metrics['lr_fab_combined'] = convert_percentiles(time_series['lr_combined'])

    # Energy usage (GW): rate_of_computation.energy_usage
    if 'energy_usage' in rate_of_computation:
        metrics['energy_usage'] = convert_percentiles(rate_of_computation['energy_usage'])

    # Covert computation rate: rate_of_computation.covert_computation
    if 'covert_computation' in rate_of_computation:
        metrics['covert_computation'] = convert_percentiles(rate_of_computation['covert_computation'])

    # H100 years (cumulative compute utilization): same as covert_computation
    # This maps to h100_years in the reference API (black_project_model.h100_years)
    if 'covert_computation' in rate_of_computation:
        metrics['h100_years'] = convert_percentiles(rate_of_computation['covert_computation'])

    # PRC operating compute: prc_compute.operating_compute
    prc_compute = response.get('prc_compute', {})
    if 'operating_compute' in prc_compute:
        metrics['prc_operating_compute'] = convert_percentiles(prc_compute['operating_compute'])

    # PRC compute stock: prc_compute.compute_stock
    if 'compute_stock' in prc_compute:
        metrics['prc_compute_stock'] = convert_percentiles(prc_compute['compute_stock'])

    # PRC capacity (GW): black_datacenters.prc_capacity_gw
    if 'prc_capacity_gw' in black_datacenters:
        metrics['prc_capacity_gw'] = convert_percentiles(black_datacenters['prc_capacity_gw'])

    return metrics


def clear_local_cache():
    """Clear all cached local API responses."""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob("local_*.json"):
            cache_file.unlink()
            print(f"  Deleted {cache_file.name}")
