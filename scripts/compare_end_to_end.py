"""
END-TO-END SIMULATION COMPARISON

Comprehensive comparison of the continuous ODE-based model vs the discrete reference model.
Compares all major components:
1. Initial conditions (compute, datacenter capacity)
2. Time evolution (survival, capacity growth, operating compute)
3. Fab production (when enabled)
4. Likelihood ratios from all sources
5. Cumulative LR and detection probability

Optionally calls the reference API to validate against the actual discrete model.
"""

import sys
import os
import math
import json
import urllib.request
import time
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')


@dataclass
class SimulationConfig:
    """Configuration for end-to-end simulation comparison."""
    # Time settings
    agreement_year: float = 2030.0
    num_years: int = 7

    # Initial conditions
    prc_compute_stock_at_agreement: float = 1e6  # H100e
    proportion_to_divert: float = 0.05

    # Hazard rates
    initial_hazard_rate: float = 0.05
    hazard_rate_increase_per_year: float = 0.02

    # Labor allocation
    total_labor: int = 11300
    frac_datacenter_construction: float = 0.88  # ~10,000 workers
    frac_ai_research: float = 0.04  # ~500 workers
    frac_fab_construction: float = 0.04
    frac_fab_operation: float = 0.04

    # Datacenter settings
    years_before_agreement_building_dc: float = 1.0
    max_proportion_prc_energy: float = 0.05
    mw_per_worker_per_year: float = 1.0
    total_prc_energy_gw: float = 1100.0
    energy_efficiency_relative_to_sota: float = 0.2
    h100_power_w: float = 700.0

    # Fab settings
    build_fab: bool = True
    fab_process_node_nm: float = 28.0
    chips_per_wafer: float = 28.0
    transistor_density_exponent: float = 1.49
    architecture_efficiency_per_year: float = 1.23
    wafers_per_month_per_worker: float = 24.64
    wafers_per_month_per_scanner: float = 1000.0
    construction_time_5k_wafers: float = 1.4
    construction_time_100k_wafers: float = 2.41

    # Scanner production
    prc_scanners_first_year: float = 20.0
    prc_scanners_per_year: float = 16.0
    prc_localization_year: float = 2028.0
    proportion_scanners_diverted: float = 0.10

    # Detection parameters
    mean_detection_time_100_workers: float = 6.95
    mean_detection_time_1000_workers: float = 3.42
    variance_detection_time: float = 3.88
    median_error_compute: float = 0.10
    median_error_energy: float = 0.10
    median_error_satellite: float = 0.10
    median_error_fab_inventory: float = 0.07

    # Prior probability
    prior_probability: float = 0.10


# =============================================================================
# CORE CALCULATIONS
# =============================================================================

def calculate_survival_rate(t: float, h0: float, h1: float) -> float:
    """S(t) = exp(-(h0*t + h1*t²/2))"""
    if t <= 0:
        return 1.0
    return math.exp(-(h0 * t + h1 * t ** 2 / 2))


def calculate_concealed_capacity(
    years_since_construction_start: float,
    construction_rate_gw_per_year: float,
    max_concealed_gw: float,
) -> float:
    """Linear growth capped at max."""
    if years_since_construction_start <= 0:
        return 0.0
    return min(construction_rate_gw_per_year * years_since_construction_start, max_concealed_gw)


def calculate_operating_compute(
    functional_compute: float,
    datacenter_capacity_gw: float,
    watts_per_h100e: float,
) -> float:
    """Operating = min(functional, capacity / power)"""
    max_from_capacity = datacenter_capacity_gw * 1e9 / watts_per_h100e
    return min(functional_compute, max_from_capacity)


def lr_from_discrepancy(
    true_if_project: float,
    true_if_no_project: float,
    us_estimate: float,
    median_error: float,
) -> float:
    """Core LR calculation from discrepancy."""
    if true_if_project < 1e-10 or true_if_no_project < 1e-10:
        return 1.0

    error_if_project = abs(us_estimate - true_if_project) / true_if_project
    error_if_no_project = abs(us_estimate - true_if_no_project) / true_if_no_project

    k = -np.log(0.5) / median_error
    p_project = k * np.exp(-k * error_if_project)
    p_no_project = k * np.exp(-k * error_if_no_project)

    return p_project / p_no_project if p_no_project > 0 else 1e6


def lr_from_workers(
    t: float,
    labor: int,
    mean_100: float,
    mean_1000: float,
    variance: float,
) -> float:
    """Worker detection LR using Gamma survival."""
    if labor <= 0 or t <= 0:
        return 1.0

    B = np.log(mean_100 / mean_1000) / np.log(1.5)
    A = mean_100 * (2 ** B)
    mu = A / (np.log10(max(labor, 1)) ** B)
    k = mu / variance

    p_not_detected = stats.gamma.sf(t, a=k, scale=variance)
    return max(p_not_detected, 0.001)


def calculate_total_scanners(year: float, config: SimulationConfig) -> float:
    """Calculate total PRC scanners at a given year."""
    n = year - config.prc_localization_year
    if n < 0:
        return 0.0
    return config.prc_scanners_first_year * (n + 1) + config.prc_scanners_per_year * n * (n + 1) / 2


def calculate_fab_construction_duration(
    wafer_starts: float,
    construction_labor: float,
    config: SimulationConfig,
) -> float:
    """Calculate fab construction duration."""
    capacity1, time1 = 5000, config.construction_time_5k_wafers
    capacity2, time2 = 100000, config.construction_time_100k_wafers

    slope = (time2 - time1) / (math.log10(capacity2) - math.log10(capacity1))
    intercept = time1 - slope * math.log10(capacity1)

    base_duration = slope * math.log10(max(wafer_starts, 1)) + intercept

    workers_per_1000 = 14.1
    labor_required = (workers_per_1000 / 1000) * wafer_starts

    if construction_labor < labor_required:
        base_duration *= (labor_required / construction_labor)

    return base_duration


def calculate_h100e_per_chip(node_nm: float, year: float, config: SimulationConfig) -> float:
    """H100e per chip = density * architecture efficiency."""
    density = (4.0 / node_nm) ** config.transistor_density_exponent
    arch_eff = config.architecture_efficiency_per_year ** (year - 2022)
    return density * arch_eff


# =============================================================================
# FULL SIMULATION
# =============================================================================

def run_simulation(config: SimulationConfig, seed: int = 42) -> Dict:
    """Run complete simulation with all components."""
    np.random.seed(seed)

    results = {
        'years': [],
        'years_since_start': [],
        # Compute metrics
        'initial_surviving_compute': [],
        'fab_surviving_compute': [],
        'total_surviving_compute': [],
        'datacenter_capacity_gw': [],
        'operating_compute': [],
        # LR components
        'lr_workers': [],
        'lr_energy': [],
        'lr_satellite': [],
        'lr_compute_accounting': [],
        'lr_fab_inventory': [],
        'cumulative_lr': [],
        'detection_probability': [],
        # Fab metrics
        'fab_is_operational': [],
        'fab_annual_production': [],
    }

    # Initial values
    initial_diverted = config.prc_compute_stock_at_agreement * config.proportion_to_divert
    watts_per_h100e = config.h100_power_w / config.energy_efficiency_relative_to_sota

    # Labor allocation
    dc_construction_labor = config.total_labor * config.frac_datacenter_construction
    ai_research_labor = config.total_labor * config.frac_ai_research
    fab_construction_labor = config.total_labor * config.frac_fab_construction
    fab_operating_labor = config.total_labor * config.frac_fab_operation

    # Datacenter construction
    construction_rate_gw = (config.mw_per_worker_per_year * dc_construction_labor) / 1000.0
    max_concealed_gw = config.max_proportion_prc_energy * config.total_prc_energy_gw
    dc_construction_start = config.agreement_year - config.years_before_agreement_building_dc

    # Fab setup
    total_scanners = calculate_total_scanners(config.agreement_year, config)
    fab_scanners = total_scanners * config.proportion_scanners_diverted

    fab_wafer_starts = min(
        fab_operating_labor * config.wafers_per_month_per_worker,
        fab_scanners * config.wafers_per_month_per_scanner,
    )

    fab_construction_duration = calculate_fab_construction_duration(
        fab_wafer_starts, fab_construction_labor, config
    )
    fab_operational_year = config.agreement_year + fab_construction_duration

    h100e_per_chip = calculate_h100e_per_chip(
        config.fab_process_node_nm, config.agreement_year, config
    )
    fab_annual_production = fab_wafer_starts * config.chips_per_wafer * h100e_per_chip * 12

    # Fab compute tracking (cohort model for fairness)
    fab_cohorts = {}
    cumulative_log_lr = 0.0

    # Sample US estimates (with error)
    us_estimate_energy = config.total_prc_energy_gw * (1 + np.random.normal(0, config.median_error_energy))
    us_estimate_compute = config.prc_compute_stock_at_agreement * (1 + np.random.normal(0, config.median_error_compute))

    for year_offset in range(config.num_years + 1):
        year = config.agreement_year + year_offset
        t = year_offset  # years since start

        results['years'].append(year)
        results['years_since_start'].append(t)

        # === INITIAL COMPUTE SURVIVAL ===
        initial_survival = calculate_survival_rate(t, config.initial_hazard_rate, config.hazard_rate_increase_per_year)
        initial_surviving = initial_diverted * initial_survival
        results['initial_surviving_compute'].append(initial_surviving)

        # === DATACENTER CAPACITY ===
        years_since_dc_start = year - dc_construction_start
        concealed_gw = calculate_concealed_capacity(years_since_dc_start, construction_rate_gw, max_concealed_gw)
        results['datacenter_capacity_gw'].append(concealed_gw)

        # === FAB PRODUCTION ===
        fab_operational = year >= fab_operational_year and config.build_fab
        results['fab_is_operational'].append(fab_operational)

        if fab_operational:
            results['fab_annual_production'].append(fab_annual_production)
            if year not in fab_cohorts:
                fab_cohorts[year] = fab_annual_production
        else:
            results['fab_annual_production'].append(0.0)

        # Fab surviving compute (cohort model)
        fab_surviving = 0.0
        for prod_year, prod_amount in fab_cohorts.items():
            cohort_age = year - prod_year
            fab_survival = calculate_survival_rate(cohort_age, config.initial_hazard_rate, config.hazard_rate_increase_per_year)
            fab_surviving += prod_amount * fab_survival
        results['fab_surviving_compute'].append(fab_surviving)

        # Total surviving compute
        total_surviving = initial_surviving + fab_surviving
        results['total_surviving_compute'].append(total_surviving)

        # Operating compute
        operating = calculate_operating_compute(total_surviving, concealed_gw, watts_per_h100e)
        results['operating_compute'].append(operating)

        # === LIKELIHOOD RATIOS ===

        # LR from workers (total labor involved)
        lr_w = lr_from_workers(
            t, config.total_labor,
            config.mean_detection_time_100_workers,
            config.mean_detection_time_1000_workers,
            config.variance_detection_time
        )
        results['lr_workers'].append(lr_w)

        # LR from energy consumption
        covert_energy_gw = total_surviving * watts_per_h100e / 1e9
        reported_energy = config.total_prc_energy_gw - covert_energy_gw
        lr_e = lr_from_discrepancy(
            config.total_prc_energy_gw, reported_energy,
            us_estimate_energy, config.median_error_energy
        )
        results['lr_energy'].append(lr_e)

        # LR from satellite (diverted capacity = 0 since we use concealed, not diverted)
        # For satellite, diverted capacity is the unconcealed capacity that was diverted
        # In our config, this is 0 (we only build concealed datacenters)
        lr_s = 1.0  # No satellite evidence if no diverted unconcealed capacity
        results['lr_satellite'].append(lr_s)

        # LR from compute accounting
        reported_compute = config.prc_compute_stock_at_agreement * (1 - config.proportion_to_divert)
        true_compute = config.prc_compute_stock_at_agreement
        lr_c = lr_from_discrepancy(
            true_compute, reported_compute,
            us_estimate_compute, config.median_error_compute
        )
        results['lr_compute_accounting'].append(lr_c)

        # LR from fab inventory (scanner accounting)
        if config.build_fab and fab_scanners > 0:
            reported_scanners = total_scanners * (1 - config.proportion_scanners_diverted)
            us_estimate_scanners = total_scanners * (1 + np.random.normal(0, config.median_error_fab_inventory))
            lr_fab = lr_from_discrepancy(
                total_scanners, reported_scanners,
                us_estimate_scanners, config.median_error_fab_inventory
            )
        else:
            lr_fab = 1.0
        results['lr_fab_inventory'].append(lr_fab)

        # Cumulative LR (product of all LRs)
        # For time-dependent LRs, we accumulate the log-LRs
        if t > 0:
            # Worker LR is cumulative (survival up to time t)
            # Other LRs are instantaneous evidence, multiply into cumulative
            cumulative_log_lr += np.log(lr_e) + np.log(lr_c) + np.log(lr_fab)

        # Total cumulative LR combines worker survival with accounting evidence
        total_cumulative_lr = lr_w * np.exp(cumulative_log_lr)
        results['cumulative_lr'].append(total_cumulative_lr)

        # Detection probability
        prior_odds = config.prior_probability / (1 - config.prior_probability)
        posterior_odds = prior_odds * total_cumulative_lr
        detection_prob = posterior_odds / (1 + posterior_odds)
        results['detection_probability'].append(detection_prob)

    # Add metadata
    results['metadata'] = {
        'initial_diverted_compute': initial_diverted,
        'fab_construction_duration': fab_construction_duration,
        'fab_operational_year': fab_operational_year,
        'fab_annual_production': fab_annual_production,
        'h100e_per_chip': h100e_per_chip,
        'fab_wafer_starts': fab_wafer_starts,
        'construction_rate_gw_per_year': construction_rate_gw,
        'max_concealed_gw': max_concealed_gw,
        'watts_per_h100e': watts_per_h100e,
    }

    return results


def call_reference_api(config: SimulationConfig, num_samples: int = 50, timeout: int = 180) -> Optional[Dict]:
    """Call the reference model API."""
    print("Calling reference API...")
    url = 'https://dark-compute.onrender.com/run_simulation'
    data = json.dumps({
        'num_samples': num_samples,
        'start_year': int(config.agreement_year),
        'total_labor': config.total_labor,
    }).encode()

    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode())
            elapsed = time.time() - start_time
            print(f"  API response received in {elapsed:.1f}s")
            return result
    except Exception as e:
        print(f"  ERROR calling API: {e}")
        return None


def compare_with_reference(sim_results: Dict, ref_data: Dict) -> Dict:
    """Compare simulation results with reference API data."""
    comparison = {}

    # Extract reference data
    ref_model = ref_data.get('black_project_model', {})
    ref_years = ref_model.get('years', [])

    # Print available reference metrics for debugging
    print(f"\n  Available reference metrics: {list(ref_model.keys())}")

    # Map metrics - try multiple possible names
    metric_mapping = {
        'total_surviving_compute': [
            ('black_project_compute', 'H100e'),
            ('surviving_compute', 'H100e'),
            ('total_compute', 'H100e'),
        ],
        'cumulative_lr': [
            ('cumulative_likelihood_ratio', 'LR'),
            ('cumulative_lr', 'LR'),
            ('lr', 'LR'),
        ],
        'detection_probability': [
            ('detection_probability', '%'),
            ('p_detection', '%'),
        ],
        'datacenter_capacity_gw': [
            ('datacenter_capacity', 'GW'),
            ('covert_datacenter_capacity', 'GW'),
        ],
    }

    for our_metric, ref_options in metric_mapping.items():
        ref_values = None
        unit = 'units'

        for ref_metric, u in ref_options:
            ref_data_metric = ref_model.get(ref_metric, {})
            if isinstance(ref_data_metric, dict):
                ref_values = ref_data_metric.get('median', [])
            elif isinstance(ref_data_metric, list):
                ref_values = ref_data_metric

            if ref_values:
                unit = u
                print(f"  Found {our_metric} -> {ref_metric}")
                break

        if not ref_values:
            continue

        our_values = sim_results.get(our_metric, [])

        # Compare at matching time points (use indices since ref may have different time steps)
        diffs = []
        num_compare = min(len(our_values), len(ref_values))

        for i in range(num_compare):
            our_val = our_values[i]
            ref_val = ref_values[i]
            year = sim_results['years'][i] if i < len(sim_results['years']) else i

            if abs(ref_val) > 1e-10:
                diff_pct = (our_val - ref_val) / abs(ref_val) * 100
            else:
                diff_pct = 0.0 if abs(our_val) < 1e-10 else float('inf')

            diffs.append({
                'year': year,
                'ours': our_val,
                'ref': ref_val,
                'diff_pct': diff_pct,
            })

        comparison[our_metric] = {
            'unit': unit,
            'values': diffs,
            'max_diff_pct': max(abs(d['diff_pct']) for d in diffs) if diffs else 0,
        }

    return comparison


def print_results(results: Dict, config: SimulationConfig):
    """Print simulation results in a formatted table."""
    print("\n" + "=" * 100)
    print("SIMULATION RESULTS")
    print("=" * 100)

    meta = results['metadata']
    print(f"\nMetadata:")
    print(f"  Initial diverted compute: {meta['initial_diverted_compute']:,.0f} H100e")
    print(f"  Fab construction duration: {meta['fab_construction_duration']:.2f} years")
    print(f"  Fab operational year: {meta['fab_operational_year']:.2f}")
    print(f"  Fab annual production: {meta['fab_annual_production']:,.0f} H100e/year")
    print(f"  DC construction rate: {meta['construction_rate_gw_per_year']:.2f} GW/year")
    print(f"  Max concealed DC: {meta['max_concealed_gw']:.2f} GW")

    # Main results table
    print("\n" + "-" * 100)
    headers = ['Year', 'Initial', 'Fab', 'Total', 'DC(GW)', 'Operating', 'LR_w', 'LR_e', 'Cum_LR', 'P(det)']
    print(f"{headers[0]:<6} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<8} {headers[5]:<12} {headers[6]:<8} {headers[7]:<8} {headers[8]:<10} {headers[9]:<8}")
    print("-" * 100)

    for i, year in enumerate(results['years']):
        initial = results['initial_surviving_compute'][i]
        fab = results['fab_surviving_compute'][i]
        total = results['total_surviving_compute'][i]
        dc = results['datacenter_capacity_gw'][i]
        operating = results['operating_compute'][i]
        lr_w = results['lr_workers'][i]
        lr_e = results['lr_energy'][i]
        cum_lr = results['cumulative_lr'][i]
        p_det = results['detection_probability'][i]

        print(f"{year:<6.0f} {initial:<12,.0f} {fab:<12,.0f} {total:<12,.0f} {dc:<8.2f} {operating:<12,.0f} {lr_w:<8.4f} {lr_e:<8.4f} {cum_lr:<10.4f} {p_det:<8.2%}")


def main():
    """Run end-to-end simulation comparison."""
    print("=" * 100)
    print("END-TO-END SIMULATION COMPARISON")
    print("Continuous ODE Model vs Discrete Reference Model")
    print("=" * 100)

    config = SimulationConfig()

    print(f"\nConfiguration:")
    print(f"  Agreement year: {config.agreement_year}")
    print(f"  Simulation years: {config.num_years}")
    print(f"  Total labor: {config.total_labor:,}")
    print(f"  PRC compute stock: {config.prc_compute_stock_at_agreement:,.0f} H100e")
    print(f"  Diversion proportion: {config.proportion_to_divert:.0%}")
    print(f"  Build fab: {config.build_fab}")
    print(f"  Prior probability: {config.prior_probability:.0%}")

    # Run our simulation
    print("\n" + "-" * 50)
    print("Running continuous model simulation...")
    results = run_simulation(config)
    print_results(results, config)

    # Call reference API
    print("\n" + "-" * 50)
    ref_data = call_reference_api(config, num_samples=100)

    if ref_data:
        comparison = compare_with_reference(results, ref_data)

        print("\n" + "=" * 100)
        print("COMPARISON WITH REFERENCE API")
        print("=" * 100)

        for metric, data in comparison.items():
            print(f"\n{metric} ({data['unit']}):")
            print(f"  {'Year':<6} {'Ours':<15} {'Reference':<15} {'Diff %':<10}")
            print("  " + "-" * 50)

            for v in data['values'][:8]:  # Show first 8 years
                print(f"  {v['year']:<6.0f} {v['ours']:<15.4f} {v['ref']:<15.4f} {v['diff_pct']:<10.1f}")

            print(f"\n  Max difference: {data['max_diff_pct']:.1f}%")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    final_idx = -1
    print(f"\nFinal year ({results['years'][final_idx]:.0f}):")
    print(f"  Total surviving compute: {results['total_surviving_compute'][final_idx]:,.0f} H100e")
    print(f"  Operating compute: {results['operating_compute'][final_idx]:,.0f} H100e")
    print(f"  Datacenter capacity: {results['datacenter_capacity_gw'][final_idx]:.2f} GW")
    print(f"  Cumulative LR: {results['cumulative_lr'][final_idx]:.4f}")
    print(f"  Detection probability: {results['detection_probability'][final_idx]:.2%}")

    if ref_data:
        ref_model = ref_data.get('black_project_model', {})
        ref_compute = ref_model.get('black_project_compute', {}).get('median', [])
        ref_lr = ref_model.get('cumulative_likelihood_ratio', {}).get('median', [])
        ref_det = ref_model.get('detection_probability', {}).get('median', [])

        if ref_compute:
            print(f"\nReference API (final year):")
            print(f"  Black project compute: {ref_compute[-1]:,.0f} H100e")
            if ref_lr:
                print(f"  Cumulative LR: {ref_lr[-1]:.4f}")
            if ref_det:
                print(f"  Detection probability: {ref_det[-1]:.2%}")

    print("\n" + "-" * 100)
    print("✓ End-to-end simulation complete")
    print("-" * 100)

    return results, ref_data


if __name__ == "__main__":
    main()
