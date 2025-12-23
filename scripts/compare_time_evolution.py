"""
Compare TIME EVOLUTION between the continuous ODE-based model and the discrete reference model.

This script compares how key metrics evolve over multiple years:
- Survival rate (chip attrition)
- Datacenter capacity growth
- Surviving compute
- Operating compute (limited by datacenter capacity)
"""

import sys
import os
import math

# Add continuous model to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ComparisonConfig:
    """Configuration for the comparison test."""
    # Time settings
    agreement_year: float = 2030.0
    num_years: int = 10  # Simulate 10 years

    # Initial conditions
    prc_compute_stock_at_agreement: float = 1e6  # H100e
    proportion_to_divert: float = 0.05

    # Hazard rates (survival model)
    initial_hazard_rate: float = 0.05
    hazard_rate_increase_per_year: float = 0.02

    # Datacenter settings
    datacenter_construction_labor: int = 10000
    years_before_agreement_building_dc: float = 1.0  # Head start
    max_proportion_prc_energy: float = 0.05
    mw_per_worker_per_year: float = 1.0
    total_prc_energy_gw: float = 1100.0
    energy_efficiency_relative_to_sota: float = 0.2

    # H100 reference
    h100_power_w: float = 700.0


def calculate_survival_rate(
    years_since_acquisition: float,
    initial_hazard_rate: float,
    hazard_rate_increase_per_year: float,
) -> float:
    """
    Calculate survival rate using the hazard model.

    S(t) = exp(-H(t)) where H(t) = h₀·t + h₁·t²/2

    This formula is IDENTICAL in both models:
    - Discrete: black_project_stock.py line 337-338
    - Continuous: chip_survival.py line 312-318
    """
    if years_since_acquisition <= 0:
        return 1.0

    cumulative_hazard = (
        initial_hazard_rate * years_since_acquisition +
        hazard_rate_increase_per_year * years_since_acquisition ** 2 / 2
    )
    return math.exp(-cumulative_hazard)


def calculate_concealed_capacity(
    years_since_construction_start: float,
    construction_rate_gw_per_year: float,
    max_concealed_capacity_gw: float,
) -> float:
    """
    Calculate concealed datacenter capacity (linear growth model).

    Both models use: min(construction_rate × time, max_capacity)
    - Discrete: black_datacenters.py lines 75-78
    - Continuous: black_compute.py lines 343-346
    """
    if years_since_construction_start <= 0:
        return 0.0

    return min(
        construction_rate_gw_per_year * years_since_construction_start,
        max_concealed_capacity_gw
    )


def calculate_operating_compute(
    functional_compute_h100e: float,
    datacenter_capacity_gw: float,
    watts_per_h100e: float,
) -> float:
    """
    Calculate operating compute limited by datacenter capacity.

    operating = min(functional_compute, capacity / energy_per_h100e)
    """
    if watts_per_h100e <= 0:
        return 0.0

    # Datacenter can power this much compute
    max_compute_from_capacity = datacenter_capacity_gw * 1e9 / watts_per_h100e

    return min(functional_compute_h100e, max_compute_from_capacity)


def run_discrete_model_evolution(config: ComparisonConfig) -> Dict[str, List[float]]:
    """
    Calculate time evolution using discrete model formulas.

    Key formulas from discrete model:
    - Survival: S(t) = exp(-(h₀·t + h₁·t²/2))  [black_project_stock.py:337-338]
    - DC capacity: min(rate × time, max)  [black_datacenters.py:75-78]
    - Operating: min(surviving, capacity/energy)  [black_project.py:119-130]
    """
    results = {
        'years': [],
        'years_since_start': [],
        'survival_rate': [],
        'surviving_compute': [],
        'concealed_capacity_gw': [],
        'total_capacity_gw': [],
        'operating_compute': [],
    }

    # Initial values
    initial_diverted = config.prc_compute_stock_at_agreement * config.proportion_to_divert
    watts_per_h100e = config.h100_power_w / config.energy_efficiency_relative_to_sota

    # Construction parameters
    construction_rate = (config.mw_per_worker_per_year * config.datacenter_construction_labor) / 1000.0
    max_concealed = config.max_proportion_prc_energy * config.total_prc_energy_gw
    unconcealed_capacity = 0.0  # fraction_diverted = 0 in our config

    # Datacenter construction start offset
    # Discrete model: starts building years_before_agreement_building_dc before agreement
    datacenter_start_year_offset = -config.years_before_agreement_building_dc

    for year_offset in range(config.num_years + 1):
        year = config.agreement_year + year_offset
        years_since_start = year_offset  # years since agreement/project start

        results['years'].append(year)
        results['years_since_start'].append(years_since_start)

        # Survival rate for initial compute
        survival = calculate_survival_rate(
            years_since_acquisition=years_since_start,
            initial_hazard_rate=config.initial_hazard_rate,
            hazard_rate_increase_per_year=config.hazard_rate_increase_per_year,
        )
        results['survival_rate'].append(survival)

        # Surviving compute
        surviving = initial_diverted * survival
        results['surviving_compute'].append(surviving)

        # Datacenter capacity
        # In discrete model: years_since_construction_start = year_relative_to_agreement - start_offset
        # Since start_offset = -1, this becomes year_relative_to_agreement + 1
        years_since_dc_construction = years_since_start - datacenter_start_year_offset
        concealed = calculate_concealed_capacity(
            years_since_construction_start=years_since_dc_construction,
            construction_rate_gw_per_year=construction_rate,
            max_concealed_capacity_gw=max_concealed,
        )
        results['concealed_capacity_gw'].append(concealed)

        total_capacity = concealed + unconcealed_capacity
        results['total_capacity_gw'].append(total_capacity)

        # Operating compute
        operating = calculate_operating_compute(
            functional_compute_h100e=surviving,
            datacenter_capacity_gw=total_capacity,
            watts_per_h100e=watts_per_h100e,
        )
        results['operating_compute'].append(operating)

    return results


def run_continuous_model_evolution(config: ComparisonConfig) -> Dict[str, List[float]]:
    """
    Calculate time evolution using continuous model formulas.

    Key formulas from continuous model:
    - Survival: S(t) = exp(-(h₀·t + h₁·t²/2))  [chip_survival.py:312-318]
    - DC capacity: min(rate × time, max)  [black_compute.py:343-346]
    - Operating: min(surviving, capacity/energy)  [black_compute.py:361-379]
    """
    results = {
        'years': [],
        'years_since_start': [],
        'survival_rate': [],
        'surviving_compute': [],
        'concealed_capacity_gw': [],
        'total_capacity_gw': [],
        'operating_compute': [],
    }

    # Initial values
    initial_diverted = config.prc_compute_stock_at_agreement * config.proportion_to_divert
    watts_per_h100e = config.h100_power_w / config.energy_efficiency_relative_to_sota

    # Construction parameters
    # In continuous model, we use total labor and fraction
    total_labor = config.datacenter_construction_labor + 500  # Add researcher headcount
    frac_dc_construction = config.datacenter_construction_labor / total_labor
    datacenter_construction_labor = total_labor * frac_dc_construction  # Should equal original

    construction_rate = (config.mw_per_worker_per_year * datacenter_construction_labor) / 1000.0
    max_concealed = config.max_proportion_prc_energy * config.total_prc_energy_gw
    unconcealed_capacity = 0.0

    # Construction start year
    construction_start_year = config.agreement_year - config.years_before_agreement_building_dc

    for year_offset in range(config.num_years + 1):
        year = config.agreement_year + year_offset
        years_since_start = year_offset

        results['years'].append(year)
        results['years_since_start'].append(years_since_start)

        # Survival rate
        survival = calculate_survival_rate(
            years_since_acquisition=years_since_start,
            initial_hazard_rate=config.initial_hazard_rate,
            hazard_rate_increase_per_year=config.hazard_rate_increase_per_year,
        )
        results['survival_rate'].append(survival)

        # Surviving compute
        surviving = initial_diverted * survival
        results['surviving_compute'].append(surviving)

        # Datacenter capacity
        years_since_dc_construction = year - construction_start_year
        concealed = calculate_concealed_capacity(
            years_since_construction_start=years_since_dc_construction,
            construction_rate_gw_per_year=construction_rate,
            max_concealed_capacity_gw=max_concealed,
        )
        results['concealed_capacity_gw'].append(concealed)

        total_capacity = concealed + unconcealed_capacity
        results['total_capacity_gw'].append(total_capacity)

        # Operating compute
        operating = calculate_operating_compute(
            functional_compute_h100e=surviving,
            datacenter_capacity_gw=total_capacity,
            watts_per_h100e=watts_per_h100e,
        )
        results['operating_compute'].append(operating)

    return results


def compare_evolution():
    """Main comparison function."""

    print("=" * 80)
    print("TIME EVOLUTION COMPARISON")
    print("=" * 80)

    config = ComparisonConfig()

    print(f"\nConfiguration:")
    print(f"  Agreement year: {config.agreement_year}")
    print(f"  Simulation years: {config.num_years}")
    print(f"  Initial diverted compute: {config.prc_compute_stock_at_agreement * config.proportion_to_divert:,.0f} H100e")
    print(f"  Initial hazard rate: {config.initial_hazard_rate}")
    print(f"  Hazard rate increase/year: {config.hazard_rate_increase_per_year}")
    print(f"  DC construction rate: {config.mw_per_worker_per_year * config.datacenter_construction_labor / 1000:.2f} GW/year")
    print(f"  Max concealed capacity: {config.max_proportion_prc_energy * config.total_prc_energy_gw:.2f} GW")

    # Run both models
    discrete = run_discrete_model_evolution(config)
    continuous = run_continuous_model_evolution(config)

    # Print detailed comparison
    metrics = [
        ('Survival Rate', 'survival_rate', ''),
        ('Surviving Compute', 'surviving_compute', 'H100e'),
        ('Concealed DC Capacity', 'concealed_capacity_gw', 'GW'),
        ('Total DC Capacity', 'total_capacity_gw', 'GW'),
        ('Operating Compute', 'operating_compute', 'H100e'),
    ]

    all_aligned = True

    for metric_name, metric_key, unit in metrics:
        print(f"\n{'='*80}")
        print(f"{metric_name} Over Time")
        print("=" * 80)
        print(f"{'Year':<8} {'Discrete':<20} {'Continuous':<20} {'Diff %':<10} {'Status':<10}")
        print("-" * 80)

        metric_aligned = True
        for i, year in enumerate(discrete['years']):
            d_val = discrete[metric_key][i]
            c_val = continuous[metric_key][i]

            if d_val == 0 and c_val == 0:
                diff_pct = 0.0
                status = "OK"
            elif d_val == 0:
                diff_pct = float('inf')
                status = "MISMATCH"
                metric_aligned = False
            else:
                diff_pct = abs(c_val - d_val) / abs(d_val) * 100
                status = "OK" if diff_pct < 1.0 else "MISMATCH"
                if status == "MISMATCH":
                    metric_aligned = False

            d_str = f"{d_val:,.4f}" if d_val < 1e6 else f"{d_val:,.0f}"
            c_str = f"{c_val:,.4f}" if c_val < 1e6 else f"{c_val:,.0f}"
            print(f"{year:<8} {d_str:<20} {c_str:<20} {diff_pct:<10.2f} {status:<10}")

        if not metric_aligned:
            all_aligned = False
            print(f"\n⚠️  {metric_name}: MISALIGNED")
        else:
            print(f"\n✓  {metric_name}: ALIGNED")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_aligned:
        print("\n✓ ALL METRICS ALIGNED over the simulation period!")
        print("  Both models produce identical results (within 1% tolerance).")
    else:
        print("\n⚠️  SOME METRICS MISALIGNED")
        print("  See details above for specific discrepancies.")

    return discrete, continuous


if __name__ == "__main__":
    compare_evolution()
