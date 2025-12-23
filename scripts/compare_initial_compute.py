"""
Compare INITIAL BLACK PROJECT COMPUTE between the continuous ODE-based model
and the discrete reference model.

This script compares initial values using deterministic calculations for both models.
"""

import sys
import os

# Add continuous model to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class ComparisonConfig:
    """Configuration for the comparison test."""
    # Time settings
    agreement_year: float = 2030.0
    num_years: float = 1.0

    # Initial conditions - these are the key parameters
    prc_compute_stock_at_agreement: float = 1e6  # H100e
    proportion_to_divert: float = 0.05

    # Fixed hazard rates (no sampling uncertainty)
    initial_hazard_rate: float = 0.05
    hazard_rate_increase_per_year: float = 0.02

    # Datacenter settings
    datacenter_construction_labor: int = 10000
    years_before_agreement_building_dc: float = 1.0
    max_proportion_prc_energy: float = 0.05
    mw_per_worker_per_year: float = 1.0
    operating_labor_per_mw: float = 0.1
    total_prc_energy_gw: float = 1100.0
    energy_efficiency_relative_to_sota: float = 0.2

    # Fab disabled for simplicity
    build_fab: bool = False

    # Detection parameters (not relevant for initial compute, but used for total labor)
    researcher_headcount: int = 500


def calculate_discrete_model_values(config: ComparisonConfig) -> Dict:
    """
    Calculate the expected values from the discrete model using known formulas.

    This is a deterministic calculation based on the discrete model's logic in:
    - black_project_stock.py for compute calculations
    - black_datacenters.py for datacenter capacity calculations
    """
    # Constants from black_project_stock.py
    H100_POWER_W = 700.0

    # PRC stock at agreement year
    initial_prc_stock = config.prc_compute_stock_at_agreement

    # Initial black project compute
    initial_prc_black_project = initial_prc_stock * config.proportion_to_divert

    # At t=0, survival rate is 1.0 because no time has passed
    survival_rate_at_t0 = 1.0
    surviving_compute_at_t0 = initial_prc_black_project * survival_rate_at_t0

    # Energy efficiency
    energy_efficiency_relative_to_h100 = config.energy_efficiency_relative_to_sota

    # Energy consumption of initial black project stock in GW
    watts_per_h100e = H100_POWER_W / config.energy_efficiency_relative_to_sota
    energy_consumption_of_initial_stock_gw = initial_prc_black_project * watts_per_h100e / 1e9

    # Datacenter capacity at t=0
    # Unconcealed = 0 (since fraction_not_built_for_concealment_diverted = 0 in our config)
    unconcealed_capacity_gw = 0.0

    # Concealed capacity: built over years_before_agreement_building_dc
    # capacity = mw_per_worker_per_year * workers * years / 1000 (convert MW to GW)
    concealed_capacity_gw = (config.mw_per_worker_per_year * config.datacenter_construction_labor *
                             config.years_before_agreement_building_dc) / 1000.0

    # Max concealed capacity
    max_concealed_capacity_gw = config.max_proportion_prc_energy * config.total_prc_energy_gw
    concealed_capacity_gw = min(concealed_capacity_gw, max_concealed_capacity_gw)

    datacenter_capacity_at_t0 = unconcealed_capacity_gw + concealed_capacity_gw

    return {
        'initial_prc_stock': initial_prc_stock,
        'initial_prc_black_project': initial_prc_black_project,
        'surviving_compute_h100e': surviving_compute_at_t0,
        'survival_rate_at_t0': survival_rate_at_t0,
        'energy_efficiency_relative_to_h100': energy_efficiency_relative_to_h100,
        'energy_consumption_initial_stock_gw': energy_consumption_of_initial_stock_gw,
        'datacenter_capacity_gw_at_t0': datacenter_capacity_at_t0,
        'unconcealed_capacity_gw': unconcealed_capacity_gw,
        'concealed_capacity_gw': concealed_capacity_gw,
        'watts_per_h100e': watts_per_h100e,
    }


def calculate_continuous_model_values(config: ComparisonConfig) -> Dict:
    """
    Calculate values from the continuous model using known formulas.

    This follows the logic in:
    - world_updaters/black_project.py (initialize_black_project function)
    """
    H100_POWER_W = 700.0

    # Initial PRC stock
    initial_prc_stock = config.prc_compute_stock_at_agreement

    # Initial diverted compute (same formula as discrete)
    # From initialize_black_project line 478:
    # diverted_compute = initial_prc_compute_stock * props.fraction_of_initial_compute_stock_to_divert_at_black_project_start
    initial_prc_black_project = initial_prc_stock * config.proportion_to_divert

    # At t=0, survival rate is 1.0
    survival_rate_at_t0 = 1.0
    surviving_compute_at_t0 = initial_prc_black_project * survival_rate_at_t0

    # Energy efficiency
    energy_efficiency = config.energy_efficiency_relative_to_sota
    watts_per_h100e = H100_POWER_W / energy_efficiency

    # Energy consumption
    energy_consumption_gw = initial_prc_black_project * watts_per_h100e / 1e9

    # Datacenter capacity calculation from initialize_black_project
    # Calculate total labor and fractions
    total_labor = config.datacenter_construction_labor + config.researcher_headcount
    frac_dc_construction = config.datacenter_construction_labor / total_labor

    # From initialize_black_project lines 499-509:
    gw_per_worker_per_year = config.mw_per_worker_per_year / 1000.0
    datacenter_construction_labor = total_labor * frac_dc_construction
    construction_rate = gw_per_worker_per_year * datacenter_construction_labor

    # Head start years from black_project_properties
    head_start_years = config.years_before_agreement_building_dc

    # Max concealed capacity
    max_concealed_gw = config.max_proportion_prc_energy * config.total_prc_energy_gw

    # Initial concealed capacity at project start
    # From lines 505-509:
    # initial_concealed = min(construction_rate * head_start_years, max(0, max_capacity_gw - unconcealed_capacity_gw))
    unconcealed_capacity_gw = 0.0  # fraction_diverted = 0
    initial_concealed = min(
        construction_rate * head_start_years,
        max(0, max_concealed_gw - unconcealed_capacity_gw)
    )

    datacenter_capacity_at_t0 = initial_concealed + unconcealed_capacity_gw

    return {
        'initial_prc_stock': initial_prc_stock,
        'initial_prc_black_project': initial_prc_black_project,
        'surviving_compute_h100e': surviving_compute_at_t0,
        'survival_rate_at_t0': survival_rate_at_t0,
        'energy_efficiency_relative_to_h100': energy_efficiency,
        'energy_consumption_initial_stock_gw': energy_consumption_gw,
        'datacenter_capacity_gw_at_t0': datacenter_capacity_at_t0,
        'unconcealed_capacity_gw': unconcealed_capacity_gw,
        'concealed_capacity_gw': initial_concealed,
        'watts_per_h100e': watts_per_h100e,
        'datacenter_construction_labor_effective': datacenter_construction_labor,
        'construction_rate_gw_per_year': construction_rate,
    }


def compare_initial_compute():
    """Main comparison function."""

    print("=" * 70)
    print("INITIAL BLACK PROJECT COMPUTE COMPARISON")
    print("=" * 70)

    config = ComparisonConfig()

    print(f"\nConfiguration:")
    print(f"  Agreement year: {config.agreement_year}")
    print(f"  PRC compute stock at agreement: {config.prc_compute_stock_at_agreement:,.0f} H100e")
    print(f"  Proportion to divert: {config.proportion_to_divert:.2%}")
    print(f"  Expected initial black project compute: {config.prc_compute_stock_at_agreement * config.proportion_to_divert:,.0f} H100e")
    print(f"  Datacenter construction labor: {config.datacenter_construction_labor:,}")
    print(f"  Researcher headcount: {config.researcher_headcount:,}")
    print(f"  Total labor: {config.datacenter_construction_labor + config.researcher_headcount:,}")

    # Calculate expected values from discrete model formulas
    print("\n" + "-" * 50)
    print("DISCRETE MODEL (Reference - Calculated)")
    print("-" * 50)
    discrete = calculate_discrete_model_values(config)
    print(f"  Initial PRC stock: {discrete['initial_prc_stock']:,.2f} H100e")
    print(f"  Initial black project compute: {discrete['initial_prc_black_project']:,.2f} H100e")
    print(f"  Surviving compute at t=0: {discrete['surviving_compute_h100e']:,.2f} H100e")
    print(f"  Survival rate at t=0: {discrete['survival_rate_at_t0']:.4f}")
    print(f"  Energy efficiency (rel to H100): {discrete['energy_efficiency_relative_to_h100']:.4f}")
    print(f"  Watts per H100e: {discrete['watts_per_h100e']:.2f} W")
    print(f"  Energy consumption initial stock: {discrete['energy_consumption_initial_stock_gw']:.6f} GW")
    print(f"  Datacenter capacity at t=0: {discrete['datacenter_capacity_gw_at_t0']:.6f} GW")
    print(f"    - Unconcealed capacity: {discrete['unconcealed_capacity_gw']:.6f} GW")
    print(f"    - Concealed capacity: {discrete['concealed_capacity_gw']:.6f} GW")

    # Calculate values from continuous model formulas
    print("\n" + "-" * 50)
    print("CONTINUOUS MODEL (ODE-based - Calculated)")
    print("-" * 50)
    continuous = calculate_continuous_model_values(config)
    print(f"  Initial PRC stock: {continuous['initial_prc_stock']:,.2f} H100e")
    print(f"  Initial black project compute: {continuous['initial_prc_black_project']:,.2f} H100e")
    print(f"  Surviving compute at t=0: {continuous['surviving_compute_h100e']:,.2f} H100e")
    print(f"  Survival rate at t=0: {continuous['survival_rate_at_t0']:.4f}")
    print(f"  Energy efficiency (rel to H100): {continuous['energy_efficiency_relative_to_h100']:.4f}")
    print(f"  Watts per H100e: {continuous['watts_per_h100e']:.2f} W")
    print(f"  Energy consumption initial stock: {continuous['energy_consumption_initial_stock_gw']:.6f} GW")
    print(f"  Datacenter capacity at t=0: {continuous['datacenter_capacity_gw_at_t0']:.6f} GW")
    print(f"    - Unconcealed capacity: {continuous['unconcealed_capacity_gw']:.6f} GW")
    print(f"    - Concealed capacity: {continuous['concealed_capacity_gw']:.6f} GW")
    print(f"  (Debug: DC construction labor used: {continuous['datacenter_construction_labor_effective']:,.2f})")
    print(f"  (Debug: Construction rate: {continuous['construction_rate_gw_per_year']:.6f} GW/year)")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    metrics = [
        ('Initial PRC stock', 'initial_prc_stock', 'H100e'),
        ('Initial black project compute', 'initial_prc_black_project', 'H100e'),
        ('Surviving compute at t=0', 'surviving_compute_h100e', 'H100e'),
        ('Datacenter capacity at t=0', 'datacenter_capacity_gw_at_t0', 'GW'),
        ('Concealed capacity', 'concealed_capacity_gw', 'GW'),
    ]

    all_aligned = True
    for name, key, unit in metrics:
        d_val = discrete.get(key, 0)
        c_val = continuous.get(key, 0)

        if d_val == 0 and c_val == 0:
            diff_pct = 0.0
            status = "ALIGNED"
        elif d_val == 0:
            diff_pct = float('inf')
            status = "MISMATCH"
            all_aligned = False
        else:
            diff_pct = abs(c_val - d_val) / abs(d_val) * 100
            status = "ALIGNED" if diff_pct < 1.0 else "MISMATCH"
            if status == "MISMATCH":
                all_aligned = False

        print(f"\n{name}:")
        print(f"  Discrete:   {d_val:,.4f} {unit}")
        print(f"  Continuous: {c_val:,.4f} {unit}")
        print(f"  Difference: {diff_pct:.2f}%")
        print(f"  Status:     {status}")

    print("\n" + "=" * 70)
    if all_aligned:
        print("OVERALL: ALIGNED (all metrics within 1% tolerance)")
    else:
        print("OVERALL: MISALIGNED (some metrics differ by >1%)")
    print("=" * 70)

    return discrete, continuous


if __name__ == "__main__":
    compare_initial_compute()
