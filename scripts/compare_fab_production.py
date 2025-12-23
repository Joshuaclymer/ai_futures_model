"""
Compare black fab compute production between discrete and continuous models.

Identifies exact differences in:
1. Lithography scanner calculation
2. Wafer starts per month
3. H100e per chip (transistor density + architecture efficiency)
4. Construction duration
5. Annual production rate
"""

import sys
import os
import numpy as np

# Add both project roots to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')
sys.path.insert(0, '/Users/joshuaclymer/github/covert_compute_production_model')

# Import from discrete model
from black_project_backend.black_project_parameters import (
    ProcessNode,
    BlackFabParameters as DiscreteBlackFabParameters,
)
from black_project_backend.classes.black_fab import (
    estimate_wafer_starts_per_month as discrete_estimate_wafer_starts,
    estimate_construction_duration as discrete_estimate_construction_duration,
    estimate_transistor_density_relative_to_h100 as discrete_transistor_density,
    estimate_architecture_efficiency_relative_to_h100 as discrete_architecture_efficiency,
    estimate_total_prc_lithography_scanners_for_node as discrete_estimate_scanners,
    H100_PROCESS_NODE_NM,
)

# Import from continuous model
from world_updaters.compute.black_compute import (
    calculate_fab_construction_duration as continuous_construction_duration,
    calculate_fab_wafer_starts_per_month as continuous_wafer_starts,
    calculate_fab_h100e_per_chip as continuous_h100e_per_chip,
)
from parameters.compute_parameters import (
    ExogenousComputeTrends,
    PRCComputeParameters,
)


def compare_transistor_density():
    """Compare transistor density calculation."""
    print("\n" + "=" * 60)
    print("TRANSISTOR DENSITY COMPARISON")
    print("=" * 60)

    # Default scaling exponent
    scaling_exp = DiscreteBlackFabParameters.transistor_density_scaling_exponent
    print(f"Discrete model scaling exponent: {scaling_exp}")

    for node_nm in [130, 28, 14, 7]:
        # Discrete model
        discrete_density = discrete_transistor_density(node_nm)

        # Continuous model (embedded in calculate_fab_h100e_per_chip)
        # density_ratio = (h100_reference_nm / fab_process_node_nm) ** scaling_exp
        h100_ref = 4.0  # H100 is 4nm
        continuous_density = (h100_ref / node_nm) ** scaling_exp

        print(f"\n  {node_nm}nm:")
        print(f"    Discrete:   {discrete_density:.6f}")
        print(f"    Continuous: {continuous_density:.6f}")
        print(f"    Match: {np.isclose(discrete_density, continuous_density)}")


def compare_architecture_efficiency():
    """Compare architecture efficiency calculation."""
    print("\n" + "=" * 60)
    print("ARCHITECTURE EFFICIENCY COMPARISON")
    print("=" * 60)

    improvement_per_year = 1.23
    h100_release_year = 2022

    for year in [2025, 2028, 2030, 2035]:
        # Discrete model
        discrete_eff = discrete_architecture_efficiency(year, improvement_per_year)

        # Continuous model
        years_since_h100 = year - h100_release_year
        continuous_eff = improvement_per_year ** years_since_h100

        print(f"\n  Year {year}:")
        print(f"    Discrete:   {discrete_eff:.6f}")
        print(f"    Continuous: {continuous_eff:.6f}")
        print(f"    Match: {np.isclose(discrete_eff, continuous_eff)}")


def compare_h100e_per_chip():
    """Compare H100e per chip calculation."""
    print("\n" + "=" * 60)
    print("H100e PER CHIP COMPARISON")
    print("=" * 60)

    exogenous_trends = ExogenousComputeTrends(
        transistor_density_scaling_exponent=1.49,
        state_of_the_art_architecture_efficiency_improvement_per_year=1.23,
        transistor_density_at_end_of_dennard_scaling_m_per_mm2=10.0,
        watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended=-1.0,
        watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended=-0.33,
        state_of_the_art_energy_efficiency_improvement_per_year=1.26,
    )

    improvement_per_year = 1.23

    for node_nm in [28, 14, 7]:
        for year in [2030, 2035]:
            # Discrete model
            density = discrete_transistor_density(node_nm)
            arch_eff = discrete_architecture_efficiency(year, improvement_per_year)
            discrete_h100e = density * arch_eff

            # Continuous model
            continuous_h100e = continuous_h100e_per_chip(
                fab_process_node_nm=node_nm,
                year=year,
                exogenous_trends=exogenous_trends,
            )

            print(f"\n  {node_nm}nm @ {year}:")
            print(f"    Discrete:   {discrete_h100e:.6f}")
            print(f"    Continuous: {continuous_h100e:.6f}")
            print(f"    Match: {np.isclose(discrete_h100e, continuous_h100e)}")


def compare_wafer_starts():
    """Compare wafer starts per month calculation."""
    print("\n" + "=" * 60)
    print("WAFER STARTS PER MONTH COMPARISON")
    print("=" * 60)

    # Parameters
    operation_labor = 2000
    num_scanners = 5

    # Get default parameter values
    wafers_per_worker = DiscreteBlackFabParameters.wafers_per_month_per_worker
    wafers_per_scanner = DiscreteBlackFabParameters.wafers_per_month_per_lithography_scanner

    print(f"Default parameters:")
    print(f"  wafers_per_worker: {wafers_per_worker}")
    print(f"  wafers_per_scanner: {wafers_per_scanner}")
    print(f"  operation_labor: {operation_labor}")
    print(f"  num_scanners: {num_scanners}")

    # Discrete model - uses stochastic sampling
    np.random.seed(42)
    discrete_results = []
    for _ in range(100):
        result = discrete_estimate_wafer_starts(operation_labor, num_scanners)
        discrete_results.append(result)
    discrete_median = np.median(discrete_results)
    discrete_mean = np.mean(discrete_results)

    # Continuous model - deterministic
    continuous_result = continuous_wafer_starts(
        fab_operating_labor=operation_labor,
        fab_number_of_lithography_scanners=num_scanners,
        wafers_per_month_per_worker=wafers_per_worker,
        wafers_per_month_per_scanner=wafers_per_scanner,
    )

    # Expected deterministic value (min of labor and scanner capacity)
    labor_capacity = operation_labor * wafers_per_worker
    scanner_capacity = num_scanners * wafers_per_scanner
    expected_deterministic = min(labor_capacity, scanner_capacity)

    print(f"\nResults:")
    print(f"  Labor capacity: {labor_capacity:.0f} wafers/month")
    print(f"  Scanner capacity: {scanner_capacity:.0f} wafers/month")
    print(f"  Expected deterministic (min): {expected_deterministic:.0f} wafers/month")
    print(f"  Continuous model: {continuous_result:.0f} wafers/month")
    print(f"  Discrete model median (100 samples): {discrete_median:.0f} wafers/month")
    print(f"  Discrete model mean (100 samples): {discrete_mean:.0f} wafers/month")
    print(f"\n  KEY DIFFERENCE: Discrete model samples from log-normal distribution")
    print(f"    with relative sigma = {DiscreteBlackFabParameters.labor_productivity_relative_sigma} (labor)")
    print(f"    and relative sigma = {DiscreteBlackFabParameters.scanner_productivity_relative_sigma} (scanner)")


def compare_construction_duration():
    """Compare fab construction duration calculation."""
    print("\n" + "=" * 60)
    print("FAB CONSTRUCTION DURATION COMPARISON")
    print("=" * 60)

    # Get reference parameters
    time_5k = DiscreteBlackFabParameters.construction_time_for_5k_wafers_per_month
    time_100k = DiscreteBlackFabParameters.construction_time_for_100k_wafers_per_month
    workers_per_1000_wafers = DiscreteBlackFabParameters.construction_workers_per_1000_wafers_per_month

    print(f"Parameters:")
    print(f"  construction_time_for_5k_wafers: {time_5k}")
    print(f"  construction_time_for_100k_wafers: {time_100k}")
    print(f"  construction_workers_per_1000_wafers: {workers_per_1000_wafers}")

    # Test case 1: Labor is NOT constraining (500 workers, 10000 wafers/month)
    # Required labor = 14.1/1000 * 10000 = 141 workers, we have 500 - not constraining
    wafer_capacity = 10000
    construction_labor = 500

    print(f"\n--- Case 1: Labor NOT constraining ---")
    print(f"  wafer_capacity: {wafer_capacity}")
    print(f"  construction_labor: {construction_labor}")
    print(f"  required_labor: {workers_per_1000_wafers / 1000 * wafer_capacity:.1f}")

    # Discrete model
    np.random.seed(42)
    discrete_results = []
    for _ in range(100):
        result = discrete_estimate_construction_duration(wafer_capacity, construction_labor)
        discrete_results.append(result)
    discrete_median = np.median(discrete_results)

    # Continuous model
    continuous_result = continuous_construction_duration(
        fab_construction_labor=construction_labor,
        target_wafer_starts_per_month=wafer_capacity,
        construction_time_for_5k_wafers=time_5k,
        construction_time_for_100k_wafers=time_100k,
        construction_workers_per_1000_wafers_per_month=workers_per_1000_wafers,
    )

    print(f"\nResults:")
    print(f"  Continuous model: {continuous_result:.3f} years")
    print(f"  Discrete model median (100 samples): {discrete_median:.3f} years")

    # Test case 2: Labor IS constraining (50 workers, 10000 wafers/month)
    # Required labor = 141 workers, we have 50 - constraining!
    wafer_capacity = 10000
    construction_labor = 50

    print(f"\n--- Case 2: Labor IS constraining ---")
    print(f"  wafer_capacity: {wafer_capacity}")
    print(f"  construction_labor: {construction_labor}")
    print(f"  required_labor: {workers_per_1000_wafers / 1000 * wafer_capacity:.1f}")

    # Discrete model
    np.random.seed(42)
    discrete_results = []
    for _ in range(100):
        result = discrete_estimate_construction_duration(wafer_capacity, construction_labor)
        discrete_results.append(result)
    discrete_median = np.median(discrete_results)

    # Continuous model
    continuous_result = continuous_construction_duration(
        fab_construction_labor=construction_labor,
        target_wafer_starts_per_month=wafer_capacity,
        construction_time_for_5k_wafers=time_5k,
        construction_time_for_100k_wafers=time_100k,
        construction_workers_per_1000_wafers_per_month=workers_per_1000_wafers,
    )

    print(f"\nResults:")
    print(f"  Continuous model: {continuous_result:.3f} years")
    print(f"  Discrete model median (100 samples): {discrete_median:.3f} years")
    print(f"  Expected base duration: ~1.63 years")
    print(f"  Expected with labor constraint: ~{1.63 * (141/50):.2f} years")

    print(f"\n  NOTE: Discrete model samples from log-normal distribution")
    print(f"    with relative sigma = {DiscreteBlackFabParameters.construction_time_relative_sigma}")

    # Test case 3: With uncertainty multiplier (matches discrete model behavior)
    print(f"\n--- Case 3: With uncertainty multiplier (simulating Monte Carlo) ---")
    wafer_capacity = 10000
    construction_labor = 500

    # Simulate what happens with the multiplier sampled from lognormal
    np.random.seed(42)
    continuous_with_multiplier = []
    for _ in range(100):
        # Sample multiplier from lognormal with relative_sigma=0.35
        relative_sigma = 0.35
        sigma_log = np.sqrt(np.log(1 + relative_sigma**2))
        multiplier = np.random.lognormal(0, sigma_log)

        result = continuous_construction_duration(
            fab_construction_labor=construction_labor,
            target_wafer_starts_per_month=wafer_capacity,
            construction_time_for_5k_wafers=time_5k,
            construction_time_for_100k_wafers=time_100k,
            construction_workers_per_1000_wafers_per_month=workers_per_1000_wafers,
            construction_time_multiplier=multiplier,
        )
        continuous_with_multiplier.append(result)

    # Discrete model
    np.random.seed(42)
    discrete_results = []
    for _ in range(100):
        result = discrete_estimate_construction_duration(wafer_capacity, construction_labor)
        discrete_results.append(result)

    print(f"\nResults with sampled uncertainty:")
    print(f"  Continuous (with multiplier) median: {np.median(continuous_with_multiplier):.3f} years")
    print(f"  Discrete median: {np.median(discrete_results):.3f} years")
    print(f"  Continuous (with multiplier) mean: {np.mean(continuous_with_multiplier):.3f} years")
    print(f"  Discrete mean: {np.mean(discrete_results):.3f} years")
    print(f"  Both should be similar when uncertainty is applied via multiplier!")


def compare_lithography_scanner_count():
    """Compare lithography scanner accumulation calculation."""
    print("\n" + "=" * 60)
    print("LITHOGRAPHY SCANNER COUNT COMPARISON")
    print("=" * 60)

    first_year_prod = DiscreteBlackFabParameters.prc_lithography_scanners_produced_in_first_year
    additional_per_year = DiscreteBlackFabParameters.prc_additional_lithography_scanners_produced_per_year

    print(f"Parameters:")
    print(f"  prc_lithography_scanners_produced_in_first_year: {first_year_prod}")
    print(f"  prc_additional_lithography_scanners_produced_per_year: {additional_per_year}")

    for years_since_localization in [0, 1, 3, 5, 10]:
        # Analytical formula (used by continuous model)
        # Total = first_year_prod * (n+1) + additional_per_year * n * (n+1) / 2
        n = years_since_localization
        analytical = first_year_prod * (n + 1) + additional_per_year * n * (n + 1) / 2

        # Discrete model - uses stochastic sampling
        current_year = 2030 + years_since_localization
        localization_year = 2030

        np.random.seed(42)
        discrete_results = []
        for _ in range(100):
            result = discrete_estimate_scanners(current_year, localization_year)
            discrete_results.append(result)
        discrete_median = np.median(discrete_results)

        print(f"\n  {years_since_localization} years since localization:")
        print(f"    Analytical (continuous): {analytical:.1f} scanners")
        print(f"    Discrete median (100 samples): {discrete_median:.1f} scanners")
        print(f"    Expected: {analytical:.1f}")

    print(f"\n  KEY DIFFERENCE: Discrete model samples from log-normal distribution")
    print(f"    with relative sigma = {DiscreteBlackFabParameters.prc_scanner_production_relative_sigma}")


def main():
    """Run all comparisons."""
    print("=" * 60)
    print("BLACK FAB COMPUTE PRODUCTION COMPARISON")
    print("Discrete Model vs Continuous Model")
    print("=" * 60)

    compare_transistor_density()
    compare_architecture_efficiency()
    compare_h100e_per_chip()
    compare_wafer_starts()
    compare_construction_duration()
    compare_lithography_scanner_count()

    print("\n" + "=" * 60)
    print("SUMMARY OF KEY DIFFERENCES")
    print("=" * 60)
    print("""
The continuous model uses DETERMINISTIC calculations while the discrete model
uses STOCHASTIC sampling from log-normal distributions for:

1. Wafer starts per month (uncertainty in labor and scanner productivity)
2. Construction duration (uncertainty in construction time)
3. Lithography scanner count (uncertainty in production rate)

The core formulas are aligned, but the uncertainty sampling should be moved
to monte_carlo_parameters.yaml for the continuous model to match the
statistical distribution of the discrete model.
""")


if __name__ == "__main__":
    main()
