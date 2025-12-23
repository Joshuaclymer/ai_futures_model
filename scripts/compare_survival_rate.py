"""
Compare compute survival rate between ai_futures_simulator and black_project_backend.

The survival rate models the proportion of initial compute that survives over time
due to chip attrition/failure. Both models use:
    S(t) = exp(-H(t))
where H(t) = h₀·t + h₁·t²/2 is the cumulative hazard.

Key parameters:
    h₀ = initial_hazard_rate (per year)
    h₁ = hazard_rate_increase_per_year (per year²)

Both models can sample a multiplier applied to both hazard rates.
"""

import sys
import os

# Add both project roots to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')
sys.path.insert(0, '/Users/joshuaclymer/github/covert_compute_production_model')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict

# Import from discrete model
from black_project_backend.model import BlackProjectModel
from black_project_backend.black_project_parameters import (
    ModelParameters as DiscreteModelParameters,
    SimulationSettings as DiscreteSimulationSettings,
    BlackProjectProperties as DiscreteBlackProjectProperties,
    BlackProjectParameters as DiscreteBlackProjectParameters,
    SurvivalRateParameters as DiscreteSurvivalRateParameters,
    ExogenousTrends as DiscreteExogenousTrends,
    BlackDatacenterParameters as DiscreteBlackDatacenterParameters,
    DetectionParameters as DiscreteDetectionParameters,
    BlackFabParameters as DiscreteBlackFabParameters,
)
from black_project_backend.util import _cache as discrete_cache

# Import from continuous model
from world_updaters.compute.chip_survival import calculate_survival_rate
from parameters.compute_parameters import SurvivalRateParameters as ContinuousSurvivalRateParameters


@dataclass
class ComparisonConfig:
    """Configuration for survival rate comparison."""
    # Hazard rate parameters (p50 values from discrete model)
    initial_hazard_rate_p50: float = 0.01
    increase_of_hazard_rate_per_year_p50: float = 0.0035
    # For simplicity, fix the multiplier to 1.0 (no uncertainty)
    hazard_rate_p25_relative_to_p50: float = 1.0
    hazard_rate_p75_relative_to_p50: float = 1.0

    # Time settings
    agreement_year: float = 2030.0
    num_years: float = 10.0
    time_step: float = 0.5

    # Initial conditions
    initial_prc_compute: float = 1e6  # H100e
    proportion_to_divert: float = 0.05


def run_discrete_model_single(config: ComparisonConfig, seed: int) -> Dict:
    """Run discrete model and extract survival rate over time."""

    discrete_cache.clear()
    np.random.seed(seed)

    sim_settings = DiscreteSimulationSettings(
        agreement_start_year=int(config.agreement_year),
        num_years_to_simulate=config.num_years,
        time_step_years=config.time_step,
        num_simulations=1,
    )

    bp_properties = DiscreteBlackProjectProperties(
        run_a_black_project=True,
        proportion_of_initial_compute_stock_to_divert=config.proportion_to_divert,
        datacenter_construction_labor=10000,
        years_before_agreement_year_prc_starts_building_black_datacenters=1,
        max_proportion_of_PRC_energy_consumption=0.05,
        fraction_of_datacenter_capacity_not_built_for_concealment_diverted_to_black_project_at_agreement_start=0.0,
        build_a_black_fab=False,
        researcher_headcount=500,
    )

    survival_params = DiscreteSurvivalRateParameters(
        initial_hazard_rate_p50=config.initial_hazard_rate_p50,
        increase_of_hazard_rate_per_year_p50=config.increase_of_hazard_rate_per_year_p50,
        hazard_rate_p25_relative_to_p50=config.hazard_rate_p25_relative_to_p50,
        hazard_rate_p75_relative_to_p50=config.hazard_rate_p75_relative_to_p50,
    )

    exogenous_trends = DiscreteExogenousTrends(
        total_prc_compute_stock_in_2025=config.initial_prc_compute / (2.2 ** (config.agreement_year - 2025)),
        annual_growth_rate_of_prc_compute_stock_p10=2.2,
        annual_growth_rate_of_prc_compute_stock_p50=2.2,
        annual_growth_rate_of_prc_compute_stock_p90=2.2,
    )

    bp_params = DiscreteBlackProjectParameters(
        survival_rate_parameters=survival_params,
        datacenter_model_parameters=DiscreteBlackDatacenterParameters(),
        black_fab_parameters=DiscreteBlackFabParameters(),
        detection_parameters=DiscreteDetectionParameters(),
        exogenous_trends=exogenous_trends,
    )

    model_params = DiscreteModelParameters(
        simulation_settings=sim_settings,
        black_project_properties=bp_properties,
        black_project_parameters=bp_params,
    )

    model = BlackProjectModel(model_params)
    model.run_simulations(1)

    project = model.simulation_results[0]['prc_black_project']
    stock = project.black_project_stock

    years = np.arange(
        config.agreement_year,
        config.agreement_year + config.num_years + config.time_step,
        config.time_step
    )

    results = {
        'years': years,
        'years_since_start': [],
        'survival_rate': [],
        'surviving_compute_h100e': [],
        'initial_compute_h100e': stock.initial_prc_black_project,
        'initial_hazard_rate': stock.initial_hazard_rate,
        'increase_in_hazard_rate_per_year': stock.increase_in_hazard_rate_per_year,
    }

    for year in years:
        years_of_life = year - config.agreement_year
        results['years_since_start'].append(years_of_life)

        # Calculate survival rate using discrete model's formula
        cumulative_hazard = (
            stock.initial_hazard_rate * years_of_life +
            stock.increase_in_hazard_rate_per_year * years_of_life**2 / 2
        )
        survival_rate = np.exp(-cumulative_hazard)
        results['survival_rate'].append(survival_rate)

        # Get surviving compute
        surviving = stock.surviving_compute(year)
        results['surviving_compute_h100e'].append(surviving.total_h100e_tpp())

    # Convert to arrays
    for key in ['years_since_start', 'survival_rate', 'surviving_compute_h100e']:
        results[key] = np.array(results[key])

    return results


def run_continuous_model(config: ComparisonConfig, discrete_results: Dict) -> Dict:
    """Run continuous model using same hazard rates as discrete model."""

    # Use the exact same hazard rates that were sampled in the discrete model
    initial_hazard_rate = discrete_results['initial_hazard_rate']
    hazard_rate_increase = discrete_results['increase_in_hazard_rate_per_year']
    initial_compute = discrete_results['initial_compute_h100e']

    years = discrete_results['years']

    results = {
        'years': years,
        'years_since_start': [],
        'survival_rate': [],
        'surviving_compute_h100e': [],
        'initial_compute_h100e': initial_compute,
        'initial_hazard_rate': initial_hazard_rate,
        'increase_in_hazard_rate_per_year': hazard_rate_increase,
    }

    for year in years:
        years_since_start = year - config.agreement_year
        results['years_since_start'].append(years_since_start)

        # Calculate survival rate using continuous model's formula
        survival_rate = calculate_survival_rate(
            years_since_acquisition=years_since_start,
            initial_hazard_rate=initial_hazard_rate,
            hazard_rate_increase_per_year=hazard_rate_increase,
        )
        results['survival_rate'].append(survival_rate)

        # Calculate surviving compute
        surviving_compute = initial_compute * survival_rate
        results['surviving_compute_h100e'].append(surviving_compute)

    # Convert to arrays
    for key in ['years_since_start', 'survival_rate', 'surviving_compute_h100e']:
        results[key] = np.array(results[key])

    return results


def compare_survival_rates(
    discrete_results: Dict,
    continuous_results: Dict,
    output_path: str,
):
    """Compare and plot survival rates."""

    print("\n" + "=" * 70)
    print("COMPUTE SURVIVAL RATE COMPARISON")
    print("=" * 70)

    print(f"\nHazard Rate Parameters:")
    print(f"  Discrete - initial_hazard_rate: {discrete_results['initial_hazard_rate']:.6f}")
    print(f"  Continuous - initial_hazard_rate: {continuous_results['initial_hazard_rate']:.6f}")
    print(f"  Discrete - hazard_rate_increase: {discrete_results['increase_in_hazard_rate_per_year']:.6f}")
    print(f"  Continuous - hazard_rate_increase: {continuous_results['increase_in_hazard_rate_per_year']:.6f}")

    print(f"\nInitial Compute:")
    print(f"  Discrete:   {discrete_results['initial_compute_h100e']:.2f} H100e")
    print(f"  Continuous: {continuous_results['initial_compute_h100e']:.2f} H100e")

    print("\n" + "-" * 70)
    print("Survival Rate by Year:")
    print("-" * 70)
    print(f"{'Year':<8} {'t':<6} {'Discrete SR':<14} {'Continuous SR':<14} {'Diff':<12} {'%Diff':<10}")
    print("-" * 70)

    years = discrete_results['years']
    max_diff = 0
    max_percent_diff = 0

    for i, year in enumerate(years):
        t = discrete_results['years_since_start'][i]
        d_sr = discrete_results['survival_rate'][i]
        c_sr = continuous_results['survival_rate'][i]
        diff = abs(d_sr - c_sr)
        percent_diff = (diff / d_sr * 100) if d_sr > 0 else 0

        max_diff = max(max_diff, diff)
        max_percent_diff = max(max_percent_diff, percent_diff)

        print(f"{year:<8.1f} {t:<6.1f} {d_sr:<14.8f} {c_sr:<14.8f} {diff:<12.2e} {percent_diff:<10.4f}%")

    print("-" * 70)
    print(f"\nMax absolute difference: {max_diff:.2e}")
    print(f"Max percent difference: {max_percent_diff:.6f}%")

    # Check alignment
    is_aligned = max_diff < 1e-10
    print(f"\nALIGNMENT STATUS: {'✓ ALIGNED' if is_aligned else '✗ NOT ALIGNED'}")

    print("\n" + "-" * 70)
    print("Surviving Compute by Year:")
    print("-" * 70)
    print(f"{'Year':<8} {'t':<6} {'Discrete H100e':<16} {'Continuous H100e':<18} {'Diff':<12}")
    print("-" * 70)

    for i, year in enumerate(years):
        t = discrete_results['years_since_start'][i]
        d_sc = discrete_results['surviving_compute_h100e'][i]
        c_sc = continuous_results['surviving_compute_h100e'][i]
        diff = abs(d_sc - c_sc)

        print(f"{year:<8.1f} {t:<6.1f} {d_sc:<16.2f} {c_sc:<18.2f} {diff:<12.2e}")

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Survival Rate
    ax1 = axes[0]
    ax1.plot(discrete_results['years_since_start'], discrete_results['survival_rate'],
             'b-', linewidth=2, label='Discrete Model')
    ax1.plot(continuous_results['years_since_start'], continuous_results['survival_rate'],
             'g--', linewidth=2, label='Continuous Model')
    ax1.set_xlabel('Years Since Start')
    ax1.set_ylabel('Survival Rate S(t)')
    ax1.set_title('Survival Rate Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Plot 2: Surviving Compute
    ax2 = axes[1]
    ax2.plot(discrete_results['years_since_start'], discrete_results['surviving_compute_h100e'],
             'b-', linewidth=2, label='Discrete Model')
    ax2.plot(continuous_results['years_since_start'], continuous_results['surviving_compute_h100e'],
             'g--', linewidth=2, label='Continuous Model')
    ax2.set_xlabel('Years Since Start')
    ax2.set_ylabel('Surviving Compute (H100e)')
    ax2.set_title('Surviving Compute Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return is_aligned


def run_multiple_comparisons(config: ComparisonConfig, num_runs: int = 50) -> Dict:
    """Run multiple comparisons with different random seeds."""

    print(f"\n{'='*70}")
    print(f"Running {num_runs} comparisons with sampled hazard rate multipliers...")
    print(f"{'='*70}")

    # Use uncertain hazard rate multiplier
    config.hazard_rate_p25_relative_to_p50 = 0.5
    config.hazard_rate_p75_relative_to_p50 = 2.0

    all_discrete_sr = []
    all_continuous_sr = []
    all_diffs = []

    for i in range(num_runs):
        seed = 1000 + i

        discrete_results = run_discrete_model_single(config, seed)
        continuous_results = run_continuous_model(config, discrete_results)

        all_discrete_sr.append(discrete_results['survival_rate'])
        all_continuous_sr.append(continuous_results['survival_rate'])

        # Calculate max difference for this run
        diff = np.abs(discrete_results['survival_rate'] - continuous_results['survival_rate'])
        all_diffs.append(np.max(diff))

    # Convert to arrays
    all_discrete_sr = np.array(all_discrete_sr)
    all_continuous_sr = np.array(all_continuous_sr)
    all_diffs = np.array(all_diffs)

    years_since_start = discrete_results['years_since_start']

    print(f"\nResults across {num_runs} runs:")
    print(f"  Max absolute difference (any run): {np.max(all_diffs):.2e}")
    print(f"  Mean max difference: {np.mean(all_diffs):.2e}")

    # Summary statistics at specific time points
    print("\nSurvival Rate Distribution at t=5 years:")
    idx = np.argmin(np.abs(years_since_start - 5))
    print(f"  Discrete:   mean={np.mean(all_discrete_sr[:, idx]):.4f}, "
          f"std={np.std(all_discrete_sr[:, idx]):.4f}, "
          f"min={np.min(all_discrete_sr[:, idx]):.4f}, "
          f"max={np.max(all_discrete_sr[:, idx]):.4f}")
    print(f"  Continuous: mean={np.mean(all_continuous_sr[:, idx]):.4f}, "
          f"std={np.std(all_continuous_sr[:, idx]):.4f}, "
          f"min={np.min(all_continuous_sr[:, idx]):.4f}, "
          f"max={np.max(all_continuous_sr[:, idx]):.4f}")

    print("\nSurvival Rate Distribution at t=10 years:")
    idx = np.argmin(np.abs(years_since_start - 10))
    print(f"  Discrete:   mean={np.mean(all_discrete_sr[:, idx]):.4f}, "
          f"std={np.std(all_discrete_sr[:, idx]):.4f}, "
          f"min={np.min(all_discrete_sr[:, idx]):.4f}, "
          f"max={np.max(all_discrete_sr[:, idx]):.4f}")
    print(f"  Continuous: mean={np.mean(all_continuous_sr[:, idx]):.4f}, "
          f"std={np.std(all_continuous_sr[:, idx]):.4f}, "
          f"min={np.min(all_continuous_sr[:, idx]):.4f}, "
          f"max={np.max(all_continuous_sr[:, idx]):.4f}")

    # All runs should be perfectly aligned
    all_aligned = np.max(all_diffs) < 1e-10
    print(f"\nALIGNMENT STATUS: {'✓ ALL RUNS ALIGNED' if all_aligned else '✗ SOME RUNS NOT ALIGNED'}")

    return {
        'all_discrete_sr': all_discrete_sr,
        'all_continuous_sr': all_continuous_sr,
        'all_diffs': all_diffs,
        'years_since_start': years_since_start,
    }


def main():
    """Main comparison function."""

    config = ComparisonConfig()

    # Single comparison with fixed hazard rates
    print("\n" + "=" * 70)
    print("SINGLE COMPARISON (Fixed Hazard Rates)")
    print("=" * 70)

    seed = 42
    discrete_results = run_discrete_model_single(config, seed)
    continuous_results = run_continuous_model(config, discrete_results)

    output_dir = '/Users/joshuaclymer/github/ai_futures_simulator/scripts'
    is_aligned = compare_survival_rates(
        discrete_results,
        continuous_results,
        f'{output_dir}/survival_rate_comparison.png'
    )

    # Multiple comparisons with sampled hazard rates
    multi_results = run_multiple_comparisons(config, num_runs=50)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    return is_aligned


if __name__ == "__main__":
    main()
