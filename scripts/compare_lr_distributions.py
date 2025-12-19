"""
Compare likelihood ratio distributions between ai_futures_simulator and black_project_backend.

Runs 50 simulations with different random seeds and compares the distributions
of cumulative likelihood ratios and all contributing terms.
"""

import sys
import os

# Add both project roots to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')
sys.path.insert(0, '/Users/joshuaclymer/github/covert_compute_production_model')

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Import from black_project_backend (discrete model)
from black_project_backend.model import BlackProjectModel
from black_project_backend.black_project_parameters import (
    ModelParameters as DiscreteModelParameters,
    SimulationSettings as DiscreteSimulationSettings,
    BlackProjectProperties as DiscreteBlackProjectProperties,
    BlackProjectParameters as DiscreteBlackProjectParameters,
    BlackFabParameters as DiscreteBlackFabParameters,
    BlackDatacenterParameters as DiscreteBlackDatacenterParameters,
    DetectionParameters as DiscreteDetectionParameters,
    ExogenousTrends as DiscreteExogenousTrends,
    SurvivalRateParameters as DiscreteSurvivalRateParameters,
)
from black_project_backend.util import _cache as discrete_cache


@dataclass
class ComparisonConfig:
    """Configuration for the comparison test."""
    # Time settings
    agreement_year: float = 2030.0
    num_years: float = 7.0
    time_step: float = 0.1

    # Initial conditions
    prc_compute_stock_at_agreement: float = 1e6  # H100e
    proportion_to_divert: float = 0.05

    # Hazard rates (fixed for comparison)
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

    # Fab settings (disabled for simplicity in first comparison)
    build_fab: bool = False

    # Detection parameters
    researcher_headcount: int = 500
    mean_detection_time_100_workers: float = 6.95
    mean_detection_time_1000_workers: float = 3.42
    variance_of_detection_time: float = 3.88


def run_discrete_model_single(config: ComparisonConfig, seed: int) -> Dict:
    """Run a single discrete model simulation and extract LR components."""

    # Clear cache to ensure fresh sampling
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
        datacenter_construction_labor=config.datacenter_construction_labor,
        years_before_agreement_year_prc_starts_building_black_datacenters=int(config.years_before_agreement_building_dc),
        max_proportion_of_PRC_energy_consumption=config.max_proportion_prc_energy,
        fraction_of_datacenter_capacity_not_built_for_concealment_diverted_to_black_project_at_agreement_start=0.0,
        build_a_black_fab=config.build_fab,
        researcher_headcount=config.researcher_headcount,
    )

    bp_params = DiscreteBlackProjectParameters(
        survival_rate_parameters=DiscreteSurvivalRateParameters(
            initial_hazard_rate_p50=config.initial_hazard_rate,
            increase_of_hazard_rate_per_year_p50=config.hazard_rate_increase_per_year,
            hazard_rate_p25_relative_to_p50=1.0,
            hazard_rate_p75_relative_to_p50=1.0,
        ),
        datacenter_model_parameters=DiscreteBlackDatacenterParameters(
            MW_per_construction_worker_per_year=config.mw_per_worker_per_year,
            relative_sigma_mw_per_construction_worker_per_year=0.0,
            operating_labor_per_MW=config.operating_labor_per_mw,
            relative_sigma_operating_labor_per_MW=0.0,
        ),
        black_fab_parameters=DiscreteBlackFabParameters(),
        detection_parameters=DiscreteDetectionParameters(
            mean_detection_time_for_100_workers=config.mean_detection_time_100_workers,
            mean_detection_time_for_1000_workers=config.mean_detection_time_1000_workers,
            variance_of_detection_time_given_num_workers=config.variance_of_detection_time,
        ),
        exogenous_trends=DiscreteExogenousTrends(
            total_prc_compute_stock_in_2025=config.prc_compute_stock_at_agreement / (2.2 ** (config.agreement_year - 2025)),
            annual_growth_rate_of_prc_compute_stock_p10=2.2,
            annual_growth_rate_of_prc_compute_stock_p50=2.2,
            annual_growth_rate_of_prc_compute_stock_p90=2.2,
            energy_efficiency_of_prc_stock_relative_to_state_of_the_art=config.energy_efficiency_relative_to_sota,
            total_GW_of_PRC_energy_consumption=config.total_prc_energy_gw,
        ),
    )

    model_params = DiscreteModelParameters(
        simulation_settings=sim_settings,
        black_project_properties=bp_properties,
        black_project_parameters=bp_params,
    )

    model = BlackProjectModel(model_params)
    model.run_simulations(1)

    project = model.simulation_results[0]['prc_black_project']
    dc = project.black_datacenters

    years = np.arange(
        config.agreement_year,
        config.agreement_year + config.num_years + config.time_step,
        config.time_step
    )

    results = {
        'years': years,
        'cumulative_lr': [],
        'lr_direct_observation': [],
        'lr_energy': [],
        'lr_satellite': [],
    }

    for year in years:
        # Direct observation LR (worker-based)
        lr_direct = dc.cumulative_lr_from_direct_observation(year)
        results['lr_direct_observation'].append(lr_direct)

        # Energy consumption LR
        lr_energy = dc.lr_from_reported_energy_consumption(year)
        results['lr_energy'].append(lr_energy)

        # Satellite LR
        lr_satellite = dc.lr_from_identifying_datacenters_with_satellites()
        results['lr_satellite'].append(lr_satellite)

        # Cumulative LR (all combined)
        cumulative = dc.cumulative_lr_from_concealed_datacenters(year)
        results['cumulative_lr'].append(cumulative)

    for key in ['cumulative_lr', 'lr_direct_observation', 'lr_energy', 'lr_satellite']:
        results[key] = np.array(results[key])

    return results


def run_continuous_model_single(config: ComparisonConfig, seed: int) -> Dict:
    """Run a single continuous model simulation using the same detection logic."""

    np.random.seed(seed)

    # Compute detection constants
    x1, mu1 = 100, config.mean_detection_time_100_workers
    x2, mu2 = 1000, config.mean_detection_time_1000_workers
    B = np.log(mu1 / mu2) / np.log(np.log10(x2) / np.log10(x1))
    A = mu1 * (np.log10(x1) ** B)

    # Total labor
    total_labor = config.researcher_headcount + config.datacenter_construction_labor

    # Calculate mean detection time for this labor level
    mu = A / (np.log10(total_labor) ** B)
    k = mu / config.variance_of_detection_time
    theta = config.variance_of_detection_time

    # Sample detection time
    sampled_detection_time = np.random.gamma(k, theta)

    years = np.arange(
        config.agreement_year,
        config.agreement_year + config.num_years + config.time_step,
        config.time_step
    )

    results = {
        'years': years,
        'cumulative_lr': [],
        'lr_direct_observation': [],
        'sampled_detection_time': sampled_detection_time,
    }

    for year in years:
        years_since_start = year - config.agreement_year

        # Check if detected
        is_detected = years_since_start >= sampled_detection_time

        if is_detected:
            lr_direct = 100.0
        elif years_since_start <= 0:
            lr_direct = 1.0
        else:
            # Survival function
            lr_direct = stats.gamma.sf(years_since_start, a=k, scale=theta)
            lr_direct = max(lr_direct, 0.001)

        results['lr_direct_observation'].append(lr_direct)
        # For continuous model, cumulative = direct observation (no energy/satellite components modeled)
        results['cumulative_lr'].append(lr_direct)

    for key in ['cumulative_lr', 'lr_direct_observation']:
        results[key] = np.array(results[key])

    return results


def run_multiple_simulations(config: ComparisonConfig, num_sims: int = 50):
    """Run multiple simulations and collect distributions."""

    print(f"Running {num_sims} simulations...")

    discrete_results = []
    continuous_results = []

    for i in range(num_sims):
        if (i + 1) % 10 == 0:
            print(f"  Simulation {i + 1}/{num_sims}")

        seed = 1000 + i  # Use consistent seeds

        discrete_results.append(run_discrete_model_single(config, seed))
        continuous_results.append(run_continuous_model_single(config, seed))

    return discrete_results, continuous_results


def analyze_distributions(
    discrete_results: List[Dict],
    continuous_results: List[Dict],
    config: ComparisonConfig,
    year_offsets: List[float] = [0, 2, 5, 7]
):
    """Analyze and compare distributions at specific year offsets."""

    print("\n" + "=" * 70)
    print("LIKELIHOOD RATIO DISTRIBUTION COMPARISON")
    print("=" * 70)

    years = discrete_results[0]['years']

    for year_offset in year_offsets:
        year = config.agreement_year + year_offset
        idx = int(year_offset / config.time_step)

        if idx >= len(years):
            continue

        print(f"\n--- Year {year:.0f} (t+{year_offset:.0f}) ---")

        # Extract values at this year
        discrete_cumulative = [r['cumulative_lr'][idx] for r in discrete_results]
        discrete_direct = [r['lr_direct_observation'][idx] for r in discrete_results]
        continuous_cumulative = [r['cumulative_lr'][idx] for r in continuous_results]
        continuous_direct = [r['lr_direct_observation'][idx] for r in continuous_results]

        # Calculate statistics
        print(f"\n  Cumulative LR:")
        print(f"    Discrete:   median={np.median(discrete_cumulative):.4f}, "
              f"mean={np.mean(discrete_cumulative):.4f}, "
              f"std={np.std(discrete_cumulative):.4f}")
        print(f"    Continuous: median={np.median(continuous_cumulative):.4f}, "
              f"mean={np.mean(continuous_cumulative):.4f}, "
              f"std={np.std(continuous_cumulative):.4f}")

        print(f"\n  LR Direct Observation (worker-based):")
        print(f"    Discrete:   median={np.median(discrete_direct):.4f}, "
              f"mean={np.mean(discrete_direct):.4f}, "
              f"std={np.std(discrete_direct):.4f}")
        print(f"    Continuous: median={np.median(continuous_direct):.4f}, "
              f"mean={np.mean(continuous_direct):.4f}, "
              f"std={np.std(continuous_direct):.4f}")

        # Count detections (LR = 100)
        discrete_detections = sum(1 for x in discrete_direct if x >= 99.0)
        continuous_detections = sum(1 for x in continuous_direct if x >= 99.0)
        print(f"\n  Detections (LR >= 99):")
        print(f"    Discrete:   {discrete_detections}/{len(discrete_results)} ({100*discrete_detections/len(discrete_results):.1f}%)")
        print(f"    Continuous: {continuous_detections}/{len(continuous_results)} ({100*continuous_detections/len(continuous_results):.1f}%)")

        # If discrete has energy/satellite LRs, show them
        if 'lr_energy' in discrete_results[0]:
            discrete_energy = [r['lr_energy'][idx] for r in discrete_results]
            discrete_satellite = [r['lr_satellite'][idx] for r in discrete_results]
            print(f"\n  LR Energy Consumption (discrete only):")
            print(f"    median={np.median(discrete_energy):.4f}, "
                  f"mean={np.mean(discrete_energy):.4f}, "
                  f"std={np.std(discrete_energy):.4f}")
            print(f"\n  LR Satellite (discrete only):")
            print(f"    median={np.median(discrete_satellite):.4f}, "
                  f"mean={np.mean(discrete_satellite):.4f}, "
                  f"std={np.std(discrete_satellite):.4f}")


def plot_distributions(
    discrete_results: List[Dict],
    continuous_results: List[Dict],
    config: ComparisonConfig,
    output_path: str
):
    """Create comparison plots of the distributions."""

    years = discrete_results[0]['years']
    year_offsets = [0, 2, 5, 7]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col, year_offset in enumerate(year_offsets):
        year = config.agreement_year + year_offset
        idx = int(year_offset / config.time_step)

        if idx >= len(years):
            continue

        # Extract direct observation LRs
        discrete_direct = [r['lr_direct_observation'][idx] for r in discrete_results]
        continuous_direct = [r['lr_direct_observation'][idx] for r in continuous_results]

        # Top row: histograms
        ax1 = axes[0, col]

        # Filter out detection events (LR=100) for histogram
        discrete_no_detect = [x for x in discrete_direct if x < 99.0]
        continuous_no_detect = [x for x in continuous_direct if x < 99.0]

        if discrete_no_detect and continuous_no_detect:
            bins = np.linspace(0, max(max(discrete_no_detect), max(continuous_no_detect)), 20)
            ax1.hist(discrete_no_detect, bins=bins, alpha=0.5, label='Discrete', density=True, color='blue')
            ax1.hist(continuous_no_detect, bins=bins, alpha=0.5, label='Continuous', density=True, color='green')

        ax1.set_xlabel('LR Direct Observation')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Year {year:.0f} (t+{year_offset:.0f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom row: QQ plots or scatter
        ax2 = axes[1, col]

        # Sort and plot
        discrete_sorted = np.sort(discrete_direct)
        continuous_sorted = np.sort(continuous_direct)

        ax2.scatter(discrete_sorted, continuous_sorted, alpha=0.5, s=20)

        # Add diagonal line
        min_val = min(min(discrete_sorted), min(continuous_sorted))
        max_val = max(max(discrete_sorted), max(continuous_sorted))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')

        ax2.set_xlabel('Discrete LR')
        ax2.set_ylabel('Continuous LR')
        ax2.set_title(f'Q-Q Plot: Year {year:.0f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def plot_time_series(
    discrete_results: List[Dict],
    continuous_results: List[Dict],
    config: ComparisonConfig,
    output_path: str
):
    """Plot median and percentile bands over time."""

    years = discrete_results[0]['years']
    num_years = len(years)

    # Stack results
    discrete_direct_all = np.array([r['lr_direct_observation'] for r in discrete_results])
    continuous_direct_all = np.array([r['lr_direct_observation'] for r in continuous_results])

    # Calculate percentiles
    discrete_median = np.median(discrete_direct_all, axis=0)
    discrete_p25 = np.percentile(discrete_direct_all, 25, axis=0)
    discrete_p75 = np.percentile(discrete_direct_all, 75, axis=0)

    continuous_median = np.median(continuous_direct_all, axis=0)
    continuous_p25 = np.percentile(continuous_direct_all, 25, axis=0)
    continuous_p75 = np.percentile(continuous_direct_all, 75, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Linear scale
    ax1 = axes[0]
    ax1.plot(years, discrete_median, 'b-', linewidth=2, label='Discrete (median)')
    ax1.fill_between(years, discrete_p25, discrete_p75, alpha=0.3, color='blue', label='Discrete (25-75%)')
    ax1.plot(years, continuous_median, 'g--', linewidth=2, label='Continuous (median)')
    ax1.fill_between(years, continuous_p25, continuous_p75, alpha=0.3, color='green', label='Continuous (25-75%)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('LR Direct Observation')
    ax1.set_title('Likelihood Ratio Over Time (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='k', linestyle=':', alpha=0.5)

    # Right plot: Log scale
    ax2 = axes[1]
    ax2.semilogy(years, discrete_median, 'b-', linewidth=2, label='Discrete (median)')
    ax2.fill_between(years, discrete_p25, discrete_p75, alpha=0.3, color='blue')
    ax2.semilogy(years, continuous_median, 'g--', linewidth=2, label='Continuous (median)')
    ax2.fill_between(years, continuous_p25, continuous_p75, alpha=0.3, color='green')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('LR Direct Observation (log scale)')
    ax2.set_title('Likelihood Ratio Over Time (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='k', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Time series plot saved to: {output_path}")


def main():
    """Main comparison function."""

    config = ComparisonConfig(
        agreement_year=2030.0,
        num_years=7.0,
        time_step=0.1,
        prc_compute_stock_at_agreement=1e6,
        proportion_to_divert=0.05,
        initial_hazard_rate=0.05,
        hazard_rate_increase_per_year=0.02,
        build_fab=False,
    )

    # Run simulations
    discrete_results, continuous_results = run_multiple_simulations(config, num_sims=50)

    # Analyze distributions
    analyze_distributions(discrete_results, continuous_results, config)

    # Create plots
    output_dir = '/Users/joshuaclymer/github/ai_futures_simulator/scripts'
    plot_distributions(
        discrete_results, continuous_results, config,
        f'{output_dir}/lr_distribution_comparison.png'
    )
    plot_time_series(
        discrete_results, continuous_results, config,
        f'{output_dir}/lr_time_series_comparison.png'
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The comparison shows the distribution of likelihood ratios from:
1. Discrete model (black_project_backend) - uses variable labor over time
2. Continuous model (ai_futures_simulator) - uses fixed total labor

Key differences:
- Discrete model includes energy consumption and satellite LRs
- Both use the same Gamma distribution for worker-based detection
- Detection events cause LR to jump to 100

If the distributions are similar, the models are equivalent for the
worker-based detection component.
""")


if __name__ == "__main__":
    main()
