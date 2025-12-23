"""
Compare energy requirements of compute produced by covert fabs between discrete and continuous models.

This script compares the energy consumption (GW) of fab-produced compute over time,
tracking how surviving fab-produced chips accumulate energy requirements.

The key metric is: Energy requirements of surviving fab-produced compute (GW)
"""

import sys
import os

# Add both project roots to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')
sys.path.insert(0, '/Users/joshuaclymer/github/covert_compute_production_model')

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional
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
    ProcessNode,
)
from black_project_backend.util import _cache as discrete_cache


@dataclass
class ComparisonConfig:
    """Configuration for the fab energy comparison test."""
    # Time settings
    agreement_year: float = 2030.0
    num_years: float = 7.0
    time_step: float = 1.0

    # Initial conditions
    prc_compute_stock_at_agreement: float = 1e6  # H100e
    proportion_to_divert: float = 0.05

    # Hazard rates
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

    # Fab settings - ENABLED for this comparison
    build_fab: bool = True
    black_fab_construction_labor: int = 500
    black_fab_operating_labor: int = 2000
    black_fab_process_node: str = "best_indigenous"
    black_fab_proportion_of_scanners: float = 0.1

    # Detection parameters
    researcher_headcount: int = 500
    mean_detection_time_100_workers: float = 6.95
    mean_detection_time_1000_workers: float = 3.42
    variance_of_detection_time: float = 3.88

    # US intelligence median errors
    us_error_compute_stock: float = 0.1
    us_error_energy: float = 0.1
    us_error_satellite: float = 0.1
    us_error_sme_stock: float = 0.07


def run_discrete_model_single(config: ComparisonConfig, seed: int) -> Dict:
    """Run a single discrete model simulation and extract fab energy data."""

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
        black_fab_construction_labor=config.black_fab_construction_labor,
        black_fab_operating_labor=config.black_fab_operating_labor,
        black_fab_process_node=config.black_fab_process_node,
        black_fab_proportion_of_prc_lithography_scanners_devoted=config.black_fab_proportion_of_scanners,
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
            us_intelligence_median_error_in_estimate_of_prc_compute_stock=config.us_error_compute_stock,
            us_intelligence_median_error_in_energy_consumption_estimate_of_prc_datacenter_capacity=config.us_error_energy,
            us_intelligence_median_error_in_satellite_estimate_of_prc_datacenter_capacity=config.us_error_satellite,
            us_intelligence_median_error_in_estimate_of_prc_sme_stock=config.us_error_sme_stock,
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
    stock = project.black_project_stock
    black_fab = project.black_fab

    years = np.arange(
        config.agreement_year,
        config.agreement_year + config.num_years + config.time_step,
        config.time_step
    )

    results = {
        'years': years,
        'fab_energy_gw': [],
        'initial_stock_energy_gw': [],
        'total_energy_gw': [],
        'fab_h100e': [],
        'initial_stock_h100e': [],
        'fab_is_operational': [],
        'fab_process_node': None,
        'fab_construction_duration': None,
        'fab_wafer_starts_per_month': None,
    }

    # Track fab properties if built
    if black_fab is not None:
        results['fab_process_node'] = str(black_fab.process_node.value) if hasattr(black_fab.process_node, 'value') else str(black_fab.process_node)
        results['fab_construction_duration'] = black_fab.construction_duration
        results['fab_wafer_starts_per_month'] = black_fab.wafer_starts_per_month

    for year in years:
        # Get energy consumption by source
        initial_energy, fab_energy, initial_h100e, fab_h100e = stock.surviving_compute_energy_by_source(year)

        results['fab_energy_gw'].append(fab_energy)
        results['initial_stock_energy_gw'].append(initial_energy)
        results['total_energy_gw'].append(initial_energy + fab_energy)
        results['fab_h100e'].append(fab_h100e)
        results['initial_stock_h100e'].append(initial_h100e)

        # Check fab operational status
        is_operational = black_fab.is_operational(year) if black_fab is not None else False
        results['fab_is_operational'].append(is_operational)

        # Also get the monthly production rate energy (this is different - it's the energy
        # to run chips produced in ONE month, not the cumulative surviving compute)
        if black_fab is not None and black_fab.is_operational(year):
            monthly_prod_energy = black_fab.energy_consumption_per_month_gw(year)
        else:
            monthly_prod_energy = 0.0
        results.setdefault('fab_monthly_production_energy_gw', []).append(monthly_prod_energy)

    # Convert lists to numpy arrays
    for key in ['fab_energy_gw', 'initial_stock_energy_gw', 'total_energy_gw',
                'fab_h100e', 'initial_stock_h100e', 'fab_is_operational',
                'fab_monthly_production_energy_gw']:
        results[key] = np.array(results[key])

    return results


def run_multiple_simulations(config: ComparisonConfig, num_sims: int = 200):
    """Run multiple simulations and collect distributions."""

    print(f"Running {num_sims} discrete model simulations...")

    discrete_results = []

    for i in range(num_sims):
        if (i + 1) % 50 == 0:
            print(f"  Simulation {i + 1}/{num_sims}")

        seed = 1000 + i
        result = run_discrete_model_single(config, seed)
        discrete_results.append(result)

    return discrete_results


def analyze_distributions(
    discrete_results: List[Dict],
    config: ComparisonConfig,
    year_offsets: List[float] = [0, 2, 4, 6, 7]
):
    """Analyze distributions at specific year offsets."""

    print("\n" + "=" * 70)
    print("FAB ENERGY REQUIREMENTS ANALYSIS (DISCRETE MODEL)")
    print("=" * 70)

    years = discrete_results[0]['years']

    # Count how many simulations have operational fabs
    fabs_built = sum(1 for r in discrete_results if r['fab_process_node'] is not None)
    print(f"\nFabs built: {fabs_built}/{len(discrete_results)} simulations")

    # Show fab properties for simulations with fabs
    fab_results = [r for r in discrete_results if r['fab_process_node'] is not None]
    if fab_results:
        process_nodes = [r['fab_process_node'] for r in fab_results]
        unique_nodes = list(set(process_nodes))
        print(f"Process nodes: {unique_nodes}")

        construction_durations = [r['fab_construction_duration'] for r in fab_results]
        print(f"Construction duration: median={np.median(construction_durations):.2f}, "
              f"mean={np.mean(construction_durations):.2f}, "
              f"min={np.min(construction_durations):.2f}, max={np.max(construction_durations):.2f} years")

        wafer_starts = [r['fab_wafer_starts_per_month'] for r in fab_results]
        print(f"Wafer starts/month: median={np.median(wafer_starts):.0f}, "
              f"mean={np.mean(wafer_starts):.0f}, "
              f"min={np.min(wafer_starts):.0f}, max={np.max(wafer_starts):.0f}")

    for year_offset in year_offsets:
        year = config.agreement_year + year_offset
        idx = int(year_offset / config.time_step)

        if idx >= len(years):
            continue

        print(f"\n--- Year {year:.0f} (t+{year_offset:.0f}) ---")

        # Extract values at this year
        fab_energy = [r['fab_energy_gw'][idx] for r in discrete_results]
        initial_energy = [r['initial_stock_energy_gw'][idx] for r in discrete_results]
        fab_h100e = [r['fab_h100e'][idx] for r in discrete_results]
        fab_operational = [r['fab_is_operational'][idx] for r in discrete_results]

        num_operational = sum(fab_operational)

        print(f"\n  Fabs operational: {num_operational}/{len(discrete_results)}")

        print(f"\n  Fab-produced compute energy (GW):")
        print(f"    Median: {np.median(fab_energy):.6f}")
        print(f"    Mean:   {np.mean(fab_energy):.6f}")
        print(f"    25th %ile: {np.percentile(fab_energy, 25):.6f}")
        print(f"    75th %ile: {np.percentile(fab_energy, 75):.6f}")

        print(f"\n  Fab-produced compute (H100e):")
        print(f"    Median: {np.median(fab_h100e):,.0f}")
        print(f"    Mean:   {np.mean(fab_h100e):,.0f}")

        print(f"\n  Initial stock energy (GW):")
        print(f"    Median: {np.median(initial_energy):.6f}")
        print(f"    Mean:   {np.mean(initial_energy):.6f}")

        # NEW: Monthly production energy (energy to run chips produced in 1 month)
        monthly_prod_energy = [r['fab_monthly_production_energy_gw'][idx] for r in discrete_results]
        print(f"\n  Fab monthly production energy (GW/month) [energy to run 1 month's production]:")
        print(f"    Median: {np.median(monthly_prod_energy):.6f}")
        print(f"    Mean:   {np.mean(monthly_prod_energy):.6f}")
        print(f"    25th %ile: {np.percentile(monthly_prod_energy, 25):.6f}")
        print(f"    75th %ile: {np.percentile(monthly_prod_energy, 75):.6f}")


def plot_time_series(
    discrete_results: List[Dict],
    config: ComparisonConfig,
    output_path: str
):
    """Plot median and percentile bands over time."""

    years = discrete_results[0]['years']

    # Stack results
    fab_energy_all = np.array([r['fab_energy_gw'] for r in discrete_results])
    initial_energy_all = np.array([r['initial_stock_energy_gw'] for r in discrete_results])
    total_energy_all = np.array([r['total_energy_gw'] for r in discrete_results])
    fab_h100e_all = np.array([r['fab_h100e'] for r in discrete_results])

    # Calculate percentiles
    fab_energy_median = np.median(fab_energy_all, axis=0)
    fab_energy_p25 = np.percentile(fab_energy_all, 25, axis=0)
    fab_energy_p75 = np.percentile(fab_energy_all, 75, axis=0)

    initial_energy_median = np.median(initial_energy_all, axis=0)
    initial_energy_p25 = np.percentile(initial_energy_all, 25, axis=0)
    initial_energy_p75 = np.percentile(initial_energy_all, 75, axis=0)

    # Also get monthly production energy data
    monthly_prod_energy_all = np.array([r['fab_monthly_production_energy_gw'] for r in discrete_results])
    monthly_prod_energy_median = np.median(monthly_prod_energy_all, axis=0)
    monthly_prod_energy_p25 = np.percentile(monthly_prod_energy_all, 25, axis=0)
    monthly_prod_energy_p75 = np.percentile(monthly_prod_energy_all, 75, axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top left: Monthly production energy (THIS SHOULD MATCH THE IMAGE!)
    ax0 = axes[0, 0]
    ax0.plot(years, monthly_prod_energy_median, 'g-', linewidth=2, label='Median')
    ax0.fill_between(years, monthly_prod_energy_p25, monthly_prod_energy_p75, alpha=0.3, color='green', label='25-75th %ile')
    ax0.set_xlabel('Year')
    ax0.set_ylabel('GW/Month')
    ax0.set_title('Energy requirements per month (GW / month)\n[Energy to run 1 month\'s fab production]')
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.set_ylim(bottom=0)

    # Top middle: Fab surviving compute energy requirements
    ax1 = axes[0, 1]
    ax1.plot(years, fab_energy_median, 'g-', linewidth=2, label='Median')
    ax1.fill_between(years, fab_energy_p25, fab_energy_p75, alpha=0.3, color='green', label='25-75th %ile')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GW')
    ax1.set_title('Surviving fab-produced compute energy (GW)\n[Cumulative surviving fab compute]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Top right: Initial stock energy
    ax2 = axes[0, 2]
    ax2.plot(years, initial_energy_median, 'b-', linewidth=2, label='Median')
    ax2.fill_between(years, initial_energy_p25, initial_energy_p75, alpha=0.3, color='blue', label='25-75th %ile')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('GW')
    ax2.set_title('Energy requirements of initial stock (GW)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # Bottom left: Total energy
    total_energy_median = np.median(total_energy_all, axis=0)
    total_energy_p25 = np.percentile(total_energy_all, 25, axis=0)
    total_energy_p75 = np.percentile(total_energy_all, 75, axis=0)

    ax3 = axes[1, 0]
    ax3.plot(years, total_energy_median, 'r-', linewidth=2, label='Median')
    ax3.fill_between(years, total_energy_p25, total_energy_p75, alpha=0.3, color='red', label='25-75th %ile')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('GW')
    ax3.set_title('Total energy requirements (GW)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    # Bottom middle: Fab-produced H100e
    fab_h100e_median = np.median(fab_h100e_all, axis=0)
    fab_h100e_p25 = np.percentile(fab_h100e_all, 25, axis=0)
    fab_h100e_p75 = np.percentile(fab_h100e_all, 75, axis=0)

    ax4 = axes[1, 1]
    ax4.plot(years, fab_h100e_median, 'm-', linewidth=2, label='Median')
    ax4.fill_between(years, fab_h100e_p25, fab_h100e_p75, alpha=0.3, color='magenta', label='25-75th %ile')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('H100e')
    ax4.set_title('Fab-produced compute (H100e)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    # Bottom right: empty (or hide)
    ax5 = axes[1, 2]
    ax5.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    """Main comparison function."""

    config = ComparisonConfig(
        agreement_year=2030.0,
        num_years=7.0,
        time_step=1.0,
        prc_compute_stock_at_agreement=1e6,
        proportion_to_divert=0.05,
        initial_hazard_rate=0.05,
        hazard_rate_increase_per_year=0.02,
        build_fab=True,
        black_fab_construction_labor=500,
        black_fab_operating_labor=2000,
        black_fab_process_node="best_indigenous",
        black_fab_proportion_of_scanners=0.1,
    )

    # Run discrete simulations
    discrete_results = run_multiple_simulations(config, num_sims=200)

    # Analyze distributions
    analyze_distributions(discrete_results, config)

    # Create plots
    output_dir = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/comparison_output'
    os.makedirs(output_dir, exist_ok=True)

    plot_time_series(
        discrete_results, config,
        f'{output_dir}/fab_energy_requirements_comparison.png'
    )

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
