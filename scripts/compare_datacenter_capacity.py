"""
Compare datacenter capacity calculations between discrete and continuous models.

Focuses specifically on:
1. Concealed datacenter capacity over time
2. Unconcealed (diverted) capacity
3. Total datacenter capacity
"""

import sys
import numpy as np

# Add project roots to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')
sys.path.insert(0, '/Users/joshuaclymer/github/covert_compute_production_model')

# Import from discrete model
from black_project_backend.classes.black_datacenters import PRCBlackDatacenters
from black_project_backend.black_project_parameters import (
    BlackProjectParameters as DiscreteBlackProjectParameters,
    BlackProjectProperties as DiscreteBlackProjectProperties,
    BlackDatacenterParameters as DiscreteBlackDatacenterParameters,
    DetectionParameters as DiscreteDetectionParameters,
    ExogenousTrends as DiscreteExogenousTrends,
    SurvivalRateParameters as DiscreteSurvivalRateParameters,
    BlackFabParameters as DiscreteBlackFabParameters,
)

# Import continuous model functions
from world_updaters.compute.black_compute import (
    calculate_concealed_capacity_gw,
    calculate_datacenter_capacity_gw,
    calculate_operating_compute,
)
from world_updaters.compute.chip_survival import calculate_survival_rate
from world_updaters.compute.black_compute import (
    calculate_fab_construction_duration,
    calculate_fab_wafer_starts_per_month,
    calculate_fab_h100e_per_chip,
    calculate_fab_annual_production_h100e,
)


def create_discrete_datacenters(config: dict) -> PRCBlackDatacenters:
    """Create discrete model datacenter object."""

    bp_properties = DiscreteBlackProjectProperties(
        run_a_black_project=True,
        proportion_of_initial_compute_stock_to_divert=config['proportion_to_divert'],
        datacenter_construction_labor=config['datacenter_construction_labor'],
        years_before_agreement_year_prc_starts_building_black_datacenters=config['years_before_agreement_building_dc'],
        max_proportion_of_PRC_energy_consumption=config['max_proportion_prc_energy'],
        fraction_of_datacenter_capacity_not_built_for_concealment_diverted_to_black_project_at_agreement_start=config['fraction_unconcealed_diverted'],
        build_a_black_fab=False,
        researcher_headcount=config['researcher_headcount'],
    )

    bp_params = DiscreteBlackProjectParameters(
        survival_rate_parameters=DiscreteSurvivalRateParameters(
            initial_hazard_rate_p50=0.05,
            increase_of_hazard_rate_per_year_p50=0.02,
            hazard_rate_p25_relative_to_p50=1.0,
            hazard_rate_p75_relative_to_p50=1.0,
        ),
        datacenter_model_parameters=DiscreteBlackDatacenterParameters(
            MW_per_construction_worker_per_year=config['mw_per_worker_per_year'],
            relative_sigma_mw_per_construction_worker_per_year=0.0,  # No sampling
            operating_labor_per_MW=config['operating_labor_per_mw'],
            relative_sigma_operating_labor_per_MW=0.0,
        ),
        black_fab_parameters=DiscreteBlackFabParameters(),
        detection_parameters=DiscreteDetectionParameters(
            mean_detection_time_for_100_workers=6.95,
            mean_detection_time_for_1000_workers=3.42,
            variance_of_detection_time_given_num_workers=3.88,
            us_intelligence_median_error_in_estimate_of_prc_compute_stock=0.1,
            us_intelligence_median_error_in_energy_consumption_estimate_of_prc_datacenter_capacity=0.1,
            us_intelligence_median_error_in_satellite_estimate_of_prc_datacenter_capacity=0.1,
        ),
        exogenous_trends=DiscreteExogenousTrends(
            total_prc_compute_stock_in_2025=config['prc_compute_stock'] / (2.2 ** (config['agreement_year'] - 2025)),
            annual_growth_rate_of_prc_compute_stock_p10=2.2,
            annual_growth_rate_of_prc_compute_stock_p50=2.2,
            annual_growth_rate_of_prc_compute_stock_p90=2.2,
            energy_efficiency_of_prc_stock_relative_to_state_of_the_art=config['energy_efficiency'],
            total_GW_of_PRC_energy_consumption=config['total_prc_energy_gw'],
        ),
    )

    years_since_agreement = list(range(int(config['num_years']) + 1))

    datacenter = PRCBlackDatacenters(
        years_since_agreement_start=years_since_agreement,
        project_parameters=bp_params,
        black_project_properties=bp_properties,
        energy_consumption_of_prc_stock_at_agreement_start=config['energy_consumption_at_agreement_gw'],
        agreement_year=int(config['agreement_year']),
    )

    return datacenter


def calculate_continuous_datacenter_capacity(config: dict, year: float) -> dict:
    """Calculate datacenter capacity using continuous model functions."""

    # Calculate construction rate
    gw_per_worker_per_year = config['mw_per_worker_per_year'] / 1000.0
    construction_rate = gw_per_worker_per_year * config['datacenter_construction_labor']

    # Construction starts before agreement
    construction_start_year = config['agreement_year'] - config['years_before_agreement_building_dc']

    # Max concealed capacity based on energy limits
    max_energy_allocatable = config['max_proportion_prc_energy'] * config['total_prc_energy_gw']
    unconcealed_capacity = config['fraction_unconcealed_diverted'] * config['energy_consumption_at_agreement_gw']
    max_concealed_capacity = max(0, max_energy_allocatable - unconcealed_capacity)

    # Calculate concealed capacity
    concealed = calculate_concealed_capacity_gw(
        current_year=year,
        construction_start_year=construction_start_year,
        construction_rate_gw_per_year=construction_rate,
        max_concealed_capacity_gw=max_concealed_capacity,
    )

    # Calculate total
    total = calculate_datacenter_capacity_gw(
        unconcealed_capacity_gw=unconcealed_capacity,
        concealed_capacity_gw=concealed,
    )

    return {
        'concealed': concealed,
        'unconcealed': unconcealed_capacity,
        'total': total,
        'max_concealed': max_concealed_capacity,
        'construction_rate': construction_rate,
    }


def compare_datacenter_capacity():
    """Compare datacenter capacity between models."""

    config = {
        'agreement_year': 2030.0,
        'num_years': 7.0,
        'prc_compute_stock': 1e6,  # H100e at agreement
        'proportion_to_divert': 0.05,
        'datacenter_construction_labor': 10000,
        'years_before_agreement_building_dc': 1,
        'max_proportion_prc_energy': 0.05,
        'mw_per_worker_per_year': 1.0,
        'operating_labor_per_mw': 0.1,
        'total_prc_energy_gw': 1100.0,
        'energy_efficiency': 0.2,
        'fraction_unconcealed_diverted': 0.1,  # 10% diversion of unconcealed
        'researcher_headcount': 500,
        # Derived: energy consumption of PRC stock at agreement
        # For compute stock of 1e6 H100e at 0.2 efficiency relative to SOTA
        # H100 power = 700W, so at 0.2 efficiency = 700W / 0.2 = 3500W
        # Total power = 1e6 * 3500W = 3.5TW = 3.5 GW
        'energy_consumption_at_agreement_gw': 1e6 * 700 / 0.2 / 1e9,  # 3.5 GW
    }

    print("=" * 70)
    print("DATACENTER CAPACITY COMPARISON")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Agreement year: {config['agreement_year']}")
    print(f"  Construction labor: {config['datacenter_construction_labor']}")
    print(f"  MW per worker per year: {config['mw_per_worker_per_year']}")
    print(f"  Years before agreement building DC: {config['years_before_agreement_building_dc']}")
    print(f"  Max proportion PRC energy: {config['max_proportion_prc_energy']}")
    print(f"  Total PRC energy: {config['total_prc_energy_gw']} GW")
    print(f"  Energy consumption at agreement: {config['energy_consumption_at_agreement_gw']:.2f} GW")

    # Create discrete model
    np.random.seed(42)  # For reproducibility
    discrete_dc = create_discrete_datacenters(config)

    print("\nDiscrete Model Parameters (after sampling):")
    print(f"  GW_per_year_per_construction_labor: {discrete_dc.GW_per_year_per_construction_labor:.6f}")
    print(f"  GW_per_year_of_concealed_datacenters: {discrete_dc.GW_per_year_of_concealed_datacenters:.6f}")
    print(f"  datacenter_start_year_offset: {discrete_dc.datacenter_start_year_offset}")

    print("\nContinuous Model Parameters:")
    continuous_results = calculate_continuous_datacenter_capacity(config, config['agreement_year'])
    print(f"  construction_rate (GW/year): {continuous_results['construction_rate']:.6f}")
    print(f"  max_concealed_capacity (GW): {continuous_results['max_concealed']:.6f}")

    print("\n" + "-" * 70)
    print("Year-by-year comparison:")
    print("-" * 70)
    print(f"{'Year':>6} | {'Discrete':>12} {'Concealed':>10} | {'Continuous':>12} {'Concealed':>10} | {'Diff':>10}")
    print(f"{'':>6} | {'Total (GW)':>12} {'(GW)':>10} | {'Total (GW)':>12} {'(GW)':>10} | {'Total':>10}")
    print("-" * 70)

    years = np.arange(config['agreement_year'], config['agreement_year'] + config['num_years'] + 1)

    max_diff = 0
    for year in years:
        # Discrete model
        discrete_concealed = discrete_dc.get_covert_GW_capacity_concealed(year)
        discrete_unconcealed = discrete_dc.get_covert_GW_capacity_unconcealed_at_agreement_start()
        discrete_total = discrete_dc.get_covert_GW_capacity_total(year)

        # Continuous model
        cont = calculate_continuous_datacenter_capacity(config, year)

        diff = abs(discrete_total - cont['total'])
        max_diff = max(max_diff, diff)

        print(f"{year:>6.0f} | {discrete_total:>12.6f} {discrete_concealed:>10.6f} | {cont['total']:>12.6f} {cont['concealed']:>10.6f} | {diff:>10.6f}")

    print("-" * 70)
    print(f"Maximum difference: {max_diff:.6f} GW")

    # Check if there are issues
    if max_diff < 0.001:
        print("\n✓ Models are well aligned (diff < 0.001 GW)")
    else:
        print(f"\n✗ Models have significant differences")

        # Diagnose the issue
        print("\n" + "=" * 70)
        print("DIAGNOSIS")
        print("=" * 70)

        # Check construction start calculation
        print("\nConstruction start calculation:")
        discrete_construction_start = config['agreement_year'] + discrete_dc.datacenter_start_year_offset
        continuous_construction_start = config['agreement_year'] - config['years_before_agreement_building_dc']
        print(f"  Discrete:   {discrete_construction_start}")
        print(f"  Continuous: {continuous_construction_start}")

        # Check GW per year
        print("\nConstruction rate (GW/year):")
        print(f"  Discrete:   {discrete_dc.GW_per_year_of_concealed_datacenters:.6f}")
        print(f"  Continuous: {continuous_results['construction_rate']:.6f}")

        # Check max capacity calculation
        print("\nMax concealed capacity calculation:")
        discrete_max = config['max_proportion_prc_energy'] * config['total_prc_energy_gw'] - discrete_dc.get_covert_GW_capacity_unconcealed_at_agreement_start()
        print(f"  Discrete max:   {discrete_max:.6f} GW")
        print(f"  Continuous max: {continuous_results['max_concealed']:.6f} GW")


def compare_survival_rates():
    """Compare survival rate calculations between models."""

    print("\n" + "=" * 70)
    print("SURVIVAL RATE COMPARISON")
    print("=" * 70)

    # Test parameters
    initial_hazard_rate = 0.05
    hazard_rate_increase_per_year = 0.02

    print(f"\nParameters:")
    print(f"  Initial hazard rate: {initial_hazard_rate}")
    print(f"  Hazard rate increase per year: {hazard_rate_increase_per_year}")

    print("\nYear-by-year comparison:")
    print("-" * 50)
    print(f"{'Year':>6} | {'Discrete':>12} | {'Continuous':>12} | {'Diff':>12}")
    print("-" * 50)

    max_diff = 0
    for year_offset in range(8):
        years_of_life = float(year_offset)

        # Discrete model formula
        cumulative_hazard = initial_hazard_rate * years_of_life + hazard_rate_increase_per_year * years_of_life**2 / 2
        discrete_survival = np.exp(-cumulative_hazard)

        # Continuous model
        continuous_survival = calculate_survival_rate(
            years_since_acquisition=years_of_life,
            initial_hazard_rate=initial_hazard_rate,
            hazard_rate_increase_per_year=hazard_rate_increase_per_year,
        )

        diff = abs(discrete_survival - continuous_survival)
        max_diff = max(max_diff, diff)

        print(f"{year_offset:>6} | {discrete_survival:>12.8f} | {continuous_survival:>12.8f} | {diff:>12.10f}")

    print("-" * 50)
    print(f"Maximum difference: {max_diff:.10e}")

    if max_diff < 1e-10:
        print("\n✓ Survival rate calculations are aligned")
    else:
        print(f"\n✗ Survival rate calculations differ")


def compare_total_black_project_compute():
    """Compare total black project compute (initial stock * survival rate)."""

    print("\n" + "=" * 70)
    print("TOTAL BLACK PROJECT COMPUTE COMPARISON (No Fab)")
    print("=" * 70)

    config = {
        'agreement_year': 2030.0,
        'num_years': 7.0,
        'initial_compute_stock_h100e': 50000.0,  # Initial diverted compute
        'initial_hazard_rate': 0.05,
        'hazard_rate_increase_per_year': 0.02,
        'energy_efficiency': 0.2,  # Relative to H100
    }

    print(f"\nConfiguration:")
    print(f"  Agreement year: {config['agreement_year']}")
    print(f"  Initial compute stock: {config['initial_compute_stock_h100e']} H100e")
    print(f"  Initial hazard rate: {config['initial_hazard_rate']}")
    print(f"  Hazard rate increase: {config['hazard_rate_increase_per_year']}")

    H100_POWER_W = 700.0
    watts_per_h100e = H100_POWER_W / config['energy_efficiency']

    print(f"\n{'Year':>6} | {'Discrete':>14} | {'Continuous':>14} | {'Diff':>14}")
    print(f"{'':>6} | {'Surviving (H100e)':>14} | {'Surviving (H100e)':>14} | {'':>14}")
    print("-" * 60)

    max_diff = 0
    for year_offset in range(int(config['num_years']) + 1):
        years_of_life = float(year_offset)

        # Discrete model formula (initial stock only, no fab)
        cumulative_hazard = config['initial_hazard_rate'] * years_of_life + config['hazard_rate_increase_per_year'] * years_of_life**2 / 2
        discrete_survival = np.exp(-cumulative_hazard)
        discrete_surviving = config['initial_compute_stock_h100e'] * discrete_survival

        # Continuous model
        continuous_survival = calculate_survival_rate(
            years_since_acquisition=years_of_life,
            initial_hazard_rate=config['initial_hazard_rate'],
            hazard_rate_increase_per_year=config['hazard_rate_increase_per_year'],
        )
        continuous_surviving = config['initial_compute_stock_h100e'] * continuous_survival

        diff = abs(discrete_surviving - continuous_surviving)
        max_diff = max(max_diff, diff)

        print(f"{int(config['agreement_year'] + year_offset):>6} | {discrete_surviving:>14.2f} | {continuous_surviving:>14.2f} | {diff:>14.6f}")

    print("-" * 60)
    print(f"Maximum difference: {max_diff:.6f} H100e")

    if max_diff < 0.001:
        print("\n✓ Total black project compute calculations are aligned")
    else:
        print(f"\n✗ Total black project compute calculations differ")


def compare_operational_compute():
    """Compare operational compute (capped by datacenter capacity)."""

    print("\n" + "=" * 70)
    print("OPERATIONAL COMPUTE COMPARISON (Capacity-Limited)")
    print("=" * 70)

    config = {
        'agreement_year': 2030.0,
        'num_years': 7.0,
        'initial_compute_stock_h100e': 1e6,  # 1 million H100e
        'initial_hazard_rate': 0.05,
        'hazard_rate_increase_per_year': 0.02,
        'energy_efficiency': 0.2,
        'datacenter_construction_labor': 10000,
        'years_before_agreement_building_dc': 1,
        'max_proportion_prc_energy': 0.05,
        'mw_per_worker_per_year': 1.0,
        'total_prc_energy_gw': 1100.0,
        'fraction_unconcealed_diverted': 0.0,
        'energy_consumption_at_agreement_gw': 3.5,
    }

    H100_POWER_W = 700.0
    watts_per_h100e = H100_POWER_W / config['energy_efficiency']

    # Construction rate
    gw_per_worker_per_year = config['mw_per_worker_per_year'] / 1000.0
    construction_rate = gw_per_worker_per_year * config['datacenter_construction_labor']
    construction_start_year = config['agreement_year'] - config['years_before_agreement_building_dc']

    # Max concealed capacity
    max_energy_allocatable = config['max_proportion_prc_energy'] * config['total_prc_energy_gw']
    unconcealed_capacity = config['fraction_unconcealed_diverted'] * config['energy_consumption_at_agreement_gw']
    max_concealed_capacity = max(0, max_energy_allocatable - unconcealed_capacity)

    print(f"\nConfiguration:")
    print(f"  Initial compute stock: {config['initial_compute_stock_h100e']:.0f} H100e")
    print(f"  Watts per H100e: {watts_per_h100e:.0f} W")
    print(f"  Datacenter construction rate: {construction_rate:.1f} GW/year")

    print(f"\n{'Year':>6} | {'DC Cap (GW)':>12} | {'Surviving':>14} | {'Operational':>14}")
    print(f"{'':>6} | {'':>12} | {'(H100e)':>14} | {'(H100e)':>14}")
    print("-" * 60)

    for year_offset in range(int(config['num_years']) + 1):
        year = config['agreement_year'] + year_offset

        # Datacenter capacity
        dc_cap = calculate_concealed_capacity_gw(
            current_year=year,
            construction_start_year=construction_start_year,
            construction_rate_gw_per_year=construction_rate,
            max_concealed_capacity_gw=max_concealed_capacity,
        ) + unconcealed_capacity

        # Survival rate
        survival = calculate_survival_rate(
            years_since_acquisition=float(year_offset),
            initial_hazard_rate=config['initial_hazard_rate'],
            hazard_rate_increase_per_year=config['hazard_rate_increase_per_year'],
        )
        surviving = config['initial_compute_stock_h100e'] * survival

        # Operational compute
        operational = calculate_operating_compute(
            functional_compute_h100e=surviving,
            datacenter_capacity_gw=dc_cap,
            watts_per_h100e=watts_per_h100e,
        )

        max_powered = dc_cap * 1e9 / watts_per_h100e

        print(f"{int(year):>6} | {dc_cap:>12.3f} | {surviving:>14.0f} | {operational:>14.0f}")
        if operational < surviving:
            print(f"       | (max powered: {max_powered:,.0f} H100e)")

    print("-" * 60)


def compare_fab_production():
    """Compare fab production calculations between models (formulas only, no sampling)."""

    print("\n" + "=" * 70)
    print("FAB PRODUCTION FORMULA COMPARISON")
    print("=" * 70)

    # Test parameters - using fixed values (no sampling)
    config = {
        'operating_labor': 2000,
        'num_scanners': 10,
        'process_node_nm': 28.0,
        'h100_reference_nm': 4.0,
        'chips_per_wafer': 28,
        'wafers_per_month_per_worker': 24.64,
        'wafers_per_month_per_scanner': 1000.0,
        'transistor_density_scaling_exponent': 1.49,
        'architecture_efficiency_per_year': 1.23,
        'h100_release_year': 2022,
        'current_year': 2030,
    }

    print("\nConfiguration:")
    print(f"  Operating labor: {config['operating_labor']}")
    print(f"  Number of scanners: {config['num_scanners']}")
    print(f"  Process node: {config['process_node_nm']} nm")
    print(f"  Chips per wafer: {config['chips_per_wafer']}")

    # Calculate wafer starts per month
    from_labor = config['operating_labor'] * config['wafers_per_month_per_worker']
    from_scanners = config['num_scanners'] * config['wafers_per_month_per_scanner']
    discrete_wafer_starts = min(from_labor, from_scanners)

    continuous_wafer_starts = calculate_fab_wafer_starts_per_month(
        fab_operating_labor=config['operating_labor'],
        fab_number_of_lithography_scanners=config['num_scanners'],
        wafers_per_month_per_worker=config['wafers_per_month_per_worker'],
        wafers_per_month_per_scanner=config['wafers_per_month_per_scanner'],
    )

    print(f"\nWafer starts per month:")
    print(f"  Discrete:   {discrete_wafer_starts:.2f}")
    print(f"  Continuous: {continuous_wafer_starts:.2f}")
    print(f"  Diff:       {abs(discrete_wafer_starts - continuous_wafer_starts):.6f}")

    # Calculate H100e per chip
    density_ratio = (config['h100_reference_nm'] / config['process_node_nm']) ** config['transistor_density_scaling_exponent']
    years_since_h100 = config['current_year'] - config['h100_release_year']
    arch_efficiency = config['architecture_efficiency_per_year'] ** years_since_h100
    discrete_h100e_per_chip = density_ratio * arch_efficiency

    continuous_h100e_per_chip = calculate_fab_h100e_per_chip(
        fab_process_node_nm=config['process_node_nm'],
        year=config['current_year'],
        h100_reference_nm=config['h100_reference_nm'],
        transistor_density_scaling_exponent=config['transistor_density_scaling_exponent'],
        architecture_efficiency_improvement_per_year=config['architecture_efficiency_per_year'],
        h100_release_year=config['h100_release_year'],
    )

    print(f"\nH100e per chip:")
    print(f"  Discrete:   {discrete_h100e_per_chip:.6f}")
    print(f"  Continuous: {continuous_h100e_per_chip:.6f}")
    print(f"  Diff:       {abs(discrete_h100e_per_chip - continuous_h100e_per_chip):.10f}")

    # Calculate annual production
    discrete_annual = discrete_wafer_starts * config['chips_per_wafer'] * discrete_h100e_per_chip * 12.0

    continuous_annual = calculate_fab_annual_production_h100e(
        fab_wafer_starts_per_month=continuous_wafer_starts,
        fab_chips_per_wafer=config['chips_per_wafer'],
        fab_h100e_per_chip=continuous_h100e_per_chip,
        fab_is_operational=True,
    )

    print(f"\nAnnual production (H100e/year):")
    print(f"  Discrete:   {discrete_annual:,.2f}")
    print(f"  Continuous: {continuous_annual:,.2f}")
    print(f"  Diff:       {abs(discrete_annual - continuous_annual):.6f}")

    # Check alignment
    total_diff = abs(discrete_annual - continuous_annual)
    if total_diff < 0.01:
        print("\n✓ Fab production formulas are aligned")
    else:
        print(f"\n✗ Fab production formulas differ by {total_diff:.6f} H100e/year")


if __name__ == "__main__":
    compare_datacenter_capacity()
    compare_survival_rates()
    compare_total_black_project_compute()
    compare_operational_compute()
    compare_fab_production()
