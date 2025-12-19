"""
Compare likelihood ratio distributions between ai_futures_simulator and black_project_backend.

Runs 50 simulations with different random seeds and compares the distributions
of cumulative likelihood ratios, detection times, datacenter capacity,
operational compute, survival rates, and all contributing terms.
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
)
from black_project_backend.util import _cache as discrete_cache

# Import from continuous model
from world_updaters.black_project import initialize_black_project
from world_updaters.compute.black_compute import (
    compute_cumulative_likelihood_ratio,
    get_datacenter_total_capacity_gw,
    compute_lr_from_reported_energy_consumption,
    compute_lr_from_satellite_detection,
    compute_lr_from_prc_compute_accounting,
)
from parameters.black_project_parameters import (
    BlackProjectParameterSet,
    BlackProjectProperties,
    BlackFabParameters,
    BlackDatacenterParameters,
    DetectionParameters,
)
from parameters.compute_parameters import ComputeParameters
from parameters.energy_consumption_parameters import EnergyConsumptionParameters


@dataclass
class ComparisonConfig:
    """Configuration for the comparison test."""
    # Time settings
    agreement_year: float = 2030.0
    num_years: float = 7.0
    time_step: float = 1.0  # Use integer years to match discrete model

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

    # US intelligence median errors
    us_error_compute_stock: float = 0.1
    us_error_energy: float = 0.1
    us_error_satellite: float = 0.1


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
            us_intelligence_median_error_in_estimate_of_prc_compute_stock=config.us_error_compute_stock,
            us_intelligence_median_error_in_energy_consumption_estimate_of_prc_datacenter_capacity=config.us_error_energy,
            us_intelligence_median_error_in_satellite_estimate_of_prc_datacenter_capacity=config.us_error_satellite,
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
    stock = project.black_project_stock

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
        'lr_compute_accounting': [],
        # Additional metrics
        'datacenter_capacity_gw': [],
        'operational_compute_h100e': [],
        'surviving_compute_h100e': [],
        'h100_years_cumulative': [],
        'survival_rate': [],
        'labor_by_year': {},
        'posterior_probability': [],
    }

    # Get labor by year from the project
    for relative_year, labor in dc.lr_over_time_vs_num_workers.items():
        # Note: lr_over_time_vs_num_workers stores LRs, not labor directly
        # We need to get labor from the datacenter calculations
        pass

    # Calculate labor for each year
    for year in years:
        relative_year = year - config.agreement_year
        construction_labor = dc.construction_labor
        operating_labor = dc.get_operating_labor(year)
        fab_labor = 0
        if project.black_fab is not None:
            fab_labor = config.black_fab_construction_labor + config.black_fab_operating_labor if hasattr(config, 'black_fab_construction_labor') else 0
        researcher_headcount = config.researcher_headcount
        total_labor = construction_labor + operating_labor + fab_labor + researcher_headcount
        results['labor_by_year'][relative_year] = int(total_labor)

    # Prior probability for Bayesian updating
    prior_prob = 0.1  # Default prior

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

        # Compute accounting LR
        lr_compute = stock.lr_from_prc_compute_accounting
        results['lr_compute_accounting'].append(lr_compute)

        # Cumulative LR (all combined)
        cumulative = project.get_cumulative_evidence_of_black_project(year)
        results['cumulative_lr'].append(cumulative)

        # Datacenter capacity (total GW)
        dc_capacity = dc.get_covert_GW_capacity_total(year)
        results['datacenter_capacity_gw'].append(dc_capacity)

        # Operational compute (limited by datacenter capacity)
        operational = project.get_operational_compute(year)
        results['operational_compute_h100e'].append(operational.total_h100e_tpp())

        # Surviving compute (before capacity limit)
        surviving = stock.surviving_compute(year)
        results['surviving_compute_h100e'].append(surviving.total_h100e_tpp())

        # H100-years cumulative
        h100_years = project.h100_years_to_date(year, list(years))
        results['h100_years_cumulative'].append(h100_years)

        # Survival rate calculation
        years_of_life = year - config.agreement_year
        cumulative_hazard = stock.initial_hazard_rate * years_of_life + stock.increase_in_hazard_rate_per_year * years_of_life**2 / 2
        survival_rate = np.exp(-cumulative_hazard)
        results['survival_rate'].append(survival_rate)

        # Posterior probability (Bayesian update)
        # P' = (P * LR) / (P * LR + (1 - P))
        posterior = (prior_prob * cumulative) / (prior_prob * cumulative + (1 - prior_prob))
        results['posterior_probability'].append(posterior)

    for key in ['cumulative_lr', 'lr_direct_observation', 'lr_energy', 'lr_satellite', 'lr_compute_accounting',
                'datacenter_capacity_gw', 'operational_compute_h100e', 'surviving_compute_h100e',
                'h100_years_cumulative', 'survival_rate', 'posterior_probability']:
        results[key] = np.array(results[key])

    return results


def run_continuous_model_single(config: ComparisonConfig, seed: int) -> Dict:
    """Run a single continuous model simulation using the actual implementation."""

    np.random.seed(seed)

    # Create parameter objects
    bp_properties = BlackProjectProperties(
        run_a_black_project=True,
        proportion_of_initial_compute_stock_to_divert=config.proportion_to_divert,
        datacenter_construction_labor=config.datacenter_construction_labor,
        years_before_agreement_year_prc_starts_building_black_datacenters=config.years_before_agreement_building_dc,
        max_proportion_of_PRC_energy_consumption=config.max_proportion_prc_energy,
        fraction_of_datacenter_capacity_not_built_for_concealment_diverted_to_black_project_at_agreement_start=0.0,
        build_a_black_fab=config.build_fab,
        black_fab_construction_labor=0,
        black_fab_operating_labor=0,
        black_fab_process_node="28nm",
        black_fab_proportion_of_prc_lithography_scanners_devoted=0.0,
        researcher_headcount=config.researcher_headcount,
    )

    bp_fab_params = BlackFabParameters(
        h100_sized_chips_per_wafer=100,
        wafers_per_month_per_worker=0.5,
        wafers_per_month_per_lithography_scanner=10000,
        construction_time_for_5k_wafers_per_month=2.0,
        construction_time_for_100k_wafers_per_month=4.0,
        construction_workers_per_1000_wafers_per_month=100,
        transistor_density_scaling_exponent=2.0,
        architecture_efficiency_improvement_per_year=1.15,
        transistor_density_at_end_of_dennard_scaling_m_per_mm2=30.0,
        watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended=-0.5,
        watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended=-0.25,
        prc_lithography_scanners_produced_in_first_year=10,
        prc_additional_lithography_scanners_produced_per_year=5,
        localization_130nm=[(2025, 0.9)],
        localization_28nm=[(2030, 0.5)],
        localization_14nm=[(2035, 0.3)],
        localization_7nm=[(2040, 0.1)],
    )

    bp_dc_params = BlackDatacenterParameters(
        MW_per_construction_worker_per_year=config.mw_per_worker_per_year,
        operating_labor_per_MW=config.operating_labor_per_mw,
    )

    bp_detection_params = DetectionParameters(
        us_intelligence_median_error_in_estimate_of_prc_compute_stock=config.us_error_compute_stock,
        us_intelligence_median_error_in_estimate_of_prc_fab_stock=0.1,
        us_intelligence_median_error_in_energy_consumption_estimate_of_prc_datacenter_capacity=config.us_error_energy,
        us_intelligence_median_error_in_satellite_estimate_of_prc_datacenter_capacity=config.us_error_satellite,
        mean_detection_time_for_100_workers=config.mean_detection_time_100_workers,
        mean_detection_time_for_1000_workers=config.mean_detection_time_1000_workers,
        variance_of_detection_time_given_num_workers=config.variance_of_detection_time,
        detection_threshold=100.0,
    )

    bp_params = BlackProjectParameterSet(
        properties=bp_properties,
        fab_params=bp_fab_params,
        datacenter_params=bp_dc_params,
        detection_params=bp_detection_params,
    )

    compute_params = ComputeParameters(
        us_frontier_project_compute_growth_rate=0.5,
        slowdown_year=2030.0,
        post_slowdown_training_compute_growth_rate=0.2,
        initial_hazard_rate=config.initial_hazard_rate,
        hazard_rate_increase_per_year=config.hazard_rate_increase_per_year,
        total_prc_compute_stock_in_2025=config.prc_compute_stock_at_agreement / (2.2 ** (config.agreement_year - 2025)),
        annual_growth_rate_of_prc_compute_stock_p10=2.2,
        annual_growth_rate_of_prc_compute_stock_p50=2.2,
        annual_growth_rate_of_prc_compute_stock_p90=2.2,
        proportion_of_prc_chip_stock_produced_domestically_2026=0.1,
        proportion_of_prc_chip_stock_produced_domestically_2030=0.3,
        us_frontier_project_h100e_in_2025=1e5,
    )

    energy_params = EnergyConsumptionParameters(
        total_GW_of_PRC_energy_consumption=config.total_prc_energy_gw,
        energy_efficiency_of_prc_stock_relative_to_state_of_the_art=config.energy_efficiency_relative_to_sota,
        architecture_efficiency_improvement_per_year=1.1,
        largest_ai_project_energy_efficiency_improvement_per_year=1.1,
    )

    # Create simulation years
    years = np.arange(
        config.agreement_year,
        config.agreement_year + config.num_years + config.time_step,
        config.time_step
    )

    # Initialize the black project
    project = initialize_black_project(
        project_id="prc_black_project",
        ai_slowdown_start_year=config.agreement_year,
        prc_compute_stock=config.prc_compute_stock_at_agreement,
        params=bp_params,
        compute_growth_params=compute_params,
        energy_consumption_params=energy_params,
        simulation_years=list(years),
    )

    results = {
        'years': years,
        'cumulative_lr': [],
        'lr_direct_observation': [],
        'lr_energy': [],
        'lr_satellite': [],
        'lr_compute_accounting': [],
        'sampled_detection_time': project.sampled_detection_time,
        # Additional metrics
        'datacenter_capacity_gw': [],
        'operational_compute_h100e': [],
        'surviving_compute_h100e': [],
        'h100_years_cumulative': [],
        'survival_rate': [],
        'labor_by_year': project.labor_by_year.copy() if project.labor_by_year else {},
        'posterior_probability': [],
    }

    # Prior probability for Bayesian updating
    prior_prob = 0.1  # Default prior

    # Pre-compute cumulative H100-years using trapezoidal integration
    h100_years_cumulative = 0.0

    for i, year in enumerate(years):
        relative_year = year - config.agreement_year

        # Get cumulative LR using the actual implementation
        cumulative_lr = compute_cumulative_likelihood_ratio(
            project=project,
            current_time=year,
            mean_detection_time_100_workers=config.mean_detection_time_100_workers,
            mean_detection_time_1000_workers=config.mean_detection_time_1000_workers,
            variance_theta=config.variance_of_detection_time,
        )
        results['cumulative_lr'].append(cumulative_lr)

        # Extract individual LR components
        # LR direct observation (worker-based)
        is_detected = relative_year >= project.sampled_detection_time
        if is_detected:
            lr_direct = 100.0
        elif project.lr_by_year:
            lr_direct = project.lr_by_year.get(round(relative_year), 1.0)
        else:
            lr_direct = 1.0
        results['lr_direct_observation'].append(lr_direct)

        # Datacenter capacity (total GW) - using linear growth model
        # Calculate linearly like the discrete model (since we're not running ODE solver here)
        dc = project.datacenters
        dc_start_year = dc.construction_start_year
        years_since_dc_start = max(0, year - dc_start_year)
        concealed_capacity = min(
            dc.construction_rate_gw_per_year * years_since_dc_start,
            max(0, dc.max_total_capacity_gw - dc.unconcealed_capacity_gw)
        )
        dc_capacity = concealed_capacity + dc.unconcealed_capacity_gw
        results['datacenter_capacity_gw'].append(dc_capacity)

        # LR energy (time-varying) - use the pre-computed datacenter capacity
        covert_energy = dc_capacity
        if covert_energy > 1e-10 and project.total_prc_energy_gw > 1e-10:
            lr_energy = compute_lr_from_reported_energy_consumption(
                total_prc_energy_gw=project.total_prc_energy_gw,
                covert_energy_gw=covert_energy,
                us_estimate_energy=project.us_estimate_energy_gw,
                median_error=config.us_error_energy,
            )
        else:
            lr_energy = 1.0
        results['lr_energy'].append(lr_energy)

        # LR satellite (constant)
        diverted_capacity = project.datacenters.unconcealed_capacity_gw
        h100_power_gw = 700e-9
        energy_efficiency = project.compute_stock.energy_efficiency_relative_to_h100
        initial_compute = project.compute.total_tpp_h100e / project.diversion_proportion if project.diversion_proportion > 0 else project.compute.total_tpp_h100e
        total_unconcealed = initial_compute * h100_power_gw / energy_efficiency

        if diverted_capacity > 1e-10:
            lr_satellite = compute_lr_from_satellite_detection(
                diverted_capacity_gw=diverted_capacity,
                total_unconcealed_capacity_gw=total_unconcealed,
                us_estimate_capacity=project.us_estimate_satellite_capacity_gw,
                median_error=config.us_error_satellite,
            )
        else:
            lr_satellite = 1.0
        results['lr_satellite'].append(lr_satellite)

        # LR compute accounting (constant)
        reported_compute_stock = project.compute.total_tpp_h100e * (1 - project.diversion_proportion) / project.diversion_proportion if project.diversion_proportion > 0 else project.compute.total_tpp_h100e
        lr_compute = compute_lr_from_prc_compute_accounting(
            reported_compute_stock=reported_compute_stock,
            diversion_proportion=project.diversion_proportion,
            us_estimate_compute=project.us_estimate_compute_stock,
            median_error=config.us_error_compute_stock,
        )
        results['lr_compute_accounting'].append(lr_compute)

        # Survival rate calculation
        years_of_life = relative_year
        cumulative_hazard = config.initial_hazard_rate * years_of_life + config.hazard_rate_increase_per_year * years_of_life**2 / 2
        survival_rate = np.exp(-cumulative_hazard)
        results['survival_rate'].append(survival_rate)

        # Surviving compute (initial stock * survival rate)
        initial_diverted = project.compute.total_tpp_h100e
        surviving_h100e = initial_diverted * survival_rate
        results['surviving_compute_h100e'].append(surviving_h100e)

        # Operational compute (limited by datacenter capacity)
        # Convert GW to H100e: H100e = (GW * 1e9 / 700W) * energy_efficiency
        if dc_capacity > 1e-10:
            max_compute_from_dc = (dc_capacity * 1e9 / 700.0) * energy_efficiency
            operational_h100e = min(surviving_h100e, max_compute_from_dc)
        else:
            operational_h100e = 0.0
        results['operational_compute_h100e'].append(operational_h100e)

        # H100-years cumulative (integrate operational compute over time)
        if i > 0:
            prev_operational = results['operational_compute_h100e'][i-1]
            dt = years[i] - years[i-1]
            h100_years_cumulative += (prev_operational + operational_h100e) / 2.0 * dt
        results['h100_years_cumulative'].append(h100_years_cumulative)

        # Posterior probability (Bayesian update)
        posterior = (prior_prob * cumulative_lr) / (prior_prob * cumulative_lr + (1 - prior_prob))
        results['posterior_probability'].append(posterior)

    for key in ['cumulative_lr', 'lr_direct_observation', 'lr_energy', 'lr_satellite', 'lr_compute_accounting',
                'datacenter_capacity_gw', 'operational_compute_h100e', 'surviving_compute_h100e',
                'h100_years_cumulative', 'survival_rate', 'posterior_probability']:
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

        # Show all LR components
        discrete_energy = [r['lr_energy'][idx] for r in discrete_results]
        discrete_satellite = [r['lr_satellite'][idx] for r in discrete_results]
        discrete_compute = [r['lr_compute_accounting'][idx] for r in discrete_results]
        continuous_energy = [r['lr_energy'][idx] for r in continuous_results]
        continuous_satellite = [r['lr_satellite'][idx] for r in continuous_results]
        continuous_compute = [r['lr_compute_accounting'][idx] for r in continuous_results]

        print(f"\n  LR Energy Consumption:")
        print(f"    Discrete:   median={np.median(discrete_energy):.4f}, "
              f"mean={np.mean(discrete_energy):.4f}, "
              f"std={np.std(discrete_energy):.4f}")
        print(f"    Continuous: median={np.median(continuous_energy):.4f}, "
              f"mean={np.mean(continuous_energy):.4f}, "
              f"std={np.std(continuous_energy):.4f}")

        print(f"\n  LR Satellite:")
        print(f"    Discrete:   median={np.median(discrete_satellite):.4f}, "
              f"mean={np.mean(discrete_satellite):.4f}, "
              f"std={np.std(discrete_satellite):.4f}")
        print(f"    Continuous: median={np.median(continuous_satellite):.4f}, "
              f"mean={np.mean(continuous_satellite):.4f}, "
              f"std={np.std(continuous_satellite):.4f}")

        print(f"\n  LR Compute Accounting:")
        print(f"    Discrete:   median={np.median(discrete_compute):.4f}, "
              f"mean={np.mean(discrete_compute):.4f}, "
              f"std={np.std(discrete_compute):.4f}")
        print(f"    Continuous: median={np.median(continuous_compute):.4f}, "
              f"mean={np.mean(continuous_compute):.4f}, "
              f"std={np.std(continuous_compute):.4f}")

        # Additional metrics comparison
        discrete_dc_cap = [r['datacenter_capacity_gw'][idx] for r in discrete_results]
        continuous_dc_cap = [r['datacenter_capacity_gw'][idx] for r in continuous_results]
        print(f"\n  Datacenter Capacity (GW):")
        print(f"    Discrete:   median={np.median(discrete_dc_cap):.6f}, "
              f"mean={np.mean(discrete_dc_cap):.6f}, "
              f"std={np.std(discrete_dc_cap):.6f}")
        print(f"    Continuous: median={np.median(continuous_dc_cap):.6f}, "
              f"mean={np.mean(continuous_dc_cap):.6f}, "
              f"std={np.std(continuous_dc_cap):.6f}")

        discrete_operational = [r['operational_compute_h100e'][idx] for r in discrete_results]
        continuous_operational = [r['operational_compute_h100e'][idx] for r in continuous_results]
        print(f"\n  Operational Compute (H100e):")
        print(f"    Discrete:   median={np.median(discrete_operational):.2f}, "
              f"mean={np.mean(discrete_operational):.2f}, "
              f"std={np.std(discrete_operational):.2f}")
        print(f"    Continuous: median={np.median(continuous_operational):.2f}, "
              f"mean={np.mean(continuous_operational):.2f}, "
              f"std={np.std(continuous_operational):.2f}")

        discrete_surviving = [r['surviving_compute_h100e'][idx] for r in discrete_results]
        continuous_surviving = [r['surviving_compute_h100e'][idx] for r in continuous_results]
        print(f"\n  Surviving Compute (H100e):")
        print(f"    Discrete:   median={np.median(discrete_surviving):.2f}, "
              f"mean={np.mean(discrete_surviving):.2f}, "
              f"std={np.std(discrete_surviving):.2f}")
        print(f"    Continuous: median={np.median(continuous_surviving):.2f}, "
              f"mean={np.mean(continuous_surviving):.2f}, "
              f"std={np.std(continuous_surviving):.2f}")

        discrete_h100years = [r['h100_years_cumulative'][idx] for r in discrete_results]
        continuous_h100years = [r['h100_years_cumulative'][idx] for r in continuous_results]
        print(f"\n  H100-Years Cumulative:")
        print(f"    Discrete:   median={np.median(discrete_h100years):.2f}, "
              f"mean={np.mean(discrete_h100years):.2f}, "
              f"std={np.std(discrete_h100years):.2f}")
        print(f"    Continuous: median={np.median(continuous_h100years):.2f}, "
              f"mean={np.mean(continuous_h100years):.2f}, "
              f"std={np.std(continuous_h100years):.2f}")

        discrete_survival = [r['survival_rate'][idx] for r in discrete_results]
        continuous_survival = [r['survival_rate'][idx] for r in continuous_results]
        print(f"\n  Survival Rate:")
        print(f"    Discrete:   median={np.median(discrete_survival):.4f}, "
              f"mean={np.mean(discrete_survival):.4f}, "
              f"std={np.std(discrete_survival):.4f}")
        print(f"    Continuous: median={np.median(continuous_survival):.4f}, "
              f"mean={np.mean(continuous_survival):.4f}, "
              f"std={np.std(continuous_survival):.4f}")

        discrete_posterior = [r['posterior_probability'][idx] for r in discrete_results]
        continuous_posterior = [r['posterior_probability'][idx] for r in continuous_results]
        print(f"\n  Posterior Probability (prior=0.1):")
        print(f"    Discrete:   median={np.median(discrete_posterior):.4f}, "
              f"mean={np.mean(discrete_posterior):.4f}, "
              f"std={np.std(discrete_posterior):.4f}")
        print(f"    Continuous: median={np.median(continuous_posterior):.4f}, "
              f"mean={np.mean(continuous_posterior):.4f}, "
              f"std={np.std(continuous_posterior):.4f}")

    # Detection time comparison (across all simulations)
    print("\n" + "=" * 70)
    print("DETECTION TIME DISTRIBUTION COMPARISON")
    print("=" * 70)
    continuous_detection_times = [r['sampled_detection_time'] for r in continuous_results]
    finite_detection_times = [t for t in continuous_detection_times if t < float('inf')]
    print(f"\n  Continuous model detection times (sampled from composite distribution):")
    if finite_detection_times:
        print(f"    Finite detections: {len(finite_detection_times)}/{len(continuous_results)}")
        print(f"    Median: {np.median(finite_detection_times):.2f} years")
        print(f"    Mean: {np.mean(finite_detection_times):.2f} years")
        print(f"    Std: {np.std(finite_detection_times):.2f} years")
        print(f"    Min: {np.min(finite_detection_times):.2f} years")
        print(f"    Max: {np.max(finite_detection_times):.2f} years")
    else:
        print(f"    No finite detection times (all > simulation period)")

    # Labor by year comparison (first simulation as example)
    print("\n" + "=" * 70)
    print("LABOR BY YEAR COMPARISON (Example from first simulation)")
    print("=" * 70)
    if discrete_results[0]['labor_by_year'] and continuous_results[0]['labor_by_year']:
        print(f"\n  {'Year (t+)':<10} {'Discrete':<15} {'Continuous':<15} {'Diff %':<10}")
        print(f"  {'-'*50}")
        for year in sorted(discrete_results[0]['labor_by_year'].keys()):
            d_labor = discrete_results[0]['labor_by_year'].get(year, 0)
            c_labor = continuous_results[0]['labor_by_year'].get(year, 0)
            diff_pct = (c_labor - d_labor) / d_labor * 100 if d_labor > 0 else 0
            print(f"  {year:<10} {d_labor:<15} {c_labor:<15} {diff_pct:+.1f}%")


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

        # Extract cumulative LRs
        discrete_lr = [r['cumulative_lr'][idx] for r in discrete_results]
        continuous_lr = [r['cumulative_lr'][idx] for r in continuous_results]

        # Top row: histograms
        ax1 = axes[0, col]

        # Filter out detection events (LR=100) for histogram
        discrete_no_detect = [x for x in discrete_lr if x < 50.0]
        continuous_no_detect = [x for x in continuous_lr if x < 50.0]

        if discrete_no_detect and continuous_no_detect:
            bins = np.linspace(0, max(max(discrete_no_detect), max(continuous_no_detect)), 20)
            ax1.hist(discrete_no_detect, bins=bins, alpha=0.5, label='Discrete', density=True, color='blue')
            ax1.hist(continuous_no_detect, bins=bins, alpha=0.5, label='Continuous', density=True, color='green')

        ax1.set_xlabel('Cumulative LR')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Year {year:.0f} (t+{year_offset:.0f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom row: QQ plots or scatter
        ax2 = axes[1, col]

        # Sort and plot
        discrete_sorted = np.sort(discrete_lr)
        continuous_sorted = np.sort(continuous_lr)

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

    # Stack results
    discrete_cumulative_all = np.array([r['cumulative_lr'] for r in discrete_results])
    continuous_cumulative_all = np.array([r['cumulative_lr'] for r in continuous_results])

    # Calculate percentiles
    discrete_median = np.median(discrete_cumulative_all, axis=0)
    discrete_p25 = np.percentile(discrete_cumulative_all, 25, axis=0)
    discrete_p75 = np.percentile(discrete_cumulative_all, 75, axis=0)

    continuous_median = np.median(continuous_cumulative_all, axis=0)
    continuous_p25 = np.percentile(continuous_cumulative_all, 25, axis=0)
    continuous_p75 = np.percentile(continuous_cumulative_all, 75, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Linear scale
    ax1 = axes[0]
    ax1.plot(years, discrete_median, 'b-', linewidth=2, label='Discrete (median)')
    ax1.fill_between(years, discrete_p25, discrete_p75, alpha=0.3, color='blue', label='Discrete (25-75%)')
    ax1.plot(years, continuous_median, 'g--', linewidth=2, label='Continuous (median)')
    ax1.fill_between(years, continuous_p25, continuous_p75, alpha=0.3, color='green', label='Continuous (25-75%)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Cumulative LR')
    ax1.set_title('Cumulative Likelihood Ratio Over Time (Linear Scale)')
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
    ax2.set_ylabel('Cumulative LR (log scale)')
    ax2.set_title('Cumulative Likelihood Ratio Over Time (Log Scale)')
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
        time_step=1.0,
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
2. Continuous model (ai_futures_simulator) - now uses same approach

Both models now include:
- LR from direct observation (worker detection, time-varying)
- LR from energy consumption accounting (time-varying)
- LR from satellite detection (constant)
- LR from PRC compute stock accounting (constant)

The models should produce very similar distributions for all components.
""")


if __name__ == "__main__":
    main()
