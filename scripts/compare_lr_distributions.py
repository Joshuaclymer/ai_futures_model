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

# Import from continuous model (new parameter structure)
from world_updaters.black_project import initialize_black_project
from world_updaters.compute.black_compute import (
    compute_lr_over_time_vs_num_workers,
    calculate_concealed_capacity_gw,
    calculate_datacenter_capacity_gw,
    calculate_operating_compute,
)
from world_updaters.compute.chip_survival import calculate_survival_rate
from parameters.black_project_parameters import (
    BlackProjectParameters,
    BlackProjectProperties,
)
from parameters.compute_parameters import (
    ComputeParameters,
    ExogenousComputeTrends,
    SurvivalRateParameters,
    USComputeParameters,
    PRCComputeParameters,
)
from parameters.data_center_and_energy_parameters import (
    DataCenterAndEnergyParameters,
    PRCDataCenterAndEnergyParameters,
)
from parameters.perceptions_parameters import BlackProjectPerceptionsParameters
from parameters.policy_parameters import PolicyParameters
from classes.world.entities import Nation, NamedNations, AISoftwareDeveloper, ComputeAllocation
from classes.world.assets import Compute, Fabs, Datacenters
from classes.world.software_progress import AISoftwareProgress
import torch


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

    # Calculate labor for each year
    for year in years:
        relative_year = year - config.agreement_year
        construction_labor = dc.construction_labor
        operating_labor = dc.get_operating_labor(year)
        fab_labor = 0
        researcher_headcount = config.researcher_headcount
        total_labor = construction_labor + operating_labor + fab_labor + researcher_headcount
        results['labor_by_year'][int(relative_year)] = int(total_labor)

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
        posterior = (prior_prob * cumulative) / (prior_prob * cumulative + (1 - prior_prob))
        results['posterior_probability'].append(posterior)

    for key in ['cumulative_lr', 'lr_direct_observation', 'lr_energy', 'lr_satellite', 'lr_compute_accounting',
                'datacenter_capacity_gw', 'operational_compute_h100e', 'surviving_compute_h100e',
                'h100_years_cumulative', 'survival_rate', 'posterior_probability']:
        results[key] = np.array(results[key])

    return results


def create_dummy_prc_nation(compute_stock_h100e: float, energy_efficiency: float, total_energy_gw: float) -> Nation:
    """Create a minimal PRC Nation for black project initialization."""
    H100_POWER_W = 700.0

    ai_sw_progress = AISoftwareProgress(
        progress=torch.tensor(0.0),
        research_stock=torch.tensor(0.0),
        ai_coding_labor_multiplier=torch.tensor(1.0),
        ai_sw_progress_mult_ref_present_day=torch.tensor(1.0),
        progress_rate=torch.tensor(0.0),
        software_progress_rate=torch.tensor(0.0),
        research_effort=torch.tensor(0.0),
        automation_fraction=torch.tensor(0.0),
        coding_labor=torch.tensor(0.0),
        serial_coding_labor=torch.tensor(0.0),
        ai_research_taste=torch.tensor(0.0),
        ai_research_taste_sd=torch.tensor(0.0),
        aggregate_research_taste=torch.tensor(0.0),
        initial_progress=torch.tensor(0.0),
        software_efficiency=torch.tensor(0.0),
    )

    prc_compute = Compute(
        all_tpp_h100e=compute_stock_h100e,
        functional_tpp_h100e=compute_stock_h100e,
        watts_per_h100e=H100_POWER_W / energy_efficiency,
        average_functional_chip_age_years=0.0,
    )

    compute_allocation = ComputeAllocation(
        fraction_for_ai_r_and_d_inference=0.3,
        fraction_for_ai_r_and_d_training=0.3,
        fraction_for_external_deployment=0.2,
        fraction_for_alignment_research=0.1,
        fraction_for_frontier_training=0.1,
    )

    prc_developer = AISoftwareDeveloper(
        id="prc_developer",
        operating_compute=[prc_compute],
        compute_allocation=compute_allocation,
        human_ai_capability_researchers=1000.0,
        ai_software_progress=ai_sw_progress,
    )
    prc_developer._set_frozen_field('ai_r_and_d_inference_compute_tpp_h100e', 0.0)
    prc_developer._set_frozen_field('ai_r_and_d_training_compute_tpp_h100e', 0.0)
    prc_developer._set_frozen_field('external_deployment_compute_tpp_h100e', 0.0)
    prc_developer._set_frozen_field('alignment_research_compute_tpp_h100e', 0.0)
    prc_developer._set_frozen_field('frontier_training_compute_tpp_h100e', 0.0)

    fab_production_compute = Compute(
        all_tpp_h100e=0.0,
        functional_tpp_h100e=0.0,
        watts_per_h100e=H100_POWER_W,
        average_functional_chip_age_years=0.0,
    )
    prc_fabs = Fabs(monthly_production_compute=fab_production_compute)

    prc_compute_stock = Compute(
        all_tpp_h100e=compute_stock_h100e,
        functional_tpp_h100e=compute_stock_h100e,
        watts_per_h100e=H100_POWER_W / energy_efficiency,
        average_functional_chip_age_years=2.0,
    )

    total_dc_capacity = total_energy_gw * 0.05
    prc_datacenters = Datacenters(data_center_capacity_gw=total_dc_capacity)

    return Nation(
        id=NamedNations.PRC,
        ai_software_developers=[prc_developer],
        fabs=prc_fabs,
        compute_stock=prc_compute_stock,
        datacenters=prc_datacenters,
        total_energy_consumption_gw=total_energy_gw,
        leading_ai_software_developer=prc_developer,
        operating_compute_tpp_h100e=compute_stock_h100e * 0.8,
    )


def run_continuous_model_single(config: ComparisonConfig, seed: int) -> Dict:
    """Run a single continuous model simulation using the new parameter structure."""

    np.random.seed(seed)
    H100_POWER_W = 700.0

    # Calculate total labor and fractions
    total_labor = config.datacenter_construction_labor + config.researcher_headcount
    frac_dc_construction = config.datacenter_construction_labor / total_labor
    frac_ai_research = config.researcher_headcount / total_labor

    # Create parameter objects with new structure
    bp_properties = BlackProjectProperties(
        total_labor=total_labor,
        fraction_of_labor_devoted_to_datacenter_construction=frac_dc_construction,
        fraction_of_labor_devoted_to_black_fab_construction=0.0,
        fraction_of_labor_devoted_to_black_fab_operation=0.0,
        fraction_of_labor_devoted_to_ai_research=frac_ai_research,
        fraction_of_initial_compute_stock_to_divert_at_black_project_start=config.proportion_to_divert,
        fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start=0.0,
        fraction_of_lithography_scanners_to_divert_at_black_project_start=0.0,
        max_fraction_of_total_national_energy_consumption=config.max_proportion_prc_energy,
        build_a_black_fab=config.build_fab,
        black_fab_max_process_node=28.0,
    )

    # Black project start year is agreement_year - years_before_agreement_building_dc
    black_project_start_year = config.agreement_year - config.years_before_agreement_building_dc

    bp_params = BlackProjectParameters(
        run_a_black_project=True,
        black_project_start_year=black_project_start_year,
        black_project_properties=bp_properties,
    )

    exogenous_compute_trends = ExogenousComputeTrends(
        transistor_density_scaling_exponent=1.49,
        state_of_the_art_architecture_efficiency_improvement_per_year=1.23,
        transistor_density_at_end_of_dennard_scaling_m_per_mm2=10.0,
        watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended=-1.0,
        watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended=-0.33,
        state_of_the_art_energy_efficiency_improvement_per_year=1.26,
    )

    survival_rate_params = SurvivalRateParameters(
        initial_annual_hazard_rate=config.initial_hazard_rate,
        annual_hazard_rate_increase_per_year=config.hazard_rate_increase_per_year,
    )

    us_compute_params = USComputeParameters(
        us_frontier_project_compute_tpp_h100e_in_2025=120325.0,
        us_frontier_project_compute_annual_growth_rate=4.0,
    )

    prc_compute_params = PRCComputeParameters(
        total_prc_compute_tpp_h100e_in_2025=config.prc_compute_stock_at_agreement / (2.2 ** (config.agreement_year - 2025)),
        annual_growth_rate_of_prc_compute_stock=2.2,
        prc_architecture_efficiency_relative_to_state_of_the_art=1.0,
        proportion_of_prc_chip_stock_produced_domestically_2026=0.1,
        proportion_of_prc_chip_stock_produced_domestically_2030=0.4,
        prc_lithography_scanners_produced_in_first_year=20.0,
        prc_additional_lithography_scanners_produced_per_year=16.0,
        p_localization_28nm_2030=0.25,
        p_localization_14nm_2030=0.10,
        p_localization_7nm_2030=0.06,
        h100_sized_chips_per_wafer=28.0,
        wafers_per_month_per_lithography_scanner=1000.0,
        construction_time_for_5k_wafers_per_month=1.4,
        construction_time_for_100k_wafers_per_month=2.41,
        fab_wafers_per_month_per_operating_worker=24.64,
        fab_wafers_per_month_per_construction_worker_under_standard_timeline=14.1,
    )

    compute_params = ComputeParameters(
        exogenous_trends=exogenous_compute_trends,
        survival_rate_parameters=survival_rate_params,
        USComputeParameters=us_compute_params,
        PRCComputeParameters=prc_compute_params,
    )

    prc_energy_params = PRCDataCenterAndEnergyParameters(
        energy_efficiency_of_compute_stock_relative_to_state_of_the_art=config.energy_efficiency_relative_to_sota,
        total_prc_energy_consumption_gw=config.total_prc_energy_gw,
        data_center_mw_per_year_per_construction_worker=config.mw_per_worker_per_year,
        data_center_mw_per_operating_worker=1.0 / config.operating_labor_per_mw,
    )

    energy_params = DataCenterAndEnergyParameters(
        prc_energy_consumption=prc_energy_params,
    )

    perception_params = BlackProjectPerceptionsParameters(
        intelligence_median_error_in_estimate_of_compute_stock=config.us_error_compute_stock,
        intelligence_median_error_in_estimate_of_fab_stock=0.07,
        intelligence_median_error_in_energy_consumption_estimate_of_datacenter_capacity=config.us_error_energy,
        intelligence_median_error_in_satellite_estimate_of_datacenter_capacity=config.us_error_satellite,
        mean_detection_time_for_100_workers=config.mean_detection_time_100_workers,
        mean_detection_time_for_1000_workers=config.mean_detection_time_1000_workers,
        variance_of_detection_time_given_num_workers=config.variance_of_detection_time,
        detection_threshold=100.0,
        detection_thresholds=[1, 2, 4],
    )

    policy_params = PolicyParameters(
        ai_slowdown_start_year=config.agreement_year,
    )

    # Create simulation years
    years = np.arange(
        config.agreement_year,
        config.agreement_year + config.num_years + config.time_step,
        config.time_step
    )

    # Create dummy PRC nation
    prc_nation = create_dummy_prc_nation(
        config.prc_compute_stock_at_agreement,
        config.energy_efficiency_relative_to_sota,
        config.total_prc_energy_gw,
    )

    # Initialize the black project with new signature
    project, lr_by_year, sampled_detection_time = initialize_black_project(
        project_id="prc_black_project",
        parent_nation=prc_nation,
        black_project_params=bp_params,
        compute_params=compute_params,
        energy_params=energy_params,
        perception_params=perception_params,
        policy_params=policy_params,
        initial_prc_compute_stock=config.prc_compute_stock_at_agreement,
        simulation_years=list(years),
    )

    results = {
        'years': years,
        'cumulative_lr': [],
        'lr_direct_observation': [],
        'lr_energy': [],
        'lr_satellite': [],
        'lr_compute_accounting': [],
        'sampled_detection_time': sampled_detection_time,
        # Additional metrics
        'datacenter_capacity_gw': [],
        'operational_compute_h100e': [],
        'surviving_compute_h100e': [],
        'h100_years_cumulative': [],
        'survival_rate': [],
        'labor_by_year': {},
        'posterior_probability': [],
    }

    # Calculate labor for each year
    for year in years:
        relative_year = int(year - config.agreement_year)
        total_labor = config.datacenter_construction_labor + config.researcher_headcount
        results['labor_by_year'][relative_year] = int(total_labor)

    # Prior probability for Bayesian updating
    prior_prob = 0.1

    # Get initial values
    initial_stock = project.compute_stock.all_tpp_h100e if project.compute_stock else 0.0
    energy_efficiency = config.energy_efficiency_relative_to_sota
    watts_per_h100e = H100_POWER_W / energy_efficiency

    # Get datacenter construction parameters
    gw_per_worker_per_year = prc_energy_params.data_center_mw_per_year_per_construction_worker / 1000.0
    construction_rate = gw_per_worker_per_year * project.concealed_datacenter_capacity_construction_labor

    h100_years_cumulative = 0.0

    for i, year in enumerate(years):
        relative_year = year - config.agreement_year

        # Get LR from pre-computed values
        lr_direct = lr_by_year.get(int(relative_year), 1.0) if lr_by_year else 1.0

        # Check if detected
        if sampled_detection_time <= relative_year:
            lr_direct = 100.0

        results['lr_direct_observation'].append(lr_direct)

        # Datacenter capacity (linear growth)
        concealed = calculate_concealed_capacity_gw(
            current_year=year,
            construction_start_year=project.preparation_start_year,
            construction_rate_gw_per_year=construction_rate,
            max_concealed_capacity_gw=project.concealed_max_total_capacity_gw,
        )
        dc_capacity = calculate_datacenter_capacity_gw(
            unconcealed_capacity_gw=project.unconcealed_datacenter_capacity_diverted_gw,
            concealed_capacity_gw=concealed,
        )
        results['datacenter_capacity_gw'].append(dc_capacity)

        # LR energy (simplified - use 1.0 for now as we don't have all the infrastructure)
        results['lr_energy'].append(1.0)

        # LR satellite (simplified)
        results['lr_satellite'].append(1.0)

        # LR compute accounting (simplified)
        results['lr_compute_accounting'].append(1.0)

        # Cumulative LR (just use direct observation for now)
        cumulative_lr = lr_direct
        results['cumulative_lr'].append(cumulative_lr)

        # Survival rate calculation
        survival_rate = calculate_survival_rate(
            years_since_acquisition=relative_year,
            initial_hazard_rate=config.initial_hazard_rate,
            hazard_rate_increase_per_year=config.hazard_rate_increase_per_year,
        )
        results['survival_rate'].append(survival_rate)

        # Surviving compute
        surviving_h100e = initial_stock * survival_rate
        results['surviving_compute_h100e'].append(surviving_h100e)

        # Operational compute (limited by datacenter capacity)
        operational_h100e = calculate_operating_compute(
            functional_compute_h100e=surviving_h100e,
            datacenter_capacity_gw=dc_capacity,
            watts_per_h100e=watts_per_h100e,
        )
        results['operational_compute_h100e'].append(operational_h100e)

        # H100-years cumulative
        if i > 0:
            prev_operational = results['operational_compute_h100e'][i-1]
            dt = years[i] - years[i-1]
            h100_years_cumulative += (prev_operational + operational_h100e) / 2.0 * dt
        results['h100_years_cumulative'].append(h100_years_cumulative)

        # Posterior probability
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

        seed = 1000 + i

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

        # Additional metrics comparison
        discrete_dc_cap = [r['datacenter_capacity_gw'][idx] for r in discrete_results]
        continuous_dc_cap = [r['datacenter_capacity_gw'][idx] for r in continuous_results]
        print(f"\n  Datacenter Capacity (GW):")
        print(f"    Discrete:   median={np.median(discrete_dc_cap):.6f}, "
              f"mean={np.mean(discrete_dc_cap):.6f}")
        print(f"    Continuous: median={np.median(continuous_dc_cap):.6f}, "
              f"mean={np.mean(continuous_dc_cap):.6f}")

        discrete_operational = [r['operational_compute_h100e'][idx] for r in discrete_results]
        continuous_operational = [r['operational_compute_h100e'][idx] for r in continuous_results]
        print(f"\n  Operational Compute (H100e):")
        print(f"    Discrete:   median={np.median(discrete_operational):.2f}, "
              f"mean={np.mean(discrete_operational):.2f}")
        print(f"    Continuous: median={np.median(continuous_operational):.2f}, "
              f"mean={np.mean(continuous_operational):.2f}")

        discrete_survival = [r['survival_rate'][idx] for r in discrete_results]
        continuous_survival = [r['survival_rate'][idx] for r in continuous_results]
        print(f"\n  Survival Rate:")
        print(f"    Discrete:   median={np.median(discrete_survival):.4f}, "
              f"mean={np.mean(discrete_survival):.4f}")
        print(f"    Continuous: median={np.median(continuous_survival):.4f}, "
              f"mean={np.mean(continuous_survival):.4f}")

    # Detection time comparison
    print("\n" + "=" * 70)
    print("DETECTION TIME DISTRIBUTION COMPARISON")
    print("=" * 70)
    continuous_detection_times = [r['sampled_detection_time'] for r in continuous_results]
    finite_detection_times = [t for t in continuous_detection_times if t < float('inf')]
    print(f"\n  Continuous model detection times:")
    if finite_detection_times:
        print(f"    Finite detections: {len(finite_detection_times)}/{len(continuous_results)}")
        print(f"    Median: {np.median(finite_detection_times):.2f} years")
        print(f"    Mean: {np.mean(finite_detection_times):.2f} years")
    else:
        print(f"    No finite detection times")


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
    plot_time_series(
        discrete_results, continuous_results, config,
        f'{output_dir}/lr_time_series_comparison.png'
    )

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
