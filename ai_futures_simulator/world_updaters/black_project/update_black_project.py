"""
Black project existence and initialization.

This module handles:
- Creating black project entities when conditions are met
- Initializing all black project attributes

The actual updates to black project state are handled by:
- compute/black_compute.py: Fab production, compute stock, attrition
- datacenters_and_energy/update_datacenters_and_energy.py: Datacenter capacity, energy
- perceptions/black_project_perceptions.py: Detection LRs, posterior probability
"""

import math
import torch
from torch import Tensor
from typing import Dict, Tuple

from classes.world.world import World
from classes.world.entities import AIBlackProject, Nation, ComputeAllocation
from classes.world.assets import Compute, Fabs, Datacenters
from classes.world.software_progress import AISoftwareProgress
from classes.simulation_primitives import WorldUpdater
from parameters.classes import SimulationParameters
from parameters.classes import BlackProjectParameters
from parameters.classes import ComputeParameters
from parameters.classes import DataCenterAndEnergyParameters
from parameters.classes import BlackProjectPerceptionsParameters
from parameters.classes import PolicyParameters
from world_updaters.compute.black_compute import (
    calculate_fab_construction_duration,
    calculate_fab_wafer_starts_per_month,
    calculate_fab_h100e_per_chip,
    calculate_fab_watts_per_chip,
    calculate_operating_compute,
)
from world_updaters.perceptions.black_project_perceptions import (
    get_black_project_total_labor,
    compute_lr_over_time_vs_num_workers,
    sample_us_estimate_with_error,
    compute_lr_from_satellite_detection,
    compute_lr_from_prc_compute_accounting,
    compute_lr_from_sme_inventory,
)


def initialize_black_project(
    project_id: str,
    parent_nation: Nation,
    black_project_params: BlackProjectParameters,
    compute_params: ComputeParameters,
    energy_params: DataCenterAndEnergyParameters,
    perception_params: BlackProjectPerceptionsParameters,
    policy_params: PolicyParameters,
    initial_prc_compute_stock: float,
    simulation_years: list = None,
) -> Tuple[AIBlackProject, Dict, float]:
    """
    Initialize a black project with all components.

    Args:
        project_id: Identifier for the project
        parent_nation: Nation that owns the black project
        black_project_params: Black project parameters
        compute_params: Compute parameters (contains PRC compute, exogenous trends, survival rates)
        energy_params: Energy consumption parameters
        perception_params: Detection/perception parameters
        policy_params: Policy parameters (contains ai_slowdown_start_year)
        initial_prc_compute_stock: PRC's total compute stock at start
        simulation_years: List of years to simulate (for labor_by_year computation)

    Returns:
        Tuple of (AIBlackProject, lr_by_year dict, sampled_detection_time)
    """
    props = black_project_params.black_project_properties
    prc_compute = compute_params.PRCComputeParameters
    prc_energy = energy_params.prc_energy_consumption
    exogenous_trends = compute_params.exogenous_trends

    # Get preparation start year directly from parameter
    black_project_start_year = black_project_params.black_project_start_year
    # Agreement year (when AI slowdown starts) - stored for reference, but construction starts at black_project_start_year
    ai_slowdown_start_year = policy_params.ai_slowdown_start_year
    # Fab construction start year - construction begins when the black project starts
    fab_construction_start_year = black_project_start_year

    # Determine which process nodes are localized by fab construction start
    localization_years = {
        7: props.prc_localization_year_7nm,
        14: props.prc_localization_year_14nm,
        28: props.prc_localization_year_28nm,
    }

    # Find best available node that is localized by fab construction start
    best_available_node = None
    best_node_localization_year = 9999
    for node_nm in [7, 14, 28]:
        if localization_years[node_nm] <= fab_construction_start_year:
            best_available_node = node_nm
            best_node_localization_year = localization_years[node_nm]
            break

    # Fab is built if best available node meets minimum requirement
    min_node_nm = props.black_fab_min_process_node
    build_a_black_fab = (best_available_node is not None and best_available_node <= min_node_nm)
    actual_process_node_nm = best_available_node if build_a_black_fab else None

    # Calculate labor values from total_labor and fractions
    total_labor = props.total_labor
    datacenter_construction_labor = total_labor * props.fraction_of_labor_devoted_to_datacenter_construction
    ai_researcher_headcount = total_labor * props.fraction_of_labor_devoted_to_ai_research

    if build_a_black_fab:
        black_fab_construction_labor = total_labor * props.fraction_of_labor_devoted_to_black_fab_construction
        black_fab_operating_labor = total_labor * props.fraction_of_labor_devoted_to_black_fab_operation
    else:
        black_fab_construction_labor = 0.0
        black_fab_operating_labor = 0.0

    # --- Calculate derived values ---

    # Initial diverted compute
    diverted_compute = initial_prc_compute_stock * props.fraction_of_initial_compute_stock_to_divert_at_black_project_start

    # Energy efficiency calculation
    h100_power_w = prc_energy.h100_power_watts
    energy_efficiency_relative_to_sota = prc_energy.energy_efficiency_of_compute_stock_relative_to_state_of_the_art

    h100_release_year = 2022
    years_since_h100 = max(0, black_project_start_year - h100_release_year)
    sota_energy_efficiency_improvement_per_year = exogenous_trends.state_of_the_art_energy_efficiency_improvement_per_year
    sota_efficiency_relative_to_h100 = sota_energy_efficiency_improvement_per_year ** years_since_h100
    combined_energy_efficiency = energy_efficiency_relative_to_sota * sota_efficiency_relative_to_h100
    watts_per_h100e = h100_power_w / combined_energy_efficiency
    energy_per_h100e_gw = watts_per_h100e / 1e9
    initial_energy_requirement = diverted_compute * energy_per_h100e_gw
    energy_consumption_of_prc_stock_gw = initial_prc_compute_stock * energy_per_h100e_gw

    # Unconcealed capacity
    unconcealed_capacity_gw = (
        props.fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start *
        energy_consumption_of_prc_stock_gw
    )

    # Max concealed capacity
    max_capacity_gw = (
        props.max_fraction_of_total_national_energy_consumption *
        prc_energy.total_prc_energy_consumption_gw
    )

    # Construction rate for datacenters
    gw_per_worker_per_year = prc_energy.data_center_mw_per_year_per_construction_worker / 1000.0
    construction_rate = gw_per_worker_per_year * datacenter_construction_labor

    # Initial concealed capacity
    head_start_years = props.years_before_black_project_start_to_begin_datacenter_construction
    construction_start = black_project_start_year - head_start_years
    years_of_construction_at_init = max(0, black_project_start_year - construction_start)
    initial_concealed = min(
        construction_rate * years_of_construction_at_init,
        max(0, max_capacity_gw - unconcealed_capacity_gw)
    )
    initial_total_capacity = initial_concealed + unconcealed_capacity_gw

    # --- Calculate number of lithography scanners ---
    if build_a_black_fab:
        localization_year = best_node_localization_year
        years_since_localization = max(0, fab_construction_start_year - localization_year)
        n = years_since_localization
        first_year_prod = prc_compute.prc_lithography_scanners_produced_in_first_year
        additional_per_year = prc_compute.prc_additional_lithography_scanners_produced_per_year
        base_total_scanners = first_year_prod * (n + 1) + additional_per_year * n * (n + 1) / 2
        total_prc_scanners = base_total_scanners * prc_compute.prc_scanner_production_multiplier
        num_scanners = int(total_prc_scanners * props.fraction_of_lithography_scanners_to_divert_at_black_project_start)
        num_scanners = max(1, num_scanners)
    else:
        num_scanners = 0
        total_prc_scanners = 0
        localization_year = localization_years.get(28, 9999)

    # Fab construction duration
    target_wafer_starts = calculate_fab_wafer_starts_per_month(
        fab_operating_labor=black_fab_operating_labor,
        fab_number_of_lithography_scanners=float(num_scanners),
        wafers_per_month_per_worker=prc_compute.fab_wafers_per_month_per_operating_worker,
        wafers_per_month_per_scanner=prc_compute.wafers_per_month_per_lithography_scanner,
        labor_productivity_multiplier=prc_compute.fab_labor_productivity_multiplier,
        scanner_productivity_multiplier=prc_compute.fab_scanner_productivity_multiplier,
    )
    fab_construction_duration = calculate_fab_construction_duration(
        fab_construction_labor=black_fab_construction_labor,
        target_wafer_starts_per_month=target_wafer_starts,
        prc_compute_params=prc_compute,
    )

    # Fab h100e per chip
    process_node_nm = actual_process_node_nm if build_a_black_fab else 28.0
    # Chip specs fixed at fab construction time (black_project_start_year)
    fab_h100e_per_chip = calculate_fab_h100e_per_chip(
        fab_process_node_nm=process_node_nm,
        year=black_project_start_year,
        exogenous_trends=exogenous_trends,
    )

    # --- Pre-compute labor_by_year for detection ---
    if simulation_years is None:
        simulation_years = [black_project_start_year + i for i in range(21)]

    combined_labor_by_relative_year = {}
    fab_labor_by_year_since_construction = {}

    # Generate fine-grained years
    fine_grained_years = []
    for year in simulation_years:
        for i in range(10):
            fine_grained_year = year + i * 0.1
            if fine_grained_year >= black_project_start_year:
                fine_grained_years.append(round(fine_grained_year, 1))
    fine_grained_years = sorted(set(fine_grained_years))

    for year in fine_grained_years:
        relative_year = round(year - black_project_start_year, 1)

        datacenter_construction_start = black_project_start_year - head_start_years
        years_since_construction_start = year - datacenter_construction_start
        concealed_at_year = min(
            construction_rate * years_since_construction_start,
            max(0, max_capacity_gw - unconcealed_capacity_gw)
        )
        total_capacity_at_year = concealed_at_year + unconcealed_capacity_gw
        operating_labor = total_capacity_at_year * 1000.0 / prc_energy.data_center_mw_per_operating_worker

        combined_labor = datacenter_construction_labor + operating_labor
        if build_a_black_fab:
            combined_labor += black_fab_construction_labor + black_fab_operating_labor
        combined_labor += ai_researcher_headcount
        combined_labor_by_relative_year[relative_year] = int(combined_labor)

        if build_a_black_fab:
            years_since_fab_construction = round(year - fab_construction_start_year, 1)
            if years_since_fab_construction >= 0:
                fab_labor = int(black_fab_construction_labor + black_fab_operating_labor)
                fab_labor_by_year_since_construction[years_since_fab_construction] = fab_labor

    # Sample detection times
    lr_datacenters_by_year, sampled_detection_time_dc = compute_lr_over_time_vs_num_workers(
        labor_by_year=combined_labor_by_relative_year,
        mean_detection_time_100_workers=perception_params.mean_detection_time_for_100_workers,
        mean_detection_time_1000_workers=perception_params.mean_detection_time_for_1000_workers,
        variance=perception_params.variance_of_detection_time_given_num_workers,
    )

    if build_a_black_fab and fab_labor_by_year_since_construction:
        lr_fab_other_by_year, sampled_detection_time_fab = compute_lr_over_time_vs_num_workers(
            labor_by_year=fab_labor_by_year_since_construction,
            mean_detection_time_100_workers=perception_params.mean_detection_time_for_100_workers,
            mean_detection_time_1000_workers=perception_params.mean_detection_time_for_1000_workers,
            variance=perception_params.variance_of_detection_time_given_num_workers,
        )
    else:
        lr_fab_other_by_year = {}
        sampled_detection_time_fab = float('inf')

    sampled_detection_time = sampled_detection_time_dc
    lr_by_year = lr_datacenters_by_year

    # --- Create nested objects ---
    ai_software_progress = AISoftwareProgress(
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

    compute_allocation = ComputeAllocation(
        fraction_for_ai_r_and_d_inference=0.5,
        fraction_for_ai_r_and_d_training=0.3,
        fraction_for_external_deployment=0.0,
        fraction_for_alignment_research=0.1,
        fraction_for_frontier_training=0.1,
    )

    initial_compute = Compute(
        tpp_h100e_including_attrition=diverted_compute,
        functional_tpp_h100e=diverted_compute,
        watts_per_h100e=watts_per_h100e,
        average_functional_chip_age_years=0.0,
    )

    fab_production_compute = Compute(
        tpp_h100e_including_attrition=0.0,
        functional_tpp_h100e=0.0,
        watts_per_h100e=h100_power_w,
        average_functional_chip_age_years=0.0,
    )

    fab = Fabs(monthly_compute_production=fab_production_compute)

    training_compute_growth_rate = 0.0
    if hasattr(prc_compute, 'annual_growth_rate_of_prc_compute_stock'):
        annual_growth_rate = prc_compute.annual_growth_rate_of_prc_compute_stock
        if annual_growth_rate > 0:
            training_compute_growth_rate = math.log10(annual_growth_rate)

    # --- Create the AIBlackProject ---
    project = AIBlackProject(
        id=project_id,
        operating_compute=[initial_compute],
        compute_allocation=compute_allocation,
        human_ai_capability_researchers=float(ai_researcher_headcount),
        ai_software_progress=ai_software_progress,
        training_compute_growth_rate=training_compute_growth_rate,
        parent_nation=parent_nation,
        black_project_start_year=black_project_start_year,
        ai_slowdown_start_year=ai_slowdown_start_year,
        fab_process_node_nm=process_node_nm,
        fab_number_of_lithography_scanners=float(num_scanners),
        fab_construction_labor=float(black_fab_construction_labor),
        fab_operating_labor=float(black_fab_operating_labor),
        fab_chips_per_wafer=float(prc_compute.h100_sized_chips_per_wafer),
        unconcealed_datacenter_capacity_diverted_gw=unconcealed_capacity_gw,
        concealed_datacenter_capacity_construction_labor=float(datacenter_construction_labor),
        concealed_max_total_capacity_gw=max_capacity_gw,
        initial_diverted_compute_h100e=diverted_compute,
        fab_compute_stock_h100e=0.0,
        fab_compute_average_age_years=0.0,
        fab_total_produced_h100e=0.0,
        fab=fab,
    )

    # --- Set init=False metrics ---
    project._set_frozen_field('ai_r_and_d_inference_compute_tpp_h100e', 0.0)
    project._set_frozen_field('ai_r_and_d_training_compute_tpp_h100e', 0.0)
    project._set_frozen_field('external_deployment_compute_tpp_h100e', 0.0)
    project._set_frozen_field('alignment_research_compute_tpp_h100e', 0.0)
    project._set_frozen_field('frontier_training_compute_tpp_h100e', 0.0)

    project._set_frozen_field('fab_construction_duration', fab_construction_duration)
    project._set_frozen_field('fab_is_operational', False)
    project._set_frozen_field('fab_wafer_starts_per_month', target_wafer_starts)
    project._set_frozen_field('fab_h100e_per_chip', fab_h100e_per_chip)
    project._set_frozen_field('fab_watts_per_chip', calculate_fab_watts_per_chip(
        fab_process_node_nm=process_node_nm,
        year=black_project_start_year,
        exogenous_trends=exogenous_trends,
    ))

    datacenters = Datacenters(data_center_capacity_gw=initial_total_capacity)
    project._set_frozen_field('datacenters', datacenters)
    project._set_frozen_field('datacenters_operating_labor_per_gw', prc_energy.data_center_mw_per_operating_worker * 1000.0)

    compute_stock = Compute(
        tpp_h100e_including_attrition=diverted_compute,
        functional_tpp_h100e=diverted_compute,
        watts_per_h100e=watts_per_h100e,
        average_functional_chip_age_years=0.0,
    )
    project._set_frozen_field('compute_stock', compute_stock)

    initial_operating = calculate_operating_compute(
        functional_compute_h100e=diverted_compute,
        datacenter_capacity_gw=initial_total_capacity,
        watts_per_h100e=watts_per_h100e,
    )
    project._set_frozen_field('operating_compute_tpp_h100e', initial_operating)

    # LR components
    total_prc_energy_gw = prc_energy.total_prc_energy_consumption_gw

    us_estimate_compute = sample_us_estimate_with_error(
        initial_prc_compute_stock,
        perception_params.intelligence_median_error_in_estimate_of_compute_stock
    )
    reported_satellite_capacity = energy_consumption_of_prc_stock_gw - unconcealed_capacity_gw
    us_estimate_satellite_capacity = sample_us_estimate_with_error(
        reported_satellite_capacity,
        perception_params.intelligence_median_error_in_satellite_estimate_of_datacenter_capacity
    )
    us_estimate_energy = sample_us_estimate_with_error(
        total_prc_energy_gw,
        perception_params.intelligence_median_error_in_energy_consumption_estimate_of_datacenter_capacity
    )

    reported_compute = initial_prc_compute_stock * (1 - props.fraction_of_initial_compute_stock_to_divert_at_black_project_start)
    lr_prc_accounting = compute_lr_from_prc_compute_accounting(
        reported_compute_stock=reported_compute,
        diversion_proportion=props.fraction_of_initial_compute_stock_to_divert_at_black_project_start,
        us_estimate_compute=us_estimate_compute,
        median_error=perception_params.intelligence_median_error_in_estimate_of_compute_stock,
    )

    lr_satellite = compute_lr_from_satellite_detection(
        diverted_capacity_gw=unconcealed_capacity_gw,
        total_unconcealed_capacity_gw=energy_consumption_of_prc_stock_gw,
        us_estimate_capacity=us_estimate_satellite_capacity,
        median_error=perception_params.intelligence_median_error_in_satellite_estimate_of_datacenter_capacity,
    )

    scanner_diversion_fraction = props.fraction_of_lithography_scanners_to_divert_at_black_project_start
    if build_a_black_fab and total_prc_scanners > 0 and scanner_diversion_fraction > 0:
        us_estimate_scanners = sample_us_estimate_with_error(
            total_prc_scanners,
            perception_params.intelligence_median_error_in_estimate_of_fab_stock
        )
        reported_scanners = total_prc_scanners * (1 - scanner_diversion_fraction)
        lr_sme_inventory = compute_lr_from_sme_inventory(
            reported_scanner_count=reported_scanners,
            diversion_proportion=scanner_diversion_fraction,
            us_estimate_scanners=us_estimate_scanners,
            median_error=perception_params.intelligence_median_error_in_estimate_of_fab_stock,
        )
    else:
        lr_sme_inventory = 1.0

    project._set_frozen_field('lr_prc_accounting', lr_prc_accounting)
    project._set_frozen_field('lr_sme_inventory', lr_sme_inventory)
    project._set_frozen_field('lr_satellite_datacenter', lr_satellite)

    lr_fab_procurement = 1.0 if localization_year <= fab_construction_start_year else 10.0
    project._set_frozen_field('lr_fab_procurement', lr_fab_procurement)

    project._set_frozen_field('lr_reported_energy', 1.0)
    project._set_frozen_field('lr_other_intel', 1.0)
    lr_sme_for_cumulative = lr_sme_inventory * lr_fab_procurement if num_scanners > 0 else 1.0
    initial_cumulative_lr = lr_prc_accounting * lr_sme_for_cumulative * lr_satellite
    project._set_frozen_field('cumulative_lr', initial_cumulative_lr)

    project._set_frozen_field('lr_fab_other', 1.0)
    initial_lr_fab_combined = lr_sme_inventory * lr_fab_procurement
    project._set_frozen_field('lr_fab_combined', initial_lr_fab_combined)

    project._set_frozen_field('us_estimate_energy', us_estimate_energy)
    project._set_frozen_field('total_prc_energy_gw', total_prc_energy_gw)
    project._set_frozen_field('posterior_prob', perception_params.prior_odds_of_covert_project / (1 + perception_params.prior_odds_of_covert_project))

    project._set_frozen_field('fab_cumulative_production_h100e', 0.0)
    project._set_frozen_field('fab_monthly_production_h100e', 0.0)
    project._set_frozen_field('fab_architecture_efficiency',
        exogenous_trends.state_of_the_art_architecture_efficiency_improvement_per_year ** (black_project_start_year - 2022.0))
    project._set_frozen_field('fab_transistor_density_relative_to_h100',
        (4.0 / process_node_nm) ** exogenous_trends.transistor_density_scaling_exponent)

    project._set_frozen_field('survival_rate', 1.0)
    project._set_frozen_field('initial_compute_surviving_h100e', diverted_compute)

    initial_energy_gw = (diverted_compute * watts_per_h100e) / 1e9
    project._set_frozen_field('initial_stock_energy_gw', initial_energy_gw)
    project._set_frozen_field('fab_compute_energy_gw', 0.0)
    project._set_frozen_field('total_compute_energy_gw', initial_energy_gw)

    project._set_frozen_field('sampled_detection_time', sampled_detection_time)
    project._set_frozen_field('is_detected', False)

    project._set_frozen_field('lr_datacenters_by_year', lr_datacenters_by_year)
    project._set_frozen_field('lr_fab_other_by_year', lr_fab_other_by_year)

    return project, lr_by_year, sampled_detection_time
