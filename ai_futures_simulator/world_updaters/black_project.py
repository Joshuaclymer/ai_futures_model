"""
Black project world updater.

Updates covert compute infrastructure including:
- Fab production (when operational)
- Datacenter capacity growth
- Compute stock with attrition

Detection is handled externally by the simulation, not in the entity.
"""

import math
import torch
from torch import Tensor
from typing import Optional, Dict, Tuple

from classes.world.world import World
from classes.world.entities import AIBlackProject, AISoftwareDeveloper, Nation, ComputeAllocation
from classes.world.assets import Compute, Fabs, Datacenters
from classes.world.software_progress import AISoftwareProgress
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.simulation_parameters import SimulationParameters
from parameters.black_project_parameters import BlackProjectParameters, BlackProjectProperties
from parameters.compute_parameters import ComputeParameters, PRCComputeParameters, ExogenousComputeTrends, SurvivalRateParameters
from parameters.data_center_and_energy_parameters import DataCenterAndEnergyParameters, PRCDataCenterAndEnergyParameters
from parameters.perceptions_parameters import BlackProjectPerceptionsParameters
from parameters.policy_parameters import PolicyParameters
from world_updaters.compute.black_compute import (
    calculate_fab_construction_duration,
    calculate_fab_wafer_starts_per_month,
    calculate_fab_h100e_per_chip,
    calculate_fab_watts_per_chip,
    calculate_fab_annual_production_h100e,
    calculate_concealed_capacity_gw,
    calculate_datacenter_capacity_gw,
    calculate_datacenter_operating_labor,
    calculate_survival_rate,
    calculate_functional_compute,
    calculate_operating_compute,
    get_black_project_total_labor,
    compute_detection_constants,
    compute_mean_detection_time,
    sample_detection_time,
    compute_lr_over_time_vs_num_workers,
    sample_us_estimate_with_error,
    lr_from_discrepancy_in_us_estimate,
    compute_lr_from_reported_energy_consumption,
    compute_lr_from_satellite_detection,
    compute_lr_from_prc_compute_accounting,
    compute_detection_probability,
)


class BlackProjectUpdater(WorldUpdater):
    """
    Updates black project dynamics.

    Updates metrics based on current state:
    1. Fab metrics (wafer starts, h100e per chip, production)
    2. Datacenter capacity (concealed + unconcealed)
    3. Compute stock (with attrition)
    4. Operating compute (limited by datacenter capacity)
    """

    def __init__(
        self,
        params: SimulationParameters,
        black_project_params: BlackProjectParameters,
        compute_params: ComputeParameters,
        energy_params: DataCenterAndEnergyParameters,
    ):
        super().__init__()
        self.params = params
        self.black_project_params = black_project_params
        self.compute_params = compute_params
        self.energy_params = energy_params

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Compute continuous contributions to d(state)/dt for black projects.

        Note: The new structure uses direct state properties rather than nested
        objects with log-space tracking. This method returns zero derivatives
        since the new model uses discrete time steps.
        """
        d_world = World.zeros(world)
        # New structure doesn't use continuous ODE integration for black projects
        return StateDerivative(d_world)

    def set_state_attributes(self, t: Tensor, world: World) -> World | None:
        """
        Apply discrete state changes for black projects.

        Triggers:
        1. Fab becomes operational when construction completes
        """
        current_time = t.item() if isinstance(t, Tensor) else float(t)
        changed = False

        for _, project in world.black_projects.items():
            # Check if fab should become operational
            if not project.fab_is_operational:
                fab_operational_year = project.preparation_start_year + project.fab_construction_duration
                if current_time >= fab_operational_year:
                    project.fab_is_operational = True
                    changed = True

        return world if changed else None

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """
        Compute derived metrics for black projects.

        Updates:
        - fab_wafer_starts_per_month, fab_h100e_per_chip, fab_watts_per_chip
        - datacenters (capacity)
        - compute_stock (with survival/attrition)
        - operating_compute_tpp_h100e
        """
        current_time = t.item() if isinstance(t, Tensor) else float(t)
        prc_compute = self.compute_params.PRCComputeParameters
        prc_energy = self.energy_params.prc_energy_consumption
        survival_params = self.compute_params.survival_rate_parameters

        for _, project in world.black_projects.items():
            # Calculate years since project start
            years_since_start = current_time - project.preparation_start_year

            # --- Fab metrics ---
            project.fab_wafer_starts_per_month = calculate_fab_wafer_starts_per_month(
                fab_operating_labor=project.fab_operating_labor,
                fab_number_of_lithography_scanners=project.fab_number_of_lithography_scanners,
                wafers_per_month_per_operating_worker=prc_compute.fab_wafers_per_month_per_operating_worker,
                wafers_per_month_per_scanner=prc_compute.wafers_per_month_per_lithography_scanner,
            )

            project.fab_h100e_per_chip = calculate_fab_h100e_per_chip(
                fab_process_node_nm=project.fab_process_node_nm,
                year=current_time,
                exogenous_trends=self.compute_params.exogenous_trends,
            )

            project.fab_watts_per_chip = calculate_fab_watts_per_chip(
                fab_process_node_nm=project.fab_process_node_nm,
                exogenous_trends=self.compute_params.exogenous_trends,
            )

            # Update fab production compute
            if project.fab_is_operational:
                monthly_h100e = (
                    project.fab_wafer_starts_per_month *
                    project.fab_chips_per_wafer *
                    project.fab_h100e_per_chip
                )
                # Update the fab's monthly production
                object.__setattr__(project.fab.monthly_production_compute, 'all_tpp_h100e', monthly_h100e)
                object.__setattr__(project.fab.monthly_production_compute, 'functional_tpp_h100e', monthly_h100e)

            # --- Datacenter metrics ---
            # Get datacenter construction rate from labor
            gw_per_worker_per_year = prc_energy.data_center_mw_per_year_per_construction_worker / 1000.0
            construction_rate = gw_per_worker_per_year * project.concealed_datacenter_capacity_construction_labor

            # Calculate concealed capacity (linear growth)
            concealed_gw = calculate_concealed_capacity_gw(
                current_year=current_time,
                construction_start_year=project.preparation_start_year,
                construction_rate_gw_per_year=construction_rate,
                max_concealed_capacity_gw=project.concealed_max_total_capacity_gw,
            )

            # Total capacity
            total_capacity_gw = calculate_datacenter_capacity_gw(
                unconcealed_capacity_gw=project.unconcealed_datacenter_capacity_diverted_gw,
                concealed_capacity_gw=concealed_gw,
            )

            # Update datacenters metric
            object.__setattr__(project.datacenters, 'data_center_capacity_gw', total_capacity_gw)

            # Operating labor per GW
            project.datacenters_operating_labor_per_gw = prc_energy.data_center_mw_per_operating_worker * 1000.0

            # --- Compute stock metrics ---
            # Get initial compute stock from parent nation
            initial_compute = project.compute_stock.all_tpp_h100e if project.compute_stock else 0.0

            # Calculate survival rate using hazard model
            survival_rate = calculate_survival_rate(
                years_since_acquisition=years_since_start,
                initial_hazard_rate=survival_params.initial_annual_hazard_rate,
                hazard_rate_increase_per_year=survival_params.annual_hazard_rate_increase_per_year,
            )

            # Surviving initial compute
            surviving_initial = initial_compute * survival_rate

            # Add fab production if operational
            cumulative_fab_production = 0.0
            if project.fab_is_operational:
                years_operational = current_time - (project.preparation_start_year + project.fab_construction_duration)
                if years_operational > 0:
                    annual_production = calculate_fab_annual_production_h100e(
                        fab_wafer_starts_per_month=project.fab_wafer_starts_per_month,
                        fab_chips_per_wafer=project.fab_chips_per_wafer,
                        fab_h100e_per_chip=project.fab_h100e_per_chip,
                        fab_is_operational=True,
                    )
                    cumulative_fab_production = annual_production * years_operational

            total_compute = surviving_initial + cumulative_fab_production
            functional_compute = calculate_functional_compute(total_compute, 1.0)  # Already applied survival

            # Update compute stock
            object.__setattr__(project.compute_stock, 'all_tpp_h100e', total_compute)
            object.__setattr__(project.compute_stock, 'functional_tpp_h100e', functional_compute)

            # --- Operating compute (limited by datacenter capacity) ---
            watts_per_h100e = project.compute_stock.watts_per_h100e if project.compute_stock else 700.0
            project.operating_compute_tpp_h100e = calculate_operating_compute(
                functional_compute_h100e=functional_compute,
                datacenter_capacity_gw=total_capacity_gw,
                watts_per_h100e=watts_per_h100e,
            )

        return world


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
    preparation_start_year = black_project_params.black_project_start_year

    # Calculate labor values from total_labor and fractions
    total_labor = props.total_labor
    datacenter_construction_labor = total_labor * props.fraction_of_labor_devoted_to_datacenter_construction
    black_fab_construction_labor = total_labor * props.fraction_of_labor_devoted_to_black_fab_construction
    black_fab_operating_labor = total_labor * props.fraction_of_labor_devoted_to_black_fab_operation
    ai_researcher_headcount = total_labor * props.fraction_of_labor_devoted_to_ai_research

    # --- Calculate derived values from parameters ---

    # Initial diverted compute
    diverted_compute = initial_prc_compute_stock * props.fraction_of_initial_compute_stock_to_divert_at_black_project_start

    # Energy requirements
    h100_power_w = 700.0
    energy_efficiency = prc_energy.energy_efficiency_of_compute_stock_relative_to_state_of_the_art
    energy_per_h100e_gw = (h100_power_w / energy_efficiency) / 1e9
    initial_energy_requirement = diverted_compute * energy_per_h100e_gw

    # Unconcealed capacity diverted from existing datacenters
    unconcealed_capacity_gw = (
        initial_energy_requirement *
        props.fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start
    )

    # Max concealed capacity
    max_capacity_gw = (
        props.max_fraction_of_total_national_energy_consumption *
        prc_energy.total_prc_energy_consumption_gw
    )

    # Construction rate for datacenters
    gw_per_worker_per_year = prc_energy.data_center_mw_per_year_per_construction_worker / 1000.0
    construction_rate = gw_per_worker_per_year * datacenter_construction_labor

    # Initial concealed capacity is 0 at project start
    initial_concealed = 0.0
    initial_total_capacity = initial_concealed + unconcealed_capacity_gw

    # Fab construction duration
    target_wafer_starts = (
        prc_compute.fab_wafers_per_month_per_operating_worker * black_fab_operating_labor * 0.5
    )
    fab_construction_duration = calculate_fab_construction_duration(
        fab_construction_labor=black_fab_construction_labor,
        target_wafer_starts_per_month=target_wafer_starts,
        prc_compute_params=prc_compute,
    )

    # Fab h100e per chip calculation
    process_node_nm = props.black_fab_max_process_node
    fab_h100e_per_chip = calculate_fab_h100e_per_chip(
        fab_process_node_nm=process_node_nm,
        year=preparation_start_year,
        exogenous_trends=exogenous_trends,
    )

    # --- Pre-compute labor_by_year for detection ---
    if simulation_years is None:
        simulation_years = [preparation_start_year + i for i in range(21)]

    labor_by_relative_year = {}
    for year in simulation_years:
        if year < preparation_start_year:
            continue
        relative_year = int(year - preparation_start_year)

        # Calculate total labor at this year
        labor_at_year = datacenter_construction_labor
        labor_at_year += ai_researcher_headcount

        # Datacenter operating labor (grows with capacity)
        years_since_prep_start = year - preparation_start_year
        concealed_at_year = min(
            construction_rate * years_since_prep_start,
            max(0, max_capacity_gw - unconcealed_capacity_gw)
        )
        total_capacity_at_year = concealed_at_year + unconcealed_capacity_gw
        operating_labor = total_capacity_at_year * prc_energy.data_center_mw_per_operating_worker * 1000.0
        labor_at_year += int(operating_labor)

        # Fab labor
        if props.build_a_black_fab:
            labor_at_year += black_fab_construction_labor
            labor_at_year += black_fab_operating_labor

        labor_by_relative_year[relative_year] = int(labor_at_year)

    # --- Sample detection time ---
    lr_by_year, sampled_detection_time = compute_lr_over_time_vs_num_workers(
        labor_by_year=labor_by_relative_year,
        mean_detection_time_100_workers=perception_params.mean_detection_time_for_100_workers,
        mean_detection_time_1000_workers=perception_params.mean_detection_time_for_1000_workers,
        variance_theta=perception_params.variance_of_detection_time_given_num_workers,
    )

    # --- Create required nested objects ---

    # AISoftwareProgress (all zeros for black project)
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
    )

    # ComputeAllocation
    compute_allocation = ComputeAllocation(
        fraction_for_ai_r_and_d_inference=0.5,
        fraction_for_ai_r_and_d_training=0.3,
        fraction_for_external_deployment=0.0,
        fraction_for_alignment_research=0.1,
        fraction_for_frontier_training=0.1,
    )

    # Initial operating compute list
    initial_compute = Compute(
        all_tpp_h100e=diverted_compute,
        functional_tpp_h100e=diverted_compute,
        watts_per_h100e=h100_power_w / energy_efficiency,
        average_functional_chip_age_years=0.0,
    )

    # Fab's monthly production (0 until operational)
    fab_production_compute = Compute(
        all_tpp_h100e=0.0,
        functional_tpp_h100e=0.0,
        watts_per_h100e=h100_power_w,  # New chips at current efficiency
        average_functional_chip_age_years=0.0,
    )

    fab = Fabs(monthly_production_compute=fab_production_compute)

    # Calculate number of lithography scanners
    num_scanners = int(
        initial_prc_compute_stock * props.fraction_of_lithography_scanners_to_divert_at_black_project_start
        / 10000  # Rough scaling
    )
    num_scanners = max(1, min(num_scanners, 100))  # Reasonable bounds

    # --- Create the AIBlackProject ---
    project = AIBlackProject(
        # Entity fields
        id=project_id,

        # AISoftwareDeveloper fields (state)
        operating_compute=[initial_compute],
        compute_allocation=compute_allocation,
        human_ai_capability_researchers=float(ai_researcher_headcount),
        ai_software_progress=ai_software_progress,

        # AIBlackProject state fields
        parent_nation=parent_nation,
        preparation_start_year=preparation_start_year,

        # Fab state
        fab_process_node_nm=process_node_nm,
        fab_number_of_lithography_scanners=float(num_scanners),
        fab_construction_labor=float(black_fab_construction_labor),
        fab_operating_labor=float(black_fab_operating_labor),
        fab_chips_per_wafer=float(prc_compute.h100_sized_chips_per_wafer),

        # Datacenter state
        unconcealed_datacenter_capacity_diverted_gw=unconcealed_capacity_gw,
        concealed_datacenter_capacity_construction_labor=float(datacenter_construction_labor),
        concealed_max_total_capacity_gw=max_capacity_gw,

        # Fab (constructor param, not init=False)
        fab=fab,
    )

    # --- Set init=False metrics after construction ---
    # Use object.__setattr__ to bypass frozen dataclass protection

    # AISoftwareDeveloper metrics
    object.__setattr__(project, 'ai_r_and_d_inference_compute_tpp_h100e', 0.0)
    object.__setattr__(project, 'ai_r_and_d_training_compute_tpp_h100e', 0.0)
    object.__setattr__(project, 'external_deployment_compute_tpp_h100e', 0.0)
    object.__setattr__(project, 'alignment_research_compute_tpp_h100e', 0.0)
    object.__setattr__(project, 'frontier_training_compute_tpp_h100e', 0.0)

    # Fab metrics
    object.__setattr__(project, 'fab_construction_duration', fab_construction_duration)
    object.__setattr__(project, 'fab_is_operational', False)
    object.__setattr__(project, 'fab_wafer_starts_per_month', target_wafer_starts)
    object.__setattr__(project, 'fab_h100e_per_chip', fab_h100e_per_chip)
    object.__setattr__(project, 'fab_watts_per_chip', calculate_fab_watts_per_chip(process_node_nm, exogenous_trends))

    # Datacenter metrics
    datacenters = Datacenters(data_center_capacity_gw=initial_total_capacity)
    object.__setattr__(project, 'datacenters', datacenters)
    object.__setattr__(project, 'datacenters_operating_labor_per_gw', prc_energy.data_center_mw_per_operating_worker * 1000.0)

    # Compute stock metrics
    compute_stock = Compute(
        all_tpp_h100e=diverted_compute,
        functional_tpp_h100e=diverted_compute,
        watts_per_h100e=h100_power_w / energy_efficiency,
        average_functional_chip_age_years=0.0,
    )
    object.__setattr__(project, 'compute_stock', compute_stock)

    # Operating compute
    initial_operating = calculate_operating_compute(
        functional_compute_h100e=diverted_compute,
        datacenter_capacity_gw=initial_total_capacity,
        watts_per_h100e=h100_power_w / energy_efficiency,
    )
    object.__setattr__(project, 'operating_compute_tpp_h100e', initial_operating)

    return project, lr_by_year, sampled_detection_time


def get_detection_info_for_simulation(
    project: AIBlackProject,
    black_project_params: BlackProjectParameters,
    energy_params: DataCenterAndEnergyParameters,
    perception_params: BlackProjectPerceptionsParameters,
    initial_prc_compute_stock: float,
    simulation_years: list,
) -> Tuple[Dict, float, Dict]:
    """
    Get detection-related information for running a simulation.

    Since detection is no longer part of the entity, this function computes
    detection parameters externally.

    Args:
        project: The black project
        black_project_params: Black project parameters
        energy_params: Energy parameters
        perception_params: Detection/perception parameters
        initial_prc_compute_stock: PRC compute stock
        simulation_years: Years to simulate

    Returns:
        Tuple of (lr_by_year, sampled_detection_time, us_estimates)
    """
    props = black_project_params.black_project_properties
    prc_energy = energy_params.prc_energy_consumption

    # Calculate labor by year
    labor_by_relative_year = {}
    gw_per_worker_per_year = prc_energy.data_center_mw_per_year_per_construction_worker / 1000.0
    construction_rate = gw_per_worker_per_year * project.concealed_datacenter_capacity_construction_labor

    for year in simulation_years:
        if year < project.preparation_start_year:
            continue
        relative_year = int(year - project.preparation_start_year)
        labor_at_year = get_black_project_total_labor(project)
        labor_by_relative_year[relative_year] = labor_at_year

    # Sample detection time
    lr_by_year, sampled_detection_time = compute_lr_over_time_vs_num_workers(
        labor_by_year=labor_by_relative_year,
        mean_detection_time_100_workers=perception_params.mean_detection_time_for_100_workers,
        mean_detection_time_1000_workers=perception_params.mean_detection_time_for_1000_workers,
        variance_theta=perception_params.variance_of_detection_time_given_num_workers,
    )

    # Sample US estimates
    total_prc_energy_gw = prc_energy.total_prc_energy_consumption_gw

    us_estimates = {
        'us_estimate_energy_gw': sample_us_estimate_with_error(
            total_prc_energy_gw,
            perception_params.intelligence_median_error_in_energy_consumption_estimate_of_datacenter_capacity
        ),
        'us_estimate_satellite_capacity_gw': sample_us_estimate_with_error(
            project.unconcealed_datacenter_capacity_diverted_gw,
            perception_params.intelligence_median_error_in_satellite_estimate_of_datacenter_capacity
        ),
        'us_estimate_compute_stock': sample_us_estimate_with_error(
            initial_prc_compute_stock,
            perception_params.intelligence_median_error_in_estimate_of_compute_stock
        ),
        'total_prc_energy_gw': total_prc_energy_gw,
        'diversion_proportion': props.fraction_of_initial_compute_stock_to_divert_at_black_project_start,
    }

    return lr_by_year, sampled_detection_time, us_estimates
