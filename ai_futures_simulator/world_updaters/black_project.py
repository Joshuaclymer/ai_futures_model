"""
Black project world updater.

Updates covert compute infrastructure including:
- Fab production (when operational)
- Datacenter capacity growth
- Compute stock with attrition
- Detection likelihood ratios
"""

import math
import torch
from torch import Tensor
from typing import Optional

from classes.world.world import World
from classes.world.entities import AIBlackProject
from classes.world.assets import BlackFabs, BlackDatacenters, BlackCompute
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.simulation_parameters import SimulationParameters
from parameters.black_project_parameters import BlackProjectParameterSet
from parameters.compute_parameters import ComputeParameters
from parameters.energy_consumption_parameters import EnergyConsumptionParameters


class BlackProjectUpdater(WorldUpdater):
    """
    Updates black project dynamics.

    Continuous dynamics:
    1. Datacenter capacity: d(log_capacity)/dt = construction_rate / capacity
    2. Compute stock: d(log_stock)/dt = (production - hazard * stock) / stock

    Discrete events:
    1. Fab becomes operational when construction completes
    2. Initial compute diversion at agreement year
    """

    def __init__(self, params: SimulationParameters, black_project_params: BlackProjectParameterSet = None):
        super().__init__()
        self.params = params
        self.black_project_params = black_project_params or BlackProjectParameterSet()

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Compute continuous contributions to d(state)/dt for black projects.

        Updates:
        - d(log_concealed_capacity)/dt for datacenter growth
        - d(log_compute_stock)/dt for production and attrition
        """
        d_world = World.zeros(world)
        current_time = t.item() if isinstance(t, Tensor) else float(t)

        for project_id, project in world.black_projects.items():
            d_project = d_world.black_projects[project_id]

            # --- Datacenter capacity growth ---
            if project.datacenters is not None:
                d_dc = d_project.datacenters
                dc = project.datacenters

                # Only grow if after construction start
                if current_time >= dc.construction_start_year:
                    current_capacity = dc.concealed_capacity_gw
                    max_concealed = dc.max_total_capacity_gw - dc.unconcealed_capacity_gw
                    headroom = max_concealed - current_capacity

                    if headroom > 0 and dc.construction_rate_gw_per_year > 0:
                        # Asymptotic growth: dC/dt = rate * (1 - exp(-(max - C) / scale))
                        # This is ~linear when C << max, and asymptotes to max
                        # Scale is 5% of max capacity for smooth transition
                        asymptote_scale = max_concealed * 0.05
                        asymptote_factor = 1.0 - math.exp(-headroom / asymptote_scale)
                        effective_rate = dc.construction_rate_gw_per_year * asymptote_factor

                        # Convert to log space: d(log(C))/dt = (dC/dt) / C
                        if current_capacity > 1e-10:
                            d_dc.log_concealed_capacity_gw = torch.tensor(
                                effective_rate / current_capacity
                            )
                        else:
                            # For very small capacity, use large growth rate
                            d_dc.log_concealed_capacity_gw = torch.tensor(10.0)
                    else:
                        d_dc.log_concealed_capacity_gw = torch.tensor(0.0)
                else:
                    d_dc.log_concealed_capacity_gw = torch.tensor(0.0)

            # --- Compute stock dynamics ---
            if project.compute_stock is not None:
                d_stock = d_project.compute_stock
                stock = project.compute_stock
                current_stock = stock.compute_stock_h100e

                # Production rate (only if fab is operational)
                production_rate = 0.0
                if project.fab is not None and project.fab.is_operational:
                    production_rate = project.fab.annual_production_h100e

                # Hazard rate (attrition)
                # For accurate survival modeling, use time since agreement as the age
                # of the initial compute cohort (simplified model without cohort tracking)
                years_since_agreement = current_time - project.ai_slowdown_start_year
                effective_age = max(0.0, years_since_agreement)

                # If there's fab production, we need to account for younger chips too
                # For now, use a weighted average: older initial stock decays faster
                if production_rate > 0 and project.fab is not None:
                    # With production, some chips are newer - use average_age which
                    # approximates the weighted mean age of all cohorts
                    effective_age = stock.average_age_years
                # else: use full time since agreement (all chips are from initial cohort)

                hazard_rate = stock.initial_hazard_rate + stock.hazard_rate_increase_per_year * effective_age

                # Stock dynamics: d(S)/dt = P - H*S
                # In log space: d(log(S))/dt = (P - H*S) / S = P/S - H

                if current_stock > 1e-6:
                    # Normal case: stock is non-negligible
                    d_log_stock = (production_rate / current_stock) - hazard_rate
                    d_stock.log_compute_stock = torch.tensor(d_log_stock)
                elif production_rate > 0:
                    # Stock is ~0 but we have production: rapid growth
                    d_stock.log_compute_stock = torch.tensor(10.0)
                else:
                    # No stock and no production: stay at zero
                    d_stock.log_compute_stock = torch.tensor(0.0)

        return StateDerivative(d_world)

    def set_state_attributes(self, t: Tensor, world: World) -> World | None:
        """
        Apply discrete state changes for black projects.

        Triggers:
        1. Fab becomes operational when construction completes
        """
        current_time = t.item() if isinstance(t, Tensor) else float(t)
        changed = False

        for project_id, project in world.black_projects.items():
            # Check if fab should become operational
            if project.fab is not None and not project.fab.is_operational:
                if current_time >= project.fab.operational_year:
                    project.fab.is_operational = True
                    changed = True

        return world if changed else None

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """Compute derived metrics for black projects."""
        current_time = t.item() if isinstance(t, Tensor) else float(t)

        for project_id, project in world.black_projects.items():
            # Update compute stock average age
            if project.compute_stock is not None:
                years_since_agreement = current_time - project.ai_slowdown_start_year
                if years_since_agreement > 0:
                    # Simplified age tracking: assume average age is half the time since start
                    # (This is a simplification; proper tracking would need cohort modeling)
                    project.compute_stock.average_age_years = years_since_agreement / 2.0

            # Update operating labor for datacenters
            if project.datacenters is not None:
                project.human_datacenter_operating_labor = int(project.datacenters.operating_labor)

        return world


def initialize_black_project(
    project_id: str,
    ai_slowdown_start_year: float,
    prc_compute_stock: float,
    params: BlackProjectParameterSet,
    compute_growth_params: ComputeParameters,
    energy_consumption_params: EnergyConsumptionParameters,
    sampled_values: dict = None,
) -> AIBlackProject:
    """
    Initialize a black project with all components.

    Args:
        project_id: Identifier for the project
        ai_slowdown_start_year: Year when AI slowdown/black project starts
        prc_compute_stock: PRC's total compute stock at slowdown start year
        params: Black project parameter set
        compute_growth_params: Compute growth parameters (contains hazard rates)
        energy_consumption_params: Energy consumption parameters
        sampled_values: Optional dict of pre-sampled values for Monte Carlo

    Returns:
        Initialized AIBlackProject
    """
    props = params.properties
    fab_params = params.fab_params
    dc_params = params.datacenter_params

    # --- Initialize compute stock with initial diversion ---
    diverted_compute = prc_compute_stock * props.proportion_of_initial_compute_stock_to_divert

    # Get hazard rates from compute_growth_params (or use provided values)
    if sampled_values and 'initial_hazard_rate' in sampled_values:
        initial_hazard = sampled_values['initial_hazard_rate']
        hazard_increase = sampled_values['hazard_rate_increase']
    else:
        initial_hazard = compute_growth_params.initial_hazard_rate
        hazard_increase = compute_growth_params.hazard_rate_increase_per_year

    compute_stock = BlackCompute(
        log_compute_stock=torch.tensor(math.log(max(diverted_compute, 1e-10))),
        initial_hazard_rate=initial_hazard,
        hazard_rate_increase_per_year=hazard_increase,
        average_age_years=0.0,
        energy_efficiency_relative_to_h100=energy_consumption_params.energy_efficiency_of_prc_stock_relative_to_state_of_the_art,
    )

    # --- Initialize datacenters ---
    dc_start_year = ai_slowdown_start_year - props.years_before_agreement_year_prc_starts_building_black_datacenters

    # Calculate construction rate
    gw_per_worker_per_year = dc_params.MW_per_construction_worker_per_year / 1000.0
    construction_rate = gw_per_worker_per_year * props.datacenter_construction_labor

    # Calculate unconcealed capacity (diverted at agreement start)
    # Based on energy consumption of diverted compute
    # H100 power ~700W, so H100e needs 700W / efficiency
    h100_power_w = 700.0
    energy_per_h100e_gw = (h100_power_w / energy_consumption_params.energy_efficiency_of_prc_stock_relative_to_state_of_the_art) / 1e9
    initial_energy_requirement = diverted_compute * energy_per_h100e_gw
    unconcealed_capacity = initial_energy_requirement * props.fraction_of_datacenter_capacity_not_built_for_concealment_diverted_to_black_project_at_agreement_start

    max_capacity = props.max_proportion_of_PRC_energy_consumption * energy_consumption_params.total_GW_of_PRC_energy_consumption

    # Calculate initial concealed capacity (built before agreement)
    years_of_construction = props.years_before_agreement_year_prc_starts_building_black_datacenters
    initial_concealed = min(
        construction_rate * years_of_construction,
        max(0, max_capacity - unconcealed_capacity)
    )

    datacenters = BlackDatacenters(
        log_concealed_capacity_gw=torch.tensor(math.log(max(initial_concealed, 1e-10))),
        unconcealed_capacity_gw=unconcealed_capacity,
        construction_start_year=dc_start_year,
        construction_rate_gw_per_year=construction_rate,
        max_total_capacity_gw=max_capacity,
        operating_labor_per_gw=dc_params.operating_labor_per_MW * 1000.0,
    )

    # --- Initialize fab (if enabled) ---
    fab = None
    if props.build_a_black_fab:
        # Sample or use provided fab parameters
        if sampled_values and 'wafer_starts_per_month' in sampled_values:
            wafer_starts = sampled_values['wafer_starts_per_month']
            construction_duration = sampled_values['fab_construction_duration']
            process_node_nm = sampled_values.get('process_node_nm', 28.0)
            h100e_per_chip = sampled_values.get('h100e_per_chip', 0.1)
        else:
            # Use median values
            wafer_starts = (
                fab_params.wafers_per_month_per_worker * props.black_fab_operating_labor * 0.5
            )  # Assume ~50% scanner constraint
            construction_duration = 2.0  # Default 2 years
            process_node_nm = 28.0
            # Calculate h100e_per_chip from transistor density and architecture
            density_ratio = (4.0 / process_node_nm) ** fab_params.transistor_density_scaling_exponent
            arch_efficiency = fab_params.architecture_efficiency_improvement_per_year ** (ai_slowdown_start_year - 2022)
            h100e_per_chip = density_ratio * arch_efficiency

        fab = BlackFabs(
            is_operational=False,
            process_node_nm=process_node_nm,
            construction_start_year=ai_slowdown_start_year,
            construction_duration=construction_duration,
            wafer_starts_per_month=wafer_starts,
            h100e_per_chip=h100e_per_chip,
            chips_per_wafer=fab_params.h100_sized_chips_per_wafer,
        )

    # --- Create the black project ---
    # Need to create minimal required fields for AISoftwareDeveloper parent
    from classes.world.entities import ComputeAllocation
    from classes.world.assets import Compute
    from classes.world.software_progress import AISoftwareProgress

    # Create AISoftwareProgress with all required fields
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

    project = AIBlackProject(
        id=project_id,
        is_primarily_controlled_by_misaligned_AI=False,
        compute=Compute(
            total_tpp_h100e=diverted_compute,
            total_energy_requirements_watts=diverted_compute * h100_power_w / energy_consumption_params.energy_efficiency_of_prc_stock_relative_to_state_of_the_art,
        ),
        compute_allocation=ComputeAllocation(
            fraction_for_ai_r_and_d_inference=0.5,
            fraction_for_ai_r_and_d_training=0.3,
            fraction_for_external_deployment=0.0,
            fraction_for_alignment_research=0.1,
            fraction_for_frontier_training=0.1,
        ),
        ai_software_progress=ai_software_progress,
        human_ai_capability_researchers=props.researcher_headcount,
        log_compute=torch.tensor(math.log(max(diverted_compute, 1e-10))),
        log_researchers=torch.tensor(math.log(max(props.researcher_headcount, 1))),
        parent_entity_id="PRC",
        human_datacenter_construction_labor=props.datacenter_construction_labor,
        human_fab_construction_labor=props.black_fab_construction_labor,
        human_fab_operating_labor=props.black_fab_operating_labor,
        researcher_headcount=props.researcher_headcount,
        fab=fab,
        datacenters=datacenters,
        compute_stock=compute_stock,
        ai_slowdown_start_year=ai_slowdown_start_year,
    )

    return project
