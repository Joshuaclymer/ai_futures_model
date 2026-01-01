"""
Black project world updater.

Updates covert compute infrastructure including:
- Fab production (when operational)
- Datacenter capacity growth
- Compute stock with attrition

Detection is handled externally by the simulation, not in the entity.
"""

import math
import numpy as np
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
from world_updaters.compute.chip_survival import (
    calculate_survival_rate,
    calculate_functional_compute,
    calculate_hazard_rate,
    calculate_compute_derivative,
    calculate_average_age_derivative,
)
from world_updaters.compute.black_compute import (
    calculate_fab_construction_duration,
    calculate_fab_wafer_starts_per_month,
    calculate_fab_h100e_per_chip,
    calculate_fab_watts_per_chip,
    calculate_fab_annual_production_h100e,
    calculate_concealed_capacity_gw,
    calculate_datacenter_capacity_gw,
    calculate_datacenter_operating_labor,
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
    compute_lr_from_sme_inventory,
    compute_detection_probability,
    # Continuous detection model functions
    compute_log_survival_probability,
    compute_cumulative_log_lr,
    compute_posterior_probability,
    compute_worker_detection_hazard_rate,
)


class BlackProjectUpdater(WorldUpdater):
    """
    Updates black project dynamics.

    Updates metrics based on current state:
    1. Fab metrics (wafer starts, h100e per chip, production)
    2. Datacenter capacity (concealed + unconcealed)
    3. Compute stock (with attrition)
    4. Operating compute (limited by datacenter capacity)
    5. Detection likelihood ratios (continuous model)
    """

    def __init__(
        self,
        params: SimulationParameters,
        black_project_params: BlackProjectParameters,
        compute_params: ComputeParameters,
        energy_params: DataCenterAndEnergyParameters,
        perception_params: BlackProjectPerceptionsParameters = None,
    ):
        super().__init__()
        self.params = params
        self.black_project_params = black_project_params
        self.compute_params = compute_params
        self.energy_params = energy_params
        self.perception_params = perception_params

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Compute continuous contributions to d(state)/dt for black projects.

        Computes derivatives for fab-produced compute stock using the average-age
        attrition model (see chip_survival.py for mathematical derivation).

        For fab-produced compute:
            dC_fab/dt = F - h(ā_fab)·C_fab
            dā_fab/dt = 1 - (ā_fab·F)/C_fab

        where:
            F = annual fab production rate (H100e/year)
            h(ā) = h₀ + h₁·ā (hazard rate at average age)
            C_fab = fab compute stock
            ā_fab = average age of fab chips
        """
        current_time = t.item() if isinstance(t, Tensor) else float(t)
        d_world = World.zeros(world)
        survival_params = self.compute_params.survival_rate_parameters
        prc_compute = self.compute_params.PRCComputeParameters
        exogenous_trends = self.compute_params.exogenous_trends

        for project_id, project in world.black_projects.items():
            d_project = d_world.black_projects.get(project_id)
            if d_project is None:
                continue

            # Convert preparation_start_year to float (may be a 0-d Tensor after ODE unpacking)
            prep_start_year = float(project.preparation_start_year.item() if hasattr(project.preparation_start_year, 'item') else project.preparation_start_year)

            # Compute whether fab is operational from state variables
            # (can't rely on fab_is_operational metric which isn't updated during ODE integration)
            # First check if a fab was actually built (scanners > 0)
            fab_was_built = project.fab_number_of_lithography_scanners > 0

            fab_is_operational = False
            if fab_was_built:
                # Calculate fab construction duration from state variables
                target_wafer_starts = calculate_fab_wafer_starts_per_month(
                    fab_operating_labor=project.fab_operating_labor,
                    fab_number_of_lithography_scanners=project.fab_number_of_lithography_scanners,
                    wafers_per_month_per_worker=prc_compute.fab_wafers_per_month_per_operating_worker,
                    wafers_per_month_per_scanner=prc_compute.wafers_per_month_per_lithography_scanner,
                )
                fab_construction_duration = calculate_fab_construction_duration(
                    fab_construction_labor=project.fab_construction_labor,
                    target_wafer_starts_per_month=target_wafer_starts,
                    prc_compute_params=prc_compute,
                )
                # Fab construction starts at agreement_year (preparation_start_year + 1)
                # This matches the reference model which uses agreement_year as construction start
                agreement_year = prep_start_year + 1
                fab_operational_year = agreement_year + fab_construction_duration
                fab_is_operational = current_time >= fab_operational_year

            # Only compute derivatives if fab is operational
            if fab_is_operational:
                # Calculate fab metrics inline (can't rely on set_metric_attributes during ODE integration)
                wafer_starts = calculate_fab_wafer_starts_per_month(
                    fab_operating_labor=project.fab_operating_labor,
                    fab_number_of_lithography_scanners=project.fab_number_of_lithography_scanners,
                    wafers_per_month_per_worker=prc_compute.fab_wafers_per_month_per_operating_worker,
                    wafers_per_month_per_scanner=prc_compute.wafers_per_month_per_lithography_scanner,
                )

                # Use preparation_start_year (fab construction time) for architecture efficiency
                # This matches discrete model: chip specs are fixed when fab is built
                h100e_per_chip = calculate_fab_h100e_per_chip(
                    fab_process_node_nm=project.fab_process_node_nm,
                    year=prep_start_year,  # Fixed at construction time, not current_time!
                    exogenous_trends=exogenous_trends,
                )

                # Calculate annual fab production rate (H100e per year)
                annual_production = calculate_fab_annual_production_h100e(
                    fab_wafer_starts_per_month=wafer_starts,
                    fab_chips_per_wafer=project.fab_chips_per_wafer,
                    fab_h100e_per_chip=h100e_per_chip,
                    fab_is_operational=True,
                )

                # Get current fab compute stock and average age
                fab_compute = project.fab_compute_stock_h100e
                fab_avg_age = project.fab_compute_average_age_years

                # Compute derivatives using average-age attrition model
                # dC_fab/dt = F - h(ā)·C_fab
                dC_fab_dt = calculate_compute_derivative(
                    functional_compute=fab_compute,
                    average_age=fab_avg_age,
                    production_rate=annual_production,
                    initial_hazard_rate=survival_params.effective_initial_hazard_rate,
                    hazard_rate_increase_per_year=survival_params.effective_hazard_rate_increase,
                )

                # dā_fab/dt = 1 - (ā·F)/C
                da_fab_dt = calculate_average_age_derivative(
                    functional_compute=fab_compute,
                    average_age=fab_avg_age,
                    production_rate=annual_production,
                )

                # Set derivatives on the zero-initialized derivative world
                d_project._set_frozen_field('fab_compute_stock_h100e', dC_fab_dt)
                d_project._set_frozen_field('fab_compute_average_age_years', da_fab_dt)
                # Cumulative production only increases (no attrition subtracted)
                d_project._set_frozen_field('fab_total_produced_h100e', annual_production)

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
            # Only mark as operational if a fab was actually built (scanners > 0)
            if not project.fab_is_operational and project.fab_number_of_lithography_scanners > 0:
                # Convert preparation_start_year to float (may be a 0-d Tensor after ODE unpacking)
                prep_start_year = float(project.preparation_start_year.item() if hasattr(project.preparation_start_year, 'item') else project.preparation_start_year)
                # Fab construction starts at agreement_year (preparation_start_year + 1)
                agreement_year = prep_start_year + 1
                fab_operational_year = agreement_year + project.fab_construction_duration
                if current_time >= fab_operational_year:
                    project._set_frozen_field('fab_is_operational', True)
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
            # Convert preparation_start_year to float (may be a 0-d Tensor after ODE unpacking)
            prep_start_year = float(project.preparation_start_year.item() if hasattr(project.preparation_start_year, 'item') else project.preparation_start_year)
            # Calculate years since project start
            years_since_start = current_time - prep_start_year
            # Calculate years since agreement (for detection model, matches reference model)
            # agreement_year = preparation_start_year + 1
            years_since_agreement = current_time - (prep_start_year + 1)

            # --- Fab metrics ---
            project._set_frozen_field('fab_wafer_starts_per_month', calculate_fab_wafer_starts_per_month(
                fab_operating_labor=project.fab_operating_labor,
                fab_number_of_lithography_scanners=project.fab_number_of_lithography_scanners,
                wafers_per_month_per_worker=prc_compute.fab_wafers_per_month_per_operating_worker,
                wafers_per_month_per_scanner=prc_compute.wafers_per_month_per_lithography_scanner,
            ))

            # Use preparation_start_year (fab construction time) for architecture efficiency
            # This matches discrete model: chip specs are fixed when fab is built
            project._set_frozen_field('fab_h100e_per_chip', calculate_fab_h100e_per_chip(
                fab_process_node_nm=project.fab_process_node_nm,
                year=prep_start_year,  # Fixed at construction time, not current_time!
                exogenous_trends=self.compute_params.exogenous_trends,
            ))

            project._set_frozen_field('fab_watts_per_chip', calculate_fab_watts_per_chip(
                fab_process_node_nm=project.fab_process_node_nm,
                year=prep_start_year,  # Fixed at construction time, matches fab_h100e_per_chip
                exogenous_trends=self.compute_params.exogenous_trends,
            ))

            # Update fab production compute
            if project.fab_is_operational:
                monthly_h100e = (
                    project.fab_wafer_starts_per_month *
                    project.fab_chips_per_wafer *
                    project.fab_h100e_per_chip
                )
                # Update the fab's monthly production
                project.fab.monthly_compute_production._set_frozen_field('tpp_h100e_including_attrition', monthly_h100e)
                project.fab.monthly_compute_production._set_frozen_field('functional_tpp_h100e', monthly_h100e)

            # --- Datacenter metrics ---
            # Get datacenter construction rate from labor
            gw_per_worker_per_year = prc_energy.data_center_mw_per_year_per_construction_worker / 1000.0
            construction_rate = gw_per_worker_per_year * project.concealed_datacenter_capacity_construction_labor

            # Calculate concealed capacity (linear growth)
            # Construction starts before agreement_year (discrete model starts 1 year before)
            props = self.black_project_params.black_project_properties
            # agreement_year = preparation_start_year + 1
            agreement_year = prep_start_year + 1
            datacenter_construction_start = agreement_year - props.years_before_black_project_start_to_begin_datacenter_construction
            concealed_gw = calculate_concealed_capacity_gw(
                current_year=current_time,
                construction_start_year=datacenter_construction_start,
                construction_rate_gw_per_year=construction_rate,
                max_concealed_capacity_gw=project.concealed_max_total_capacity_gw,
            )

            # Total capacity
            total_capacity_gw = calculate_datacenter_capacity_gw(
                unconcealed_capacity_gw=project.unconcealed_datacenter_capacity_diverted_gw,
                concealed_capacity_gw=concealed_gw,
            )

            # Update datacenters metric
            project.datacenters._set_frozen_field('data_center_capacity_gw', total_capacity_gw)

            # Operating labor per GW
            project._set_frozen_field('datacenters_operating_labor_per_gw', prc_energy.data_center_mw_per_operating_worker * 1000.0)

            # --- Compute stock metrics ---
            # Use the CONSTANT initial_diverted_compute_h100e (set at initialization, never changes)
            # This fixes the bug where we were reading from compute_stock which included fab production
            initial_diverted = project.initial_diverted_compute_h100e

            # Calculate survival rate for initial diverted compute using hazard model
            # Reference model adds initial compute at agreement_year, so survival is based on
            # years since agreement (not years since preparation_start_year)
            # If before agreement_year, chips have 0 years of life (100% survival)
            chip_years_of_life = max(0.0, years_since_agreement)
            survival_rate = calculate_survival_rate(
                years_since_acquisition=chip_years_of_life,
                initial_hazard_rate=survival_params.effective_initial_hazard_rate,
                hazard_rate_increase_per_year=survival_params.effective_hazard_rate_increase,
            )

            # Store the pure survival probability for initial stock
            # This is used to compute surviving_initial below
            initial_stock_survival_probability = survival_rate

            # Surviving initial compute = original_diverted * S(t)
            surviving_initial = initial_diverted * initial_stock_survival_probability
            project._set_frozen_field('initial_compute_surviving_h100e', surviving_initial)

            # Fab compute comes from state variable (integrated via ODE with attrition)
            # This is tracked separately with its own average age for proper attrition
            fab_compute = project.fab_compute_stock_h100e

            # Update fab production metrics for reporting
            monthly_fab_production = 0.0
            if project.fab_is_operational:
                annual_production = calculate_fab_annual_production_h100e(
                    fab_wafer_starts_per_month=project.fab_wafer_starts_per_month,
                    fab_chips_per_wafer=project.fab_chips_per_wafer,
                    fab_h100e_per_chip=project.fab_h100e_per_chip,
                    fab_is_operational=True,
                )
                monthly_fab_production = annual_production / 12.0

            project._set_frozen_field('fab_cumulative_production_h100e', project.fab_total_produced_h100e)  # Actual cumulative production (no attrition)
            project._set_frozen_field('fab_monthly_production_h100e', monthly_fab_production)

            # Total compute = surviving initial + surviving fab production
            # Both have had attrition properly applied:
            # - Initial: via cohort survival rate S(t)
            # - Fab: via ODE integration with average-age attrition model
            total_compute = surviving_initial + fab_compute
            functional_compute = total_compute  # Already applied survival to both components

            # Compute survival_rate as ratio of surviving to total ever added
            # This matches the reference model definition:
            #   survival_rate = total_surviving / total_compute_ever_added
            # Where total_compute_ever_added = initial_diverted + fab_total_produced (without attrition)
            total_ever_added = initial_diverted + project.fab_total_produced_h100e
            overall_survival_rate = total_compute / total_ever_added if total_ever_added > 0 else 1.0
            project._set_frozen_field('survival_rate', overall_survival_rate)

            # Update compute stock metric
            project.compute_stock._set_frozen_field('tpp_h100e_including_attrition', total_compute)
            project.compute_stock._set_frozen_field('functional_tpp_h100e', functional_compute)

            # --- Energy consumption by source ---
            # Calculate energy consumed by each compute source:
            # - Initial stock uses watts_per_h100e (weighted efficiency for diverted chips)
            # - Fab compute uses fab_watts_per_chip / fab_h100e_per_chip (watts per H100e for new chips)
            initial_watts_per_h100e = project.compute_stock.watts_per_h100e if project.compute_stock else prc_energy.h100_power_watts
            fab_watts_per_h100e = (project.fab_watts_per_chip / project.fab_h100e_per_chip) if project.fab_h100e_per_chip > 0 else 0.0

            # Energy in GW = H100e * watts_per_H100e / 1e9
            initial_energy_gw = (surviving_initial * initial_watts_per_h100e) / 1e9
            fab_energy_gw = (fab_compute * fab_watts_per_h100e) / 1e9
            total_energy_gw = initial_energy_gw + fab_energy_gw

            project._set_frozen_field('initial_stock_energy_gw', initial_energy_gw)
            project._set_frozen_field('fab_compute_energy_gw', fab_energy_gw)
            project._set_frozen_field('total_compute_energy_gw', total_energy_gw)

            # --- Operating compute (limited by datacenter capacity) ---
            # Reference model: uses total_energy_requirements_GW() from all chip types
            # and scales down proportionally if energy exceeds capacity.
            # We use total_energy_gw which accounts for both initial stock and fab compute.
            if total_energy_gw <= 0:
                operating_compute = 0.0
            elif total_energy_gw <= total_capacity_gw:
                operating_compute = functional_compute
            else:
                # Scale down proportionally to fit within capacity
                operating_compute = functional_compute * (total_capacity_gw / total_energy_gw)
            project._set_frozen_field('operating_compute_tpp_h100e', operating_compute)

            # --- Detection model metrics (continuous) ---
            # Use years_since_agreement for detection (matches reference model)
            if self.perception_params is not None and years_since_agreement > 0:
                # Get total labor for detection calculation
                total_labor = get_black_project_total_labor(project)

                # Compute lr_reported_energy from datacenter energy consumption (dynamic)
                # Covert energy = datacenter capacity (which consumes that amount of power)
                covert_energy_gw = total_capacity_gw
                if covert_energy_gw > 1e-10 and hasattr(project, 'total_prc_energy_gw') and hasattr(project, 'us_estimate_energy'):
                    lr_energy = compute_lr_from_reported_energy_consumption(
                        total_prc_energy_gw=project.total_prc_energy_gw,
                        covert_energy_gw=covert_energy_gw,
                        us_estimate_energy=project.us_estimate_energy,
                        median_error=self.perception_params.intelligence_median_error_in_energy_consumption_estimate_of_datacenter_capacity,
                    )
                else:
                    lr_energy = 1.0

                # Compute static LR components (already set at initialization)
                # Reference model: cumulative_lr = lr_initial * lr_sme * satellite_lr * energy_lr * lr_other
                # Where lr_sme = lr_inventory * lr_procurement IF fab exists, else 1.0
                # So: lr_prc_accounting * lr_sme * lr_satellite * lr_energy
                has_fab = project.fab_number_of_lithography_scanners > 0
                lr_sme_log = (
                    math.log(max(1e-10, project.lr_sme_inventory)) +
                    math.log(max(1e-10, project.lr_fab_procurement))
                ) if has_fab else 0.0  # log(1.0) = 0.0 if no fab
                static_log_lr = (
                    math.log(max(1e-10, project.lr_prc_accounting)) +
                    lr_sme_log +
                    math.log(max(1e-10, project.lr_satellite_datacenter)) +
                    math.log(max(1e-10, lr_energy))  # Include energy LR
                )

                # Check if detection has occurred (sampled_detection_time is relative to agreement_year)
                detected = years_since_agreement >= project.sampled_detection_time

                # Compute continuous log-LR from worker detection (time relative to agreement)
                log_lr = compute_cumulative_log_lr(
                    t=years_since_agreement,
                    labor=total_labor,
                    mean_detection_time_100_workers=self.perception_params.mean_detection_time_for_100_workers,
                    mean_detection_time_1000_workers=self.perception_params.mean_detection_time_for_1000_workers,
                    variance=self.perception_params.variance_of_detection_time_given_num_workers,
                    static_log_lr=static_log_lr,
                    detected=detected,
                )

                # Compute posterior probability
                posterior_prob_val = compute_posterior_probability(
                    cumulative_log_lr=log_lr,
                    prior_odds=self.perception_params.prior_odds_of_covert_project,
                )

                # Look up DATACENTER-specific LR from precomputed dictionary
                # This uses the sampled detection time from initialization
                # Use round(..., 1) to match float keys in the dict
                relative_year_dc = round(float(years_since_agreement), 1)
                lr_datacenters_dict = getattr(project, 'lr_datacenters_by_year', {})
                worker_lr = lr_datacenters_dict.get(relative_year_dc, 1.0)

                cumulative_lr_val = math.exp(log_lr)

                # Set point-in-time metrics (time series extracted from trajectory)
                project._set_frozen_field('lr_other_intel', worker_lr)
                project._set_frozen_field('cumulative_lr', cumulative_lr_val)
                project._set_frozen_field('posterior_prob', posterior_prob_val)
                project._set_frozen_field('lr_reported_energy', lr_energy)

                # --- Fab-specific LR calculations ---
                # Uses fab labor and time from fab construction start (different from datacenter)
                # Fab construction starts at agreement_year (preparation_start_year + 1)
                agreement_year_val = prep_start_year + 1
                # Use round(..., 1) to match float keys in the dict
                years_since_fab_construction = round(float(current_time - agreement_year_val), 1)

                if years_since_fab_construction >= 0 and project.fab_number_of_lithography_scanners > 0:
                    # Look up FAB-specific LR from precomputed dictionary
                    lr_fab_other_dict = getattr(project, 'lr_fab_other_by_year', {})
                    fab_lr_other = lr_fab_other_dict.get(years_since_fab_construction, 1.0)

                    # Fab combined LR = lr_sme_inventory × lr_fab_procurement × lr_fab_other
                    # lr_sme_inventory = scanner inventory accounting (diverted lithography equipment)
                    # lr_fab_procurement = foreign equipment imports (1.0 if localized, 10.0 if not)
                    fab_lr_combined = project.lr_sme_inventory * project.lr_fab_procurement * fab_lr_other

                    project._set_frozen_field('lr_fab_other', fab_lr_other)
                    project._set_frozen_field('lr_fab_combined', fab_lr_combined)
                else:
                    # Before fab construction or no fab - use defaults
                    project._set_frozen_field('lr_fab_other', 1.0)
                    project._set_frozen_field('lr_fab_combined', project.lr_sme_inventory * project.lr_fab_procurement)

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
    # Agreement year (when AI slowdown starts) - used for architecture efficiency calculation
    agreement_year = policy_params.ai_slowdown_start_year
    # Fab construction start year - matches reference model: construction_start_year = agreement_year
    fab_construction_start_year = agreement_year

    # Determine which process nodes are localized by fab construction start
    # Check each node from most advanced (7nm) to least advanced (28nm)
    # Reference model checks against agreement_year (fab construction starts at agreement year)
    localization_years = {
        7: props.prc_localization_year_7nm,
        14: props.prc_localization_year_14nm,
        28: props.prc_localization_year_28nm,
    }

    # Find best available node (smallest nm = most advanced) that is localized by fab construction start
    best_available_node = None
    best_node_localization_year = 9999
    for node_nm in [7, 14, 28]:  # Check from most to least advanced
        if localization_years[node_nm] <= fab_construction_start_year:
            best_available_node = node_nm
            best_node_localization_year = localization_years[node_nm]
            break

    # Fab is built if best available node meets minimum requirement
    min_node_nm = props.black_fab_min_process_node
    build_a_black_fab = (best_available_node is not None and best_available_node <= min_node_nm)

    # The actual process node used is the best available (if fab is built)
    actual_process_node_nm = best_available_node if build_a_black_fab else None

    # Calculate labor values from total_labor and fractions
    total_labor = props.total_labor
    datacenter_construction_labor = total_labor * props.fraction_of_labor_devoted_to_datacenter_construction
    ai_researcher_headcount = total_labor * props.fraction_of_labor_devoted_to_ai_research

    # Fab labor only if fab is being built
    if build_a_black_fab:
        black_fab_construction_labor = total_labor * props.fraction_of_labor_devoted_to_black_fab_construction
        black_fab_operating_labor = total_labor * props.fraction_of_labor_devoted_to_black_fab_operation
    else:
        black_fab_construction_labor = 0.0
        black_fab_operating_labor = 0.0

    # --- Calculate derived values from parameters ---

    # Initial diverted compute
    diverted_compute = initial_prc_compute_stock * props.fraction_of_initial_compute_stock_to_divert_at_black_project_start

    # Energy requirements
    # The energy efficiency needs to account for both:
    # 1. PRC efficiency relative to state-of-the-art (e.g., 0.2)
    # 2. State-of-the-art efficiency relative to H100 (which improves over time)
    # This matches the discrete model's LargestAIProject.get_energy_efficiency_relative_to_h100()
    h100_power_w = prc_energy.h100_power_watts
    energy_efficiency_relative_to_sota = prc_energy.energy_efficiency_of_compute_stock_relative_to_state_of_the_art

    # Calculate state-of-the-art efficiency relative to H100 at agreement year
    # Reference model uses agreement_year (not preparation_start_year) for initial stock efficiency
    h100_release_year = 2022
    years_since_h100 = max(0, agreement_year - h100_release_year)
    sota_energy_efficiency_improvement_per_year = exogenous_trends.state_of_the_art_energy_efficiency_improvement_per_year
    sota_efficiency_relative_to_h100 = sota_energy_efficiency_improvement_per_year ** years_since_h100

    # Combined efficiency: PRC efficiency relative to SOTA * SOTA efficiency relative to H100
    combined_energy_efficiency = energy_efficiency_relative_to_sota * sota_efficiency_relative_to_h100

    # Watts per H100e = H100 power / combined efficiency
    watts_per_h100e = h100_power_w / combined_energy_efficiency
    energy_per_h100e_gw = watts_per_h100e / 1e9
    initial_energy_requirement = diverted_compute * energy_per_h100e_gw

    # Calculate energy consumption of TOTAL PRC compute stock (used for datacenter capacity)
    # Reference model: energy_consumption_of_prc_stock = sum of energy from initial stock
    energy_consumption_of_prc_stock_gw = initial_prc_compute_stock * energy_per_h100e_gw

    # Unconcealed capacity diverted from existing datacenters
    # Reference model: fraction_diverted * energy_consumption_of_prc_stock
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

    # Initial concealed capacity at project start (preparation_start_year)
    # Construction starts at agreement_year - head_start_years (matches reference model)
    # agreement_year = preparation_start_year + 1, so construction_start = preparation_start_year
    # At preparation_start_year, construction has just started = 0 years of construction
    head_start_years = props.years_before_black_project_start_to_begin_datacenter_construction
    # Time from construction start (agreement_year - head_start) to preparation_start_year
    construction_start = agreement_year - head_start_years
    years_of_construction_at_init = max(0, preparation_start_year - construction_start)
    initial_concealed = min(
        construction_rate * years_of_construction_at_init,
        max(0, max_capacity_gw - unconcealed_capacity_gw)
    )
    initial_total_capacity = initial_concealed + unconcealed_capacity_gw

    # --- Calculate number of lithography scanners (needed for wafer_starts calculation) ---
    # This matches reference model: calculate scanners at fab_construction_start_year (= agreement_year)
    if build_a_black_fab:
        # Calculate years since PRC achieved localization for this process node
        # Scanner count is calculated at fab_construction_start_year, matching reference model
        localization_year = best_node_localization_year
        years_since_localization = max(0, fab_construction_start_year - localization_year)

        # Calculate total accumulated PRC scanners using linear production ramp-up
        # Total = first_year_production * (n+1) + additional_per_year * n * (n+1) / 2
        n = years_since_localization
        first_year_prod = prc_compute.prc_lithography_scanners_produced_in_first_year
        additional_per_year = prc_compute.prc_additional_lithography_scanners_produced_per_year
        median_total_scanners = first_year_prod * (n + 1) + additional_per_year * n * (n + 1) / 2

        # Apply lognormal uncertainty to scanner production (matches reference model)
        sigma_relative = getattr(prc_compute, 'prc_scanner_production_relative_sigma', 0.30)
        if sigma_relative > 0 and median_total_scanners > 0:
            sigma_log = np.sqrt(np.log(1 + sigma_relative**2))
            mu_log = np.log(median_total_scanners)
            total_prc_scanners = np.random.lognormal(mean=mu_log, sigma=sigma_log)
        else:
            total_prc_scanners = median_total_scanners

        # Scanners devoted to fab = total * diversion proportion
        num_scanners = int(total_prc_scanners * props.fraction_of_lithography_scanners_to_divert_at_black_project_start)
        num_scanners = max(1, num_scanners)  # At least 1 scanner if fab is built
    else:
        num_scanners = 0
        total_prc_scanners = 0
        localization_year = localization_years.get(28, 9999)  # Default for lr_fab_procurement calculation

    # Fab construction duration - uses wafer_starts from min(labor, scanner) capacity
    # This matches reference model's estimate_wafer_starts_per_month() and estimate_construction_duration()
    target_wafer_starts = calculate_fab_wafer_starts_per_month(
        fab_operating_labor=black_fab_operating_labor,
        fab_number_of_lithography_scanners=float(num_scanners),
        wafers_per_month_per_worker=prc_compute.fab_wafers_per_month_per_operating_worker,
        wafers_per_month_per_scanner=prc_compute.wafers_per_month_per_lithography_scanner,
    )
    fab_construction_duration = calculate_fab_construction_duration(
        fab_construction_labor=black_fab_construction_labor,
        target_wafer_starts_per_month=target_wafer_starts,
        prc_compute_params=prc_compute,
    )

    # Fab h100e per chip calculation
    process_node_nm = actual_process_node_nm if build_a_black_fab else 28.0  # Use best available node, default to 28nm if no fab
    fab_h100e_per_chip = calculate_fab_h100e_per_chip(
        fab_process_node_nm=process_node_nm,
        year=preparation_start_year,
        exogenous_trends=exogenous_trends,
    )

    # --- Pre-compute labor_by_year for detection ---
    # Use agreement_year as base for detection model (matches reference model)
    # agreement_year = ai_slowdown_start_year = preparation_start_year + 1
    agreement_year = policy_params.ai_slowdown_start_year

    if simulation_years is None:
        simulation_years = [preparation_start_year + i for i in range(21)]

    # Build labor dictionaries for project-level LR and fab-specific LR
    # Reference model uses COMBINED labor (datacenter + fab + researcher) for lr_other_intel
    # But uses SEPARATE fab-only labor for lr_fab_other
    combined_labor_by_relative_year = {}  # For project-level lr_other_intel
    fab_labor_by_year_since_construction = {}  # For fab-specific lr_fab_other

    # Fab construction starts at agreement_year (same as preparation_start_year + 1)
    fab_construction_start_year = agreement_year

    # Generate fine-grained years (0.1 step) for labor dict to match reference model
    # This ensures LR is computed at each time step, not just at integer years
    fine_grained_years = []
    for year in simulation_years:
        # For each integer year, add 10 fine-grained points (0.0, 0.1, ..., 0.9)
        for i in range(10):
            fine_grained_year = year + i * 0.1
            if fine_grained_year >= agreement_year:
                fine_grained_years.append(round(fine_grained_year, 1))
    # Remove duplicates and sort
    fine_grained_years = sorted(set(fine_grained_years))

    for year in fine_grained_years:
        relative_year = round(year - agreement_year, 1)

        # --- Combined labor for project-level lr_other_intel ---
        # Matches reference model: datacenter_construction + datacenter_operating + fab (if exists) + researcher
        datacenter_construction_start = agreement_year - head_start_years
        years_since_construction_start = year - datacenter_construction_start
        concealed_at_year = min(
            construction_rate * years_since_construction_start,
            max(0, max_capacity_gw - unconcealed_capacity_gw)
        )
        total_capacity_at_year = concealed_at_year + unconcealed_capacity_gw
        # Reference model: operating_labor_per_GW = operating_labor_per_MW * 1000
        # Local: data_center_mw_per_operating_worker = MW per worker (inverse of workers_per_MW)
        # So: operating_labor = capacity_GW * 1000 / mw_per_worker
        operating_labor = total_capacity_at_year * 1000.0 / prc_energy.data_center_mw_per_operating_worker

        # Combined labor includes: datacenter + fab (if exists) + researcher headcount
        combined_labor = datacenter_construction_labor + operating_labor
        if build_a_black_fab:
            combined_labor += black_fab_construction_labor + black_fab_operating_labor
        combined_labor += ai_researcher_headcount
        combined_labor_by_relative_year[relative_year] = int(combined_labor)

        # --- Fab labor (for lr_fab_other) ---
        # Uses: fab_construction_labor + fab_operating_labor ONLY
        # Time is measured from fab construction start (NOT agreement start)
        # Use float keys to match reference model's fine-grained LR computation
        if build_a_black_fab:
            years_since_fab_construction = round(year - fab_construction_start_year, 1)
            if years_since_fab_construction >= 0:
                fab_labor = int(black_fab_construction_labor + black_fab_operating_labor)
                fab_labor_by_year_since_construction[years_since_fab_construction] = fab_labor

    # --- Sample detection time for project-level LR (uses combined labor) ---
    lr_datacenters_by_year, sampled_detection_time_dc = compute_lr_over_time_vs_num_workers(
        labor_by_year=combined_labor_by_relative_year,
        mean_detection_time_100_workers=perception_params.mean_detection_time_for_100_workers,
        mean_detection_time_1000_workers=perception_params.mean_detection_time_for_1000_workers,
        variance=perception_params.variance_of_detection_time_given_num_workers,
    )

    # --- Sample detection time for FAB ---
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

    # Use the datacenter detection time as the main sampled_detection_time
    # (backward compatibility with existing code)
    sampled_detection_time = sampled_detection_time_dc
    lr_by_year = lr_datacenters_by_year

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
        initial_progress=torch.tensor(0.0),
        software_efficiency=torch.tensor(0.0),
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
        tpp_h100e_including_attrition=diverted_compute,
        functional_tpp_h100e=diverted_compute,
        watts_per_h100e=watts_per_h100e,  # Uses combined efficiency (PRC vs SOTA * SOTA vs H100)
        average_functional_chip_age_years=0.0,
    )

    # Fab's monthly production (0 until operational)
    fab_production_compute = Compute(
        tpp_h100e_including_attrition=0.0,
        functional_tpp_h100e=0.0,
        watts_per_h100e=h100_power_w,  # New chips at current efficiency
        average_functional_chip_age_years=0.0,
    )

    fab = Fabs(monthly_compute_production=fab_production_compute)

    # Note: num_scanners, total_prc_scanners, and localization_year are calculated earlier
    # (before fab_construction_duration) to enable proper wafer_starts calculation

    # Compute training_compute_growth_rate from PRC compute growth rate (OOMs/year)
    import math
    training_compute_growth_rate = 0.0
    if hasattr(prc_compute, 'annual_growth_rate_of_prc_compute_stock'):
        annual_growth_rate = prc_compute.annual_growth_rate_of_prc_compute_stock
        if annual_growth_rate > 0:
            training_compute_growth_rate = math.log10(annual_growth_rate)

    # --- Create the AIBlackProject ---
    project = AIBlackProject(
        # Entity fields
        id=project_id,

        # AISoftwareDeveloper fields (state)
        operating_compute=[initial_compute],
        compute_allocation=compute_allocation,
        human_ai_capability_researchers=float(ai_researcher_headcount),
        ai_software_progress=ai_software_progress,
        training_compute_growth_rate=training_compute_growth_rate,

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

        # Compute tracking state (for proper attrition model)
        # Initial diverted compute is constant - never changes after initialization
        initial_diverted_compute_h100e=diverted_compute,
        # Fab compute stock starts at 0 (integrated via ODE when fab becomes operational)
        fab_compute_stock_h100e=0.0,
        # Average age of fab chips starts at 0 (will evolve via ODE)
        fab_compute_average_age_years=0.0,
        # Total cumulative production starts at 0 (integrated via ODE, no attrition)
        fab_total_produced_h100e=0.0,

        # Fab (constructor param, not init=False)
        fab=fab,
    )

    # --- Set init=False metrics after construction ---
    # AISoftwareDeveloper metrics
    project._set_frozen_field('ai_r_and_d_inference_compute_tpp_h100e', 0.0)
    project._set_frozen_field('ai_r_and_d_training_compute_tpp_h100e', 0.0)
    project._set_frozen_field('external_deployment_compute_tpp_h100e', 0.0)
    project._set_frozen_field('alignment_research_compute_tpp_h100e', 0.0)
    project._set_frozen_field('frontier_training_compute_tpp_h100e', 0.0)

    # Fab metrics
    project._set_frozen_field('fab_construction_duration', fab_construction_duration)
    project._set_frozen_field('fab_is_operational', False)
    project._set_frozen_field('fab_wafer_starts_per_month', target_wafer_starts)
    project._set_frozen_field('fab_h100e_per_chip', fab_h100e_per_chip)
    project._set_frozen_field('fab_watts_per_chip', calculate_fab_watts_per_chip(
        fab_process_node_nm=process_node_nm,
        year=preparation_start_year,  # Fixed at construction time, matches fab_h100e_per_chip
        exogenous_trends=exogenous_trends,
    ))

    # Datacenter metrics
    datacenters = Datacenters(data_center_capacity_gw=initial_total_capacity)
    project._set_frozen_field('datacenters', datacenters)
    project._set_frozen_field('datacenters_operating_labor_per_gw', prc_energy.data_center_mw_per_operating_worker * 1000.0)

    # Compute stock metrics
    compute_stock = Compute(
        tpp_h100e_including_attrition=diverted_compute,
        functional_tpp_h100e=diverted_compute,
        watts_per_h100e=watts_per_h100e,  # Uses combined efficiency (PRC vs SOTA * SOTA vs H100)
        average_functional_chip_age_years=0.0,
    )
    project._set_frozen_field('compute_stock', compute_stock)

    # Operating compute
    initial_operating = calculate_operating_compute(
        functional_compute_h100e=diverted_compute,
        datacenter_capacity_gw=initial_total_capacity,
        watts_per_h100e=watts_per_h100e,  # Uses combined efficiency
    )
    project._set_frozen_field('operating_compute_tpp_h100e', initial_operating)

    # Time series metrics (initialized as empty lists, computed during simulation)
    project._set_frozen_field('years', [])
    project._set_frozen_field('cumulative_h100_years', [])
    project._set_frozen_field('operational_compute_h100e_by_year', [])
    project._set_frozen_field('survival_rate_by_year', [])
    project._set_frozen_field('datacenter_capacity_gw_by_year', [])
    project._set_frozen_field('total_black_compute_by_year', [])
    project._set_frozen_field('initial_black_compute_by_year', [])
    project._set_frozen_field('fab_flow_by_year', [])

    # LR components - compute static LR values from resource accounting evidence
    # Sample US estimates of various quantities
    total_prc_energy_gw = prc_energy.total_prc_energy_consumption_gw

    us_estimate_compute = sample_us_estimate_with_error(
        initial_prc_compute_stock,
        perception_params.intelligence_median_error_in_estimate_of_compute_stock
    )
    # Reference model: US estimates satellite capacity based on reported_capacity (total - diverted)
    # reported_capacity = energy_consumption_of_prc_stock - unconcealed_capacity_gw
    reported_satellite_capacity = energy_consumption_of_prc_stock_gw - unconcealed_capacity_gw
    us_estimate_satellite_capacity = sample_us_estimate_with_error(
        reported_satellite_capacity,
        perception_params.intelligence_median_error_in_satellite_estimate_of_datacenter_capacity
    )
    # Sample US estimate of energy for lr_reported_energy (used dynamically)
    us_estimate_energy = sample_us_estimate_with_error(
        total_prc_energy_gw,
        perception_params.intelligence_median_error_in_energy_consumption_estimate_of_datacenter_capacity
    )

    # Compute LR from PRC chip stock accounting
    # If project exists, reported stock = true stock * (1 - diversion_fraction)
    reported_compute = initial_prc_compute_stock * (1 - props.fraction_of_initial_compute_stock_to_divert_at_black_project_start)
    lr_prc_accounting = compute_lr_from_prc_compute_accounting(
        reported_compute_stock=reported_compute,
        diversion_proportion=props.fraction_of_initial_compute_stock_to_divert_at_black_project_start,
        us_estimate_compute=us_estimate_compute,
        median_error=perception_params.intelligence_median_error_in_estimate_of_compute_stock,
    )

    # Compute LR from satellite detection of diverted datacenter capacity
    # Reference model: total_capacity = energy_consumption_of_prc_stock_at_agreement_start
    # diverted_capacity = unconcealed_capacity_gw (fraction_diverted * energy_consumption_of_prc_stock)
    lr_satellite = compute_lr_from_satellite_detection(
        diverted_capacity_gw=unconcealed_capacity_gw,
        total_unconcealed_capacity_gw=energy_consumption_of_prc_stock_gw,
        us_estimate_capacity=us_estimate_satellite_capacity,
        median_error=perception_params.intelligence_median_error_in_satellite_estimate_of_datacenter_capacity,
    )

    # Compute LR from SME (lithography scanner) inventory accounting
    # Use the already-sampled total_prc_scanners from scanner allocation section (includes lognormal uncertainty)
    # This matches reference model which uses same sampled value for both allocation and LR
    scanner_diversion_fraction = props.fraction_of_lithography_scanners_to_divert_at_black_project_start
    if build_a_black_fab and total_prc_scanners > 0 and scanner_diversion_fraction > 0:
        # Sample US estimate of scanner inventory (based on noisy total)
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
        lr_sme_inventory = 1.0  # No scanner diversion evidence

    # Static LR components (set at initialization)
    project._set_frozen_field('lr_prc_accounting', lr_prc_accounting)
    project._set_frozen_field('lr_sme_inventory', lr_sme_inventory)
    project._set_frozen_field('lr_satellite_datacenter', lr_satellite)

    # Fab procurement LR: 1.0 if PRC achieved localization before construction start, 10.0 if foreign imports needed
    # If fab is built, localization was achieved (by definition), so lr_fab_procurement = 1.0
    # If fab not built but localization achieved, lr_fab_procurement = 1.0
    # If localization NOT achieved, lr_fab_procurement = 10.0 (foreign equipment imports detectable)
    # Note: Use fab_construction_start_year (agreement_year) as the cutoff, NOT preparation_start_year
    lr_fab_procurement = 1.0 if localization_year <= fab_construction_start_year else 10.0
    project._set_frozen_field('lr_fab_procurement', lr_fab_procurement)

    # Dynamic LR metrics (initialized, updated during simulation)
    project._set_frozen_field('lr_reported_energy', 1.0)
    project._set_frozen_field('lr_other_intel', 1.0)
    # Initial cumulative includes all static LR components
    # Reference model: cumulative_lr = lr_initial * lr_sme * satellite_lr * energy_lr * lr_other
    # Where lr_sme = lr_inventory * lr_procurement IF fab exists, else 1.0
    # At initialization, lr_energy=1 and lr_other=1, so:
    # initial_cumulative_lr = lr_prc_accounting * lr_sme * lr_satellite
    # lr_sme = (lr_sme_inventory * lr_fab_procurement) if fab, else 1.0
    # Note: lr_sme_inventory is already 1.0 if no fab (num_scanners=0), but lr_fab_procurement
    # might be 10.0 even with no fab, so we need to explicitly check for fab existence
    lr_sme_for_cumulative = lr_sme_inventory * lr_fab_procurement if num_scanners > 0 else 1.0
    initial_cumulative_lr = lr_prc_accounting * lr_sme_for_cumulative * lr_satellite
    project._set_frozen_field('cumulative_lr', initial_cumulative_lr)

    # Fab-specific LR metrics (uses fab labor and time from construction start)
    project._set_frozen_field('lr_fab_other', 1.0)  # Initial fab worker detection LR
    # Fab combined LR = lr_sme_inventory × lr_fab_procurement × lr_fab_other
    initial_lr_fab_combined = lr_sme_inventory * lr_fab_procurement  # lr_fab_other=1 initially
    project._set_frozen_field('lr_fab_combined', initial_lr_fab_combined)

    # Store energy parameters for dynamic lr_reported_energy calculation
    project._set_frozen_field('us_estimate_energy', us_estimate_energy)
    project._set_frozen_field('total_prc_energy_gw', total_prc_energy_gw)
    project._set_frozen_field('posterior_prob', perception_params.prior_odds_of_covert_project / (1 + perception_params.prior_odds_of_covert_project))

    # Fab production metrics (initialized, updated during simulation)
    project._set_frozen_field('fab_cumulative_production_h100e', 0.0)
    project._set_frozen_field('fab_monthly_production_h100e', 0.0)
    # Architecture efficiency uses agreement_year (defined earlier) to match reference model
    project._set_frozen_field('fab_architecture_efficiency',
        exogenous_trends.state_of_the_art_architecture_efficiency_improvement_per_year ** (agreement_year - 2022.0))
    # Use correct scaling exponent (1.49) to match reference model
    # transistor_density = (H100_node / fab_node) ^ scaling_exponent
    project._set_frozen_field('fab_transistor_density_relative_to_h100',
        (4.0 / process_node_nm) ** exogenous_trends.transistor_density_scaling_exponent)

    # Survival and initial compute metrics
    project._set_frozen_field('survival_rate', 1.0)  # Initial survival rate
    project._set_frozen_field('initial_compute_surviving_h100e', diverted_compute)  # Initially all survive

    # Energy consumption by source (initialized, updated during simulation)
    # Initial energy = diverted_compute * watts_per_h100e / 1e9
    initial_energy_gw = (diverted_compute * watts_per_h100e) / 1e9
    project._set_frozen_field('initial_stock_energy_gw', initial_energy_gw)
    project._set_frozen_field('fab_compute_energy_gw', 0.0)  # No fab compute initially
    project._set_frozen_field('total_compute_energy_gw', initial_energy_gw)

    # Legacy time series fields (kept for backward compatibility)
    project._set_frozen_field('lr_reported_energy_by_year', [])
    project._set_frozen_field('lr_other_intel_by_year', [])
    project._set_frozen_field('cumulative_lr_by_year', [])
    project._set_frozen_field('posterior_prob_by_year', [])
    project._set_frozen_field('fab_cumulative_production_h100e_by_year', [])
    project._set_frozen_field('fab_architecture_efficiency_by_year', [])

    # Detection outcome
    project._set_frozen_field('sampled_detection_time', sampled_detection_time)

    # Store separate LR dictionaries for datacenter and fab
    project._set_frozen_field('lr_datacenters_by_year', lr_datacenters_by_year)
    project._set_frozen_field('lr_fab_other_by_year', lr_fab_other_by_year)

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
        variance=perception_params.variance_of_detection_time_given_num_workers,
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
