"""
Black project compute updater and utility functions.

Contains:
- BlackProjectComputeUpdater: WorldUpdater for black project compute (fabs, stock, attrition)
- Fab calculation utilities (construction duration, wafer starts, chip performance)
- Operating compute calculations
"""

import math
from typing import TYPE_CHECKING
from torch import Tensor

from classes.world.world import World
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.classes import SimulationParameters
from parameters.classes import BlackProjectParameters
from parameters.classes import ComputeParameters
from parameters.classes import DataCenterAndEnergyParameters
from world_updaters.compute.chip_survival import (
    calculate_survival_rate,
    calculate_compute_derivative,
    calculate_average_age_derivative,
)


if TYPE_CHECKING:
    from parameters.classes import PRCComputeParameters, ExogenousComputeTrends


# =============================================================================
# FAB METRIC CALCULATIONS
# =============================================================================

def calculate_fab_construction_duration(
    fab_construction_labor: float,
    target_wafer_starts_per_month: float,
    prc_compute_params: "PRCComputeParameters" = None,
    construction_time_for_5k_wafers: float = None,
    construction_time_for_100k_wafers: float = None,
    construction_workers_per_1000_wafers_per_month: float = None,
    construction_time_multiplier: float = None,
) -> float:
    """
    Calculate fab construction duration based on labor and target capacity.

    Uses a fixed-proportions production function where construction time depends on:
    1. Fab capacity: Larger fabs take longer to build (log-linear relationship)
    2. Construction labor: Fewer workers than required extends construction time
    3. Uncertainty multiplier: fab_construction_time_multiplier (sampled from distribution in YAML)

    This matches the discrete model's estimate_construction_duration function.
    """
    # Get construction times and labor requirements from params or use defaults
    if prc_compute_params is not None:
        construction_time_for_5k_wafers = prc_compute_params.construction_time_for_5k_wafers_per_month
        construction_time_for_100k_wafers = prc_compute_params.construction_time_for_100k_wafers_per_month
        construction_time_multiplier = prc_compute_params.fab_construction_time_multiplier
        # Convert from wafers_per_worker to workers_per_1000_wafers
        # wafers_per_worker = 14.1 means we need 14.1 workers per 1000 wafers/month
        construction_workers_per_1000_wafers_per_month = 14.1  # Default from discrete model
    else:
        construction_time_for_5k_wafers = construction_time_for_5k_wafers or 1.4
        construction_time_for_100k_wafers = construction_time_for_100k_wafers or 2.41
        construction_workers_per_1000_wafers_per_month = construction_workers_per_1000_wafers_per_month or 14.1
        construction_time_multiplier = construction_time_multiplier or 1.0

    if target_wafer_starts_per_month <= 0:
        return 2.0  # Default

    # Step 1: Log-linear interpolation for base construction duration given capacity
    # Uses log10 to match discrete model formula: time = slope * log10(wafers) + intercept
    log10_5k = math.log10(5000)
    log10_100k = math.log10(100000)
    log10_target = math.log10(target_wafer_starts_per_month)

    # Calculate slope and intercept for log-linear extrapolation
    slope = (construction_time_for_100k_wafers - construction_time_for_5k_wafers) / (log10_100k - log10_5k)
    intercept = construction_time_for_5k_wafers - slope * log10_5k

    # Calculate base construction duration from capacity
    construction_duration = slope * log10_target + intercept

    # Step 2: Calculate construction labor requirement given wafer capacity
    # The discrete model uses: workers_needed = (workers_per_1000_wafers / 1000) * wafer_capacity
    construction_labor_requirement = (
        construction_workers_per_1000_wafers_per_month / 1000.0
    ) * target_wafer_starts_per_month

    # Step 3: If actual labor < required labor, extend construction duration proportionally
    if fab_construction_labor < construction_labor_requirement and fab_construction_labor > 0:
        construction_duration *= (construction_labor_requirement / fab_construction_labor)

    # Step 4: Apply uncertainty multiplier (sampled from distribution in YAML config)
    construction_duration *= construction_time_multiplier

    return construction_duration


def calculate_fab_wafer_starts_per_month(
    fab_operating_labor: float,
    fab_number_of_lithography_scanners: float,
    wafers_per_month_per_worker: float,
    wafers_per_month_per_scanner: float,
    labor_productivity_multiplier: float = 1.0,
    scanner_productivity_multiplier: float = 1.0,
) -> float:
    """Calculate wafer starts per month based on labor and scanner constraints.

    Uses fixed-proportions production where output is limited by the binding constraint
    (either labor or scanners). Uncertainty is captured by the productivity multipliers
    which are sampled from log-normal distributions in Monte Carlo mode:
    - labor_productivity_multiplier: relative_sigma=0.62 (matches reference model)
    - scanner_productivity_multiplier: relative_sigma=0.20 (matches reference model)

    This matches the reference model's estimate_wafer_starts_per_month() which samples
    labor capacity and scanner capacity independently from log-normal distributions.
    """
    # Labor capacity with uncertainty multiplier
    from_labor = fab_operating_labor * wafers_per_month_per_worker * labor_productivity_multiplier

    # Scanner capacity with uncertainty multiplier
    from_scanners = fab_number_of_lithography_scanners * wafers_per_month_per_scanner * scanner_productivity_multiplier

    # Return minimum of labor and scanner capacity (fixed-proportions production)
    result = min(from_labor, from_scanners)
    return max(result, 1.0)  # Floor at 1 to prevent zero


def calculate_fab_h100e_per_chip(
    fab_process_node_nm: float,
    year: float,
    exogenous_trends: "ExogenousComputeTrends",
    h100_reference_nm: float = 4.0,
    h100_release_year: float = 2022.0,
) -> float:
    """Calculate H100-equivalent per chip based on process node and architecture improvements.

    Args:
        fab_process_node_nm: Process node in nanometers
        year: Current simulation year
        exogenous_trends: ExogenousComputeTrends containing transistor_density_scaling_exponent
            and state_of_the_art_architecture_efficiency_improvement_per_year (required)
        h100_reference_nm: H100 reference node (default 4nm)
        h100_release_year: H100 release year (default 2022)
    """
    transistor_density_scaling_exponent = exogenous_trends.transistor_density_scaling_exponent
    architecture_efficiency_improvement_per_year = exogenous_trends.state_of_the_art_architecture_efficiency_improvement_per_year

    # Density ratio relative to H100
    density_ratio = (h100_reference_nm / fab_process_node_nm) ** transistor_density_scaling_exponent

    # Architecture efficiency improvement since H100 release
    years_since_h100 = year - h100_release_year
    arch_efficiency = architecture_efficiency_improvement_per_year ** years_since_h100

    return density_ratio * arch_efficiency


def calculate_transistor_density_from_process_node(
    fab_process_node_nm: float,
    exogenous_trends: "ExogenousComputeTrends",
    h100_reference_nm: float = 4.0,
    h100_transistor_density_m_per_mm2: float = 98.28,
) -> float:
    """
    Calculate transistor density from process node.

    Uses the formula: density = H100_DENSITY * (process_node / H100_NODE)^(-exponent)

    This matches the discrete model's calculate_transistor_density_from_process_node().

    Args:
        fab_process_node_nm: Process node in nanometers
        exogenous_trends: ExogenousComputeTrends containing transistor_density_scaling_exponent (required)
        h100_reference_nm: H100 reference node (default 4nm)
        h100_transistor_density_m_per_mm2: H100 transistor density (default 98.28 M/mm²)

    Returns:
        Transistor density in M/mm²
    """
    transistor_density_scaling_exponent = exogenous_trends.transistor_density_scaling_exponent

    node_ratio = fab_process_node_nm / h100_reference_nm
    return h100_transistor_density_m_per_mm2 * (node_ratio ** (-transistor_density_scaling_exponent))


def calculate_watts_per_tpp_from_transistor_density(
    transistor_density_m_per_mm2: float,
    exogenous_trends: "ExogenousComputeTrends",
    h100_transistor_density_m_per_mm2: float = 98.28,
    h100_watts_per_tpp: float = 0.326493,
) -> float:
    """
    Calculate watts per TPP from transistor density using Dennard scaling model.

    Uses a piecewise power law with different exponents before/after Dennard scaling ended:
    - Before Dennard: watts_per_tpp scales with density^exponent_before (steeper, -2.0)
    - After Dennard: watts_per_tpp scales with density^exponent_after (shallower, -0.91)

    The transition point is anchored to the H100 (post-Dennard) and the pre-Dennard
    line is connected to ensure continuity.

    This matches the discrete model's predict_watts_per_tpp_from_transistor_density().

    Args:
        transistor_density_m_per_mm2: Transistor density in M/mm²
        exogenous_trends: ExogenousComputeTrends containing Dennard scaling parameters (required)
        h100_transistor_density_m_per_mm2: H100 transistor density (default 98.28)
        h100_watts_per_tpp: H100 watts per TPP (default 0.326493)

    Returns:
        Watts per TPP for the given transistor density
    """
    transistor_density_at_end_of_dennard = exogenous_trends.transistor_density_at_end_of_dennard_scaling_m_per_mm2
    watts_per_tpp_exponent_before_dennard = exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended
    watts_per_tpp_exponent_after_dennard = exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended

    # Calculate watts_per_tpp at the Dennard transition point using post-Dennard relationship
    transition_density_ratio = transistor_density_at_end_of_dennard / h100_transistor_density_m_per_mm2
    transition_watts_per_tpp = h100_watts_per_tpp * (transition_density_ratio ** watts_per_tpp_exponent_after_dennard)

    if transistor_density_m_per_mm2 < transistor_density_at_end_of_dennard:
        # Before Dennard scaling ended - anchor to transition point
        exponent = watts_per_tpp_exponent_before_dennard
        density_ratio = transistor_density_m_per_mm2 / transistor_density_at_end_of_dennard
        return transition_watts_per_tpp * (density_ratio ** exponent)
    else:
        # After Dennard scaling ended - anchor to H100
        exponent = watts_per_tpp_exponent_after_dennard
        density_ratio = transistor_density_m_per_mm2 / h100_transistor_density_m_per_mm2
        return h100_watts_per_tpp * (density_ratio ** exponent)


def calculate_fab_watts_per_chip(
    fab_process_node_nm: float,
    year: float,
    exogenous_trends: "ExogenousComputeTrends",
    h100_reference_nm: float = 4.0,
    h100_release_year: float = 2022.0,
    h100_transistor_density_m_per_mm2: float = 98.28,
    h100_watts_per_tpp: float = 0.326493,
    h100_tpp_per_chip: float = 2144.0,
) -> float:
    """
    Calculate watts per chip based on process node and year.

    This matches the discrete model's calculation in get_monthly_production_rate():
    1. Calculate h100e_per_chip (from density ratio and architecture efficiency)
    2. Calculate tpp_per_chip = h100e_per_chip * H100_TPP_PER_CHIP
    3. Calculate transistor_density from process node
    4. Calculate watts_per_tpp from transistor density
    5. watts_per_chip = tpp_per_chip * watts_per_tpp

    Args:
        fab_process_node_nm: Process node in nanometers (e.g., 28 for 28nm)
        year: Current simulation year (for architecture efficiency calculation)
        exogenous_trends: ExogenousComputeTrends containing all scaling parameters (required)
        h100_reference_nm: H100 reference node (default 4nm)
        h100_release_year: H100 release year (default 2022)
        h100_transistor_density_m_per_mm2: H100 transistor density (default 98.28)
        h100_watts_per_tpp: H100 watts per TPP (default 0.326493)
        h100_tpp_per_chip: H100 TPP per chip (default 2144.0)

    Returns:
        Watts per chip for the given process node and year
    """
    # Step 1: Calculate h100e_per_chip (performance relative to H100)
    h100e_per_chip = calculate_fab_h100e_per_chip(
        fab_process_node_nm=fab_process_node_nm,
        year=year,
        exogenous_trends=exogenous_trends,
        h100_reference_nm=h100_reference_nm,
        h100_release_year=h100_release_year,
    )

    # Step 2: Calculate TPP per chip
    tpp_per_chip = h100e_per_chip * h100_tpp_per_chip

    # Step 3: Calculate transistor density from process node
    transistor_density = calculate_transistor_density_from_process_node(
        fab_process_node_nm=fab_process_node_nm,
        exogenous_trends=exogenous_trends,
        h100_reference_nm=h100_reference_nm,
        h100_transistor_density_m_per_mm2=h100_transistor_density_m_per_mm2,
    )

    # Step 4: Calculate watts per TPP
    watts_per_tpp = calculate_watts_per_tpp_from_transistor_density(
        transistor_density_m_per_mm2=transistor_density,
        exogenous_trends=exogenous_trends,
        h100_transistor_density_m_per_mm2=h100_transistor_density_m_per_mm2,
        h100_watts_per_tpp=h100_watts_per_tpp,
    )

    # Step 5: Calculate watts per chip
    return tpp_per_chip * watts_per_tpp


def calculate_fab_annual_production_h100e(
    fab_wafer_starts_per_month: float,
    fab_chips_per_wafer: float,
    fab_h100e_per_chip: float,
    fab_is_operational: bool,
) -> float:
    """Calculate annual production in H100e when operational."""
    if not fab_is_operational:
        return 0.0
    return fab_wafer_starts_per_month * fab_chips_per_wafer * fab_h100e_per_chip * 12.0


# =============================================================================
# COMPUTE METRIC CALCULATIONS
# =============================================================================

def calculate_operating_compute(
    functional_compute_h100e: float,
    datacenter_capacity_gw: float,
    watts_per_h100e: float,
) -> float:
    """Calculate operating compute limited by datacenter capacity."""
    if datacenter_capacity_gw <= 0 or watts_per_h100e <= 0:
        return 0.0

    # Max compute that can be powered
    max_powered = datacenter_capacity_gw * 1e9 / watts_per_h100e

    return min(functional_compute_h100e, max_powered)


# =============================================================================
# BLACK PROJECT COMPUTE UPDATER
# =============================================================================

class BlackProjectComputeUpdater(WorldUpdater):
    """
    Updates compute attributes for black projects.

    Handles:
    1. Fab production ODE (when operational)
    2. Fab metrics (wafer starts, h100e per chip, production)
    3. Compute stock with attrition
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
        Compute continuous contributions to d(state)/dt for black project compute.

        Computes derivatives for fab-produced compute stock using the average-age
        attrition model.

        For fab-produced compute:
            dC_fab/dt = F - h(a_fab)*C_fab
            da_fab/dt = 1 - (a_fab*F)/C_fab

        where:
            F = annual fab production rate (H100e/year)
            h(a) = h0 + h1*a (hazard rate at average age)
            C_fab = fab compute stock
            a_fab = average age of fab chips
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

            # Get black_project_start_year (when fab construction starts)
            black_project_start_year = float(
                project.preparation_start_year.item()
                if hasattr(project.preparation_start_year, 'item')
                else project.preparation_start_year
            )

            # Check if fab is operational (can't rely on metric during ODE integration)
            fab_was_built = project.fab_number_of_lithography_scanners > 0

            fab_is_operational = False
            if fab_was_built:
                # Calculate fab construction duration from state variables
                target_wafer_starts = calculate_fab_wafer_starts_per_month(
                    fab_operating_labor=project.fab_operating_labor,
                    fab_number_of_lithography_scanners=project.fab_number_of_lithography_scanners,
                    wafers_per_month_per_worker=prc_compute.fab_wafers_per_month_per_operating_worker,
                    wafers_per_month_per_scanner=prc_compute.wafers_per_month_per_lithography_scanner,
                    labor_productivity_multiplier=prc_compute.fab_labor_productivity_multiplier,
                    scanner_productivity_multiplier=prc_compute.fab_scanner_productivity_multiplier,
                )
                fab_construction_duration = calculate_fab_construction_duration(
                    fab_construction_labor=project.fab_construction_labor,
                    target_wafer_starts_per_month=target_wafer_starts,
                    prc_compute_params=prc_compute,
                )
                fab_operational_year = black_project_start_year + fab_construction_duration
                fab_is_operational = current_time >= fab_operational_year

            # Only compute derivatives if fab is operational
            if fab_is_operational:
                # Calculate fab metrics inline
                wafer_starts = calculate_fab_wafer_starts_per_month(
                    fab_operating_labor=project.fab_operating_labor,
                    fab_number_of_lithography_scanners=project.fab_number_of_lithography_scanners,
                    wafers_per_month_per_worker=prc_compute.fab_wafers_per_month_per_operating_worker,
                    wafers_per_month_per_scanner=prc_compute.wafers_per_month_per_lithography_scanner,
                    labor_productivity_multiplier=prc_compute.fab_labor_productivity_multiplier,
                    scanner_productivity_multiplier=prc_compute.fab_scanner_productivity_multiplier,
                )

                # Chip specs fixed at fab construction time (black_project_start_year)
                h100e_per_chip = calculate_fab_h100e_per_chip(
                    fab_process_node_nm=project.fab_process_node_nm,
                    year=black_project_start_year,
                    exogenous_trends=exogenous_trends,
                )

                # Calculate annual fab production rate
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
                dC_fab_dt = calculate_compute_derivative(
                    functional_compute=fab_compute,
                    average_age=fab_avg_age,
                    production_rate=annual_production,
                    initial_hazard_rate=survival_params.effective_initial_hazard_rate,
                    hazard_rate_increase_per_year=survival_params.effective_hazard_rate_increase,
                )

                da_fab_dt = calculate_average_age_derivative(
                    functional_compute=fab_compute,
                    average_age=fab_avg_age,
                    production_rate=annual_production,
                )

                # Set derivatives
                d_project._set_frozen_field('fab_compute_stock_h100e', dC_fab_dt)
                d_project._set_frozen_field('fab_compute_average_age_years', da_fab_dt)
                d_project._set_frozen_field('fab_total_produced_h100e', annual_production)

        return StateDerivative(d_world)

    def set_state_attributes(self, t: Tensor, world: World) -> World | None:
        """
        Apply discrete state changes for black project compute.

        Triggers:
        1. Fab becomes operational when construction completes
        """
        current_time = t.item() if isinstance(t, Tensor) else float(t)
        changed = False

        for _, project in world.black_projects.items():
            # Check if fab should become operational
            if not project.fab_is_operational and project.fab_number_of_lithography_scanners > 0:
                black_project_start_year = float(
                    project.preparation_start_year.item()
                    if hasattr(project.preparation_start_year, 'item')
                    else project.preparation_start_year
                )
                fab_operational_year = black_project_start_year + project.fab_construction_duration
                if current_time >= fab_operational_year:
                    project._set_frozen_field('fab_is_operational', True)
                    changed = True

        return world if changed else None

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """
        Compute derived compute metrics for black projects.

        Updates:
        - Fab metrics (wafer_starts, h100e_per_chip, watts_per_chip, production)
        - Compute stock (survival, total compute)
        - Operating compute (limited by datacenter capacity)
        """
        current_time = t.item() if isinstance(t, Tensor) else float(t)
        prc_compute = self.compute_params.PRCComputeParameters
        prc_energy = self.energy_params.prc_energy_consumption
        survival_params = self.compute_params.survival_rate_parameters
        exogenous_trends = self.compute_params.exogenous_trends

        for _, project in world.black_projects.items():
            black_project_start_year = float(
                project.preparation_start_year.item()
                if hasattr(project.preparation_start_year, 'item')
                else project.preparation_start_year
            )
            years_since_project_start = current_time - black_project_start_year

            # --- Fab metrics ---
            project._set_frozen_field('fab_wafer_starts_per_month', calculate_fab_wafer_starts_per_month(
                fab_operating_labor=project.fab_operating_labor,
                fab_number_of_lithography_scanners=project.fab_number_of_lithography_scanners,
                wafers_per_month_per_worker=prc_compute.fab_wafers_per_month_per_operating_worker,
                wafers_per_month_per_scanner=prc_compute.wafers_per_month_per_lithography_scanner,
                labor_productivity_multiplier=prc_compute.fab_labor_productivity_multiplier,
                scanner_productivity_multiplier=prc_compute.fab_scanner_productivity_multiplier,
            ))

            # Chip specs fixed at fab construction time (black_project_start_year)
            project._set_frozen_field('fab_h100e_per_chip', calculate_fab_h100e_per_chip(
                fab_process_node_nm=project.fab_process_node_nm,
                year=black_project_start_year,
                exogenous_trends=exogenous_trends,
            ))

            project._set_frozen_field('fab_watts_per_chip', calculate_fab_watts_per_chip(
                fab_process_node_nm=project.fab_process_node_nm,
                year=black_project_start_year,
                exogenous_trends=exogenous_trends,
            ))

            # Fab production metrics
            if project.fab_is_operational:
                monthly_h100e = (
                    project.fab_wafer_starts_per_month *
                    project.fab_chips_per_wafer *
                    project.fab_h100e_per_chip
                )
                project.fab.monthly_compute_production._set_frozen_field('tpp_h100e_including_attrition', monthly_h100e)
                project.fab.monthly_compute_production._set_frozen_field('functional_tpp_h100e', monthly_h100e)

            # --- Compute stock ---
            initial_diverted = project.initial_diverted_compute_h100e

            # Survival rate for initial diverted compute
            chip_years_of_life = max(0.0, years_since_project_start)
            survival_rate = calculate_survival_rate(
                years_since_acquisition=chip_years_of_life,
                initial_hazard_rate=survival_params.effective_initial_hazard_rate,
                hazard_rate_increase_per_year=survival_params.effective_hazard_rate_increase,
            )

            surviving_initial = initial_diverted * survival_rate
            project._set_frozen_field('initial_compute_surviving_h100e', surviving_initial)

            # Fab compute from ODE state
            fab_compute = project.fab_compute_stock_h100e

            # Fab production metrics for reporting
            monthly_fab_production = 0.0
            if project.fab_is_operational:
                annual_production = calculate_fab_annual_production_h100e(
                    fab_wafer_starts_per_month=project.fab_wafer_starts_per_month,
                    fab_chips_per_wafer=project.fab_chips_per_wafer,
                    fab_h100e_per_chip=project.fab_h100e_per_chip,
                    fab_is_operational=True,
                )
                monthly_fab_production = annual_production / 12.0

            project._set_frozen_field('fab_cumulative_production_h100e', project.fab_total_produced_h100e)
            project._set_frozen_field('fab_monthly_production_h100e', monthly_fab_production)

            # Total compute
            total_compute = surviving_initial + fab_compute
            functional_compute = total_compute

            # Overall survival rate
            total_ever_added = initial_diverted + project.fab_total_produced_h100e
            overall_survival_rate = total_compute / total_ever_added if total_ever_added > 0 else 1.0
            project._set_frozen_field('survival_rate', overall_survival_rate)

            # Update compute stock metric
            project.compute_stock._set_frozen_field('tpp_h100e_including_attrition', total_compute)
            project.compute_stock._set_frozen_field('functional_tpp_h100e', functional_compute)

            # --- Operating compute (limited by datacenter capacity) ---
            # Get energy values (computed by BlackProjectDatacenterUpdater)
            total_energy_gw = getattr(project, 'total_compute_energy_gw', 0.0)
            total_capacity_gw = project.datacenters.data_center_capacity_gw if project.datacenters else 0.0

            if total_energy_gw <= 0:
                operating_compute = 0.0
            elif total_energy_gw <= total_capacity_gw:
                operating_compute = functional_compute
            else:
                # Scale down proportionally to fit within capacity
                operating_compute = functional_compute * (total_capacity_gw / total_energy_gw)
            project._set_frozen_field('operating_compute_tpp_h100e', operating_compute)

        return world
