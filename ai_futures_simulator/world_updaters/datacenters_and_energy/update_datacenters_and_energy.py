"""
Datacenters and energy world updater for black projects.

Contains:
- Utility functions for datacenter capacity and operating labor calculations
- BlackProjectDatacenterUpdater: WorldUpdater for black project datacenter metrics

Updates datacenter capacity and energy consumption for black projects:
- Concealed datacenter capacity (linear growth)
- Total datacenter capacity (concealed + unconcealed)
- Operating labor for datacenters
- Energy consumption by compute source (initial stock, fab production)
"""

from torch import Tensor

from classes.world.world import World
from classes.simulation_primitives import WorldUpdater
from parameters.classes import SimulationParameters
from parameters.classes import BlackProjectParameters
from parameters.classes import DataCenterAndEnergyParameters


# =============================================================================
# DATACENTER UTILITY FUNCTIONS
# =============================================================================

def calculate_datacenter_capacity_gw(
    unconcealed_capacity_gw: float,
    concealed_capacity_gw: float,
) -> float:
    """Calculate total datacenter capacity."""
    return unconcealed_capacity_gw + concealed_capacity_gw


def calculate_concealed_capacity_gw(
    current_year: float,
    construction_start_year: float,
    construction_rate_gw_per_year: float,
    max_concealed_capacity_gw: float,
) -> float:
    """Calculate concealed datacenter capacity at a given time (linear growth model)."""
    years_since_start = current_year - construction_start_year
    if years_since_start <= 0:
        return 0.0

    return min(
        construction_rate_gw_per_year * years_since_start,
        max_concealed_capacity_gw
    )


def calculate_datacenter_operating_labor(
    datacenter_capacity_gw: float,
    operating_labor_per_gw: float,
) -> float:
    """Calculate operating labor required for datacenter capacity."""
    return datacenter_capacity_gw * operating_labor_per_gw


# =============================================================================
# BLACK PROJECT DATACENTER UPDATER
# =============================================================================


class BlackProjectDatacenterUpdater(WorldUpdater):
    """
    Updates datacenter and energy metrics for black projects.

    Updates:
    - Datacenter capacity (concealed + unconcealed)
    - Operating labor for datacenters
    - Energy consumption by source (initial stock, fab compute)
    """

    def __init__(
        self,
        params: SimulationParameters,
        black_project_params: BlackProjectParameters,
        energy_params: DataCenterAndEnergyParameters,
    ):
        super().__init__()
        self.params = params
        self.black_project_params = black_project_params
        self.energy_params = energy_params

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """
        Compute datacenter and energy metrics for black projects.

        Updates:
        - datacenters.data_center_capacity_gw (concealed + unconcealed)
        - datacenters_operating_labor_per_gw
        - initial_stock_energy_gw, fab_compute_energy_gw, total_compute_energy_gw
        """
        current_time = t.item() if isinstance(t, Tensor) else float(t)
        prc_energy = self.energy_params.prc_energy_consumption
        props = self.black_project_params.black_project_properties

        for _, project in world.black_projects.items():
            # Get black_project_start_year (when black project and construction starts)
            black_project_start_year = float(
                project.preparation_start_year.item()
                if hasattr(project.preparation_start_year, 'item')
                else project.preparation_start_year
            )

            # --- Datacenter capacity ---
            # Get datacenter construction rate from labor
            gw_per_worker_per_year = prc_energy.data_center_mw_per_year_per_construction_worker / 1000.0
            construction_rate = gw_per_worker_per_year * project.concealed_datacenter_capacity_construction_labor

            # Calculate concealed capacity (linear growth)
            # Datacenter construction starts years_before_black_project_start before the black project starts
            datacenter_construction_start = black_project_start_year - props.years_before_black_project_start_to_begin_datacenter_construction
            concealed_gw = calculate_concealed_capacity_gw(
                current_year=current_time,
                construction_start_year=datacenter_construction_start,
                construction_rate_gw_per_year=construction_rate,
                max_concealed_capacity_gw=project.concealed_max_total_capacity_gw,
            )

            # Total capacity = unconcealed + concealed
            total_capacity_gw = calculate_datacenter_capacity_gw(
                unconcealed_capacity_gw=project.unconcealed_datacenter_capacity_diverted_gw,
                concealed_capacity_gw=concealed_gw,
            )

            # Update datacenters metric
            project.datacenters._set_frozen_field('data_center_capacity_gw', total_capacity_gw)

            # Operating labor per GW
            project._set_frozen_field(
                'datacenters_operating_labor_per_gw',
                prc_energy.data_center_mw_per_operating_worker * 1000.0
            )

            # --- Energy consumption by source ---
            # Get surviving compute values (computed by BlackProjectComputeUpdater)
            surviving_initial = getattr(project, 'initial_compute_surviving_h100e', 0.0)
            fab_compute = project.fab_compute_stock_h100e

            # Energy efficiency for each source
            initial_watts_per_h100e = (
                project.compute_stock.watts_per_h100e
                if project.compute_stock
                else prc_energy.h100_power_watts
            )
            fab_watts_per_h100e = (
                (project.fab_watts_per_chip / project.fab_h100e_per_chip)
                if project.fab_h100e_per_chip > 0
                else 0.0
            )

            # Energy in GW = H100e * watts_per_H100e / 1e9
            initial_energy_gw = (surviving_initial * initial_watts_per_h100e) / 1e9
            fab_energy_gw = (fab_compute * fab_watts_per_h100e) / 1e9
            total_energy_gw = initial_energy_gw + fab_energy_gw

            project._set_frozen_field('initial_stock_energy_gw', initial_energy_gw)
            project._set_frozen_field('fab_compute_energy_gw', fab_energy_gw)
            project._set_frozen_field('total_compute_energy_gw', total_energy_gw)

        return world
