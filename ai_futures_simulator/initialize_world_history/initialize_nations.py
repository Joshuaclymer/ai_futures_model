"""
Nation initialization.

Initializes Nation entities for a given year.
"""

from classes.world.entities import Nation, NamedNations, AISoftwareDeveloper
from classes.world.assets import Compute, Fabs, Datacenters
from parameters.simulation_parameters import SimulationParameters


def initialize_usa(
    params: SimulationParameters,
    year: int,
    ai_software_developers: list = None,
) -> Nation:
    """
    Initialize the USA nation for a given year.

    Args:
        params: SimulationParameters containing model configuration
        year: The integer year to initialize for (e.g., 2026)
        ai_software_developers: List of AISoftwareDeveloper entities for this nation

    Returns:
        Initialized Nation for the USA
    """
    if ai_software_developers is None:
        ai_software_developers = []

    # Get the leading AI developer (first one in the list if available)
    leading_developer = ai_software_developers[0] if ai_software_developers else None

    # Compute total operating compute from all developers
    total_compute = 0.0
    if ai_software_developers:
        for dev in ai_software_developers:
            for compute in dev.operating_compute:
                total_compute += compute.functional_tpp_h100e

    # Create compute stock (national level)
    compute_stock = Compute(
        all_tpp_h100e=total_compute,
        functional_tpp_h100e=total_compute,
        watts_per_h100e=700.0,
        average_functional_chip_age_years=0.0,
    )

    # Create empty fabs (not modeling fab production for US currently)
    fabs = Fabs(monthly_production_compute=Compute(
        all_tpp_h100e=0.0,
        functional_tpp_h100e=0.0,
        watts_per_h100e=700.0,
        average_functional_chip_age_years=0.0,
    ))

    # Create datacenters (energy capacity based on compute)
    # Assuming 700W per H100e, convert to GW
    datacenter_capacity_gw = (total_compute * 700.0) / 1e9 if total_compute > 0 else 0.0
    datacenters = Datacenters(data_center_capacity_gw=datacenter_capacity_gw)

    return Nation(
        id=NamedNations.USA,
        ai_software_developers=ai_software_developers,
        fabs=fabs,
        compute_stock=compute_stock,
        datacenters=datacenters,
        total_energy_consumption_gw=datacenter_capacity_gw,  # Simplified: just AI compute
        leading_ai_software_developer=leading_developer,
        operating_compute_tpp_h100e=total_compute,
    )
