"""
Nation initialization.

Initializes Nation entities for a given year.
"""

from classes.world.entities import Nation, NamedNations
from parameters.simulation_parameters import SimulationParameters


def initialize_usa(
    params: SimulationParameters,
    year: int,
    leading_ai_developer_id: str = "us_frontier_lab",
) -> Nation:
    """
    Initialize the USA nation for a given year.

    Args:
        params: SimulationParameters containing model configuration
        year: The integer year to initialize for (e.g., 2026)
        leading_ai_developer_id: ID of the leading AI developer in the USA

    Returns:
        Initialized Nation for the USA
    """
    return Nation(
        id=NamedNations.USA,
        leading_ai_software_developer_id=leading_ai_developer_id,
    )
