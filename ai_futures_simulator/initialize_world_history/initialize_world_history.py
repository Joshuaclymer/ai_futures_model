"""
World history initialization.

Combines all entity initializers to create world states for multiple years.
"""

import torch
from typing import Dict

from classes.world.world import World
from classes.world.entities import NamedNations
from parameters.simulation_parameters import SimulationParameters
from initialize_world_history.initialize_nations import initialize_usa
from initialize_world_history.initialize_ai_software_developers.initialize_ai_software_developers import initialize_us_frontier_lab

FIRST_YEAR_IN_HISTORY = 2012
LAST_YEAR_IN_HISTORY = 2026


def initialize_world_for_year(params: SimulationParameters, year: int) -> World:
    """Initialize the world state for a specific year."""
    us_developer = initialize_us_frontier_lab(params, year)
    usa = initialize_usa(params, year, leading_ai_developer_id=us_developer.id)

    return World(
        current_time=torch.tensor(float(year)),
        coalitions={},
        nations={NamedNations.USA: usa},
        ai_software_developers={us_developer.id: us_developer},
        ai_policies={},
    )


def initialize_world_history(
    params: SimulationParameters,
    start_year: int = FIRST_YEAR_IN_HISTORY,
    end_year: int = LAST_YEAR_IN_HISTORY,
) -> Dict[int, World]:
    """Initialize world states for a range of historical years."""
    return {
        year: initialize_world_for_year(params, year)
        for year in range(start_year, end_year + 1)
    }
