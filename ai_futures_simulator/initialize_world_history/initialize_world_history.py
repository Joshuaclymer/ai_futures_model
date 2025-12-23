"""
World history initialization.

Combines all entity initializers to create world states for multiple years.
"""

import torch
from typing import Dict

from classes.world.world import World
from classes.world.entities import NamedNations
from parameters.simulation_parameters import SimulationParameters
from initialize_world_history.initialize_nations import initialize_usa, initialize_prc
from initialize_world_history.initialize_ai_software_developers.initialize_ai_software_developers import initialize_us_frontier_lab

FIRST_YEAR_IN_HISTORY = 2012
LAST_YEAR_IN_HISTORY = 2026


def initialize_world_for_year(params: SimulationParameters, year: int) -> World:
    """Initialize the world state for a specific year."""
    # Initialize AI developer
    us_developer = initialize_us_frontier_lab(params, year)

    # Compute total operating compute from developer
    total_compute = sum(c.functional_tpp_h100e for c in us_developer.operating_compute) if us_developer.operating_compute else 0.0

    # Initialize USA with total compute
    usa = initialize_usa(params, year, total_compute=total_compute)

    # Initialize PRC
    prc = initialize_prc(params, year)

    # Create nations dict
    nations = {
        NamedNations.USA: usa,
        NamedNations.PRC: prc,
    }

    # Initialize black projects dict
    black_projects = {}

    # Add black project if enabled (project tracks its own start year internally)
    if (params.black_project is not None and
        params.black_project.run_a_black_project):

        from world_updaters.black_project import initialize_black_project

        # Get PRC compute stock at black project start year for diversion calculation
        bp_start_year = params.black_project.black_project_start_year
        prc_compute_params = params.compute.PRCComputeParameters
        base_compute = prc_compute_params.total_prc_compute_tpp_h100e_in_2025
        growth_rate = prc_compute_params.annual_growth_rate_of_prc_compute_stock
        prc_compute_at_bp_start = base_compute * (growth_rate ** (bp_start_year - 2025))

        # Generate simulation years for detection calculation
        end_year = params.settings.simulation_end_year
        simulation_years = [bp_start_year + i * 0.25 for i in range(int((end_year - bp_start_year) / 0.25) + 1)]

        project, lr_by_year, sampled_detection_time = initialize_black_project(
            project_id="prc_black_project",
            parent_nation=prc,
            black_project_params=params.black_project,
            compute_params=params.compute,
            energy_params=params.datacenter_and_energy,
            perception_params=params.perceptions.black_project_perception_parameters,
            policy_params=params.policy,
            initial_prc_compute_stock=prc_compute_at_bp_start,
            simulation_years=simulation_years,
        )
        black_projects["prc_black_project"] = project

    return World(
        current_time=torch.tensor(float(year)),
        coalitions={},
        nations=nations,
        ai_software_developers={us_developer.id: us_developer},
        ai_policies={},
        black_projects=black_projects,
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
