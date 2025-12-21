"""
AI Software Developer world updater.

Updates compute and researcher headcount for all AI developers in the world.
Uses historical data when simulating before LAST_YEAR_IN_HISTORY,
otherwise uses modeled growth dynamics.
"""

import csv
import math
import torch
from torch import Tensor
from pathlib import Path

from classes.world.world import World
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.simulation_parameters import SimulationParameters
from initialize_world_history.initialize_world_history import LAST_YEAR_IN_HISTORY

# Load historical data from CSV
_DATA_PATH = Path(__file__).resolve().parent.parent / "initialize_world_history" / "initialize_ai_software_developers" / "largest_ai_developer.csv"
_historical_data = {}
with open(_DATA_PATH, 'r') as f:
    rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        year = int(float(row['time']))
        _historical_data[year] = {
            'training_compute_growth_rate': float(row['training_compute_growth_rate']),
            'researchers': float(row['L_HUMAN']),
        }
    # Compute researcher growth rates from year-over-year changes
    for i, row in enumerate(rows):
        year = int(float(row['time']))
        if i > 0:
            prev_researchers = float(rows[i-1]['L_HUMAN'])
            curr_researchers = float(row['L_HUMAN'])
            # Growth rate as d(ln(researchers))/dt
            _historical_data[year]['researcher_growth_rate'] = math.log(curr_researchers / prev_researchers)
        else:
            _historical_data[year]['researcher_growth_rate'] = 0.0


class AISoftwareDeveloperUpdater(WorldUpdater):
    """
    Updates AI software developers (compute and researcher headcount).

    Before LAST_YEAR_IN_HISTORY (2026): uses historical growth rates from data.
    After LAST_YEAR_IN_HISTORY: uses modeled growth dynamics.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

    def get_training_compute_growth_rate(self, current_time: float) -> float:
        """Get training compute growth rate (OOMs/year) at a given time."""
        year = int(current_time)

        # Use historical data if before LAST_YEAR_IN_HISTORY
        if year < LAST_YEAR_IN_HISTORY and year in _historical_data:
            return _historical_data[year]['training_compute_growth_rate']

        # Otherwise use modeled growth rate from compute parameters
        # The new structure has compute.USComputeParameters.us_frontier_project_compute_annual_growth_rate
        us_compute = self.params.compute.USComputeParameters
        return us_compute.us_frontier_project_compute_annual_growth_rate

    def get_researcher_growth_rate(self, current_time: float) -> float:
        """Get researcher growth rate (natural log units/year) at a given time."""
        year = int(current_time)

        # Use historical data if before LAST_YEAR_IN_HISTORY
        if year < LAST_YEAR_IN_HISTORY and year in _historical_data:
            return _historical_data[year]['researcher_growth_rate']

        # After historical period, assume no researcher growth (could be parameterized)
        return 0.0

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Compute contribution to d(state)/dt for AI software developers.

        Note: The new entity structure no longer has log_compute/log_researchers as state.
        Time-varying compute and researcher counts are now read from CSV data by
        the SoftwareRAndD updater. This updater returns empty derivatives.
        """
        d_world = World.zeros(world)
        return StateDerivative(d_world)

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """Update metric attributes for AI software developers.

        Note: With the new structure, compute allocation metrics are computed from
        operating_compute list. This is a no-op for now.
        """
        return world
