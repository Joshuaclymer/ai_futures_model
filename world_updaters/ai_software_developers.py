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

        # Otherwise use modeled growth with slowdown
        cg = self.params.compute_growth
        if current_time < cg.slowdown_year:
            return cg.constant_training_compute_growth_rate
        else:
            return cg.post_slowdown_training_compute_growth_rate

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

        Updates:
        - d(log_compute)/dt = training_compute_growth_rate * ln(10)
        - d(log_researchers)/dt = researcher_growth_rate
        """
        d_world = World.zeros(world)

        current_time = t.item() if isinstance(t, Tensor) else float(t)

        # Get growth rates
        compute_growth_rate = self.get_training_compute_growth_rate(current_time)
        researcher_growth_rate = self.get_researcher_growth_rate(current_time)

        # Convert OOMs/year to natural log growth rate for compute
        log_compute_growth_rate = compute_growth_rate * 2.302585  # ln(10)

        for dev_id in world.ai_software_developers:
            d_dev = d_world.ai_software_developers[dev_id]
            d_dev.log_compute = torch.tensor(log_compute_growth_rate)
            d_dev.log_researchers = torch.tensor(researcher_growth_rate)

        return StateDerivative(d_world)

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """Update compute and researcher count from log values."""
        for dev_id, dev in world.ai_software_developers.items():
            dev.compute.total_tpp_h100e = float(torch.exp(dev.log_compute).item())
            dev.human_ai_capability_researchers = int(torch.exp(dev.log_researchers).item())

        return world
