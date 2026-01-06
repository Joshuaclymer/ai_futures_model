"""
Nation AI researcher headcount world updater.

Updates researcher headcount for nations (US, PRC) using continuous growth dynamics.
"""

import math
import torch
from torch import Tensor

from classes.world.world import World
from classes.world.entities import NamedNations
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.classes import SimulationParameters


class NationResearcherUpdater(WorldUpdater):
    """
    Updates AI researcher headcount for nations (US and PRC).

    Implements continuous exponential growth:
        d(researchers)/dt = researchers * ln(growth_rate)

    Where growth_rate is the annual multiplier from parameters.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

    def _get_growth_rate(self, nation_id: str) -> float:
        """Get annual researcher growth rate multiplier for a nation from parameters."""
        researcher_params = getattr(self.params, 'ai_researcher_headcount', None)
        if researcher_params is None:
            return 1.0  # No growth if parameters not set

        if nation_id == NamedNations.PRC:
            return researcher_params.prc_researchers.annual_growth_rate
        elif nation_id == NamedNations.USA:
            return researcher_params.us_researchers.annual_growth_rate
        return 1.0  # No growth for unknown nations

    def _get_nation_researcher_count(self, nation_id: str, world: World) -> float:
        """Get the current researcher count for a nation."""
        if nation_id not in world.nations:
            return 0.0

        nation = world.nations[nation_id]
        researcher_count = getattr(nation, 'ai_researcher_headcount', None)
        if researcher_count is None:
            return 0.0

        if isinstance(researcher_count, Tensor):
            return researcher_count.item()
        return float(researcher_count)

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Compute contribution to d(state)/dt for nation researcher headcounts.

        Updates:
        - d(researchers)/dt = researchers * ln(growth_rate) for each nation
        """
        d_world = World.zeros(world)

        for nation_id, nation in world.nations.items():
            # Get nation's researcher growth rate from parameters
            growth_rate = self._get_growth_rate(nation_id)

            # Skip if no meaningful growth
            if growth_rate <= 1.0:
                continue

            # Current researcher count
            current_researchers = self._get_nation_researcher_count(nation_id, world)
            if current_researchers <= 0:
                continue

            # Compute derivative: d(R)/dt = R * ln(growth_rate)
            log_growth_rate = math.log(growth_rate)
            derivative = current_researchers * log_growth_rate

            # Update nation's researcher headcount derivative
            d_nation = d_world.nations[nation_id]
            if hasattr(d_nation, 'ai_researcher_headcount'):
                d_nation._set_frozen_field('ai_researcher_headcount', torch.tensor(derivative))

        return StateDerivative(d_world)

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """Update derived metrics from state (no-op for researchers)."""
        return world
