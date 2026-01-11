"""
Combined AI researcher updater that orchestrates all researcher-related updates.

Updates in order:
1. Nation researcher headcount (exponential growth)
2. AI software developer researchers (fraction of nation's researchers)
3. Black project researchers (if enabled)
"""

import torch.nn as nn
from torch import Tensor

from classes.world.world import World
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.classes import SimulationParameters

from world_updaters.ai_researchers.nation_researchers import NationResearcherUpdater
from world_updaters.ai_researchers.ai_sw_developer_ai_researchers import AISoftwareDeveloperResearcherUpdater


class AIResearcherUpdater(WorldUpdater):
    """
    Combined updater for all AI researcher-related state and metrics.

    Orchestrates updates in the correct order:
    1. NationResearcherUpdater - updates nation researcher headcount growth
    2. AISoftwareDeveloperResearcherUpdater - allocates researchers to AI developers

    Note: BlackProjectResearcherUpdater is NOT included here because it has
    conditional enablement. It should be added to the CombinedUpdater separately
    when black projects are enabled.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

        # Create sub-updaters in execution order
        self.nation_researcher_updater = NationResearcherUpdater(params)
        self.ai_sw_developer_researcher_updater = AISoftwareDeveloperResearcherUpdater(params)

        # Register as submodules for proper parameter tracking
        self._updaters = nn.ModuleList([
            self.nation_researcher_updater,
            self.ai_sw_developer_researcher_updater,
        ])

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Combine state derivative contributions from researcher updaters.

        NationResearcherUpdater contributes derivatives for exponential growth.
        Other updaters only set metric attributes.
        """
        total_derivative = StateDerivative.zeros(world)

        # Nation researchers contribute derivatives for exponential growth
        total_derivative = total_derivative + self.nation_researcher_updater.contribute_state_derivatives(t, world)

        return total_derivative

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """
        Update researcher metrics in order.

        Order matters:
        1. Nation researcher metrics
        2. AI software developer researchers (fraction of nation's researchers)
        """
        # 1. Update nation researcher metrics
        world = self.nation_researcher_updater.set_metric_attributes(t, world)

        # 2. Update AI software developer researchers
        world = self.ai_sw_developer_researcher_updater.set_metric_attributes(t, world)

        return world

    def set_state_attributes(self, t: Tensor, world: World) -> World | None:
        """
        Apply discrete state changes from researcher updaters.

        Returns None if no changes were made.
        """
        # Currently no discrete state changes for researchers
        return None
