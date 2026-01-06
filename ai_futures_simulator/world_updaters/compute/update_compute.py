"""
Combined compute updater that orchestrates all compute-related updates.

Updates in order:
1. Nation compute stock (exponential growth)
2. Chip survival (attrition over time)
3. Black project compute (if enabled)
4. AI software developer compute (fraction of nation's functional compute)
"""

import torch
import torch.nn as nn
from torch import Tensor

from classes.world.world import World
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.classes import SimulationParameters

from world_updaters.compute.nation_compute import NationComputeUpdater
from world_updaters.compute.ai_sw_developer_compute import AISoftwareDeveloperComputeUpdater


class ComputeUpdater(WorldUpdater):
    """
    Combined updater for all compute-related state and metrics.

    Orchestrates updates in the correct order:
    1. NationComputeUpdater - updates nation compute stock growth
    2. Chip survival - applies attrition to compute stocks
    3. BlackProjectUpdater - updates covert compute (if enabled, added externally)
    4. AISoftwareDeveloperComputeUpdater - allocates compute to AI developers

    Note: BlackProjectUpdater is NOT included here because it has complex
    initialization requirements and conditional enablement. It should be
    added to the CombinedUpdater separately when enabled.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

        # Create sub-updaters in execution order
        self.nation_compute_updater = NationComputeUpdater(params)
        self.ai_sw_developer_updater = AISoftwareDeveloperComputeUpdater(params)

        # Register as submodules for proper parameter tracking
        self._updaters = nn.ModuleList([
            self.nation_compute_updater,
            self.ai_sw_developer_updater,
        ])

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Combine state derivative contributions from compute updaters.

        Only NationComputeUpdater contributes derivatives (for exponential growth).
        Other updaters only set metric attributes.
        """
        total_derivative = StateDerivative.zeros(world)

        # Nation compute contributes derivatives for exponential growth
        total_derivative = total_derivative + self.nation_compute_updater.contribute_state_derivatives(t, world)

        return total_derivative

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """
        Update compute metrics in order.

        Order matters:
        1. Nation compute metrics (operating compute limited by datacenter capacity)
        2. AI software developer compute (fraction of nation's functional compute)
        """
        # 1. Update nation compute metrics
        world = self.nation_compute_updater.set_metric_attributes(t, world)

        # 2. Update AI software developer compute
        # This uses nation's functional compute to allocate to developers
        world = self.ai_sw_developer_updater.set_metric_attributes(t, world)

        return world

    def set_state_attributes(self, t: Tensor, world: World) -> World | None:
        """
        Apply discrete state changes from compute updaters.

        Returns None if no changes were made.
        """
        changed = False

        result = self.nation_compute_updater.set_state_attributes(t, world)
        if result is not None:
            world = result
            changed = True

        result = self.ai_sw_developer_updater.set_state_attributes(t, world)
        if result is not None:
            world = result
            changed = True

        return world if changed else None
