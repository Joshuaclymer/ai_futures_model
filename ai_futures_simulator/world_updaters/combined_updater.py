"""
Combined updater that orchestrates all world updaters.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

from classes.world.world import World
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.simulation_parameters import SimulationParameters


class CombinedUpdater(WorldUpdater):
    """
    Orchestrates all world updaters.

    This class:
    - Collects contributions from all updaters for the ODE solver
    - Applies discrete changes at event times
    - Computes metrics after integration

    Which updaters are enabled is controlled by flags in the parameter objects:
    - black_project.properties.run_a_black_project: enables BlackProjectUpdater
    - perceptions.update_perceptions: enables StatePerceptionsOfCovertComputeUpdater
    """

    def __init__(
        self,
        params: SimulationParameters,
        updaters: List[WorldUpdater] = None,
    ):
        super().__init__()
        self.params = params

        # If no updaters provided, create default set based on params
        if updaters is None:
            from world_updaters.software_r_and_d import SoftwareRAndD
            from world_updaters.ai_software_developers import AISoftwareDeveloperUpdater
            updaters = [SoftwareRAndD(params), AISoftwareDeveloperUpdater(params)]

            # Add black project updater if enabled in params
            if (params.black_project is not None and
                params.black_project.properties.run_a_black_project):
                from world_updaters.black_project import BlackProjectUpdater
                updaters.append(BlackProjectUpdater(params, params.black_project))

            # Add perceptions updater if enabled in params
            if (params.perceptions is not None and
                params.perceptions.update_perceptions):
                from world_updaters.state_perceptions_of_covert_compute import (
                    StatePerceptionsOfCovertComputeUpdater
                )
                updaters.append(StatePerceptionsOfCovertComputeUpdater(params, params.perceptions))

        # Register updaters as submodules for proper parameter tracking
        self.updaters = nn.ModuleList(updaters)

        # Store a template world for reconstruction from tensors
        self._world_template = None

    def set_world_template(self, world: World):
        """Set the world template used for tensor reconstruction."""
        self._world_template = world

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """Combine state derivative contributions from all updaters."""
        total_derivative = StateDerivative.zeros(world)
        for updater in self.updaters:
            total_derivative = total_derivative + updater.contribute_state_derivatives(t, world)
        return total_derivative

    def set_state_attributes(self, t: Tensor, world: World) -> World | None:
        """Apply discrete state changes from all updaters. Returns None if no changes."""
        changed = False
        for updater in self.updaters:
            result = updater.set_state_attributes(t, world)
            if result is not None:
                world = result
                changed = True
        return world if changed else None

    def make_event_fn(self, end_time: float):
        """
        Create an event function for odeint_event.

        Returns a function that crosses zero when either:
        1. Any updater's set_state_attributes returns non-None (discrete event)
        2. Time reaches end_time
        """
        def event_fn(t: Tensor, state_tensor: Tensor) -> Tensor:
            # Check if we've reached end time
            time_remaining = end_time - t.item()
            if time_remaining <= 0:
                return torch.tensor(0.0)

            # Check if any updater wants to make a discrete change
            world = World.from_state_tensor(state_tensor, self._world_template)
            result = self.set_state_attributes(t, world)
            if result is not None:
                return torch.tensor(0.0)

            # No event - return positive value (time remaining)
            return torch.tensor(time_remaining)

        return event_fn

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """Compute metrics from all updaters."""
        for updater in self.updaters:
            world = updater.set_metric_attributes(t, world)
        return world

    def forward(self, t: Tensor, state_tensor: Tensor) -> Tensor:
        """
        Compute total d(state)/dt for ODE solver.

        This is the main entry point called by torchdiffeq's odeint.
        """
        if self._world_template is None:
            raise RuntimeError("Must call set_world_template() before forward()")

        world = World.from_state_tensor(state_tensor, self._world_template)
        total_derivative = self.contribute_state_derivatives(t, world)
        return total_derivative.to_state_tensor()
