"""
Combined updater that orchestrates all world updaters.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, TYPE_CHECKING

from classes.world.world import World
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.simulation_parameters import SimulationParameters

if TYPE_CHECKING:
    from classes.world.flat_world import FlatWorld, FlatStateDerivative


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
            from world_updaters.compute import ComputeUpdater
            updaters = [
                ComputeUpdater(params),  # Handles nation compute, chip survival, and AI sw developer compute
            ]

            # Add software R&D updater only if enabled (calibration is slow)
            # Append at end so it runs AFTER ComputeUpdater, ensuring the time-series
            # interpolated values (used in R&D computation) are stored on the developer
            # entity for serialization rather than the nation-based compute values.
            if (params.software_r_and_d is not None and
                params.software_r_and_d.update_software_progress):
                from world_updaters.software_r_and_d import SoftwareRAndD
                updaters.append(SoftwareRAndD(params))

            # Add black project updater if enabled in params
            if (params.black_project is not None and
                params.black_project.run_a_black_project):
                from world_updaters.black_project import BlackProjectUpdater
                # Get perception params for detection model
                perception_params = None
                if (params.perceptions is not None and
                    hasattr(params.perceptions, 'black_project_perception_parameters')):
                    perception_params = params.perceptions.black_project_perception_parameters
                updaters.append(BlackProjectUpdater(
                    params=params,
                    black_project_params=params.black_project,
                    compute_params=params.compute,
                    energy_params=params.datacenter_and_energy,
                    perception_params=perception_params,
                ))

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


class FlatCombinedUpdater(nn.Module):
    """
    Fast ODE updater that operates directly on flat tensors.

    This is a wrapper around CombinedUpdater that avoids the expensive
    World.from_state_tensor() and to_state_tensor() calls in the ODE loop
    by using FlatWorld and FlatStateDerivative with proxy objects.

    The individual WorldUpdater classes continue to use the same attribute
    access interface (world.nations[id].compute_stock.functional_tpp_h100e)
    which is provided by the proxy objects.
    """

    def __init__(self, combined: CombinedUpdater, flat_world: 'FlatWorld'):
        """
        Args:
            combined: The CombinedUpdater with configured updaters
            flat_world: A FlatWorld template for proxy creation
        """
        super().__init__()
        self.combined = combined
        self._flat_template = flat_world
        self._schema = flat_world._schema
        self._metadata = flat_world._metadata
        self._world_template = flat_world._template

    def forward(self, t: Tensor, state_tensor: Tensor) -> Tensor:
        """
        Compute d(state)/dt using flat tensor operations.

        This avoids cloning the entire World dataclass on every call.
        Instead, we use FlatWorld/FlatStateDerivative with proxy objects
        that map attribute access to tensor indices.
        """
        from classes.world.flat_world import FlatWorld, FlatStateDerivative

        # Create FlatWorld wrapping the state tensor (no clone!)
        flat_world = FlatWorld(
            state_tensor, self._schema, self._metadata, self._world_template
        )

        # Create zero-initialized derivative tensor
        deriv_tensor = torch.zeros_like(state_tensor)
        flat_deriv = FlatStateDerivative(
            deriv_tensor, self._schema, self._metadata, self._world_template
        )

        # Call each updater's contribute_state_derivatives
        # They use the same world.nations[id].attr interface via proxies
        for updater in self.combined.updaters:
            updater_deriv = updater.contribute_state_derivatives(t, flat_world)
            # Add contribution to our flat derivative
            deriv_tensor = deriv_tensor + updater_deriv.to_state_tensor()

        return deriv_tensor

    def set_flat_template(self, flat_world: 'FlatWorld'):
        """Update the flat template (e.g., after a discrete event)."""
        self._flat_template = flat_world
        self._schema = flat_world._schema
        self._metadata = flat_world._metadata
        self._world_template = flat_world._template
