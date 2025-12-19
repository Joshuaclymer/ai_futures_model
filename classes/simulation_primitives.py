"""
Simulation primitive classes.

Contains base classes and data structures for the simulation framework:
- WorldUpdater: Base class for updating world state
- StateDerivative: Wrapper for d(state)/dt values
- SimulationResult: Results from a simulation run
"""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import List

from classes.world.world import World
from parameters.simulation_parameters import SimulationParameters


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    times: Tensor
    trajectory: List[World]  # World state at each time point
    params: SimulationParameters


@dataclass
class StateDerivative:
    """
    Wrapper indicating the World object contains d(state)/dt values.

    This avoids duplicating the World class for derivatives.
    Multiple StateUpdaters can contribute to the total derivative,
    and their contributions are summed.
    """
    world: World

    def to_state_tensor(self) -> Tensor:
        """Pack derivative values into a tensor."""
        return self.world.to_state_tensor()

    def __add__(self, other: 'StateDerivative') -> 'StateDerivative':
        """Add two state derivatives element-wise."""
        if not isinstance(other, StateDerivative):
            return NotImplemented
        return StateDerivative(self.world + other.world)

    def __radd__(self, other):
        """Support sum() by handling 0 + StateDerivative."""
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, scalar: float) -> 'StateDerivative':
        """Multiply derivative by a scalar."""
        return StateDerivative(self.world * scalar)

    def __rmul__(self, scalar: float) -> 'StateDerivative':
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    @classmethod
    def zeros(cls, template: World = None) -> 'StateDerivative':
        """Create a zero-initialized StateDerivative."""
        return cls(World.zeros(template))

    @classmethod
    def from_world(cls, world: World) -> 'StateDerivative':
        """Create a StateDerivative from a World instance."""
        return cls(world)


class WorldUpdater(nn.Module):
    """
    Base class for world updaters.

    Each WorldUpdater can implement three methods:
    - contribute_state_derivatives: Continuous contribution to d(state)/dt (differentiable)
    - set_state_attributes: Discrete changes to state (triggers event detection)
    - set_metric_attributes: Compute derived metrics from state

    Subclasses should override the methods they need.
    """

    def __init__(self):
        super().__init__()

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Compute continuous contribution to d(state)/dt.

        This method is called by the ODE solver and must be differentiable.

        Returns:
            StateDerivative containing this updater's contribution to d(state)/dt
        """
        return StateDerivative.zeros(world)

    def set_state_attributes(self, t: Tensor, world: World) -> World | None:
        """
        Apply discrete changes to state variables.

        Return None if no changes needed (default). Return a World to trigger
        an event that stops ODE integration, applies the state change, and
        restarts integration from the new state.

        Returns:
            None (no change) or updated World (triggers discrete event)
        """
        return None

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """
        Compute derived metrics from state.

        Called after integration to populate non-state attributes.

        Returns:
            Updated world with computed metrics
        """
        return world
