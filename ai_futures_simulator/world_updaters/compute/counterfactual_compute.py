"""
Counterfactual nation compute updater.

Updates compute stock for counterfactual nations (PRC and USA no-slowdown scenarios)
using simple exponential growth WITHOUT attrition.

These represent hypothetical "what-if" scenarios where no slowdown agreement exists,
used for computing AI R&D reduction ratios.
"""

import math
import torch
from torch import Tensor

from classes.world.world import World
from classes.world.entities import NamedNations
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.classes import SimulationParameters


# List of counterfactual nation IDs to update
COUNTERFACTUAL_NATION_IDS = [
    NamedNations.PRC_COUNTERFACTUAL_NO_SLOWDOWN,
    NamedNations.USA_COUNTERFACTUAL_NO_SLOWDOWN,
]


class CounterfactualComputeUpdater(WorldUpdater):
    """
    Updates compute stock for counterfactual nations.

    Implements simple exponential growth WITHOUT attrition:
        dC/dt = C * ln(g)

    Where:
        C = compute stock
        g = annual growth rate multiplier (from parameters)

    This simpler model is appropriate for hypothetical counterfactual scenarios
    where we don't need to track chip aging/attrition.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

    def _get_growth_rate(self, nation_id: str) -> float:
        """Get annual growth rate multiplier for a counterfactual nation from parameters."""
        if nation_id == NamedNations.PRC_COUNTERFACTUAL_NO_SLOWDOWN:
            return self.params.compute.PRCComputeParameters.annual_growth_rate_of_prc_compute_stock
        elif nation_id == NamedNations.USA_COUNTERFACTUAL_NO_SLOWDOWN:
            return self.params.compute.USComputeParameters.total_us_compute_annual_growth_rate
        return 1.0

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Compute contribution to d(state)/dt for counterfactual nation compute stocks.

        For counterfactuals, we use simple exponential growth:
            dC/dt = C * ln(g)

        This gives exponential growth: C(t) = C_0 * g^t
        """
        d_world = World.zeros(world)

        for nation_id in COUNTERFACTUAL_NATION_IDS:
            nation = world.nations.get(nation_id)
            if nation is None:
                continue

            d_nation = d_world.nations[nation_id]
            growth_rate = self._get_growth_rate(nation_id)

            # Current compute stock
            current_stock = nation.compute_stock.functional_tpp_h100e
            if isinstance(current_stock, Tensor):
                current_stock = current_stock.item()

            # Skip if no compute
            if current_stock <= 0:
                continue

            # Simple exponential growth: dC/dt = C * ln(g)
            dC_dt = current_stock * math.log(growth_rate)

            # Update compute stock derivatives
            d_nation.compute_stock._set_frozen_field('tpp_h100e_including_attrition', torch.tensor(dC_dt))
            d_nation.compute_stock._set_frozen_field('functional_tpp_h100e', torch.tensor(dC_dt))

        return StateDerivative(d_world)

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """Update derived metrics from state for counterfactual nations."""
        for nation_id in COUNTERFACTUAL_NATION_IDS:
            nation = world.nations.get(nation_id)
            if nation is None:
                continue

            # For counterfactuals, operating compute equals total compute
            # (no datacenter capacity constraints in hypothetical scenario)
            if nation.compute_stock:
                functional_compute = nation.compute_stock.functional_tpp_h100e
                if isinstance(functional_compute, Tensor):
                    functional_compute = functional_compute.item()
                nation._set_frozen_field('operating_compute_tpp_h100e', functional_compute)

        return world
