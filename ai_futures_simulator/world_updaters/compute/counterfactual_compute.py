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


# Fixed counterfactual growth rates (from reference model ExogenousTrends)
# These are intentionally NOT sampled - they represent baseline counterfactual scenarios
COUNTERFACTUAL_GROWTH_RATES = {
    NamedNations.PRC_COUNTERFACTUAL_NO_SLOWDOWN: 2.2,  # PRC p50 growth rate
    NamedNations.USA_COUNTERFACTUAL_NO_SLOWDOWN: 2.91,  # Largest AI project growth rate
}


class CounterfactualComputeUpdater(WorldUpdater):
    """
    Updates compute stock for counterfactual nations.

    Implements simple exponential growth WITHOUT attrition:
        dC/dt = C * ln(g)

    Where:
        C = compute stock
        g = annual growth rate multiplier

    This simpler model is appropriate for hypothetical counterfactual scenarios
    where we don't need to track chip aging/attrition.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

    def _get_growth_rate(self, nation_id: str) -> float:
        """Get annual growth rate multiplier for a counterfactual nation."""
        return COUNTERFACTUAL_GROWTH_RATES.get(nation_id, 1.0)

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Compute contribution to d(state)/dt for counterfactual nation compute stocks.

        For counterfactuals, we use simple exponential growth:
            dC/dt = C * ln(g)

        This gives exponential growth: C(t) = C_0 * g^t
        """
        d_world = World.zeros(world)

        for nation_id in COUNTERFACTUAL_GROWTH_RATES.keys():
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
        for nation_id in COUNTERFACTUAL_GROWTH_RATES.keys():
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
