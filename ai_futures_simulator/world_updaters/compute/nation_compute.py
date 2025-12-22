"""
Nation compute stock world updater.

Updates compute stock for nations (US, PRC) using continuous growth dynamics.
Uses actual simulation parameters, not hardcoded values.
"""

import math
import torch
from torch import Tensor

from classes.world.world import World
from classes.world.entities import NamedNations
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.simulation_parameters import SimulationParameters


class NationComputeUpdater(WorldUpdater):
    """
    Updates compute stock for nations (US and PRC).

    Implements continuous exponential growth:
        d(compute_stock)/dt = compute_stock * ln(growth_rate)

    Where growth_rate is the annual multiplier from parameters.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

    def _get_growth_rate(self, nation_id: str) -> float:
        """Get annual growth rate multiplier for a nation from parameters."""
        if nation_id == NamedNations.PRC:
            return self.params.compute.PRCComputeParameters.annual_growth_rate_of_prc_compute_stock
        elif nation_id == NamedNations.USA:
            return self.params.compute.USComputeParameters.us_frontier_project_compute_annual_growth_rate
        return 1.0  # No growth for unknown nations

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Compute contribution to d(state)/dt for nation compute stocks.

        Updates:
        - d(compute_stock)/dt = compute_stock * ln(growth_rate) for each nation
        """
        d_world = World.zeros(world)

        for nation_id, nation in world.nations.items():
            d_nation = d_world.nations[nation_id]

            # Get nation's compute growth rate from parameters
            growth_rate = self._get_growth_rate(nation_id)

            # Skip if no meaningful growth
            if growth_rate <= 1.0:
                continue

            # Current compute stock
            current_stock = nation.compute_stock.functional_tpp_h100e
            if isinstance(current_stock, Tensor):
                current_stock = current_stock.item()

            # Compute derivative: d(S)/dt = S * ln(growth_rate)
            # This gives exponential growth: S(t) = S0 * growth_rate^t
            log_growth_rate = math.log(growth_rate)
            derivative = current_stock * log_growth_rate

            # Update both all and functional compute (assuming same growth)
            d_nation.compute_stock._set_frozen_field('all_tpp_h100e', torch.tensor(derivative))
            d_nation.compute_stock._set_frozen_field('functional_tpp_h100e', torch.tensor(derivative))

        return StateDerivative(d_world)

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """Update derived metrics from state."""
        for nation_id, nation in world.nations.items():
            # Update operating compute (limited by datacenter capacity)
            if nation.compute_stock and nation.datacenters:
                functional_compute = nation.compute_stock.functional_tpp_h100e
                if isinstance(functional_compute, Tensor):
                    functional_compute = functional_compute.item()

                datacenter_gw = nation.datacenters.data_center_capacity_gw
                if isinstance(datacenter_gw, Tensor):
                    datacenter_gw = datacenter_gw.item()

                watts_per_h100e = nation.compute_stock.watts_per_h100e
                if isinstance(watts_per_h100e, Tensor):
                    watts_per_h100e = watts_per_h100e.item()

                # Max compute that can be powered
                if watts_per_h100e > 0:
                    max_operating = (datacenter_gw * 1e9) / watts_per_h100e
                    operating_compute = min(functional_compute, max_operating)
                else:
                    operating_compute = functional_compute

                nation._set_frozen_field('operating_compute_tpp_h100e', operating_compute)

        return world
