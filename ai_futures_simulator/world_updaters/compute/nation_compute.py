"""
Nation compute stock world updater.

Updates compute stock for nations (US, PRC) using continuous growth dynamics
with chip attrition modeled via a linearly increasing hazard rate.

See world_updaters/compute/chip_survival.py for the mathematical model.
"""

import torch
from torch import Tensor

from classes.world.world import World
from classes.world.entities import NamedNations
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.classes import SimulationParameters
from world_updaters.compute.chip_survival import (
    calculate_derivatives_with_attrition,
    calculate_production_rate_from_growth,
)


class NationComputeUpdater(WorldUpdater):
    """
    Updates compute stock for nations (US and PRC).

    Implements continuous growth with attrition:
        dC/dt = F - h(ā)·C  (production minus attrition)
        dā/dt = 1 - (ā·F)/C  (average age dynamics)

    Where:
        C = functional compute
        ā = average chip age
        F = production rate = C·ln(growth_rate)
        h(ā) = h₀ + h₁·ā (hazard rate at average age)

    See chip_survival.py for detailed derivation.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

    def _get_growth_rate(self, nation_id: str) -> float:
        """Get annual growth rate multiplier for a nation from parameters."""
        if nation_id == NamedNations.PRC:
            return self.params.compute.PRCComputeParameters.annual_growth_rate_of_prc_compute_stock
        elif nation_id == NamedNations.USA:
            return self.params.compute.USComputeParameters.total_us_compute_annual_growth_rate
        return 1.0  # No growth for unknown nations

    def _get_survival_params(self):
        """Get survival rate parameters."""
        return self.params.compute.survival_rate_parameters

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Compute contribution to d(state)/dt for nation compute stocks.

        Updates:
        - dC/dt = F - h(ā)·C (functional compute with attrition)
        - dā/dt = 1 - (ā·F)/C (average age dynamics)
        """
        d_world = World.zeros(world)
        survival_params = self._get_survival_params()

        for nation_id, nation in world.nations.items():
            d_nation = d_world.nations[nation_id]

            # Get nation's compute growth rate from parameters
            growth_rate = self._get_growth_rate(nation_id)

            # Current compute stock
            current_stock = nation.compute_stock.functional_tpp_h100e
            if isinstance(current_stock, Tensor):
                current_stock = current_stock.item()

            # Current average age
            average_age = nation.compute_stock.average_functional_chip_age_years
            if isinstance(average_age, Tensor):
                average_age = average_age.item()

            # Skip if no compute
            if current_stock <= 0:
                continue

            # Calculate derivatives with attrition
            dC_dt, da_dt = calculate_derivatives_with_attrition(
                functional_compute=current_stock,
                average_age=average_age,
                gross_growth_rate=growth_rate,
                initial_hazard_rate=survival_params.initial_annual_hazard_rate,
                hazard_rate_increase_per_year=survival_params.annual_hazard_rate_increase_per_year,
            )

            # Update compute stock derivatives
            d_nation.compute_stock._set_frozen_field('tpp_h100e_including_attrition', torch.tensor(dC_dt))
            d_nation.compute_stock._set_frozen_field('functional_tpp_h100e', torch.tensor(dC_dt))
            d_nation.compute_stock._set_frozen_field('average_functional_chip_age_years', torch.tensor(da_dt))

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

            # Update monthly compute production
            # Production rate F = C·ln(g) where g is the gross growth rate
            # Monthly production = F / 12
            if nation.compute_stock and nation.fabs:
                growth_rate = self._get_growth_rate(nation_id)
                if growth_rate > 1.0:
                    functional_compute = nation.compute_stock.functional_tpp_h100e
                    if isinstance(functional_compute, Tensor):
                        functional_compute = functional_compute.item()

                    # Annual production rate
                    annual_production = calculate_production_rate_from_growth(
                        functional_compute, growth_rate
                    )
                    monthly_production = annual_production / 12.0

                    nation.fabs.monthly_compute_production._set_frozen_field(
                        'tpp_h100e_including_attrition', monthly_production
                    )
                    nation.fabs.monthly_compute_production._set_frozen_field(
                        'functional_tpp_h100e', monthly_production
                    )

        return world
