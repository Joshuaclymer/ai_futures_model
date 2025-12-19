"""
Nation compute stock world updater.

Updates compute stock for nations (US, PRC, etc.) using continuous growth dynamics.
"""

import math
import torch
from torch import Tensor
from typing import Dict, TYPE_CHECKING

from classes.world.world import World
from classes.world.entities import NamedNations
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.simulation_parameters import SimulationParameters

if TYPE_CHECKING:
    from classes.world.entities import Nation


# =============================================================================
# NATION UTILITY FUNCTIONS
# =============================================================================

def get_nation_compute_stock_h100e(nation: "Nation") -> float:
    """Get compute stock in H100e TPP for a nation."""
    return float(torch.exp(nation.log_compute_stock).item())


class NationComputeUpdater(WorldUpdater):
    """
    Updates compute stock for nations.

    Implements continuous exponential growth:
        d(log_compute_stock)/dt = log(growth_rate)

    Where growth_rate is the annual multiplier (e.g., 2.2 means 2.2x per year).
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """
        Compute contribution to d(state)/dt for nation compute stocks.

        Updates:
        - d(log_compute_stock)/dt = ln(growth_rate) for each nation
        """
        d_world = World.zeros(world)

        for nation_id, nation in world.nations.items():
            d_nation = d_world.nations[nation_id]

            # Get nation's compute growth rate
            growth_rate = nation.compute_growth_rate

            # Skip if no growth rate set
            if growth_rate <= 0:
                d_nation.log_compute_stock = torch.tensor(0.0)
                continue

            # Compute derivative: d(log(S))/dt = ln(growth_rate)
            # This gives exponential growth: S(t) = S0 * growth_rate^t
            log_growth_rate = math.log(growth_rate)
            d_nation.log_compute_stock = torch.tensor(log_growth_rate)

        return StateDerivative(d_world)

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """Update derived metrics from state."""
        # The compute_stock_h100e property is computed from log_compute_stock
        # No additional metrics to set here
        return world


class NationComputeConfig:
    """
    Configuration for initializing nation compute stocks.

    Based on ExogenousTrends from black_project_parameters.
    """

    # PRC compute configuration
    PRC_COMPUTE_STOCK_2025 = 1e5  # H100e TPP
    PRC_GROWTH_RATE_P50 = 2.2  # Annual multiplier (median)

    # USA compute configuration (based on largest AI project)
    USA_COMPUTE_STOCK_2025 = 120325.0  # H100e TPP
    USA_GROWTH_RATE = 2.91  # Annual multiplier

    # PRC energy infrastructure
    PRC_TOTAL_ENERGY_GW = 1100.0

    @classmethod
    def initialize_nation_compute(
        cls,
        nation_id: str,
        year: float,
        growth_rate: float = None
    ) -> Dict[str, float]:
        """
        Calculate initial compute stock for a nation at a given year.

        Args:
            nation_id: Nation identifier (e.g., "PRC", "USA")
            year: Year to calculate compute stock for
            growth_rate: Annual growth multiplier (if None, uses default)

        Returns:
            Dict with 'compute_stock' and 'growth_rate'
        """
        if nation_id == NamedNations.PRC:
            base_stock = cls.PRC_COMPUTE_STOCK_2025
            rate = growth_rate if growth_rate is not None else cls.PRC_GROWTH_RATE_P50
            years_since_2025 = year - 2025.0
            stock = base_stock * (rate ** years_since_2025)
            return {
                'compute_stock': stock,
                'growth_rate': rate,
                'total_energy_gw': cls.PRC_TOTAL_ENERGY_GW,
            }

        elif nation_id == NamedNations.USA:
            base_stock = cls.USA_COMPUTE_STOCK_2025
            rate = growth_rate if growth_rate is not None else cls.USA_GROWTH_RATE
            years_since_2025 = year - 2025.0
            stock = base_stock * (rate ** years_since_2025)
            return {
                'compute_stock': stock,
                'growth_rate': rate,
                'total_energy_gw': 0.0,  # Not modeled for USA
            }

        else:
            # Default: no compute stock
            return {
                'compute_stock': 0.0,
                'growth_rate': 0.0,
                'total_energy_gw': 0.0,
            }
