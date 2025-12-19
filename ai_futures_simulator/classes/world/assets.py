"""
Asset classes representing physical and compute resources.
All values must be explicitly set during world initialization - no defaults.
"""

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from classes.world.tensor_dataclass import TensorDataclass


class ProductionTechnology(Enum):
    CURRENT_TECH = "current_tech"
    GENERAL_ROBOTICS = "general_robotics"
    NANO_TECH = "nanotechnology"


class EnergyGenerationMethod(Enum):
    FOSSIL_FUELS = "fossil_fuels"
    SOLAR = "solar"
    NUCLEAR_FISSION = "nuclear_fission"
    FUSION = "fusion"


class UnmannedWeaponType(Enum):
    DRONES = "drones"
    CONVENTIONAL_INTER_CONTINENTAL_MISSILES = "conventional_missiles"
    NUCLEAR_INTER_CONTINENTAL_MISSILES = "nuclear_missiles"


@dataclass
class Compute(TensorDataclass):
    """Represents the stock of a single type of chip."""
    total_tpp_h100e: float
    total_energy_requirements_watts: float
    number_of_chips: Optional[int] = None
    inter_chip_bandwidth_gbps: Optional[float] = None
    intra_chip_bandwidth_gbps: Optional[float] = None


@dataclass
class Datacenters(TensorDataclass):
    data_center_capacity_gw: float


@dataclass
class Fabs(TensorDataclass):
    monthly_production_tpp_h100e: float
    production_method: ProductionTechnology
    tsmc_process_node_equivalent_in_nm: Optional[float] = None
    number_of_lithography_scanners: Optional[int] = None
    h100_sized_chips_per_wafer: Optional[float] = None
    wafers_per_month_during_operation: Optional[float] = None


@dataclass
class EnergyGeneration(TensorDataclass):
    continuous_power_generation_GW: float
    production_method: ProductionTechnology
    energy_generation_method: EnergyGenerationMethod


@dataclass
class Robots(TensorDataclass):
    embodied_ai_labor_in_human_equivalents: int


@dataclass
class RobotFactories(TensorDataclass):
    monthly_production_robots: int
    production_method: ProductionTechnology


@dataclass
class UnmannedWeapons(TensorDataclass):
    weapon_type: UnmannedWeaponType
    metric_tons_of_explosives: Optional[float] = None


@dataclass
class UnmannedWeaponsFactory(TensorDataclass):
    weapon_type: UnmannedWeaponType
    monthly_production_count: int
    production_method: ProductionTechnology


# =============================================================================
# BLACK PROJECT ASSETS
# =============================================================================

@dataclass
class BlackFabs(Fabs):
    """
    Covert semiconductor fabrication facility for black projects.

    Use world_updaters.compute.black_compute functions for derived values:
    - get_fab_operational_year(fab)
    - get_fab_monthly_production_h100e(fab)
    - get_fab_annual_production_h100e(fab)
    """

    # Whether fab is operational (set by discrete event when construction completes)
    is_operational: bool = False

    # Configuration (set at initialization, does not change)
    process_node_nm: float = 28.0
    construction_start_year: float = 2030.0
    construction_duration: float = 2.0  # years

    # Production parameters (set at initialization based on sampling)
    wafer_starts_per_month: float = 0.0
    h100e_per_chip: float = 0.0  # H100-equivalent per chip produced
    chips_per_wafer: float = 28.0

    # Detection likelihood ratios (set at initialization)
    lr_inventory: float = 1.0
    lr_procurement: float = 1.0


@dataclass
class BlackDatacenters(Datacenters):
    """
    Covert datacenter infrastructure for black projects.

    Use world_updaters.compute.black_compute functions for derived values:
    - get_datacenter_concealed_capacity_gw(dc)
    - get_datacenter_total_capacity_gw(dc)
    - get_datacenter_operating_labor(dc)
    """

    # Concealed capacity (built for hiding) - continuous state
    log_concealed_capacity_gw: Tensor = field(
        default_factory=lambda: torch.tensor(-10.0),  # ~0 GW initially
        metadata={'is_state': True}
    )

    # Unconcealed capacity diverted at agreement start (constant after initialization)
    unconcealed_capacity_gw: float = 0.0

    # Construction parameters
    construction_start_year: float = 2029.0  # Usually 1 year before agreement
    construction_rate_gw_per_year: float = 0.0
    max_total_capacity_gw: float = 55.0  # 5% of PRC's 1100 GW

    # Operating labor per GW
    operating_labor_per_gw: float = 100.0


@dataclass
class BlackCompute(Compute):
    """
    Covert compute stock with attrition for black projects.

    Models compute stock with continuous hazard-based attrition.
    Stock dynamics: d(S)/dt = production - hazard * S

    Use world_updaters.compute.black_compute functions for derived values:
    - get_compute_stock_h100e(compute)
    - get_current_hazard_rate(compute)
    """

    # Log of compute stock for numerical stability
    log_compute_stock: Tensor = field(
        default_factory=lambda: torch.tensor(0.0),
        metadata={'is_state': True}
    )

    # Hazard rate parameters (set at initialization from sampling)
    initial_hazard_rate: float = 0.05  # Per year
    hazard_rate_increase_per_year: float = 0.02  # Per year^2

    # Tracking when compute was added (for hazard rate calculation)
    # This is a weighted average age of the compute stock
    average_age_years: float = 0.0

    # Energy efficiency relative to H100 (for power calculations)
    energy_efficiency_relative_to_h100: float = 0.2
