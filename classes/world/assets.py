"""
Asset classes representing physical and compute resources.
All values must be explicitly set during world initialization - no defaults.
"""

from dataclasses import dataclass
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
