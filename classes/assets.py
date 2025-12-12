from typing import list
from classes.policies import VerificationMeasure
from enum import Enum

class Assets():
    asset_status: list["AssetStatus"]
    under_verification_measures: VerificationMeasure | None = None

class AssetStatus(Enum):
    OPERATING = "operating"
    NOT_OPERATING = "not_operating"
    DESTROYED = "destroyed"
    DEGRADED = "degraded"
    ONEARTH = "on_earth"
    InSpace = "in_space"

class ProductionTechnology(Enum):
    CurrentTech = "current_tech"
    GeneralRobotics = "general_robotics"
    NanoTech = "nanotechnology"

class Compute(Assets):
    """Represents the stock of a single type of chip."""
    total_tpp_h100e: float
    total_energy_requirements_watts: float
    number_of_chips: int | None = None
    inter_chip_bandwidth_gbps: float | None = None
    intra_chip_bandwidth_gbps: float | None = None

class Datacenters(Assets):
    data_center_capacity_gw : float

class Fabs(Assets):
    monthly_production: Compute
    production_method: ProductionTechnology

    # Optional, more detailed info
    tsmc_process_node_equivalent_in_nm: float | None
    number_of_lithography_scanners : int | None
    h100_sized_chips_per_wafer : float | None
    wafers_per_month_during_operation: float | None

class EnergyGeneration(Assets):
    continuous_power_generation_GW: float
    production_method: ProductionTechnology
    energy_generation_method: "EnergyGenerationMethod"

class EnergyGenerationMethod(Enum):
    FOSSIL_FUELS = "fossil_fuels"
    SOLAR = "solar"
    NUCLEAR_FISSION = "nuclear_fission"
    FUSION = "fusion"

class Robots(Assets):
    embodied_ai_labor_in_human_equivalents : int

class RobotFactories(Assets):
    monthly_production: Robots
    production_method: ProductionTechnology

class UnmannedWeapons(Assets):
    type : "UnmannedWeaponType"
    metric_tons_of_explosives: float | None

class UnmannedWeaponType(Enum):
    DRONES = "drones"
    CONVENTIONAL_INTER_CONTINENTAL_MISSILES = "missiles"
    NUCLEAR_INTER_CONTINENTAL_MISSILES = "missiles"

class UnmannedWeaponsFactory(Assets):
    type : "UnmannedWeaponType"
    monthly_production: Robots
    production_method: ProductionTechnology

class Stars(Assets):
    sun_equivalents_of_energy: float