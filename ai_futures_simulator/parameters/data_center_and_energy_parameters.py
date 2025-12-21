"""
Energy consumption parameters for the AI Futures Simulator.

Contains parameters related to energy efficiency and energy consumption
for AI compute infrastructure.

NOTE: Default values are NOT stored here. All defaults are in modal_parameters.yaml.
"""

from dataclasses import dataclass

@dataclass
class PRCDataCenterAndEnergyParameters:
    energy_efficiency_of_compute_stock_relative_to_state_of_the_art: float
    total_prc_energy_consumption_gw: float

    # Converting between datacenter capacity and construction labor (for black project)
    data_center_mw_per_year_per_construction_worker: float
    data_center_mw_per_operating_worker: float

    # H100 power consumption in watts (~700W typical for H100)
    h100_power_watts: float = 700.0

@dataclass
class DataCenterAndEnergyParameters:
    prc_energy_consumption: PRCDataCenterAndEnergyParameters
