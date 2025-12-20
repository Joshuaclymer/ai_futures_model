"""
Energy consumption parameters for the AI Futures Simulator.

Contains parameters related to energy efficiency and energy consumption
for AI compute infrastructure.

NOTE: Default values are NOT stored here. All defaults are in modal_parameters.yaml.
"""

from dataclasses import dataclass


@dataclass
class ExogenousEnergyTrends:
    state_of_the_art_energy_efficiency_improvement_per_year: float

@dataclass
class PRCEnergyConsumptionParameters:
    energy_efficiency_of_compute_stock_relative_to_state_of_the_art: float
    total_prc_energy_consumption_gw: float

@dataclass
class EnergyConsumptionParameters:
    exogenous_trends : ExogenousEnergyTrends
    prc_energy_consumption : PRCEnergyConsumptionParameters
