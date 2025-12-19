"""
Energy consumption parameters for the AI Futures Simulator.

Contains parameters related to energy efficiency and energy consumption
for AI compute infrastructure.

NOTE: Default values are NOT stored here. All defaults are in modal_parameters.yaml.
"""

from dataclasses import dataclass


@dataclass
class EnergyConsumptionParameters:
    """
    Parameters for energy consumption and efficiency.

    Models energy requirements for AI compute infrastructure including
    efficiency relative to state-of-the-art and total available capacity.
    """

    # Energy efficiency
    energy_efficiency_of_prc_stock_relative_to_state_of_the_art: float
    architecture_efficiency_improvement_per_year: float

    # Energy consumption capacity
    total_GW_of_PRC_energy_consumption: float

    # Largest AI project energy efficiency
    largest_ai_project_energy_efficiency_improvement_per_year: float
