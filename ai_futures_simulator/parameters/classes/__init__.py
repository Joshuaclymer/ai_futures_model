"""
Parameter dataclass definitions.

This module exports all parameter dataclasses used by the simulation.
"""

from parameters.classes.software_r_and_d_parameters import SoftwareRAndDParameters
from parameters.classes.compute_parameters import (
    ComputeParameters,
    ExogenousComputeTrends,
    SurvivalRateParameters,
    USComputeParameters,
    PRCComputeParameters,
)
from parameters.classes.policy_parameters import PolicyParameters
from parameters.classes.data_center_and_energy_parameters import (
    DataCenterAndEnergyParameters,
    PRCDataCenterAndEnergyParameters,
)
from parameters.classes.black_project_parameters import (
    BlackProjectParameters,
    BlackProjectProperties,
    H100_PROCESS_NODE_NM,
    H100_RELEASE_YEAR,
    H100_TRANSISTOR_DENSITY_M_PER_MM2,
    H100_WATTS_PER_TPP,
    H100_TPP_PER_CHIP,
)
from parameters.classes.perceptions_parameters import (
    PerceptionsParameters,
    BlackProjectPerceptionsParameters,
)
from parameters.classes.ai_researcher_headcount_parameters import (
    AIResearcherHeadcountParameters,
    USResearcherParameters,
    PRCResearcherParameters,
)
from parameters.classes.simulation_parameters import (
    SimulationSettings,
    SimulationParameters,
)

__all__ = [
    # Software R&D
    "SoftwareRAndDParameters",
    # Compute
    "ComputeParameters",
    "ExogenousComputeTrends",
    "SurvivalRateParameters",
    "USComputeParameters",
    "PRCComputeParameters",
    # Policy
    "PolicyParameters",
    # Datacenter and Energy
    "DataCenterAndEnergyParameters",
    "PRCDataCenterAndEnergyParameters",
    # Black Project
    "BlackProjectParameters",
    "BlackProjectProperties",
    "H100_PROCESS_NODE_NM",
    "H100_RELEASE_YEAR",
    "H100_TRANSISTOR_DENSITY_M_PER_MM2",
    "H100_WATTS_PER_TPP",
    "H100_TPP_PER_CHIP",
    # Perceptions
    "PerceptionsParameters",
    "BlackProjectPerceptionsParameters",
    # AI Researcher Headcount
    "AIResearcherHeadcountParameters",
    "USResearcherParameters",
    "PRCResearcherParameters",
    # Simulation
    "SimulationSettings",
    "SimulationParameters",
]
