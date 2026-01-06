"""
Simulation parameters dataclasses.

Contains SimulationSettings and SimulationParameters - pure data structures
with no methods. Factory/sampling logic is in parameters.model_parameters.
"""

from dataclasses import dataclass
from typing import Optional

from parameters.classes.software_r_and_d_parameters import SoftwareRAndDParameters
from parameters.classes.compute_parameters import ComputeParameters
from parameters.classes.policy_parameters import PolicyParameters
from parameters.classes.data_center_and_energy_parameters import DataCenterAndEnergyParameters
from parameters.classes.black_project_parameters import BlackProjectParameters
from parameters.classes.perceptions_parameters import PerceptionsParameters
from parameters.classes.ai_researcher_headcount_parameters import AIResearcherHeadcountParameters


@dataclass
class SimulationSettings:
    """Settings for simulation execution."""
    simulation_start_year: int
    simulation_end_year: float
    n_eval_points: int
    # ODE solver settings
    ode_rtol: float
    ode_atol: float
    ode_max_step: float


@dataclass
class SimulationParameters:
    """
    Top-level simulation configuration for a single simulation run.

    Contains:
    - Simulation settings (start year, end year, eval points)
    - Model parameters for each component:
      - software_r_and_d: AI R&D dynamics
      - compute: Compute growth (exogenous trends, survival rates, US/PRC compute)
      - datacenter_and_energy: Datacenter capacity and energy consumption
      - policy: AI governance and slowdown policies
      - black_project: Covert compute infrastructure (optional)
      - perceptions: Detection/perception parameters (optional)

    Note: simulation_start_year must be a discrete year in the historical data (2012-2026).
    """
    settings: SimulationSettings
    software_r_and_d: SoftwareRAndDParameters
    compute: ComputeParameters
    datacenter_and_energy: DataCenterAndEnergyParameters
    policy: PolicyParameters
    ai_researcher_headcount: Optional[AIResearcherHeadcountParameters] = None
    black_project: Optional[BlackProjectParameters] = None
    perceptions: Optional[PerceptionsParameters] = None
