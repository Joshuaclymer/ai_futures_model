"""
Simulation parameters dataclasses.

Contains SimulationSettings and SimulationParameters - the top-level
parameter structures that can hold either distributions or concrete values.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np

from parameters.classes.base_spec import BaseSpec, ParamValue
from parameters.classes.software_r_and_d_parameters import SoftwareRAndDParameters
from parameters.classes.compute_parameters import ComputeParameters
from parameters.classes.policy_parameters import PolicyParameters
from parameters.classes.data_center_and_energy_parameters import DataCenterAndEnergyParameters
from parameters.classes.black_project_parameters import BlackProjectParameters
from parameters.classes.perceptions_parameters import PerceptionsParameters
from parameters.classes.ai_researcher_headcount_parameters import AIResearcherHeadcountParameters

if TYPE_CHECKING:
    from parameters.calibrate import CalibratedParameters


@dataclass
class SimulationSettings(BaseSpec):
    """Settings for simulation execution."""
    simulation_start_year: ParamValue = None
    simulation_end_year: ParamValue = None
    n_eval_points: ParamValue = None
    ode_rtol: ParamValue = None
    ode_atol: ParamValue = None
    ode_max_step: ParamValue = None


@dataclass
class SimulationParameters(BaseSpec):
    """
    Top-level simulation configuration for a single simulation run.

    Contains:
    - Simulation settings (start year, end year, eval points)
    - Model parameters for each component
    - Calibrated parameters (computed from software_r_and_d after sampling)
    """
    settings: Optional[SimulationSettings] = None
    software_r_and_d: Optional[SoftwareRAndDParameters] = None
    compute: Optional[ComputeParameters] = None
    datacenter_and_energy: Optional[DataCenterAndEnergyParameters] = None
    policy: Optional[PolicyParameters] = None
    ai_researcher_headcount: Optional[AIResearcherHeadcountParameters] = None
    black_project: Optional[BlackProjectParameters] = None
    perceptions: Optional[PerceptionsParameters] = None

    # Calibrated parameters - computed after sampling, before simulation
    # This is set by ModelParameters.sample() after sampling the raw parameters
    software_r_and_d_calibrated: Optional["CalibratedParameters"] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationParameters":
        return cls(
            settings=SimulationSettings.from_dict(d["settings"]) if "settings" in d else None,
            software_r_and_d=SoftwareRAndDParameters.from_dict(d["software_r_and_d"]) if "software_r_and_d" in d else None,
            compute=ComputeParameters.from_dict(d["compute"]) if "compute" in d else None,
            datacenter_and_energy=DataCenterAndEnergyParameters.from_dict(d["datacenter_and_energy"]) if "datacenter_and_energy" in d else None,
            policy=PolicyParameters.from_dict(d["policy"]) if "policy" in d else None,
            ai_researcher_headcount=AIResearcherHeadcountParameters.from_dict(d["ai_researcher_headcount"]) if d.get("ai_researcher_headcount") else None,
            black_project=BlackProjectParameters.from_dict(d["black_project"]) if d.get("black_project") else None,
            perceptions=PerceptionsParameters.from_dict(d["perceptions"]) if d.get("perceptions") else None,
        )

    def sample(self, rng: np.random.Generator) -> "SimulationParameters":
        return SimulationParameters(
            settings=self.settings.sample(rng) if self.settings else None,
            software_r_and_d=self.software_r_and_d.sample(rng) if self.software_r_and_d else None,
            compute=self.compute.sample(rng) if self.compute else None,
            datacenter_and_energy=self.datacenter_and_energy.sample(rng) if self.datacenter_and_energy else None,
            policy=self.policy.sample(rng) if self.policy else None,
            ai_researcher_headcount=self.ai_researcher_headcount.sample(rng) if self.ai_researcher_headcount else None,
            black_project=self.black_project.sample(rng) if self.black_project else None,
            perceptions=self.perceptions.sample(rng) if self.perceptions else None,
        )

    def get_modal(self) -> "SimulationParameters":
        return SimulationParameters(
            settings=self.settings.get_modal() if self.settings else None,
            software_r_and_d=self.software_r_and_d.get_modal() if self.software_r_and_d else None,
            compute=self.compute.get_modal() if self.compute else None,
            datacenter_and_energy=self.datacenter_and_energy.get_modal() if self.datacenter_and_energy else None,
            policy=self.policy.get_modal() if self.policy else None,
            ai_researcher_headcount=self.ai_researcher_headcount.get_modal() if self.ai_researcher_headcount else None,
            black_project=self.black_project.get_modal() if self.black_project else None,
            perceptions=self.perceptions.get_modal() if self.perceptions else None,
        )
