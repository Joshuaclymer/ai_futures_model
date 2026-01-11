"""
Energy consumption parameters for the AI Futures Simulator.

Contains parameters related to energy efficiency and energy consumption
for AI compute infrastructure.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from parameters.classes.base_spec import BaseSpec, ParamValue


@dataclass
class PRCDataCenterAndEnergyParameters(BaseSpec):
    energy_efficiency_of_compute_stock_relative_to_state_of_the_art: ParamValue = None
    total_prc_energy_consumption_gw: ParamValue = None
    data_center_mw_per_year_per_construction_worker: ParamValue = None
    data_center_mw_per_operating_worker: ParamValue = None
    h100_power_watts: ParamValue = None


@dataclass
class DataCenterAndEnergyParameters(BaseSpec):
    prc_energy_consumption: Optional[PRCDataCenterAndEnergyParameters] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataCenterAndEnergyParameters":
        return cls(
            prc_energy_consumption=PRCDataCenterAndEnergyParameters.from_dict(d["prc_energy_consumption"]) if "prc_energy_consumption" in d else None,
        )

    def sample(self, rng: np.random.Generator) -> "DataCenterAndEnergyParameters":
        return DataCenterAndEnergyParameters(
            prc_energy_consumption=self.prc_energy_consumption.sample(rng) if self.prc_energy_consumption else None,
        )

    def get_modal(self) -> "DataCenterAndEnergyParameters":
        return DataCenterAndEnergyParameters(
            prc_energy_consumption=self.prc_energy_consumption.get_modal() if self.prc_energy_consumption else None,
        )
