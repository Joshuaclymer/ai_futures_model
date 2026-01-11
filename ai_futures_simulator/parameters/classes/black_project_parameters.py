"""
Black project parameters for covert compute infrastructure modeling.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from parameters.classes.base_spec import BaseSpec, ParamValue, parse_param_value, sample_param, get_modal_param


# H100 reference chip (used as baseline for all compute measurements)
H100_PROCESS_NODE_NM = 4.0
H100_RELEASE_YEAR = 2022.0
H100_TRANSISTOR_DENSITY_M_PER_MM2 = 98.28  # Million transistors per mm^2
H100_WATTS_PER_TPP = 0.326493  # Watts per Tera-Parameter-Pass
H100_TPP_PER_CHIP = 2144.0  # Tera-Parameter-Passes per chip


@dataclass
class BlackProjectProperties(BaseSpec):
    """Properties of a black project."""
    # Labor
    total_labor: ParamValue = None
    fraction_of_labor_devoted_to_datacenter_construction: ParamValue = None
    fraction_of_labor_devoted_to_black_fab_construction: ParamValue = None
    fraction_of_labor_devoted_to_black_fab_operation: ParamValue = None
    fraction_of_labor_devoted_to_ai_research: ParamValue = None

    # Diverted resources
    fraction_of_initial_compute_stock_to_divert_at_black_project_start: ParamValue = None
    fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start: ParamValue = None
    fraction_of_lithography_scanners_to_divert_at_black_project_start: ParamValue = None
    max_fraction_of_total_national_energy_consumption: ParamValue = None

    # Fab configuration
    black_fab_min_process_node: ParamValue = None
    prc_localization_year_28nm: ParamValue = None
    prc_localization_year_14nm: ParamValue = None
    prc_localization_year_7nm: ParamValue = None

    # Datacenter construction timing
    years_before_black_project_start_to_begin_datacenter_construction: ParamValue = None


@dataclass
class BlackProjectParameters(BaseSpec):
    """Complete set of black project parameters."""
    run_a_black_project: ParamValue = None
    black_project_start_year: ParamValue = None
    black_project_properties: Optional[BlackProjectProperties] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BlackProjectParameters":
        return cls(
            run_a_black_project=parse_param_value(d.get("run_a_black_project")),
            black_project_start_year=parse_param_value(d.get("black_project_start_year")),
            black_project_properties=BlackProjectProperties.from_dict(d["properties"]) if "properties" in d else None,
        )

    def sample(self, rng: np.random.Generator) -> "BlackProjectParameters":
        return BlackProjectParameters(
            run_a_black_project=sample_param(self.run_a_black_project, rng, "run_a_black_project"),
            black_project_start_year=sample_param(self.black_project_start_year, rng, "black_project_start_year"),
            black_project_properties=self.black_project_properties.sample(rng) if self.black_project_properties else None,
        )

    def get_modal(self) -> "BlackProjectParameters":
        return BlackProjectParameters(
            run_a_black_project=get_modal_param(self.run_a_black_project, "run_a_black_project"),
            black_project_start_year=get_modal_param(self.black_project_start_year, "black_project_start_year"),
            black_project_properties=self.black_project_properties.get_modal() if self.black_project_properties else None,
        )
