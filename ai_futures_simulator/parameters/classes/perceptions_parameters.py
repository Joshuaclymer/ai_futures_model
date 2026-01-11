"""
Perceptions parameters for modeling entity beliefs about the world.

Contains parameters for US perceptions of covert AI projects.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from parameters.classes.base_spec import BaseSpec, ParamValue, parse_param_value, sample_param, get_modal_param


@dataclass
class BlackProjectPerceptionsParameters(BaseSpec):
    """Parameters for likelihood ratio calculations."""
    prior_odds_of_covert_project: ParamValue = None
    intelligence_median_error_in_estimate_of_compute_stock: ParamValue = None
    intelligence_median_error_in_estimate_of_fab_stock: ParamValue = None
    intelligence_median_error_in_energy_consumption_estimate_of_datacenter_capacity: ParamValue = None
    intelligence_median_error_in_satellite_estimate_of_datacenter_capacity: ParamValue = None
    mean_detection_time_for_100_workers: ParamValue = None
    mean_detection_time_for_1000_workers: ParamValue = None
    variance_of_detection_time_given_num_workers: ParamValue = None
    detection_threshold: ParamValue = None
    detection_thresholds: ParamValue = None


@dataclass
class PerceptionsParameters(BaseSpec):
    """Parameters for updating entity perceptions about covert projects."""
    update_perceptions: ParamValue = None
    black_project_perception_parameters: Optional[BlackProjectPerceptionsParameters] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerceptionsParameters":
        return cls(
            update_perceptions=parse_param_value(d.get("update_perceptions")),
            black_project_perception_parameters=BlackProjectPerceptionsParameters.from_dict(d["black_project_perception_parameters"]) if "black_project_perception_parameters" in d else None,
        )

    def sample(self, rng: np.random.Generator) -> "PerceptionsParameters":
        return PerceptionsParameters(
            update_perceptions=sample_param(self.update_perceptions, rng, "update_perceptions"),
            black_project_perception_parameters=self.black_project_perception_parameters.sample(rng) if self.black_project_perception_parameters else None,
        )

    def get_modal(self) -> "PerceptionsParameters":
        return PerceptionsParameters(
            update_perceptions=get_modal_param(self.update_perceptions, "update_perceptions"),
            black_project_perception_parameters=self.black_project_perception_parameters.get_modal() if self.black_project_perception_parameters else None,
        )
