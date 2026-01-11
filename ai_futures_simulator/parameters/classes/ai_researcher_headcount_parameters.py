"""
AI Researcher headcount parameters for modeling researcher dynamics.

Researchers are allocated to nations and then to AI software developers
proportionally (similar to how compute is allocated).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from parameters.classes.base_spec import BaseSpec, ParamValue, parse_param_value, sample_param, get_modal_param


@dataclass
class USResearcherParameters(BaseSpec):
    """US-specific researcher parameters."""
    initial_ai_researcher_headcount_2025: ParamValue = None
    annual_growth_rate: ParamValue = None
    proportion_of_researchers_in_largest_ai_sw_developer: ParamValue = None


@dataclass
class PRCResearcherParameters(BaseSpec):
    """PRC-specific researcher parameters."""
    initial_ai_researcher_headcount_2025: ParamValue = None
    annual_growth_rate: ParamValue = None
    proportion_of_researchers_in_largest_ai_sw_developer: ParamValue = None


@dataclass
class AIResearcherHeadcountParameters(BaseSpec):
    """Parameters for AI researcher headcount modeling."""
    us_researchers: Optional[USResearcherParameters] = None
    prc_researchers: Optional[PRCResearcherParameters] = None

    # Global parameters
    initial_global_ai_researcher_headcount: ParamValue = None
    annual_growth_rate_of_ai_researcher_headcount: ParamValue = None
    proportion_of_global_ai_researchers_in_us: ParamValue = None
    proportion_of_global_ai_researchers_in_prc: ParamValue = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AIResearcherHeadcountParameters":
        return cls(
            initial_global_ai_researcher_headcount=parse_param_value(d.get("initial_global_ai_researcher_headcount")),
            annual_growth_rate_of_ai_researcher_headcount=parse_param_value(d.get("annual_growth_rate_of_ai_researcher_headcount")),
            proportion_of_global_ai_researchers_in_us=parse_param_value(d.get("proportion_of_global_ai_researchers_in_us")),
            proportion_of_global_ai_researchers_in_prc=parse_param_value(d.get("proportion_of_global_ai_researchers_in_prc")),
            us_researchers=USResearcherParameters.from_dict(d["us_researchers"]) if "us_researchers" in d else None,
            prc_researchers=PRCResearcherParameters.from_dict(d["prc_researchers"]) if "prc_researchers" in d else None,
        )

    def sample(self, rng: np.random.Generator) -> "AIResearcherHeadcountParameters":
        return AIResearcherHeadcountParameters(
            initial_global_ai_researcher_headcount=sample_param(self.initial_global_ai_researcher_headcount, rng, "initial_global_ai_researcher_headcount"),
            annual_growth_rate_of_ai_researcher_headcount=sample_param(self.annual_growth_rate_of_ai_researcher_headcount, rng, "annual_growth_rate_of_ai_researcher_headcount"),
            proportion_of_global_ai_researchers_in_us=sample_param(self.proportion_of_global_ai_researchers_in_us, rng, "proportion_of_global_ai_researchers_in_us"),
            proportion_of_global_ai_researchers_in_prc=sample_param(self.proportion_of_global_ai_researchers_in_prc, rng, "proportion_of_global_ai_researchers_in_prc"),
            us_researchers=self.us_researchers.sample(rng) if self.us_researchers else None,
            prc_researchers=self.prc_researchers.sample(rng) if self.prc_researchers else None,
        )

    def get_modal(self) -> "AIResearcherHeadcountParameters":
        return AIResearcherHeadcountParameters(
            initial_global_ai_researcher_headcount=get_modal_param(self.initial_global_ai_researcher_headcount, "initial_global_ai_researcher_headcount"),
            annual_growth_rate_of_ai_researcher_headcount=get_modal_param(self.annual_growth_rate_of_ai_researcher_headcount, "annual_growth_rate_of_ai_researcher_headcount"),
            proportion_of_global_ai_researchers_in_us=get_modal_param(self.proportion_of_global_ai_researchers_in_us, "proportion_of_global_ai_researchers_in_us"),
            proportion_of_global_ai_researchers_in_prc=get_modal_param(self.proportion_of_global_ai_researchers_in_prc, "proportion_of_global_ai_researchers_in_prc"),
            us_researchers=self.us_researchers.get_modal() if self.us_researchers else None,
            prc_researchers=self.prc_researchers.get_modal() if self.prc_researchers else None,
        )
