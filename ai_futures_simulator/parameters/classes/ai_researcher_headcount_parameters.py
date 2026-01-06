"""
AI Researcher headcount parameters for modeling researcher dynamics.

Researchers are allocated to nations and then to AI software developers
proportionally (similar to how compute is allocated).
"""

from dataclasses import dataclass, field


@dataclass
class USResearcherParameters:
    """US-specific researcher parameters."""
    initial_ai_researcher_headcount_2025: float
    annual_growth_rate: float
    proportion_of_researchers_in_largest_ai_sw_developer: float


@dataclass
class PRCResearcherParameters:
    """PRC-specific researcher parameters."""
    initial_ai_researcher_headcount_2025: float
    annual_growth_rate: float
    proportion_of_researchers_in_largest_ai_sw_developer: float


@dataclass
class AIResearcherHeadcountParameters:
    """
    Parameters for AI researcher headcount modeling.

    Researchers are modeled at the nation level and then allocated to
    AI software developers proportionally (similar to compute allocation).
    """
    us_researchers: USResearcherParameters
    prc_researchers: PRCResearcherParameters

    # Global parameters
    initial_global_ai_researcher_headcount: float
    annual_growth_rate_of_ai_researcher_headcount: float

    # Legacy parameters (kept for backwards compatibility)
    proportion_of_global_ai_researchers_in_us: float
    proportion_of_global_ai_researchers_in_prc: float
