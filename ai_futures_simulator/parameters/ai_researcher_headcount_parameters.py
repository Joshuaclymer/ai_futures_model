"""
AI Researcher headcount parameters for modeling researcher dynamics.

Researchers are allocated to nations and then to AI software developers
proportionally (similar to how compute is allocated).
"""

from dataclasses import dataclass, field


@dataclass
class USResearcherParameters:
    """US-specific researcher parameters."""
    initial_ai_researcher_headcount_2025: float = 50000.0  # Estimated US AI researchers in 2025
    annual_growth_rate: float = 1.10  # 10% annual growth rate
    proportion_of_researchers_in_largest_ai_sw_developer: float = 0.05  # 5% in largest lab


@dataclass
class PRCResearcherParameters:
    """PRC-specific researcher parameters."""
    initial_ai_researcher_headcount_2025: float = 40000.0  # Estimated PRC AI researchers in 2025
    annual_growth_rate: float = 1.15  # 15% annual growth rate (faster growth)
    proportion_of_researchers_in_largest_ai_sw_developer: float = 0.05  # 5% in largest lab


@dataclass
class AIResearcherHeadcountParameters:
    """
    Parameters for AI researcher headcount modeling.

    Researchers are modeled at the nation level and then allocated to
    AI software developers proportionally (similar to compute allocation).
    """
    us_researchers: USResearcherParameters = field(default_factory=USResearcherParameters)
    prc_researchers: PRCResearcherParameters = field(default_factory=PRCResearcherParameters)

    # Global parameters
    initial_global_ai_researcher_headcount: float = 90000.0  # US + PRC + others
    annual_growth_rate_of_ai_researcher_headcount: float = 1.12  # 12% global growth

    # Legacy parameters (kept for backwards compatibility)
    proportion_of_global_ai_researchers_in_us: float = 0.55  # ~55% in US
    proportion_of_global_ai_researchers_in_prc: float = 0.44  # ~44% in PRC
