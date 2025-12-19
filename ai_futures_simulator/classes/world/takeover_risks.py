"""
Takeover risk and AI alignment status classes.
"""

from dataclasses import dataclass
from enum import Enum


class AIPropensities(Enum):
    """Possible AI alignment propensities."""
    WOULD_NOT_AID_TAKEOVER = "ai_wont_aid_takeover"
    CATASTROPHICALLY_MISALIGNED = "misaligned"
    MOSTLY_ALIGNED_TO_COMPANY_MEMBERS = "mostly_aligned_to_company_members"
    MOSTLY_ALIGNED_TO_US_PRESIDENT = "mostly_aligned_to_us_president"
    MOSTLY_ALIGNED_TO_FOREIGN_CYBER_ATTACKERS = "mostly_aligned_to_foreign_cyber_attackers"


@dataclass
class AIAlignmentStatus:
    """Alignment status of an AI system."""
    probability_of_handing_off_to_misaligned_ai_with_current_research: float = 0.0
    probability_of_misalignment_at_ASI_if_current_alignment_compute_expenditure_is_maintained: float = 0.0
    probability_of_human_power_grab_if_ASI_achieved_now: float = 0.0
    current_ai_propensities: AIPropensities = AIPropensities.WOULD_NOT_AID_TAKEOVER
