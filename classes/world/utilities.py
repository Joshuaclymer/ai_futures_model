"""
Utility classes for entity preferences.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Utilities:
    """Utility values for different outcomes."""
    utility_of_status_quo: float = 1.0
    utility_of_misaligned_AI_takeover: float = 0.0
    utility_of_foreign_state_AI_takeover: Dict[str, float] = field(default_factory=dict)
    utility_of_domestic_ai_company_power_grab: float = 0.0  # (not the company in question)
