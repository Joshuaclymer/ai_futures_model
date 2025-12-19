"""
Perception classes for entity beliefs about the world.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum


class LevelsOfAINationalSecurityImportance(Enum):
    """Peak annual spend in billions USD."""
    CURRENT = 10
    SPACE_RACE = 60
    WAR_ON_TERROR = 200
    WW2 = 1000


@dataclass
class Perceptions:
    """
    Perceptions are with respect to an Entity (State, Coalition, AISoftwareDeveloper, etc).
    """
    # Assessment of near-term AI impact on national security
    ai_national_security_importance: LevelsOfAINationalSecurityImportance = (
        LevelsOfAINationalSecurityImportance.CURRENT
    )

    # Assessment of takeover risk from internal sources
    probability_of_internal_takeover: float = 0.0

    # Assessment of takeover risk from external sources
    probability_of_external_takeover: float = 0.0

    # Covert AI project assessments
    distribution_over_entity_covert_compute: Dict[str, List[float]] = field(default_factory=dict)
    distribution_over_entity_covert_compute_production_capacity: Dict[str, List[float]] = field(
        default_factory=dict
    )
    distribution_over_entity_covert_frontier_ai_researcher_headcount: Dict[str, List[float]] = (
        field(default_factory=dict)
    )
    probability_entity_has_covert_AI_project: Dict[str, float] = field(default_factory=dict)
