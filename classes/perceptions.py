from typing import Dict, List
from classes.entities import Entity
from enum import Enum

class LevelsOfAINationalSecurityImportance(Enum):
    SIMILAR_TO_LEADING_IN_CONSUMER_ELECTRONICS = 1
    SIMILAR_TO_BEATING_RUSSIA_TO_THE_MOON = 2
    SIMILAR_TO_FIGHTING_THE_WAR_ON_TERROR = 3
    SIMILAR_TO_WINNING_WW2 = 4

class Perceptions: # Perceptions are with respect to an Entity (State, Coalition, AISoftwareDeveloper, etc)

    # Assessment of near-term AI impact on national security
    ai_national_security_importance : LevelsOfAINationalSecurityImportance

    # Assessment of takeover risk from internal sources
    probability_of_internal_takeover: float

    # Assessment of takeover risk from external sources
    probability_of_external_takeover: float

    # Covert AI project assessments
    distribution_over_entity_covert_compute : Dict[Entity, List[float]]
    distribution_over_entity_covert_compute_production_capacity : Dict[Entity, List[float]]
    distribution_over_entity_covert_frontier_ai_researcher_headcount : Dict[Entity, List[float]]
    probability_entity_has_covert_AI_project : Dict[Entity, float]
