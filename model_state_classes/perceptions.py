from classes.entities import Entity
from enum import Enum

class LevelsOfAINationalSecurityImportance(Enum): #Peak annual spend in billions USD
    CURRENT = 10
    SPACE_RACE = 60
    WAR_ON_TERROR = 200
    WW2 = 1000

class Perceptions: # Perceptions are with respect to an Entity (State, Coalition, AISoftwareDeveloper, etc)

    # Assessment of near-term AI impact on national security
    ai_national_security_importance : LevelsOfAINationalSecurityImportance

    # Assessment of takeover risk from internal sources
    probability_of_internal_takeover: float

    # Assessment of takeover risk from external sources
    probability_of_external_takeover: float

    # Covert AI project assessments
    distribution_over_entity_covert_compute : dict[Entity, list[float]]
    distribution_over_entity_covert_compute_production_capacity : dict[Entity, list[float]]
    distribution_over_entity_covert_frontier_ai_researcher_headcount : dict[Entity, list[float]]
    probability_entity_has_covert_AI_project : dict[Entity, float]
