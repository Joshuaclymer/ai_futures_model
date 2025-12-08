from typing import List, Optional
from entity import Entity
from assets import Assets, UnmannedWeapons
from entities import AISoftwareDeveloper
from enum import Enum

class Attack:
    """Represents an attack action taken by an entity."""
    pass

class CyberAttack(Attack):
    operational_capacity_level_of_attack : "OperationalCapacityLevel"
    attack_success : bool
    exfiltrate_weights_of_developer : Optional[AISoftwareDeveloper] = None
    hijack_developer : Optional[AISoftwareDeveloper] = None
    hijack_what_fraction_of_compute : Optional[float] = None
    sabotage_developer : Optional[AISoftwareDeveloper] = None

class OperationalCapacityLevel(Enum):
    OC1 = "attacker_at_RAND_Operational_Capacity_level_1"
    OC2 = "attacker_at_RAND_Operational_Capacity_level_2"
    OC3 = "attacker_at_RAND_Operational_Capacity_level_3"
    OC4 = "attacker_at_RAND_Operational_Capacity_level_4"
    OC5 = "attacker_at_RAND_Operational_Capacity_level_5"
    OC6 = "attacker_at_RAND_Operational_Capacity_level_6"

class SoftwareSecurityLevel(Enum): # Software Security Defense Levels
    OC1_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_1"
    OC2_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_2"
    OC3_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_3"
    OC4_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_4"
    OC5_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_5"
    OC6_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_6"

class KineticAttack(Attack):
    target_entity: Entity
    target_assets: Assets
    strike_vehicles: List[UnmannedWeapons]

class UntargetedNovelBioAttack(Attack):
    pass # This attack reduces the human population of every nation by 20x

class PoliticalInfluenceOperation(Attack):
    target_entity: Entity
    proportion_of_population_persuadable_absent_AI_defense: float # (represents the strength of the attack)