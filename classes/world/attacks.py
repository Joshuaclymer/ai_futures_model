"""
Attack classes representing offensive actions.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class OperationalCapacityLevel(Enum):
    """RAND Operational Capacity levels for cyber attacks."""
    OC1 = "attacker_at_RAND_Operational_Capacity_level_1"
    OC2 = "attacker_at_RAND_Operational_Capacity_level_2"
    OC3 = "attacker_at_RAND_Operational_Capacity_level_3"
    OC4 = "attacker_at_RAND_Operational_Capacity_level_4"
    OC5 = "attacker_at_RAND_Operational_Capacity_level_5"
    OC6 = "attacker_at_RAND_Operational_Capacity_level_6"


class SoftwareSecurityLevel(Enum):
    """Software Security Defense Levels."""
    OC1_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_1"
    OC2_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_2"
    OC3_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_3"
    OC4_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_4"
    OC5_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_5"
    OC6_RESILIENT = "resilient_to_attacker_at_RAND_Operational_Capacity_level_6"


@dataclass
class Attack:
    """Base class for attack actions."""
    pass


@dataclass
class CyberAttack(Attack):
    """A cyber attack action."""
    operational_capacity_level: OperationalCapacityLevel = OperationalCapacityLevel.OC1
    attack_success: bool = False
    exfiltrate_weights_of_developer_id: Optional[str] = None
    hijack_developer_id: Optional[str] = None
    hijack_what_fraction_of_compute: Optional[float] = None
    sabotage_developer_id: Optional[str] = None


@dataclass
class KineticAttack(Attack):
    """A kinetic (physical) attack action."""
    target_entity_id: str = ""
    target_asset_ids: List[str] = None
    strike_vehicle_ids: List[str] = None

    def __post_init__(self):
        if self.target_asset_ids is None:
            self.target_asset_ids = []
        if self.strike_vehicle_ids is None:
            self.strike_vehicle_ids = []


@dataclass
class UntargetedNovelBioAttack(Attack):
    """
    An untargeted novel biological attack.
    This attack reduces the human population of every nation.
    """
    population_reduction_factor: float = 20.0


@dataclass
class PoliticalInfluenceOperation(Attack):
    """A political influence operation."""
    target_entity_id: str = ""
    # Represents the strength of the attack
    proportion_of_population_persuadable_absent_AI_defense: float = 0.0
