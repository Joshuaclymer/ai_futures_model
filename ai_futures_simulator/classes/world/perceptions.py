"""
Perception classes for entity beliefs about the world.
"""

from dataclasses import dataclass, field
from typing import Dict
from enum import Enum
from classes.world.assets import ProcessNode
from classes.world.entities import Nation

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
    ai_national_security_importance: LevelsOfAINationalSecurityImportance 

    # Assessment of takeover risk from internal sources
    probability_of_internal_takeover: float

    # Assessment of takeover risk from external sources
    probability_of_external_takeover: float

    # Black project assessments
    black_project_perceptions : dict[Nation, "BlackProjectPerceptions"]

@dataclass
class BlackProjectPerceptions:

    # State
    prior_probability_of_black_project_existence : float = field(metadata={'is_state': True})
    estimate_of_compute_stock_tpp_h100e_at_agreement_start : float = field(metadata={'is_state': True})
    estimate_of_number_of_lithography_machines_at_agreement_start : Dict[ProcessNode, float] = field(metadata={'is_state': True})
    estimate_of_datacenter_capacity_gw_from_satellites_at_agreement_start : float = field(metadata={'is_state': True})
    estimate_of_total_energy_consumption_gw : float = field(metadata={'is_state': True})

    lr_from_compute_accounting : float = field(metadata={'is_state': True})
    lr_from_sme_accounting : float = field(metadata={'is_state': True})
    lr_from_datacenter_accounting : float = field(metadata={'is_state': True})
    cumulative_lr_from_energy_accounting : float = field(metadata={'is_state': True})

    cumulative_lr_from_direct_observations : float = field(metadata={'is_state': True})

    # Metrics
    cumulative_lr : float = field(init=False)
    is_detected : bool = field(init=False)
    probability_of_black_project_existence : float = field(init=False)