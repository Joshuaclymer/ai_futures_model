from typing import List, Optional, Dict
from classes.assets import Compute, Assets
from classes.policies import AIPolicy, VerificationCapacity
from classes.software_progress import AISoftwareProgress
from classes.perceptions import Perceptions
from classes.economy import StateEconomy
from classes.attacks import Attack, SoftwareSecurityLevel
from classes.takeover_risks import AIAlignmentStatus
from classes.technology_progress import Technologies
from classes.utilities import Utilities

class Entity: # Can be a Coalition or State, Organization, legal person, etc
    id : str
    verification_capacity : Optional[VerificationCapacity] = None
    policies_entity_is_verifying : Optional[List[AIPolicy]] = None
    policies_entity_is_subject_to : Optional[List[AIPolicy]] = None
    assets_under_ownership: Optional[List[Assets]] = None
    perceptions : Optional[Perceptions] = None
    utilities : Optional[Utilities] = None
    current_attacks : Optional[List[Attack]] = None

class Coalition(Entity):
    id: str
    member_states: List["State"]

class NamedCoalitions():
    USA_ALLIES = "USA_Allies"
    USA_RIVALS = "USA_Rivals"
    SLOWDOWN_COOPERATORS = "Slowdown_Cooperators"
    SLOWDOWN_HOLDOUTS = "Slowdown_Holdouts"

class State(Entity):
    id: str
    leading_ai_software_developer : Optional["AISoftwareDeveloper"] = None
    black_project : Optional["BlackProject"] = None
    all_entities_under_jurisdiction: List[Entity]

    economy : StateEconomy
    technologies_unlocked : List[Technologies]
    division_of_political_influence : List[Entity]

    proportion_of_GDP_spent_on_kinetic_offense: float
    proportion_of_GDP_spent_on_kinetic_defense: float
    proportion_of_population_defended_from_novel_bio_attack : float
    proportion_of_population_defended_from_superhuman_influence_operations : float

class NamedStates():
    PRC = "PRC"
    USA = "USA"

class AISoftwareDeveloper(Entity):
    id: str
    is_primarily_controlled_by_misaligned_AI : bool
    compute_in_use: List[Compute]
    compute_allocation: List["ComputeAllocation"]
    ai_software_progress: AISoftwareProgress
    ai_alignment_status: AIAlignmentStatus
    software_security_level: SoftwareSecurityLevel
    human_ai_capability_researchers : int

class ComputeAllocation:
    fraction_of_compute_used_for_ai_rnd_inference: float
    fraction_of_compute_used_for_ai_rnd_training: float
    fraction_of_compute_used_for_external_deployment: float
    fraction_of_compute_used_for_alignment_research: float
    fraction_of_compute_used_for_frontier_training: float
    compute_hijacked_by_other_actors: Dict[Entity, float]

class AIBlackProject(AISoftwareDeveloper):
    parent_entity: Entity

    human_datacenter_construction_labor : int 
    human_datacenter_operating_labor : int
    human_fab_construction_labor : int
    human_fab_operating_labor : int
    human_ai_capability_researchers : int

    proportion_of_compute_taken_from_parent_entity_initially : Optional[float]
    proportion_of_SME_taken_from_parent_entity_initially : Optional[float]
    proportion_of_unconcealed_datacenters_taken_from_parent_entity_initially : Optional[float]
    proportion_of_researcher_headcount_taken_from_parent_entity_initially : Optional[float] 
    max_proportion_of_energy_taken_from_parent_entity_ongoingly: Optional[float]