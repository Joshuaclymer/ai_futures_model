from typing import list, dict
from classes.assets import Compute, Assets
from classes.policies import AIPolicy, VerificationCapacity
from classes.software_progress import AISoftwareProgress
from classes.perceptions import Perceptions
from classes.state_economy import StateEconomy
from classes.attacks import Attack, SoftwareSecurityLevel
from classes.takeover_risks import AIAlignmentStatus
from classes.technology_progress import Technologies
from classes.utilities import Utilities

class Entity: # Can be a Coalition or State, Organization, legal person, etc
    id : str
    verification_capacity : VerificationCapacity | None = None
    policies_entity_is_verifying : list[AIPolicy] | None = None
    policies_entity_is_subject_to : list[AIPolicy] | None = None
    assets_under_ownership: list[Assets] | None = None
    perceptions : Perceptions | None = None
    utilities : Utilities | None = None
    current_attacks : list[Attack] | None = None

class Coalition(Entity):
    id: str
    member_states: list["State"]

class NamedCoalitions():
    USA_ALLIES = "USA_Allies"
    USA_RIVALS = "USA_Rivals"
    SLOWDOWN_COOPERATORS = "Slowdown_Cooperators"
    SLOWDOWN_HOLDOUTS = "Slowdown_Holdouts"

class State(Entity):
    id: str
    leading_ai_software_developer : "AISoftwareDeveloper | None" = None
    black_project : "BlackProject | None" = None
    all_entities_under_jurisdiction: list[Entity]

    economy : StateEconomy
    technologies_unlocked : list[Technologies]
    division_of_political_influence : list[Entity]

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
    compute_in_use: list[Compute]
    compute_allocation: list["ComputeAllocation"]
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
    compute_hijacked_by_other_actors: dict[Entity, float]

class AIBlackProject(AISoftwareDeveloper):
    parent_entity: Entity

    human_datacenter_construction_labor : int 
    human_datacenter_operating_labor : int
    human_fab_construction_labor : int
    human_fab_operating_labor : int
    human_ai_capability_researchers : int

    proportion_of_compute_taken_from_parent_entity_initially : float | None
    proportion_of_SME_taken_from_parent_entity_initially : float | None
    proportion_of_unconcealed_datacenters_taken_from_parent_entity_initially : float | None
    proportion_of_researcher_headcount_taken_from_parent_entity_initially : float | None
    max_proportion_of_energy_taken_from_parent_entity_ongoingly: float | None