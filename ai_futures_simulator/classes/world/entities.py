"""
Entity classes representing actors in the simulation.

Entities are legal entities which can have assets and be subject to policies.
All values must be explicitly set during world initialization - no defaults.
"""

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional, List
from classes.world.tensor_dataclass import TensorDataclass
from classes.world.assets import Assets, Compute, Fabs, Datacenters
from classes.world.software_progress import AISoftwareProgress

@dataclass
class Entity(TensorDataclass): # Can be a Coalition or State, Organization, legal person, etc
    id : str


@dataclass
class Coalition(Entity):
    """A coalition of nations."""
    member_nation_ids: List[str] = field(metadata={'is_state': True})


class NamedCoalitions:
    """Named coalition identifiers."""
    USA_ALLIES = "USA_Allies"
    USA_RIVALS = "USA_Rivals"
    SLOWDOWN_COOPERATORS = "Slowdown_Cooperators"
    SLOWDOWN_HOLDOUTS = "Slowdown_Holdouts"

@dataclass
class Nation(Entity):
    """
    A nation with compute stock tracking.

    Use world_updaters.compute.nation_compute functions for derived values:
    - get_nation_compute_stock_h100e(nation)
    """
    # State: these are the minimal core properties that can be used to update metrics
    ai_software_developers: List["AISoftwareDeveloper"] = field(metadata={'is_state': True})
    fabs: Fabs = field(metadata={'is_state': True})
    compute_stock: Compute = field(metadata={'is_state': True}) # Note, this includes chips that cannot be powered bc there is not enough datacenter capacity
    datacenters : Datacenters = field(metadata={'is_state': True})
    total_energy_consumption_gw: float = field(metadata={'is_state': True}) # Includes all energy, not just AI!

    # Metrics: these are metrics derived from core properties
    leading_ai_software_developer: "AISoftwareDeveloper"
    operating_compute_tpp_h100e: float # This only includes compute that is actually being powered by datacenters


class NamedNations:
    """Named nation identifiers."""
    PRC = "PRC"
    USA = "USA"


@dataclass
class ComputeAllocation:
    """How compute is allocated across different uses."""
    fraction_for_ai_r_and_d_inference: float = field(metadata={'is_state': True})
    fraction_for_ai_r_and_d_training: float = field(metadata={'is_state': True})
    fraction_for_external_deployment: float = field(metadata={'is_state': True})
    fraction_for_alignment_research: float = field(metadata={'is_state': True})
    fraction_for_frontier_training: float = field(metadata={'is_state': True})

@dataclass
class AISoftwareDeveloper(Entity):
    """
    An AI software development organization.

    Contains nested state (via ai_software_progress) that gets integrated.
    State fields have metadata={'is_state': True}.
    """
    # State: these are the minimal core properties that can be used to update metrics
    operating_compute: List[Compute] = field(metadata={'is_state': True})
    compute_allocation: ComputeAllocation = field(metadata={'is_state': True})
    human_ai_capability_researchers: float = field(metadata={'is_state': True})
    ai_software_progress: AISoftwareProgress = field(metadata={'is_state': True})

    # Metrics (computed from state)
    ai_r_and_d_inference_compute_tpp_h100e: float = field(init=False)
    ai_r_and_d_training_compute_tpp_h100e: float = field(init=False)
    external_deployment_compute_tpp_h100e: float = field(init=False)
    alignment_research_compute_tpp_h100e: float = field(init=False)
    frontier_training_compute_tpp_h100e: float = field(init=False)

@dataclass
class AIBlackProject(AISoftwareDeveloper):
    # State: All black project properties at any given point in time 
    parent_nation: Nation = field(metadata={'is_state': True})
    preparation_start_year : float = field(metadata={'is_state': True})

    ## Fab
    fab_process_node_nm: float = field(metadata={'is_state': True})
    fab_number_of_lithography_scanners: float = field(metadata={'is_state': True})
    fab_construction_labor: float = field(metadata={'is_state': True})
    fab_operating_labor: float = field(metadata={'is_state': True})
    fab_chips_per_wafer: float = field(metadata={'is_state': True})

    ## Datacenters
    unconcealed_datacenter_capacity_diverted_gw : float = field(metadata={'is_state': True})
    concealed_datacenter_capacity_construction_labor: float = field(metadata={'is_state': True})
    concealed_max_total_capacity_gw: float = field(metadata={'is_state': True})

    # Metrics

    ## Fab
    fab: Fabs
    fab_construction_duration: float = field(init=False)
    fab_is_operational: bool = field(init=False)
    fab_wafer_starts_per_month: float = field(init=False)
    fab_h100e_per_chip: float = field(init=False)
    fab_watts_per_chip: float = field(init=False)
    
    ## Datacenters
    datacenters: Datacenters = field(init=False)
    datacenters_operating_labor_per_gw: float = field(init=False) # Used to determine detection evidence
    
    ## Compute
    compute_stock: Compute = field(init=False)
    operating_compute_tpp_h100e: float = field(init=False) # includes compute that is both functional and operating in datacenters

    ## Time series metrics (computed per simulation for frontend plots)
    years: List[float] = field(init=False)  # Simulation years
    cumulative_h100_years: List[float] = field(init=False)  # Cumulative covert computation
    operational_compute_h100e_by_year: List[float] = field(init=False)
    survival_rate_by_year: List[float] = field(init=False)
    datacenter_capacity_gw_by_year: List[float] = field(init=False)
    total_black_compute_by_year: List[float] = field(init=False)  # Total covert chip stock (surviving)
    initial_black_compute_by_year: List[float] = field(init=False)  # Initial diverted compute (surviving)
    fab_flow_by_year: List[float] = field(init=False)  # Fab cumulative production

    ## LR components (for detection breakdown)
    lr_prc_accounting: float = field(init=False)  # From chip stock discrepancy
    lr_sme_inventory: float = field(init=False)  # From SME diversion
    lr_satellite_datacenter: float = field(init=False)  # From satellite imagery
    lr_reported_energy_by_year: List[float] = field(init=False)  # Energy consumption LR over time
    lr_other_intel_by_year: List[float] = field(init=False)  # Direct evidence (workers, etc) over time
    cumulative_lr_by_year: List[float] = field(init=False)  # Combined LR over time
    posterior_prob_by_year: List[float] = field(init=False)  # P(project exists) over time

    ## Fab production metrics
    fab_cumulative_production_h100e_by_year: List[float] = field(init=False)
    fab_architecture_efficiency_by_year: List[float] = field(init=False)
    fab_transistor_density_relative_to_h100: float = field(init=False)

    ## Detection outcomes (per threshold)
    sampled_detection_time: float = field(init=False)  # Sampled from composite distribution
