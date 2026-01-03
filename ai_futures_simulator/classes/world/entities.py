"""
Entity classes representing actors in the simulation.

Entities are legal entities which can have assets and be subject to policies.
All values must be explicitly set during world initialization - no defaults.
"""

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional, List, Dict
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
    fabs: Fabs = field(metadata={'is_state': True})
    compute_stock: Compute = field(metadata={'is_state': True}) # Note, this includes chips that cannot be powered bc there is not enough datacenter capacity
    datacenters : Datacenters = field(metadata={'is_state': True})
    total_energy_consumption_gw: float = field(metadata={'is_state': True}) # Includes all energy, not just AI!

    # Metrics: these are metrics derived from core properties
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
    training_compute_growth_rate: float = field(metadata={'is_state': True})  # OOMs/year contribution to progress

    # Metrics (computed from state, set during initialization and updated by world_updaters)
    # These are init=False to avoid dataclass inheritance issues with default values
    ai_r_and_d_inference_compute_tpp_h100e: float = field(init=False, default=0.0)
    ai_r_and_d_training_compute_tpp_h100e: float = field(init=False, default=0.0)
    external_deployment_compute_tpp_h100e: float = field(init=False, default=0.0)
    alignment_research_compute_tpp_h100e: float = field(init=False, default=0.0)
    frontier_training_compute_tpp_h100e: float = field(init=False, default=0.0)

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

    ## Compute tracking (state variables for attrition model)
    # Initial diverted compute (constant, set at project start, never changes)
    initial_diverted_compute_h100e: float = field(metadata={'is_state': True})
    # Fab-produced compute stock (integrated via ODE, includes attrition)
    fab_compute_stock_h100e: float = field(metadata={'is_state': True})
    # Average age of fab-produced chips (for computing attrition rate)
    fab_compute_average_age_years: float = field(metadata={'is_state': True})
    # Total cumulative H100e ever produced by fab (no attrition, monotonically increasing)
    fab_total_produced_h100e: float = field(metadata={'is_state': True})

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

    ## Fab production metrics (point-in-time, extracted from trajectory for time series)
    fab_cumulative_production_h100e: float = field(init=False)  # Cumulative H100e produced by fab
    fab_monthly_production_h100e: float = field(init=False)  # Current monthly production rate
    fab_architecture_efficiency: float = field(init=False)  # Current architecture efficiency
    fab_transistor_density_relative_to_h100: float = field(init=False)  # Transistor density relative to H100

    ## Survival and compute metrics (point-in-time)
    survival_rate: float = field(init=False)  # Current chip survival rate
    initial_compute_surviving_h100e: float = field(init=False)  # Initial diverted compute * survival rate

    ## Energy consumption by source (point-in-time, for energy stacked charts)
    initial_stock_energy_gw: float = field(init=False)  # Energy consumed by initial diverted compute (GW)
    fab_compute_energy_gw: float = field(init=False)  # Energy consumed by fab-produced compute (GW)
    total_compute_energy_gw: float = field(init=False)  # Total energy consumption (GW)

    ## LR components (for detection breakdown) - all point-in-time metrics
    lr_prc_accounting: float = field(init=False)  # From chip stock discrepancy (static)
    lr_sme_inventory: float = field(init=False)  # From SME diversion (static)
    lr_satellite_datacenter: float = field(init=False)  # From satellite imagery (static)
    lr_reported_energy: float = field(init=False)  # Energy consumption LR (dynamic)
    lr_other_intel: float = field(init=False)  # Direct evidence LR from workers (dynamic)
    cumulative_lr: float = field(init=False)  # Combined likelihood ratio (dynamic)
    posterior_prob: float = field(init=False)  # P(project exists | evidence) (dynamic)

    ## Fab-specific LR components (for black_fab section - uses fab labor and time from construction)
    lr_fab_other: float = field(init=False)  # Fab worker detection LR (based on fab labor)
    lr_fab_procurement: float = field(init=False)  # Fab procurement LR (1.0 if localized, 10.0 if foreign imports needed)
    lr_fab_combined: float = field(init=False)  # Fab's combined LR = lr_sme_inventory × lr_fab_procurement × lr_fab_other

    ## Precomputed LR time series (separate for datacenters and fab)
    lr_datacenters_by_year: Dict[float, float] = field(init=False)  # Datacenter worker LR by year (since agreement)
    lr_fab_other_by_year: Dict[float, float] = field(init=False)  # Fab worker LR by year (since fab construction)

    ## Legacy time series fields (kept for backward compatibility, will be deprecated)
    years: List[float] = field(init=False)
    cumulative_h100_years: List[float] = field(init=False)
    operational_compute_h100e_by_year: List[float] = field(init=False)
    survival_rate_by_year: List[float] = field(init=False)
    datacenter_capacity_gw_by_year: List[float] = field(init=False)
    total_black_compute_by_year: List[float] = field(init=False)
    initial_black_compute_by_year: List[float] = field(init=False)
    fab_flow_by_year: List[float] = field(init=False)
    lr_reported_energy_by_year: List[float] = field(init=False)
    lr_other_intel_by_year: List[float] = field(init=False)
    cumulative_lr_by_year: List[float] = field(init=False)
    posterior_prob_by_year: List[float] = field(init=False)
    fab_cumulative_production_h100e_by_year: List[float] = field(init=False)
    fab_architecture_efficiency_by_year: List[float] = field(init=False)

    ## Detection outcomes (per threshold)
    sampled_detection_time: float = field(init=False)  # Sampled from composite distribution
    is_detected: bool = field(init=False)  # Whether project has been detected (set by world_updaters)

    ## Energy parameters for dynamic LR calculation
    us_estimate_energy: float = field(init=False)  # US intelligence estimate of total PRC energy
    total_prc_energy_gw: float = field(init=False)  # Total PRC energy consumption (GW)
