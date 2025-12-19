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
from classes.world.assets import Compute, Fabs, Datacenters, BlackFabs, BlackDatacenters, BlackCompute
from classes.world.software_progress import AISoftwareProgress


@dataclass
class Entity(TensorDataclass):
    """Base class for all entities (Coalition, Nation, Organization, etc)."""
    id: str


@dataclass
class Coalition(Entity):
    """A coalition of nations."""
    member_nation_ids: List[str]


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
    leading_ai_software_developer_id: Optional[str] = None

    # Compute stock state (H100e TPP in log space for numerical stability)
    log_compute_stock: Tensor = field(default_factory=lambda: torch.tensor(0.0), metadata={'is_state': True})

    # Compute stock growth rate (for nations like PRC with independent growth)
    compute_growth_rate: float = 0.0  # Multiplier per year (e.g., 2.2 means 2.2x per year)

    # Energy infrastructure
    total_energy_consumption_gw: float = 0.0


class NamedNations:
    """Named nation identifiers."""
    PRC = "PRC"
    USA = "USA"


@dataclass
class ComputeAllocation:
    """How compute is allocated across different uses."""
    fraction_for_ai_r_and_d_inference: float
    fraction_for_ai_r_and_d_training: float
    fraction_for_external_deployment: float
    fraction_for_alignment_research: float
    fraction_for_frontier_training: float


@dataclass
class AISoftwareDeveloper(Entity):
    """
    An AI software development organization.

    Contains nested state (via ai_software_progress) that gets integrated.
    State fields have metadata={'is_state': True}.
    """
    is_primarily_controlled_by_misaligned_AI: bool
    compute: Compute
    compute_allocation: ComputeAllocation
    ai_software_progress: AISoftwareProgress
    human_ai_capability_researchers: int
    log_compute: Tensor = field(metadata={'is_state': True})
    log_researchers: Tensor = field(metadata={'is_state': True})
    # Experiment compute (for R&D experiments) - separate from inference compute
    experiment_compute: float = 0.0


@dataclass
class AIBlackProject(AISoftwareDeveloper):
    """
    A covert AI development project with integrated infrastructure.

    Extends AISoftwareDeveloper with:
    - BlackFabs: Covert semiconductor fabrication
    - BlackDatacenters: Covert power infrastructure
    - BlackCompute: Compute stock with attrition

    Use world_updaters.compute.black_compute functions for derived values:
    - get_black_project_total_labor(project)
    - get_black_project_operational_compute(project)
    """
    parent_entity_id: Optional[str] = None

    # Labor allocation
    human_datacenter_construction_labor: int = 0
    human_datacenter_operating_labor: int = 0
    human_fab_construction_labor: int = 0
    human_fab_operating_labor: int = 0
    researcher_headcount: int = 500

    # Infrastructure state
    fab: Optional[BlackFabs] = None
    datacenters: Optional[BlackDatacenters] = None
    compute_stock: Optional[BlackCompute] = None

    # AI slowdown start year (when black project starts due to international agreements)
    ai_slowdown_start_year: float = 2030.0

    # Detection state
    # Sampled at initialization from Gamma distribution based on labor
    sampled_detection_time: float = float('inf')  # Years since ai_slowdown_start_year
    is_detected: bool = False  # Set to True when current_time >= detection_time
    cumulative_likelihood_ratio: float = 1.0
    # Continuous detection probability: P(detected by current time) = Gamma.cdf(t, k, θ)
    # This is differentiable and can be used for backpropagation optimization
    detection_probability: float = 0.0

    # Pre-computed labor by year (relative to ai_slowdown_start_year)
    # Used for composite detection distribution - matches discrete model
    labor_by_year: Optional[dict] = None

    # Pre-computed likelihood ratios by year (relative to ai_slowdown_start_year)
    # These are sampled once at initialization and reused
    lr_by_year: Optional[dict] = None

    # Sampled US intelligence estimates (sampled once at initialization)
    # These drive the energy, satellite, and compute stock LR calculations
    us_estimate_energy_gw: float = 0.0
    us_estimate_satellite_capacity_gw: float = 0.0
    us_estimate_compute_stock: float = 0.0

    # Parameters needed for LR calculations (stored from initialization)
    total_prc_energy_gw: float = 0.0
    diversion_proportion: float = 0.0
