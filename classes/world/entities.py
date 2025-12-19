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
from classes.world.assets import Compute, Fabs, Datacenters
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
    """A nation with compute stock tracking."""
    leading_ai_software_developer_id: Optional[str] = None

    # Compute stock state (H100e TPP in log space for numerical stability)
    log_compute_stock: Tensor = field(default_factory=lambda: torch.tensor(0.0), metadata={'is_state': True})

    # Compute stock growth rate (for nations like PRC with independent growth)
    compute_growth_rate: float = 0.0  # Multiplier per year (e.g., 2.2 means 2.2x per year)

    # Energy infrastructure
    total_energy_consumption_gw: float = 0.0

    @property
    def compute_stock_h100e(self) -> float:
        """Get compute stock in H100e TPP."""
        return float(torch.exp(self.log_compute_stock).item())


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
class BlackFabState(TensorDataclass):
    """State of a covert semiconductor fabrication facility."""

    # Whether fab is operational (set by discrete event when construction completes)
    is_operational: bool = False

    # Configuration (set at initialization, does not change)
    process_node_nm: float = 28.0
    construction_start_year: float = 2030.0
    construction_duration: float = 2.0  # years

    # Production parameters (set at initialization based on sampling)
    wafer_starts_per_month: float = 0.0
    h100e_per_chip: float = 0.0  # H100-equivalent per chip produced
    chips_per_wafer: float = 28.0

    # Detection likelihood ratios (set at initialization)
    lr_inventory: float = 1.0
    lr_procurement: float = 1.0

    @property
    def operational_year(self) -> float:
        """Year when fab becomes operational."""
        return self.construction_start_year + self.construction_duration

    @property
    def monthly_production_h100e(self) -> float:
        """Monthly production rate in H100e when operational."""
        if not self.is_operational:
            return 0.0
        return self.wafer_starts_per_month * self.chips_per_wafer * self.h100e_per_chip

    @property
    def annual_production_h100e(self) -> float:
        """Annual production rate in H100e when operational."""
        return self.monthly_production_h100e * 12.0


@dataclass
class BlackDatacenterState(TensorDataclass):
    """State of covert datacenter infrastructure."""

    # Concealed capacity (built for hiding) - continuous state
    log_concealed_capacity_gw: Tensor = field(
        default_factory=lambda: torch.tensor(-10.0),  # ~0 GW initially
        metadata={'is_state': True}
    )

    # Unconcealed capacity diverted at agreement start (constant after initialization)
    unconcealed_capacity_gw: float = 0.0

    # Construction parameters
    construction_start_year: float = 2029.0  # Usually 1 year before agreement
    construction_rate_gw_per_year: float = 0.0
    max_total_capacity_gw: float = 55.0  # 5% of PRC's 1100 GW

    # Operating labor per GW
    operating_labor_per_gw: float = 100.0

    @property
    def concealed_capacity_gw(self) -> float:
        """Get concealed capacity in GW."""
        return float(torch.exp(self.log_concealed_capacity_gw).item())

    @property
    def total_capacity_gw(self) -> float:
        """Total covert datacenter capacity (concealed + unconcealed)."""
        return self.concealed_capacity_gw + self.unconcealed_capacity_gw

    @property
    def operating_labor(self) -> float:
        """Operating labor required for current capacity."""
        return self.total_capacity_gw * self.operating_labor_per_gw


@dataclass
class BlackComputeStock(TensorDataclass):
    """
    State of covert compute stock with attrition.

    Models compute stock with continuous hazard-based attrition.
    Stock dynamics: d(S)/dt = production - hazard * S
    """

    # Log of compute stock for numerical stability
    log_compute_stock: Tensor = field(
        default_factory=lambda: torch.tensor(0.0),
        metadata={'is_state': True}
    )

    # Hazard rate parameters (set at initialization from sampling)
    initial_hazard_rate: float = 0.05  # Per year
    hazard_rate_increase_per_year: float = 0.02  # Per year^2

    # Tracking when compute was added (for hazard rate calculation)
    # This is a weighted average age of the compute stock
    average_age_years: float = 0.0

    # Energy efficiency relative to H100 (for power calculations)
    energy_efficiency_relative_to_h100: float = 0.2

    @property
    def compute_stock_h100e(self) -> float:
        """Get current compute stock in H100e TPP."""
        return float(torch.exp(self.log_compute_stock).item())

    def current_hazard_rate(self) -> float:
        """Calculate current hazard rate based on average age."""
        return self.initial_hazard_rate + self.hazard_rate_increase_per_year * self.average_age_years


@dataclass
class AIBlackProject(AISoftwareDeveloper):
    """
    A covert AI development project with integrated infrastructure.

    Extends AISoftwareDeveloper with:
    - BlackFabState: Covert semiconductor fabrication
    - BlackDatacenterState: Covert power infrastructure
    - BlackComputeStock: Compute stock with attrition
    """
    parent_entity_id: Optional[str] = None

    # Labor allocation
    human_datacenter_construction_labor: int = 0
    human_datacenter_operating_labor: int = 0
    human_fab_construction_labor: int = 0
    human_fab_operating_labor: int = 0
    researcher_headcount: int = 500

    # Infrastructure state
    fab: Optional[BlackFabState] = None
    datacenters: Optional[BlackDatacenterState] = None
    compute_stock: Optional[BlackComputeStock] = None

    # AI slowdown start year (when black project starts due to international agreements)
    ai_slowdown_start_year: float = 2030.0

    # Detection state
    cumulative_likelihood_ratio: float = 1.0

    @property
    def total_labor(self) -> int:
        """Total labor involved in the black project."""
        labor = self.researcher_headcount
        labor += self.human_datacenter_construction_labor
        if self.datacenters is not None:
            labor += int(self.datacenters.operating_labor)
        if self.fab is not None and self.fab.is_operational:
            labor += self.human_fab_operating_labor
        else:
            labor += self.human_fab_construction_labor
        return labor

    @property
    def operational_compute_h100e(self) -> float:
        """
        Compute that can actually operate given datacenter capacity.

        Limited by min(compute_stock, datacenter_capacity / energy_per_h100e).
        """
        if self.compute_stock is None or self.datacenters is None:
            return 0.0

        stock = self.compute_stock.compute_stock_h100e
        capacity_gw = self.datacenters.total_capacity_gw

        # Energy required per H100e (in GW)
        # H100 is ~700W, so H100e TPP needs power proportional to efficiency
        h100_power_gw = 700e-9  # 700W in GW
        power_per_h100e_gw = h100_power_gw / self.compute_stock.energy_efficiency_relative_to_h100

        # Maximum compute that can be powered
        max_powered_compute = capacity_gw / power_per_h100e_gw

        return min(stock, max_powered_compute)
