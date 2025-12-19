"""
Black Project parameters for modeling covert compute infrastructure.

Adapted from black_project_backend for continuous ODE simulation.

NOTE: Default values are NOT stored here. All defaults are in modal_parameters.yaml.
These dataclasses define the structure only.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from classes.world.entities import BlackProjectProperties
from enum import Enum


class ProcessNodes:
    """Semiconductor process nodes available for covert fab."""
    nm130 = "130nm"
    nm28 = "28nm"
    nm14 = "14nm"
    nm7 = "7nm"

    def to_nm(self) -> float:
        """Convert to numeric nanometers."""
        return {
            ProcessNode.nm130: 130.0,
            ProcessNode.nm28: 28.0,
            ProcessNode.nm14: 14.0,
            ProcessNode.nm7: 7.0,
        }[self]


class ProcessNodeStrategy(Enum):
    """Strategy for selecting process node based on indigenous capability."""
    BEST_INDIGENOUS = "best_indigenous"
    BEST_INDIGENOUS_GTE_28NM = "best_indigenous_gte_28nm"
    BEST_INDIGENOUS_GTE_14NM = "best_indigenous_gte_14nm"
    BEST_INDIGENOUS_GTE_7NM = "best_indigenous_gte_7nm"


# =============================================================================
# REFERENCE CONSTANTS
# =============================================================================

# H100 reference chip (used as baseline for all compute measurements)
H100_PROCESS_NODE_NM = 4.0
H100_RELEASE_YEAR = 2022.0
H100_TRANSISTOR_DENSITY_M_PER_MM2 = 98.28  # Million transistors per mm^2
H100_WATTS_PER_TPP = 0.326493  # Watts per Tera-Parameter-Pass
H100_TPP_PER_CHIP = 2144.0  # Tera-Parameter-Passes per chip


# =============================================================================
# PARAMETER DATACLASSES
# =============================================================================

@dataclass
class BlackFabParameters:
    """Parameters for covert semiconductor fabrication facility."""

    # Production parameters
    h100_sized_chips_per_wafer: float
    wafers_per_month_per_worker: float
    wafers_per_month_per_lithography_scanner: float

    # Construction time parameters (log-linear model)
    construction_time_for_5k_wafers_per_month: float
    construction_time_for_100k_wafers_per_month: float
    construction_workers_per_1000_wafers_per_month: float

    # Technology parameters
    transistor_density_scaling_exponent: float
    architecture_efficiency_improvement_per_year: float

    # Energy efficiency parameters (Dennard scaling)
    transistor_density_at_end_of_dennard_scaling_m_per_mm2: float
    watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended: float
    watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended: float

    # PRC lithography scanner production
    prc_lithography_scanners_produced_in_first_year: float
    prc_additional_lithography_scanners_produced_per_year: float

    # PRC localization probability curves: List of (year, cumulative_probability) tuples
    localization_130nm: List[Tuple[float, float]]
    localization_28nm: List[Tuple[float, float]]
    localization_14nm: List[Tuple[float, float]]
    localization_7nm: List[Tuple[float, float]]

    def get_localization_probabilities(self, node: ProcessNode) -> List[Tuple[float, float]]:
        """Get localization probability curve for a given process node."""
        return {
            ProcessNode.nm130: self.localization_130nm,
            ProcessNode.nm28: self.localization_28nm,
            ProcessNode.nm14: self.localization_14nm,
            ProcessNode.nm7: self.localization_7nm,
        }[node]


@dataclass
class BlackDatacenterParameters:
    """Parameters for covert datacenter construction."""

    # Construction productivity
    MW_per_construction_worker_per_year: float

    # Operating requirements
    operating_labor_per_MW: float


@dataclass
class DetectionParameters:
    """Parameters for likelihood ratio calculations."""

    # US intelligence estimation error (median absolute relative error)
    us_intelligence_median_error_in_estimate_of_prc_compute_stock: float
    us_intelligence_median_error_in_estimate_of_prc_fab_stock: float
    us_intelligence_median_error_in_energy_consumption_estimate_of_prc_datacenter_capacity: float
    us_intelligence_median_error_in_satellite_estimate_of_prc_datacenter_capacity: float

    # Worker detection model (Gamma distribution parameters)
    mean_detection_time_for_100_workers: float
    mean_detection_time_for_1000_workers: float
    variance_of_detection_time_given_num_workers: float

    # Likelihood ratio threshold for detection
    # When cumulative_likelihood_ratio exceeds this, the project is considered detected
    detection_threshold: float


@dataclass
class PRCBlackProjectParameters:
    """
    User-configurable properties for the black project scenario.

    Note: The black project starts at the AI slowdown start year, which is
    defined in policy_parameters.py as ai_slowdown_start_year. This represents
    when international agreements take effect and covert development begins.
    """
    run_a_black_project: bool
    black_project_properties : BlackProjectProperties

@dataclass
class BlackProjectProperties:
    """
    Properties of a black project
    (separated into a separate class see we can specify black project properties in /parameters)
    """
    # Initial compute diversion
    proportion_of_initial_compute_stock_to_divert: float

    # Datacenter configuration
    datacenter_construction_labor: float
    years_before_agreement_year_prc_starts_building_black_datacenters: float
    max_proportion_of_PRC_energy_consumption: float
    fraction_of_datacenter_capacity_not_built_for_concealment_diverted_to_black_project_at_agreement_start: float

    # Fab configuration
    build_a_black_fab: bool
    black_fab_construction_labor: int
    black_fab_operating_labor: int
    black_fab_process_node: str
    black_fab_proportion_of_prc_lithography_scanners_devoted: float

    # Workforce
    researcher_headcount: int

@dataclass
class BlackProjectParameterSet:
    """Complete set of black project parameters."""

    properties: BlackProjectProperties
    fab_params: BlackFabParameters
    datacenter_params: BlackDatacenterParameters
    detection_params: DetectionParameters

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BlackProjectParameterSet":
        """Create parameter set from dictionary configuration."""
        return cls(
            properties=BlackProjectProperties(**config.get("properties", {})),
            fab_params=BlackFabParameters(**config.get("fab", {})),
            datacenter_params=BlackDatacenterParameters(**config.get("datacenter", {})),
            detection_params=DetectionParameters(**config.get("detection", {})),
        )
