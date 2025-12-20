from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

# H100 reference chip (used as baseline for all compute measurements)
H100_PROCESS_NODE_NM = 4.0
H100_RELEASE_YEAR = 2022.0
H100_TRANSISTOR_DENSITY_M_PER_MM2 = 98.28  # Million transistors per mm^2
H100_WATTS_PER_TPP = 0.326493  # Watts per Tera-Parameter-Pass
H100_TPP_PER_CHIP = 2144.0  # Tera-Parameter-Passes per chip

@dataclass
class BlackProjectProperties:
    """
    Properties of a black project
    (separated into a separate class so we can specify black project properties in /parameters)
    """
    # Labor
    total_labor: float
    fraction_of_labor_devoted_to_datacenter_construction: float
    fraction_of_labor_devoted_to_black_fab_construction: float
    fraction_of_labor_devoted_to_black_fab_operation: float
    fraction_of_labor_devoted_to_ai_research: float

    # Diverted resources
    fraction_of_initial_compute_stock_to_divert_at_black_project_start: float
    fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start: float
    fraction_of_lithography_scanners_to_divert_at_black_project_start: float
    max_fraction_of_total_national_energy_consumption: float

    # Fab configuration
    build_a_black_fab: bool
    black_fab_max_process_node: float

@dataclass
class BlackProjectParameters:
    """Complete set of black project parameters."""

    run_a_black_project: bool
    black_project_start_year: float
    black_project_properties: BlackProjectProperties