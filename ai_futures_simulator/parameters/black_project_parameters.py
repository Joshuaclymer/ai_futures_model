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
    black_fab_min_process_node: float  # Min process node required (nm), e.g., 28nm
    # Localization years for each process node (sampled independently)
    # 9999 = never achieved. Fab uses best available node meeting minimum threshold.
    prc_localization_year_28nm: float  # Year PRC achieves 28nm localization
    prc_localization_year_14nm: float  # Year PRC achieves 14nm localization
    prc_localization_year_7nm: float   # Year PRC achieves 7nm localization
    # Note: build_a_black_fab is derived: True if min_node localization <= black_project_start_year

    # Datacenter construction timing (has default, so must come after non-default fields)
    # How many years before black_project_start_year to begin datacenter construction
    # Discrete model uses 1.0 (starts 1 year before agreement)
    years_before_black_project_start_to_begin_datacenter_construction: float = 1.0

@dataclass
class BlackProjectParameters:
    """Complete set of black project parameters."""

    run_a_black_project: bool
    black_project_start_year: float
    black_project_properties: BlackProjectProperties