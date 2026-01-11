"""
Compute-related world updaters and utility functions.

This module contains:
- ComputeUpdater: Combined updater that orchestrates all compute updates
- NationComputeUpdater: Updates compute stock for nations
- AISoftwareDeveloperComputeUpdater: Updates compute for AI software developers
- BlackProjectComputeUpdater: Updates compute for black projects (fabs, stock, attrition)
- Chip survival utilities: Functions for modeling chip attrition
- Fab calculation utilities: Functions for fab construction and production

Datacenter utilities have been moved to world_updaters.datacenters_and_energy.
Detection utilities have been moved to world_updaters.perceptions.black_project_perceptions.
"""

from world_updaters.compute.update_compute import ComputeUpdater
from world_updaters.compute.nation_compute import NationComputeUpdater
from world_updaters.compute.ai_sw_developer_compute import AISoftwareDeveloperComputeUpdater
from world_updaters.compute.black_compute import BlackProjectComputeUpdater
from world_updaters.compute.chip_survival import (
    # Core attrition model functions
    calculate_hazard_rate,
    calculate_attrition_rate,
    calculate_compute_derivative,
    calculate_average_age_derivative,
    calculate_production_rate_from_growth,
    calculate_derivatives_with_attrition,
    # Legacy functions for single-cohort survival
    calculate_survival_rate,
    calculate_functional_compute,
    calculate_survival_rate_from_params,
)
from world_updaters.compute.black_compute import (
    # Fab calculations
    calculate_fab_construction_duration,
    calculate_fab_wafer_starts_per_month,
    calculate_fab_h100e_per_chip,
    calculate_transistor_density_from_process_node,
    calculate_watts_per_tpp_from_transistor_density,
    calculate_fab_watts_per_chip,
    calculate_fab_annual_production_h100e,
    # Compute calculations
    calculate_operating_compute,
)

__all__ = [
    # Combined updater
    'ComputeUpdater',
    # Individual updaters
    'NationComputeUpdater',
    'AISoftwareDeveloperComputeUpdater',
    'BlackProjectComputeUpdater',
    # Chip attrition model functions
    'calculate_hazard_rate',
    'calculate_attrition_rate',
    'calculate_compute_derivative',
    'calculate_average_age_derivative',
    'calculate_production_rate_from_growth',
    'calculate_derivatives_with_attrition',
    # Legacy single-cohort survival functions
    'calculate_survival_rate',
    'calculate_functional_compute',
    'calculate_survival_rate_from_params',
    # Fab calculations
    'calculate_fab_construction_duration',
    'calculate_fab_wafer_starts_per_month',
    'calculate_fab_h100e_per_chip',
    'calculate_transistor_density_from_process_node',
    'calculate_watts_per_tpp_from_transistor_density',
    'calculate_fab_watts_per_chip',
    'calculate_fab_annual_production_h100e',
    # Compute calculations
    'calculate_operating_compute',
]
