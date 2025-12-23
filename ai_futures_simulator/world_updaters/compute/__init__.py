"""
Compute-related world updaters and utility functions.

This module contains:
- ComputeUpdater: Combined updater that orchestrates all compute updates
- NationComputeUpdater: Updates compute stock for nations
- AISoftwareDeveloperComputeUpdater: Updates compute for AI software developers
- Chip survival utilities: Functions for modeling chip attrition
- Black project compute utilities: Functions for covert compute calculations
"""

from world_updaters.compute.update_compute import ComputeUpdater
from world_updaters.compute.nation_compute import NationComputeUpdater
from world_updaters.compute.ai_sw_developer_compute import AISoftwareDeveloperComputeUpdater
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
    # Datacenter calculations
    calculate_concealed_capacity_gw,
    calculate_datacenter_capacity_gw,
    calculate_datacenter_operating_labor,
    # Compute calculations (operating compute only - survival moved to chip_survival.py)
    calculate_operating_compute,
    # Labor utility
    get_black_project_total_labor,
    # Detection utilities
    compute_detection_constants,
    compute_mean_detection_time,
    sample_detection_time,
    compute_lr_over_time_vs_num_workers,
    sample_us_estimate_with_error,
    lr_from_discrepancy_in_us_estimate,
    compute_lr_from_reported_energy_consumption,
    compute_lr_from_satellite_detection,
    compute_lr_from_prc_compute_accounting,
    compute_detection_probability,
)

__all__ = [
    # Combined updater
    'ComputeUpdater',
    # Individual updaters
    'NationComputeUpdater',
    'AISoftwareDeveloperComputeUpdater',
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
    # Datacenter calculations
    'calculate_concealed_capacity_gw',
    'calculate_datacenter_capacity_gw',
    'calculate_datacenter_operating_labor',
    # Compute calculations
    'calculate_operating_compute',
    # Labor utility
    'get_black_project_total_labor',
    # Detection utilities
    'compute_detection_constants',
    'compute_mean_detection_time',
    'sample_detection_time',
    'compute_lr_over_time_vs_num_workers',
    'sample_us_estimate_with_error',
    'lr_from_discrepancy_in_us_estimate',
    'compute_lr_from_reported_energy_consumption',
    'compute_lr_from_satellite_detection',
    'compute_lr_from_prc_compute_accounting',
    'compute_detection_probability',
]
