"""
Compute-related world updaters and utility functions.

This module contains:
- NationComputeUpdater: Updates compute stock for nations
- Black project compute utilities: Functions for covert compute calculations
"""

from world_updaters.compute.nation_compute import NationComputeUpdater
from world_updaters.compute.black_compute import (
    # Fab calculations
    calculate_fab_construction_duration,
    calculate_fab_wafer_starts_per_month,
    calculate_fab_h100e_per_chip,
    calculate_fab_watts_per_chip,
    calculate_fab_annual_production_h100e,
    # Datacenter calculations
    calculate_concealed_capacity_gw,
    calculate_datacenter_capacity_gw,
    calculate_datacenter_operating_labor,
    # Compute calculations
    calculate_survival_rate,
    calculate_functional_compute,
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
    'NationComputeUpdater',
    # Fab calculations
    'calculate_fab_construction_duration',
    'calculate_fab_wafer_starts_per_month',
    'calculate_fab_h100e_per_chip',
    'calculate_fab_watts_per_chip',
    'calculate_fab_annual_production_h100e',
    # Datacenter calculations
    'calculate_concealed_capacity_gw',
    'calculate_datacenter_capacity_gw',
    'calculate_datacenter_operating_labor',
    # Compute calculations
    'calculate_survival_rate',
    'calculate_functional_compute',
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
