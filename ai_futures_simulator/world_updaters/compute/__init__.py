"""
Compute-related world updaters and utility functions.

This module contains:
- NationComputeUpdater: Updates compute stock for nations
- Black project compute utilities: Functions for covert compute calculations
"""

from world_updaters.compute.nation_compute import (
    NationComputeUpdater,
    NationComputeConfig,
    get_nation_compute_stock_h100e,
)
from world_updaters.compute.black_compute import (
    get_fab_operational_year,
    get_fab_monthly_production_h100e,
    get_fab_annual_production_h100e,
    get_datacenter_concealed_capacity_gw,
    get_datacenter_total_capacity_gw,
    get_datacenter_operating_labor,
    get_compute_stock_h100e,
    get_current_hazard_rate,
    get_black_project_total_labor,
    get_black_project_operational_compute,
    # Detection utilities
    compute_detection_constants,
    compute_mean_detection_time,
    sample_detection_time,
    compute_lr_other,
    compute_cumulative_likelihood_ratio,
    compute_detection_probability,
)

__all__ = [
    'NationComputeUpdater',
    'NationComputeConfig',
    'get_nation_compute_stock_h100e',
    'get_fab_operational_year',
    'get_fab_monthly_production_h100e',
    'get_fab_annual_production_h100e',
    'get_datacenter_concealed_capacity_gw',
    'get_datacenter_total_capacity_gw',
    'get_datacenter_operating_labor',
    'get_compute_stock_h100e',
    'get_current_hazard_rate',
    'get_black_project_total_labor',
    'get_black_project_operational_compute',
    # Detection utilities
    'compute_detection_constants',
    'compute_mean_detection_time',
    'sample_detection_time',
    'compute_lr_other',
    'compute_cumulative_likelihood_ratio',
    'compute_detection_probability',
]
