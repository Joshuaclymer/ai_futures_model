"""
Perceptions world updaters.

This module contains:
- StatePerceptionsOfCovertComputeUpdater: Updates entity perceptions about covert AI projects
- BlackProjectPerceptionsUpdater: Updates detection/perception metrics for black projects
- Detection utilities: Functions for likelihood ratios, detection time sampling, Bayesian updates
"""

from world_updaters.perceptions.update_perceptions import (
    StatePerceptionsOfCovertComputeUpdater,
    probability_to_odds,
    odds_to_probability,
    bayesian_update_probability,
)
from world_updaters.perceptions.black_project_perceptions import (
    BlackProjectPerceptionsUpdater,
    # Labor utility
    get_black_project_total_labor,
    # Detection utilities
    compute_detection_constants,
    compute_mean_detection_time,
    sample_detection_time,
    sample_us_estimate_with_error,
    lr_from_discrepancy_in_us_estimate,
    compute_lr_from_reported_energy_consumption,
    compute_lr_from_satellite_detection,
    compute_lr_from_prc_compute_accounting,
    compute_lr_from_sme_inventory,
    compute_lr_over_time_vs_num_workers,
    compute_detection_probability,
    # Continuous detection model
    compute_gamma_hazard_rate,
    compute_worker_detection_hazard_rate,
    compute_log_survival_probability,
    compute_cumulative_log_lr,
    compute_posterior_probability,
)

__all__ = [
    # Updaters
    'StatePerceptionsOfCovertComputeUpdater',
    'BlackProjectPerceptionsUpdater',
    # Bayesian utilities
    'probability_to_odds',
    'odds_to_probability',
    'bayesian_update_probability',
    # Labor utility
    'get_black_project_total_labor',
    # Detection utilities
    'compute_detection_constants',
    'compute_mean_detection_time',
    'sample_detection_time',
    'sample_us_estimate_with_error',
    'lr_from_discrepancy_in_us_estimate',
    'compute_lr_from_reported_energy_consumption',
    'compute_lr_from_satellite_detection',
    'compute_lr_from_prc_compute_accounting',
    'compute_lr_from_sme_inventory',
    'compute_lr_over_time_vs_num_workers',
    'compute_detection_probability',
    # Continuous detection model
    'compute_gamma_hazard_rate',
    'compute_worker_detection_hazard_rate',
    'compute_log_survival_probability',
    'compute_cumulative_log_lr',
    'compute_posterior_probability',
]
