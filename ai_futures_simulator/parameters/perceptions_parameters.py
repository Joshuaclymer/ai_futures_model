"""
Perceptions parameters for modeling entity beliefs about the world.

Contains parameters for US perceptions of covert AI projects.

NOTE: Default values are NOT stored here. All defaults are in modal_parameters.yaml.
These dataclasses define the structure only.
"""

from dataclasses import dataclass


@dataclass
class PerceptionsParameters:
    """
    Parameters for updating entity perceptions about covert projects.

    These parameters control how the US updates its belief about the
    probability that the PRC has a covert AI project based on
    intelligence indicators (cumulative likelihood ratio).
    """

    # Enable/disable perceptions updating
    update_perceptions: bool
    black_project_perception_parameters : "BlackProjectPerceptionsParameters"

@dataclass
class BlackProjectPerceptionsParameters:
    """Parameters for likelihood ratio calculations."""

    # Prior odds that a covert project exists
    prior_odds_of_covert_project: float

    # Intelligence estimation error (median absolute relative error)
    intelligence_median_error_in_estimate_of_compute_stock: float
    intelligence_median_error_in_estimate_of_fab_stock: float
    intelligence_median_error_in_energy_consumption_estimate_of_datacenter_capacity: float
    intelligence_median_error_in_satellite_estimate_of_datacenter_capacity: float

    # Worker detection model (Gamma distribution parameters)
    mean_detection_time_for_100_workers: float
    mean_detection_time_for_1000_workers: float
    variance_of_detection_time_given_num_workers: float

    # Likelihood ratio threshold for detection
    # When cumulative_likelihood_ratio exceeds this, the project is considered detected
    detection_threshold: float
    # Multiple detection thresholds for CCDF plots (1x, 2x, 4x update in odds)
    detection_thresholds: list[float]