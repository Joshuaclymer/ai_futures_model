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

    # Prior probability that PRC has a covert AI project
    # This is the US's initial belief before observing any evidence
    prior_probability_prc_has_covert_project: float

    # Minimum probability floor (prevents P from going to exactly 0)
    min_probability: float = 0.001

    # Maximum probability ceiling (prevents P from going to exactly 1)
    max_probability: float = 0.999
