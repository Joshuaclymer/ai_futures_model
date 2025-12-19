"""
Compute parameters for modeling AI training compute dynamics.

Contains parameters for training compute growth rates, slowdown scenarios,
and chip survival/attrition rates.

NOTE: Default values are NOT stored here. All defaults are in modal_parameters.yaml.
"""

from dataclasses import dataclass


@dataclass
class ComputeParameters:
    """
    Parameters for training compute dynamics.

    Models exponential growth in training compute with optional slowdown.
    Growth is measured in orders of magnitude (OOMs) per year.
    Also includes chip survival/attrition parameters.
    """

    # Pre-slowdown growth rate (OOMs/year)
    # Historical rate has been ~0.5-0.7 OOMs/year
    constant_training_compute_growth_rate: float

    # Year when slowdown begins (if any)
    slowdown_year: float

    # Post-slowdown growth rate (OOMs/year)
    # Could be lower due to energy/chip constraints, policy, etc.
    post_slowdown_training_compute_growth_rate: float

    # Chip survival/attrition parameters
    # Hazard rate model: H(t) = initial + increase * t
    initial_hazard_rate: float  # Per year (median estimate)
    hazard_rate_increase_per_year: float  # Per year^2 (median estimate)

    # PRC compute stock
    total_prc_compute_stock_in_2025: float
    annual_growth_rate_of_prc_compute_stock_p10: float
    annual_growth_rate_of_prc_compute_stock_p50: float
    annual_growth_rate_of_prc_compute_stock_p90: float

    # PRC domestic production capability
    proportion_of_prc_chip_stock_produced_domestically_2026: float
    proportion_of_prc_chip_stock_produced_domestically_2030: float

    # Largest AI project compute
    largest_ai_project_h100e_in_2025: float
    largest_ai_project_growth_rate: float
