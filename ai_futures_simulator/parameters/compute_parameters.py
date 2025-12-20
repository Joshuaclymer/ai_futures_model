"""
Compute parameters for modeling AI training compute dynamics.

"""

from dataclasses import dataclass

@dataclass
class ExogenousComputeTrends:
    transistor_density_scaling_exponent: float # with respect to process node
    state_of_the_art_architecture_efficiency_improvement_per_year: float

    # Energy efficiency parameters (Dennard scaling)
    transistor_density_at_end_of_dennard_scaling_m_per_mm2: float
    watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended: float
    watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended: float
    state_of_the_art_energy_efficiency_improvement_per_year: float

@dataclass
class USComputeParameters:
    us_frontier_project_compute_tpp_h100e_in_2025: float
    us_frontier_project_compute_annual_growth_rate: float # annual multiplier

@dataclass
class PRCComputeParameters:
    total_prc_compute_tpp_h100e_in_2025: float
    annual_growth_rate_of_prc_compute_stock: float
    prc_architecture_efficiency_relative_to_state_of_the_art : float # This should be 1.0

    proportion_of_prc_chip_stock_produced_domestically_2026: float
    proportion_of_prc_chip_stock_produced_domestically_2030: float

    # PRC lithography scanner production
    prc_lithography_scanners_produced_in_first_year: float
    prc_additional_lithography_scanners_produced_per_year: float

    # PRC localization probability curves: List of (year, cumulative_probability) tuples
    p_localization_28nm_2030: float
    p_localization_14nm_2030: float
    p_localization_7nm_2030: float

    # Fab production
    h100_sized_chips_per_wafer: float
    wafers_per_month_per_lithography_scanner: float
    construction_time_for_5k_wafers_per_month: float
    construction_time_for_100k_wafers_per_month: float
    
    # Converting between number of workers and fab capacity (for black projects)
    fab_wafers_per_month_per_operating_worker: float
    fab_wafers_per_month_per_construction_worker_under_standard_timeline: float

@dataclass
class SurvivalRateParameters:
    initial_annual_hazard_rate: float  # Per year (median estimate)
    annual_hazard_rate_increase_per_year: float  # Per year^2 (median estimate)

@dataclass
class ComputeParameters:
    exogenous_trends: ExogenousComputeTrends
    survival_rate_parameters : SurvivalRateParameters
    USComputeParameters: USComputeParameters
    PRCComputeParameters: PRCComputeParameters