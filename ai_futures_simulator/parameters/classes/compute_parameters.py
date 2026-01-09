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
    total_us_compute_tpp_h100e_in_2025: float
    total_us_compute_annual_growth_rate: float # annual multiplier
    proportion_of_compute_in_largest_ai_sw_developer: float

@dataclass
class ComputeAllocations:
    fraction_for_ai_r_and_d_inference: float 
    fraction_for_ai_r_and_d_training: float 
    fraction_for_external_deployment: float
    fraction_for_alignment_research: float
    fraction_for_frontier_training: float

@dataclass
class PRCComputeParameters:
    total_prc_compute_tpp_h100e_in_2025: float
    annual_growth_rate_of_prc_compute_stock: float
    proportion_of_compute_in_largest_ai_sw_developer: float
    prc_architecture_efficiency_relative_to_state_of_the_art : float # This should be 1.0

    proportion_of_prc_chip_stock_produced_domestically_2026: float
    proportion_of_prc_chip_stock_produced_domestically_2030: float

    # PRC lithography scanner production
    prc_lithography_scanners_produced_in_first_year: float
    prc_additional_lithography_scanners_produced_per_year: float
    # Uncertainty multiplier for total PRC scanner production (sampled from lognormal, median=1.0)
    # Reference: prc_scanner_production_relative_sigma=0.30
    prc_scanner_production_multiplier: float

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

    # Fab construction time uncertainty multiplier (sampled from lognormal, median=1.0)
    fab_construction_time_multiplier: float

    # Fab productivity multipliers (sampled from lognormal, median=1.0)
    # These capture uncertainty in wafer production capacity
    # Reference: labor_productivity_relative_sigma=0.62, scanner_productivity_relative_sigma=0.20
    fab_labor_productivity_multiplier: float
    fab_scanner_productivity_multiplier: float

@dataclass
class SurvivalRateParameters:
    # Base hazard rate parameters (p50 values)
    initial_annual_hazard_rate: float  # Per year (base value, will be multiplied by hazard_rate_multiplier)
    annual_hazard_rate_increase_per_year: float  # Per year^2 (base value, will be multiplied by hazard_rate_multiplier)
    # Multiplier applied to both hazard rates (sampled from distribution in Monte Carlo mode)
    # Discrete model uses metalog with p25=0.1, p50=1.0, p75=6.0
    hazard_rate_multiplier: float

    @property
    def effective_initial_hazard_rate(self) -> float:
        """Initial hazard rate with multiplier applied."""
        return self.initial_annual_hazard_rate * self.hazard_rate_multiplier

    @property
    def effective_hazard_rate_increase(self) -> float:
        """Hazard rate increase per year with multiplier applied."""
        return self.annual_hazard_rate_increase_per_year * self.hazard_rate_multiplier

@dataclass
class ComputeParameters:
    exogenous_trends: ExogenousComputeTrends
    survival_rate_parameters : SurvivalRateParameters
    USComputeParameters: USComputeParameters
    PRCComputeParameters: PRCComputeParameters
    compute_allocations: ComputeAllocations