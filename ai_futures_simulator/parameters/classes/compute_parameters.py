"""
Compute parameters for modeling AI training compute dynamics.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from parameters.classes.base_spec import BaseSpec, ParamValue


@dataclass
class ExogenousComputeTrends(BaseSpec):
    transistor_density_scaling_exponent: ParamValue = None
    state_of_the_art_architecture_efficiency_improvement_per_year: ParamValue = None
    transistor_density_at_end_of_dennard_scaling_m_per_mm2: ParamValue = None
    watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended: ParamValue = None
    watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended: ParamValue = None
    state_of_the_art_energy_efficiency_improvement_per_year: ParamValue = None


@dataclass
class USComputeParameters(BaseSpec):
    total_us_compute_tpp_h100e_in_2025: ParamValue = None
    total_us_compute_annual_growth_rate: ParamValue = None
    proportion_of_compute_in_largest_ai_sw_developer: ParamValue = None


@dataclass
class ComputeAllocations(BaseSpec):
    fraction_for_ai_r_and_d_inference: ParamValue = None
    fraction_for_ai_r_and_d_training: ParamValue = None
    fraction_for_external_deployment: ParamValue = None
    fraction_for_alignment_research: ParamValue = None
    fraction_for_frontier_training: ParamValue = None


@dataclass
class PRCComputeParameters(BaseSpec):
    total_prc_compute_tpp_h100e_in_2025: ParamValue = None
    annual_growth_rate_of_prc_compute_stock: ParamValue = None
    proportion_of_compute_in_largest_ai_sw_developer: ParamValue = None
    prc_architecture_efficiency_relative_to_state_of_the_art: ParamValue = None

    proportion_of_prc_chip_stock_produced_domestically_2026: ParamValue = None
    proportion_of_prc_chip_stock_produced_domestically_2030: ParamValue = None

    # PRC lithography scanner production
    prc_lithography_scanners_produced_in_first_year: ParamValue = None
    prc_additional_lithography_scanners_produced_per_year: ParamValue = None
    prc_scanner_production_multiplier: ParamValue = None

    # PRC localization probabilities
    p_localization_28nm_2030: ParamValue = None
    p_localization_14nm_2030: ParamValue = None
    p_localization_7nm_2030: ParamValue = None

    # Fab production
    h100_sized_chips_per_wafer: ParamValue = None
    wafers_per_month_per_lithography_scanner: ParamValue = None
    construction_time_for_5k_wafers_per_month: ParamValue = None
    construction_time_for_100k_wafers_per_month: ParamValue = None
    fab_construction_time_multiplier: ParamValue = None
    fab_wafers_per_month_per_operating_worker: ParamValue = None
    fab_wafers_per_month_per_construction_worker_under_standard_timeline: ParamValue = None
    fab_labor_productivity_multiplier: ParamValue = None
    fab_scanner_productivity_multiplier: ParamValue = None


@dataclass
class SurvivalRateParameters(BaseSpec):
    initial_annual_hazard_rate: ParamValue = None
    annual_hazard_rate_increase_per_year: ParamValue = None
    hazard_rate_multiplier: ParamValue = None

    @property
    def effective_initial_hazard_rate(self) -> float:
        """Initial hazard rate with multiplier applied."""
        return self.initial_annual_hazard_rate * self.hazard_rate_multiplier

    @property
    def effective_hazard_rate_increase(self) -> float:
        """Hazard rate increase per year with multiplier applied."""
        return self.annual_hazard_rate_increase_per_year * self.hazard_rate_multiplier


@dataclass
class ComputeParameters(BaseSpec):
    """All compute parameters (nested structure)."""
    exogenous_trends: Optional[ExogenousComputeTrends] = None
    survival_rate_parameters: Optional[SurvivalRateParameters] = None
    USComputeParameters: Optional[USComputeParameters] = None
    PRCComputeParameters: Optional[PRCComputeParameters] = None
    compute_allocations: Optional[ComputeAllocations] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComputeParameters":
        return cls(
            exogenous_trends=ExogenousComputeTrends.from_dict(d["exogenous_trends"]) if "exogenous_trends" in d else None,
            survival_rate_parameters=SurvivalRateParameters.from_dict(d["survival_rate_parameters"]) if "survival_rate_parameters" in d else None,
            USComputeParameters=USComputeParameters.from_dict(d["us_compute"]) if "us_compute" in d else None,
            PRCComputeParameters=PRCComputeParameters.from_dict(d["prc_compute"]) if "prc_compute" in d else None,
            compute_allocations=ComputeAllocations.from_dict(d["compute_allocations"]) if "compute_allocations" in d else None,
        )

    def sample(self, rng: np.random.Generator) -> "ComputeParameters":
        return ComputeParameters(
            exogenous_trends=self.exogenous_trends.sample(rng) if self.exogenous_trends else None,
            survival_rate_parameters=self.survival_rate_parameters.sample(rng) if self.survival_rate_parameters else None,
            USComputeParameters=self.USComputeParameters.sample(rng) if self.USComputeParameters else None,
            PRCComputeParameters=self.PRCComputeParameters.sample(rng) if self.PRCComputeParameters else None,
            compute_allocations=self.compute_allocations.sample(rng) if self.compute_allocations else None,
        )

    def get_modal(self) -> "ComputeParameters":
        return ComputeParameters(
            exogenous_trends=self.exogenous_trends.get_modal() if self.exogenous_trends else None,
            survival_rate_parameters=self.survival_rate_parameters.get_modal() if self.survival_rate_parameters else None,
            USComputeParameters=self.USComputeParameters.get_modal() if self.USComputeParameters else None,
            PRCComputeParameters=self.PRCComputeParameters.get_modal() if self.PRCComputeParameters else None,
            compute_allocations=self.compute_allocations.get_modal() if self.compute_allocations else None,
        )
