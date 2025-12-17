from classes.perceptions import LevelsOfAINationalSecurityImportance

import enum as Enum
from dataclasses import dataclass

class ModelParameters:
    pass # Placeholder for class implementation

@dataclass
class MethodOfSettingCaps():
    ten_times_reduction = "ten_times_reduction"
    one_hundred_times_reduction = "one_hundred_times_reduction"
    covert_stock_percentile_heuristic = "covert_stock_percentile_heuristic"
    model_based_optimization = "model_based_optimization"

class ProcessNode():
    nm130 = "130nm"
    nm28 = "28nm"
    nm14 = "14nm"
    nm7 = "7nm"

@dataclass
class SimulationParameters:

    ##### Constants ##### -------------------------------------
    
    # Calculating compute capacity of fabs
    h100_node: str = ProcessNode.nm7
    transistor_density_vs_node_exponent: float = 1.49
    wafers_per_month_per_lithography_scanner: float = 1000

    # Calculating power consumption of chips
    watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended: float = -2.00
    watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended: float = -0.91
    transistor_density_at_end_of_dennard_scaling_m_per_mm2: float = 1.98
    h100_power_watts: float = 700

    ##### USA Allies Assets ##### -------------------------------------
    
    # Compute
    total_usa_allies_compute_stock_in_2026_h100e_tpp: float = 5e5
    total_usa_allies_compute_production_capacity_in_2026_h100e_tpp_per_month: float = 2e4
    annual_growth_multiplier_of_usa_allies_compute_production_capacity: float = 2.4

    # Energy production
    total_usa_allies_energy_consumption_in_2026_GW: float = 1500
    annual_status_quo_growth_multiplier_of_usa_allies_energy_consumption: float = 1.04

    # Energy requirements (used to determine datacenter capacity)
    usa_allies_status_quo_improvement_in_energy_efficiency_per_year: float = 1.26  # Moved from InitialPRCBlackComputeStockParameters

    ##### PRC Assets ##### -------------------------------------

    # Compute
    total_prc_compute_stock_in_2026_h100e_tpp: float = 1e5

    # Smuggling
    current_h100e_smuggled_annually_h100e_tpp: float = 1e5
    annual_growth_multiplier_in_h100e_smuggled_annually = 1.1
    max_h100e_smuggled_annually = 2e5

    # Purchasing
    prc_compute_capax_in_2026 : float = 3e5
    annual_growth_of_prc_compute_capax : float = 1.2

    # Fabs
    total_prc_compute_production_capacity_in_2026_h100e_tpp_per_month: float = 1e4
    total_7nm_DUV_immersion_machines_in_prc_in_2026: int = 5
    probability_prc_localizes_28nm_by_2031: float = 0.6
    probability_prc_localizes_14nm_by_2031: float = 0.10
    probability_prc_localizes_7nm_by_2031: float = 0.06
    prc_lithography_scanners_produced_in_first_year_node_is_achieved: float = 20.0
    prc_additional_lithography_scanners_produced_per_year: float = 16.0
    yield_h100_sized_chips_per_wafer: float = 28

    # Energy production
    total_prc_energy_consumption_in_2026_GW: float = 1000
    annual_status_quo_growth_multiplier_of_prc_energy_consumption: float = 1.05






    annual_growth_multiplier_of_prc_compute_production_capacity: float = 2.2

    total_prc_compute_stock_in_2026_h100e_tpp: float = 1e5

    annual_growth_multiplier_in_h100e_smuggled_annually = 1.1
    max_h100e_smuggled_annually = 2e5

    total_prc_energy_consumption_in_2026_GW: float = 1000
    annual_no_ai_growth_multiplier_of_prc_energy_consumption: float = 1.05
    energy_efficiency_of_chips_produced_in_prc_relative_to_usa_allies: float = 0.2

    prc_researcher_headcount_in_2026: int = 10000
    prc_researcher_headcount_annual_growth_multiplier: float = 1.10

    ##### Policies ##### -------------------------------------

    # Export controls
    required_natsec_importance_for_inspection_backed_export_controls = LevelsOfAINationalSecurityImportance.SPACE_RACE

    # AI slowdown agreements
    required_natsec_importance_for_ai_slowdown = LevelsOfAINationalSecurityImportance.WAR_ON_TERROR
    required_perception_of_internal_takeover_risk_for_ai_slowdown = 0.1
    required_natsec_importance_for_shutdown = LevelsOfAINationalSecurityImportance.WW2

    cap_chip_production : bool = True
    stop_fab_construction : bool = True
    cap_ai_researcher_headcount : bool = True

    method_of_setting_caps : str = MethodOfSettingCaps.ten_times_reduction
    number_of_auditors_per_state_required_for_workload_monitoring: int = 30
    zero_trust_capability_cap_starts_at_what_ai_rnd_speedup: float = 4.0
    zero_trust_capability_cap_rises_by_what_multiple_per_year: float = 1.5
    maximum_zero_trust_capability_cap: float = 10.0  # e.g., 10x speedup
    when_is_perceived_takeover_risk_low_enough_for_benefits_to_justify_handoff : float = 0.005 # i.e., 0.5%

    # AI alignment agreements
    required_natsec_importance_for_international_ai_alignment_standard = LevelsOfAINationalSecurityImportance.WW2
    required_perception_of_internal_ai_takeover_risk_for_international_ai_alignment_standard = 0

    # Verification 
    verification_mechanisms_inference_only_package_development_time: float = 1.5
    verification_mechanisms_off_chip_checkpoint_monitoring_development_time: float = 3.0
    verification_mechanisms_off_chip_workload_monitoring_development_time: float = 2.5
    verification_mechanisms_on_chip_workload_monitoring_development_time: float = 2.0

    inspector_headcount_growth_rate_per_year: float = 4 # 4x growth per year
    auditor_headcount_growth_rate_per_year: float = 4 # 4x growth per year
    installer_headcount_growth_rate_per_year: float = 4 # 4x growth per year
