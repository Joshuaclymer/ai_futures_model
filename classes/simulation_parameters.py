from classes.perceptions import LevelsOfAINationalSecurityImportance
import enum as Enum
from dataclasses import dataclass

class DistributionOverSimulationParameters:
    pass # Placeholder for class implementation

@dataclass
class MethodOfSettingCaps():
    ten_times_reduction = "ten_times_reduction"
    one_hundred_times_reduction = "one_hundred_times_reduction"
    covert_stock_percentile_heuristic = "covert_stock_percentile_heuristic"
    model_based_optimization = "model_based_optimization"

@dataclass
class SimulationParameters:

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
    when_is_takeover_risk_low_enough_for_benefits_to_justify_handoff : float = 0.005 # i.e., 0.5%

    # AI alignment agreements
    required_natsec_importance_for_international_ai_alignment_standard = LevelsOfAINationalSecurityImportance.WW2
    required_perception_of_internal_ai_takeover_risk_for_international_ai_alignment_standard = 0

    # Verification Mechanisms
    verification_mechanisms_inference_only_package_development_time: float = 1.5
    verification_mechanisms_off_chip_checkpoint_monitoring_development_time: float = 3.0
    verification_mechanisms_off_chip_workload_monitoring_development_time: float = 2.5
    verification_mechanisms_on_chip_workload_monitoring_development_time: float = 2.0

    inspector_headcount_growth_rate_per_year: float = 4 # 4x growth per year
    auditor_headcount_growth_rate_per_year: float = 4 # 4x growth per year
    installer_headcount_growth_rate_per_year: float = 4 # 4x growth per year