from typing import List, Optional, Dict
from classes.entities import Entity
from classes.software_progress import AISoftwareCapabilityCap
from abc import ABC, abstractmethod
from enum import Enum

class AIPolicy(ABC):
    id : str
    entities_subject_to_policy: List[Entity]
    entities_verifying_compliance: List[Entity]
    verification_measures: Optional["VerificationMeasure"] = []

    # Export controls 
    compute_export_blacklist : List[Entity] = []
    SME_export_blacklist: List[Entity] = []

    # Location reporting (e.g. for trust building or to help enforce export controls)
    compute_location_reporting : bool = False

    # Inference-only
    non_inference_compute_cap_tpp_h100e: Optional[float] = None  # in H100e TPP
    shut_down_sites_without_inference_verification_installed: bool = False

    # AI R&D caps
    experiment_compute_cap_tpp_h100e: Optional[float] = None  # in H100e TPP
    ai_researcher_headcount_cap: Optional[int] = None  # number of researchers
    ai_capability_cap: Optional[AISoftwareCapabilityCap] = None # AI R&D speedup or time horizon
    
    # Compute production caps
    compute_production_cap_monthly_tpp_h100e: Optional[float] = None  # units of H100e TPP per month
    compute_production_capacity_cap_monthly_tpp_h100e: Optional[float] = None # Can states keep building fabs?

    # Alignment standard
    alignment_to_international_controllability_standard: bool = False

    @abstractmethod
    def is_politically_viable(self) -> bool:
        pass

    @abstractmethod
    def is_technically_viable(self) -> bool:
        pass

class AIPolicyIDs():
    CURRENT_EXPORT_CONTROLS = "export_controls"
    AGGRESSIVE_EXPORT_CONTROLS = "aggressive_export_controls"
    COMPUTE_LOCATION_REPORTING = "compute_location_reporting"
    INFERENCE_ONLY_POLICY = "inference_only_policy"
    AI_RESEARCHER_HEADCOUNT_CAP = "ai_researcher_headcount_cap"
    EXPERIMENT_COMPUTE_CAP = "experiment_compute_cap"
    AI_CAPABILITY_CAP = "ai_capability_cap"
    COMPUTE_PRODUCTION_CAP = "compute_production_cap"
    ALIGNMENT_TO_INTERNATIONAL_CONTROLLABILITY_STANDARD = "alignment_to_international_controllability_standard"

class VerificationMeasure(Enum):
    ON_CHIP_TRACKING_DEVICES = "on_chip_tracking_devices"
    ONSITE_INSPECTION = "onsite_inspection"
    INFERENCE_ONLY_PACKAGE = "inference_only_package"
    WORKLOAD_AUDITING = "workload_auditing"
    EVALUATION_BASED_CAPABILITY_CAP = "evaluation_based_capability_cap"

class VerificationCapacity:
    years_spent_developing_or_securing_protocols: Dict[VerificationMeasure, float]
    national_intelligence_spend_USD: float 
    verification_installer_headcount: int
    site_inspector_headcount: int
    workload_auditor_headcount: int
