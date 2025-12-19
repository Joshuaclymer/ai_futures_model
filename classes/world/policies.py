"""
AI Policy classes.

Defines policy structures that can restrict entities.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any

from classes.world.software_progress import AISoftwareCapabilityCap


@dataclass
class AIPolicy:
    """An AI policy with various provisions."""
    template_id: str = ""
    entities_subject_to_policy_ids: List[str] = field(default_factory=list)
    entities_verifying_compliance_ids: List[str] = field(default_factory=list)
    verification_protocol: Optional[Any] = None

    # Export controls
    compute_export_blacklist_ids: List[str] = field(default_factory=list)
    SME_export_blacklist_ids: List[str] = field(default_factory=list)

    # Location reporting
    compute_location_reporting: bool = False

    # AI researcher headcount caps
    ai_researcher_headcount_cap: Optional[int] = None

    # AI R&D caps
    non_inference_compute_cap_tpp_h100e: Optional[float] = None
    experiment_compute_cap_tpp_h100e: Optional[float] = None
    ai_capability_cap: Optional[AISoftwareCapabilityCap] = None

    # Compute production caps
    compute_production_cap_monthly_tpp_h100e: Optional[float] = None
    compute_production_capacity_cap_monthly_tpp_h100e: Optional[float] = None

    # Alignment standard
    international_alignment_standard: bool = False


class AIPolicyTemplateIDs:
    """Standard policy template identifiers."""
    US_EXPORT_CONTROLS = "US_export_controls"
    INTERNATIONAL_AI_R_AND_D_SLOWDOWN = "international_ai_r_and_d_slowdown"
    INTERNATIONAL_ALIGNMENT_STANDARD = "international_alignment_standard"


@dataclass
class VerificationProtocol:
    """A verification protocol for policy compliance."""
    max_inspector_headcount: Optional[int] = None
    max_workload_auditor_headcount: Optional[int] = None
    max_installer_headcount: Optional[int] = None
    verification_mechanism_ids: List[str] = field(default_factory=list)


class VerificationMechanismIDs:
    """Standard verification mechanism identifiers."""
    ON_CHIP_TRACKING_DEVICES = "on_chip_tracking_devices"
    SHUTDOWN = "shutdown"
    INFERENCE_ONLY_PACKAGE = "inference_only_package"
    OFF_CHIP_WORKLOAD_MONITORING = "off_chip_workload_monitoring"
    ON_CHIP_AND_OFF_CHIP_WORKLOAD_MONITORING = "on_chip_and_off_chip_workload_monitoring"


@dataclass
class VerificationCapacity:
    """Verification capacity of an entity."""
    national_intelligence_spend_USD: float = 0.0
    verification_installer_headcount: int = 0
    site_inspector_headcount: int = 0
    workload_auditor_headcount: int = 0
