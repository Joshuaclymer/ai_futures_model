from abc import ABC, abstractmethod
from dataclasses import dataclass
from classes.entities import Entity
from classes.software_progress import AISoftwareCapabilityCap
from classes.simulations import SimulationParameters, Time, WorldState

class AIPolicy:
    template_id : str

    entities_subject_to_policy: list[Entity]
    entities_verifying_compliance: list[Entity]
    verification_protocol: "VerificationProtocol | None"

    # Export controls 
    compute_export_blacklist : list[Entity] = []
    SME_export_blacklist: list[Entity] = []

    # Location reporting (e.g. for trust building or to help enforce export controls)
    compute_location_reporting : bool = False

    # AI researcher headcount caps
    ai_researcher_headcount_cap: int | None = None  # number of researchers

    # AI R&D caps
    non_inference_compute_cap_tpp_h100e: float | None = None  # in H100e TPP
    experiment_compute_cap_tpp_h100e: float | None = None  # in H100e TPP
    ai_capability_cap: AISoftwareCapabilityCap | None = None # AI R&D speedup or time horizon

    # Compute production caps
    compute_production_cap_monthly_tpp_h100e: float | None = None  # units of H100e TPP per month
    compute_production_capacity_cap_monthly_tpp_h100e: float | None = None # Can states keep building fabs?

    # Alignment standard
    international_alignment_standard: bool = False

@dataclass
class AIPolicyTemplate(ABC):
    id : str
    simulation_parameters : SimulationParameters
    current_time : Time
    previous_world_state : WorldState
    previous_world_states : dict[Time, WorldState]

    @abstractmethod
    def entities_preparing_verification_protocol(self) -> dict[Entity, "VerificationProtocol"]:
        pass

    @abstractmethod
    def get_policies(self) -> list[AIPolicy]:
        pass

class AIPolicyTemplateIDs():
    US_EXPORT_CONTROLS = "US_export_controls"
    INTERNATIONAL_AI_RND_SLOWDOWN = "international_ai_rnd_slowdown"
    INTERNATIONAL_ALIGNMENT_STANDARD = "international_alignment_standard"

class VerificationProtocol():
    max_inspector_headcount : int | None = None  # max number of inspectors to hire
    max_workload_auditor_headcount : int | None = None  # max number of inspectors to hire
    max_installer_headcount : int | None = None  # max number of installers to hire
    verification_mechanisms_in_order_of_precedence: list["VerificationMechanism"]
    # Verification mechanisms with higher precedence replace verification mechanisms with lower precedence as they are rolled out.
    # In the list above, verification mechanisms with higher precedence appear earlier in the list.

class VerificationMechanism():
    id : str
    securitization_and_development_latency_years : float

class VerificationMechanismIDs:
    ON_CHIP_TRACKING_DEVICES = "on_chip_tracking_devices"
    SHUTDOWN = "shutdown"
    INFERENCE_ONLY_PACKAGE = "inference_only_package"
    OFF_CHIP_WORKLOAD_MONITORING = "off_chip_workload_monitoring"
    ON_CHIP_AND_OFF_CHIP_WORKLOAD_MONITORING = "on_chip_and_off_chip_workload_monitoring"

class VerificationCapacity:
    national_intelligence_spend_USD: float 

    years_spent_developing_or_securing_verification_mechanisms: dict[VerificationMechanism, float]
    verification_installer_headcount: int
    site_inspector_headcount: int
    workload_auditor_headcount: int