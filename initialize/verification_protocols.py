from enum import Enum
from typing import List
from entities import Entity
from classes.verification_protocols import VerificationProtocol, VerificationCapacity

class CurrentUSExportControls(VerificationProtocol):
    name = "current_us_export_controls"
    parties_to_protocol: List[Entity]

    def minumum_verification_capacity_requirements_met(self) -> bool:

        if (state_verification_capacity.verification_rand_and_securitization_person_years >= required_person_years and
            state_verification_capacity.verification_installer_headcount >= required_installer_headcount and
            state_verification_capacity.site_inspector_headcount >= required_site_inspector_headcount and
            state_verification_capacity.workload_auditor_headcount >= required_workload_auditor_headcount):
            return True
        return False
    
class AggressiveUSExportControls(VerificationProtocol):
class VerificationProtocol(ABC):
    name: str
    trusted_by: List[Entity]

    @abstractmethod
    def can_protocol_be_trusted(self, state_verification_capacity: "VerificationCapacity", other_parties_to_policy: List[Entity]) -> bool:
        pass

class VerificationProtocols(Enum):
    chip_tracking = ChipTracking()
