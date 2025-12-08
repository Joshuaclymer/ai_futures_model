from classes.policies import AIPolicies, AIPolicy, VerificationMeasure, AIPolicyIDs
from classes.entities import Entity
from classes.perceptions import LevelsOfAINationalSecurityImportance
from typing import List

class AggressiveUSExportControls(AIPolicy):
    id = AIPolicyIDs.AGGRESSIVE_EXPORT_CONTROLS

    def __init__(self, USA: Entity, USA_allies: List[Entity], USA_rivals: List[Entity]):
        self.USA = USA
        self.entities_subject_to_policy = USA_allies
        self.entities_verifying_compliance = [USA]
        self.compute_export_blacklist = USA_rivals
        self.SME_export_blacklist = USA_rivals
        self.verification_measures = [VerificationMeasure.ONSITE_INSPECTION]
        self.compute_location_reporting = True

    def is_politically_viable(self) -> bool:
        return self.USA.assessments.ai_national_security_importance == LevelsOfAINationalSecurityImportance.SIMILAR_TO_BEATING_RUSSIA_TO_THE_MOON

    def is_technically_viable(self) -> bool:
        return True  # Placeholder logic
