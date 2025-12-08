from classes.policies import VerificationMeasure
from classes.entities import Entity
from classes.perceptions import LevelsOfAINationalSecurityPrioritization
from typing import List


class ComputeReporting:
    id = "compute_reporting"

    def __init__(self, participating_states: List[Entity]):
        self.entities_subject_to_policy = participating_states
        self.entities_verifying_compliance = participating_states
        self.verification_measures = [VerificationMeasure.ONSITE_INSPECTION]
        self.compute_location_reporting = True

    def is_politically_viable(self) -> bool:
        for state in self.entities_subject_to_policy:
            if state.assessments.ai_national_security_prioritization < LevelsOfAINationalSecurityPrioritization.SIMILAR_TO_FIGHTING_THE_WAR_ON_TERROR:
                return False

    def is_technically_viable(self) -> bool:
        return True
