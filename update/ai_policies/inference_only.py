from classes.policies import VerificationMeasure
from classes.entities import Entity
from classes.perceptions import LevelsOfAINationalSecurityPrioritization
from typing import List


class InferenceOnly:
    id = "inference_only"

    def __init__(self, participating_states: List[Entity], non_inference_cap_tpp_h100e: float):
        self.entities_subject_to_policy = participating_states
        self.entities_verifying_compliance = participating_states
        self.verification_measures = [VerificationMeasure.INFERENCE_ONLY_PACKAGE, VerificationMeasure.WORKLOAD_AUDITING]
        self.non_inference_compute_cap_tpp_h100e = non_inference_cap_tpp_h100e

    def is_politically_viable(self) -> bool:
        for state in self.entities_subject_to_policy:
            if state.assessments.ai_national_security_prioritization < LevelsOfAINationalSecurityPrioritization.SIMILAR_TO_FIGHTING_THE_WAR_ON_TERROR:
                return False

    def is_technically_viable(self) -> bool:
        return True  # Placeholder logic
