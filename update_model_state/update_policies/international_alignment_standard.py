from classes.policies import AIPolicyTemplateIDs, AIPolicyTemplate, AIPolicy, AIPolicyTemplateIDs, VerificationProtocol, VerificationMechanismIDs
from classes.entities import Entity, NamedStates, NamedCoalitions
from dataclasses import dataclass, field
from verification_mechanisms import get_verification_mechanisms, get_available_verification_mechanisms

@dataclass
class InternationalAlignmentStandard(AIPolicyTemplate):
    id : str = field(default=AIPolicyTemplateIDs.INTERNATIONAL_ALIGNMENT_STANDARD, init=False)

    # Alignment agreements rely on on-chip workload monitoring
    def get_verification_protocol(self) -> VerificationProtocol:
        return VerificationProtocol(
                max_inspector_headcount=100,
                max_workload_auditor_headcount=100,
                max_installer_headcount=100,
                verification_mechanisms = [get_verification_mechanisms(
                    self.simulation_parameters
                )[VerificationMechanismIDs.ON_CHIP_AND_OFF_CHIP_WORKLOAD_MONITORING]]
            )
    # The US and PRC start preparing on-chip workload monitoring after AI becomes sufficiently important to national security
    def entities_preparing_verification_protocol(self) -> dict[Entity, VerificationProtocol]:
        minimum_importance = self.simulation_parameters.required_natsec_importance_for_international_ai_alignment_standard
        states = [self.previous_world_state.states[NamedStates.USA], self.previous_world_state.states[NamedStates.PRC]]
        return {state: self.get_verification_protocol() for state in states if state.perceptions.ai_national_security_importance.value >= minimum_importance}
    
    def get_policies(self) -> list[AIPolicy]:
        # First check if verification capacity is sufficient

        # is on chip workload monitoring ready?
        states = self.entities_preparing_verification_protocol().keys()
        verification_mechanisms_ready = all([VerificationMechanismIDs.ON_CHIP_AND_OFF_CHIP_WORKLOAD_MONITORING in get_available_verification_mechanisms(state) for state in states])

        # is there enough auditor capacity?
        sufficient_auditor_capacity = all([state.verification_capacity.workload_auditors >= self.simulation_parameters.number_of_auditors_per_state_required_for_workload_monitoring for state in states])

        if not (verification_mechanisms_ready and sufficient_auditor_capacity):
            return []
        
         # If so, implement the alignment standard policy
        return [
            AIPolicy(
                template_id = self.id,
                entities_subject_to_policy = [self.previous_world_state.coalitions[NamedCoalitions.USA_ALLIES]],
                entities_verifying_compliance = [self.previous_world_state.states[NamedStates.PRC]],
                compute_reporting=True,
                international_alignment_standard = True,
                verification_protocol = self.get_verification_protocol(),
            ),
            AIPolicy(
                template_id = self.id,
                entities_subject_to_policy = [self.previous_world_state.coalitions[NamedStates.PRC]],
                entities_verifying_compliance = [self.previous_world_state.states[NamedStates.USA]],
                compute_reporting=True,
                international_alignment_standard = True,
                verification_protocol = self.get_verification_protocol(),
            ),
        ]