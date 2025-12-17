from classes.policies import AIPolicyTemplateIDs, AIPolicyTemplate, AIPolicy, AIPolicyTemplateIDs, VerificationProtocol
from classes.entities import Entity, NamedStates, NamedCoalitions
from dataclasses import dataclass, field

@dataclass
class USExportControls(AIPolicyTemplate):
    id : str = field(default=AIPolicyTemplateIDs.US_EXPORT_CONTROLS, init=False)

    # First, we cache some entities we'll need later.
    def __post_init__(self):
        self.USA = self.previous_world_state.states[NamedStates.USA]
        self.USA_allies = self.previous_world_state.coalitions[NamedCoalitions.USA_ALLIES].members
        self.USA_rivals = self.previous_world_state.coalitions[NamedCoalitions.USA_RIVALS].members

    # US export controls might start to rely on inspection-based verification mechanisms as AI becomes more important to national security.
    def get_verification_protocol(self) -> VerificationProtocol:
        inspection_is_politically_viable = \
            self.previous_world_state.states[NamedStates.USA].perceptions.ai_national_security_importance.value >= self.simulation_parameters.required_natsec_importance_for_inspection_backed_export_controls
        if inspection_is_politically_viable:
            return VerificationProtocol(
                max_inspector_headcount=100,
            )
        else:
            return None
    
    # The US starts hiring inspectors after it perceives AI as sufficiently important
    def entities_preparing_verification_protocol(self) -> dict[Entity, VerificationProtocol]:
        verification_protocol = self.get_verification_protocol()
        if verification_protocol:
            return {entity: verification_protocol for entity in self.USA_allies}
        else:
            return {}
    
    # Finally, we specify the export control policy itself
    def get_policies(self) -> list[AIPolicy]:
        return [AIPolicy(
            template_id = self.id,
            entities_subject_to_policy = self.USA_allies,
            entities_verifying_compliance = [self.USA],
            compute_export_blacklist = self.USA_rivals,
            SME_export_blacklist = self.USA_rivals,
            verification_protocol = self.get_verification_protocol(),
        )]