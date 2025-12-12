from classes.simulations import Updater, SimulationParameters, Time, WorldState
from us_export_controls import USExportControls
from international_ai_rnd_slowdown import InternationalAIRndSlowdown
from international_alignment_standard import InternationalAlignmentStandard

class UpdatePolicies(Updater):
    simulation_parameters : SimulationParameters
    previous_world_states : dict[Time, WorldState] 
    next_world_state : WorldState

    def get_next_world_state(self) -> WorldState:
        previous_world_state = self.simulation_run.world_states[max(self.simulation_run.world_states.keys())]
        policy_templates = [
            USExportControls(simulation_parameters=self.simulation_parameters, previous_world_state=previous_world_state, previous_world_states=self.previous_world_states, current_time=self.next_world_state.current_time),
            InternationalAIRndSlowdown(simulation_parameters=self.simulation_parameters, previous_world_state=previous_world_state, previous_world_states=self.previous_world_states, current_time=self.next_world_state.current_time),
            InternationalAlignmentStandard(simulation_parameters=self.simulation_parameters, previous_world_state=previous_world_state, previous_world_states=self.previous_world_states, current_time=self.next_world_state.current_time),
        ]

        # Update verification capacity

        entities_building_verification_capacity = [
            entity
            for policy_template in policy_templates
            for entity in policy_template.entities_preparing_verification_protocol().keys()
        ]

        for entity in entities_building_verification_capacity:
            # Get protocols entity is preparing
            protocols_entity_is_preparing = [
                policy_template.entities_preparing_verification_protocol()[entity]
                for policy_template in policy_templates if policy_template.entities_preparing_verification_protocol().get(entity) is not None
            ]

            # Get change in year (for growth rates)
            time_delta_years = self.next_world_state.current_time.year - previous_world_state.current_time.year

            # Increase inspector headcount
            current_inspector_headcount = entity.verification_capacity.inspector_headcount
            global_max = max([protocol.max_inspector_headcount for protocol in protocols_entity_is_preparing])
            growth_factor = self.simulation_parameters.inspector_headcount_growth_rate_per_year ** time_delta_years
            new_inspector_headcount_given_growth = growth_factor * (current_inspector_headcount if current_inspector_headcount is not None else 10)
            entity.verification_capacity.inspector_headcount = min(global_max, new_inspector_headcount_given_growth)

            # Increase auditor headcount
            current_auditor_headcount = entity.verification_capacity.auditor_headcount
            global_max = max([protocol.max_auditor_headcount for protocol in protocols_entity_is_preparing])
            growth_factor = self.simulation_parameters.auditor_headcount_growth_rate_per_year ** time_delta_years
            new_auditor_headcount_given_growth = growth_factor * (current_auditor_headcount if current_auditor_headcount is not None else 10) 
            entity.verification_capacity.auditor_headcount = min(global_max, new_auditor_headcount_given_growth)

            # Increase installer headcount
            current_installer_headcount = entity.verification_capacity.installer_headcount
            global_max = max([protocol.max_installer_headcount for protocol in protocols_entity_is_preparing()])
            growth_factor = self.simulation_parameters.installer_headcount_growth_rate_per_year ** time_delta_years
            new_installer_headcount_given_growth = growth_factor * (current_installer_headcount if current_installer_headcount is not None else 10) 
            entity.verification_capacity.installer_headcount = min(global_max, new_installer_headcount_given_growth)

            # Increase verification mechanism development and securitization
            verification_mechanisms_entity_is_preparing = [protocol.verification_mechanisms_in_order_of_precedence for protocol in protocols_entity_is_preparing]
            for mechanism in verification_mechanisms_entity_is_preparing:
                entity.verification_capacity.years_spent_developing_or_securing_verification_mechanisms[mechanism] = \
                    entity.verification_capacity.years_spent_developing_or_securing_verification_mechanisms.get(mechanism, 0) + time_delta_years

        # Update policies
        policies = []
        for template in policy_templates:
            policies.extend(template.get_policies())
        self.next_world_state.ai_policies = policies