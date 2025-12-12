from classes.policies import AIPolicyTemplateIDs, AIPolicyTemplate, AIPolicy, VerificationMechanismIDs, VerificationProtocol
from classes.simulations import SimulationParameters, WorldState
from classes.entities import Entity, NamedStates, NamedCoalitions, State
from dataclasses import dataclass, field
from verification_mechanisms import get_verification_mechanisms, get_available_verification_mechanisms
from classes.simulation_parameters import MethodOfSettingCaps
from classes.software_progress import AISoftwareCapabilityCap

@dataclass
class InternationalAIRndSlowdown(AIPolicyTemplate):
    id : str = field(default=AIPolicyTemplateIDs.INTERNATIONAL_AI_RND_SLOWDOWN, init=False)

    max_inspector_headcount=100
    max_workload_auditor_headcount=100
    max_installer_headcount=100

    # Once states are willing to slow down AI, they start preparing the following verification mechanisms.
    def get_verification_protocol_to_prepare(self) -> VerificationProtocol:
        verification_mechanisms_list = get_verification_mechanisms(self.simulation_parameters)
        return VerificationProtocol(
            max_inspector_headcount=self.max_inspector_headcount,
            max_workload_auditor_headcount=self.max_workload_auditor_headcount,
            max_installer_headcount=self.max_installer_headcount,
            verification_mechanisms_in_order_of_precedence=[
                verification_mechanisms_list[VerificationMechanismIDs.ON_CHIP_AND_OFF_CHIP_WORKLOAD_MONITORING],
                verification_mechanisms_list[VerificationMechanismIDs.OFF_CHIP_WORKLOAD_MONITORING],
                verification_mechanisms_list[VerificationMechanismIDs.INFERENCE_ONLY_PACKAGE],
            ]
        )
 
    # The US and China start preparing verification capacity once they think AI is sufficiently important and dangerous
    def entities_preparing_verification_protocol(self) -> dict[Entity, VerificationProtocol]:

        # We're only considering the USA and PRC for now
        states = [self.previous_world_state.states[NamedStates.USA], self.previous_world_state.states[NamedStates.PRC]]

        states_preparing_verification_protocol = {}

        for state in states:
            # States need to perceive AI as sufficiently important to warrant slowdown measures
            minimum_importance_for_slowdown = self.simulation_parameters.required_natsec_importance_for_ai_slowdown
            if state.perceptions.ai_national_security_importance.value <= minimum_importance_for_slowdown:
                continue

            # States also need to perceive AI as sufficiently dangerous to warrant slowdown measures
            minimum_risk_for_slowdown = self.simulation_parameters.required_perception_of_internal_takeover_risk_for_ai_slowdown
            if state.perceptions.internal_ai_takeover_risk.value <= minimum_risk_for_slowdown:
                continue

            # If both conditions are met, the state prepares the verification protocol for the slowdown phase
            states_preparing_verification_protocol[state] = self.get_verification_protocol_to_prepare()
    
        return states_preparing_verification_protocol
    
    def get_policies(self) -> list[AIPolicy]:

        # We're only considering the USA and PRC for now
        states = [self.previous_world_state.states[NamedStates.USA], self.previous_world_state.states[NamedStates.PRC]]

        # Before states start preparing verification, no agreement is in force.
        entities_preparing = self.entities_preparing_verification_protocol()
        if len(entities_preparing) < 2:
            return None
        
        # After states start preparing verification, they only actually slow down AI after two conditions are met:
        # Condition 1. States have been reporting compute for at least 6 months
        # Condition 2. At least one verification mechanism for slowing down AI is politically and technically feasible to implement

        # Condition 1. States have been reporting compute for at least 6 months
        compute_reporting_start_time = min([time.year for time, world_state in self.previous_world_states.items() if world_state.ai_policies.get(AIPolicyTemplateIDs.INTERNATIONAL_AI_RND_SLOWDOWN) is not None])
        months_since_compute_reporting_started = (self.current_time.year - compute_reporting_start_time) * 12
        compute_reporting_for_at_least_6_months = months_since_compute_reporting_started > 6

        # Condition 2. At least one verification mechanism for slowing down AI is politically and technically feasible to implement
        verification_mechanisms_list = get_verification_mechanisms(self.simulation_parameters)
        verification_mechanisms_for_slowdown = [
            verification_mechanisms_list[VerificationMechanismIDs.ON_CHIP_AND_OFF_CHIP_WORKLOAD_MONITORING],
            verification_mechanisms_list[VerificationMechanismIDs.OFF_CHIP_WORKLOAD_MONITORING],
            verification_mechanisms_list[VerificationMechanismIDs.INFERENCE_ONLY_PACKAGE]
        ]
        verification_mechanisms_available = [mechanism for mechanism in verification_mechanisms_for_slowdown if all([mechanism in get_available_verification_mechanisms(state) for state in states])]
        states_are_willing_to_shut_down_compute = all([state.perceptions.willingness_to_shutdown_unverified_compute.value >= self.simulation_parameters.required_natsec_importance_for_shutdown for state in states])
        if states_are_willing_to_shut_down_compute:
            verification_mechanisms_available.append(verification_mechanisms_list[VerificationMechanismIDs.SHUTDOWN])
        at_least_one_verification_mechanism_is_available = len(verification_mechanisms_available) > 0

        slowdown_has_begun = compute_reporting_for_at_least_6_months and at_least_one_verification_mechanism_is_available

        # The form of the agreement depends on whether states can monitor workloads or not.
        # If states can monitor workloads, they can enforce a capability cap and limit experiment compute.

        # States can monitor workloads if two conditions are met:
        # Condition 1. Off-chip workload monitoring technology is available
        # Condition 2. States have sufficient auditor capacity to monitor workloads

        off_chip_workload_monitoring_available = verification_mechanisms_list[VerificationMechanismIDs.OFF_CHIP_WORKLOAD_MONITORING] in verification_mechanisms_available
        sufficient_auditor_capacity = all([state.verification_capacity.workload_auditors >= self.simulation_parameters.number_of_auditors_per_state_required_for_workload_monitoring for state in states])
        monitor_workloads = off_chip_workload_monitoring_available and sufficient_auditor_capacity

        # Now we can specify the verification protocol for the AI slowdown agreement

        verification_protocol = VerificationProtocol(
            max_inspector_headcount=self.max_inspector_headcount,
            max_installer_headcount=self.max_installer_headcount if slowdown_has_begun else 0,
            max_workload_auditor_headcount=self.max_workload_auditor_headcount if (slowdown_has_begun and monitor_workloads) else 0,
            verification_mechanisms_in_order_of_precedence=[verification_mechanisms_available]
        ),

        # Before we can specify the agreement itself, we need to calculate the compute and capability caps (if applicable)
        us_experiment_compute_cap_tpp_h100e, us_ai_researcher_headcount_cap, \
            us_compute_production_cap_tpp_h100e, us_compute_production_capacity_cap_tpp_h100e = self.calculate_resource_caps(NamedCoalitions.USA_ALLIES)

        prc_experiment_compute_cap_tpp_h100e, prc_ai_researcher_headcount_cap, \
            prc_compute_production_cap_tpp_h100e, prc_compute_production_capacity_cap_tpp_h100e = self.calculate_resource_caps(NamedStates.PRC)
        
        capability_cap = self.calculate_ai_capability_cap() if monitor_workloads else None

        # Now we format the final agreement policies - one for the US and one for the PRC
        return [
                AIPolicy(
                    entities_subject_to_policy = [self.previous_world_state.states[NamedStates.PRC]],
                    entities_verifying_compliance = [self.previous_world_state.states[NamedStates.USA]],
                    compute_location_reporting = True,
                    non_inference_compute_cap_tpp_h100e = prc_experiment_compute_cap_tpp_h100e if (not monitor_workloads) and slowdown_has_begun else None,
                    experiment_compute_cap_tpp_h100e = prc_experiment_compute_cap_tpp_h100e if monitor_workloads and slowdown_has_begun else None,
                    ai_capability_cap = capability_cap if monitor_workloads and slowdown_has_begun else None,
                    ai_researcher_headcount_cap = prc_ai_researcher_headcount_cap if monitor_workloads and slowdown_has_begun else None,
                    compute_production_cap_monthly_tpp_h100e = prc_compute_production_cap_tpp_h100e if slowdown_has_begun else None,
                    compute_production_capacity_cap_monthly_tpp_h100e = prc_compute_production_capacity_cap_tpp_h100e if slowdown_has_begun else None,
                    verification_protocol = verification_protocol
                ),
                AIPolicy(
                    entities_subject_to_policy = [self.previous_world_state.states[NamedCoalitions.USA_ALLIES]],
                    entities_verifying_compliance = [self.previous_world_state.states[NamedStates.PRC]],
                    compute_location_reporting = True,
                    non_inference_compute_cap_tpp_h100e = us_experiment_compute_cap_tpp_h100e if (not monitor_workloads) and slowdown_has_begun else None,
                    experiment_compute_cap_tpp_h100e = us_experiment_compute_cap_tpp_h100e if monitor_workloads and slowdown_has_begun else None,
                    ai_capability_cap = capability_cap if monitor_workloads and slowdown_has_begun else None,
                    ai_researcher_headcount_cap = us_ai_researcher_headcount_cap if monitor_workloads and slowdown_has_begun else None,
                    compute_production_cap_monthly_tpp_h100e = us_compute_production_cap_tpp_h100e if slowdown_has_begun else None,
                    compute_production_capacity_cap_monthly_tpp_h100e = us_compute_production_capacity_cap_tpp_h100e if slowdown_has_begun else None,
                    verification_protocol = verification_protocol
                ),
        ]

    def calculate_resource_caps(self, entity) -> tuple[float]:

        # --- Determine reduction factor based on method_of_setting_caps ---
        method = self.simulation_parameters.method_of_setting_caps
        if method == MethodOfSettingCaps.ten_times_reduction:
            reduction_factor = 10.0
        elif method == MethodOfSettingCaps.one_hundred_times_reduction:
            reduction_factor = 100.0
        else:
            reduction_factor = 10.0

        # TODO write logic after writing assets code
        experiment_cap = 1e6 / reduction_factor  # Placeholder value
        experiment_plus_training_cap = 5e6 / reduction_factor  # Placeholder value
        researcher_cap = 1000 / reduction_factor  # Placeholder value
        chip_production_cap = 2e6 / reduction_factor  # Placeholder value
        return experiment_cap, experiment_plus_training_cap, researcher_cap, chip_production_cap

    def calculate_ai_capability_cap(self) -> AISoftwareCapabilityCap:

        previous_ai_slowdown_policy = self.previous_world_state.ai_policies.get(AIPolicyTemplateIDs.INTERNATIONAL_AI_RND_SLOWDOWN)
        previous_capability_cap = previous_ai_slowdown_policy[0].ai_capability_cap if previous_ai_slowdown_policy else None

        if previous_capability_cap is None:
            return AISoftwareCapabilityCap(
            cap_of_ai_sw_progress_mult_ref_present_day=self.simulation_parameters.zero_trust_capability_cap_starts_at_what_ai_rnd_speedup,
            cap_of_time_horizon=None
        )
        else: 
            years_since_previous_cap = self.current_time.year - self.previous_world_state.current_time.year
            growth_rate = self.simulation_parameters.zero_trust_capability_cap_rises_by_what_multiple_per_year
            cap_given_growth = previous_capability_cap.cap_of_ai_sw_progress_mult_ref_present_day * (growth_rate ** years_since_previous_cap)
            max_cap = self.simulation_parameters.maximum_zero_trust_capability_cap
            new_cap = min(cap_given_growth, max_cap)
        return AISoftwareCapabilityCap(
            cap_of_ai_sw_progress_mult_ref_present_day=new_cap,
        )