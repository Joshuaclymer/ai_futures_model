from classes.policies import VerificationMechanism, VerificationMechanismIDs
from classes.simulations import SimulationParameters
from classes.entities import Entity
import enum as Enum

def get_verification_mechanisms(simulation_parameters: SimulationParameters) -> None:
    return {
        # VerificationMechanismIDs.ON_CHIP_TRACKING_DEVICES: VerificationMechanism(
        #     id = VerificationMechanismIDs.ON_CHIP_TRACKING_DEVICES,
        #     securitization_and_development_latency_years = simulation_parameters.development_time_of_on_chip_tracking_years
        # ),
        VerificationMechanismIDs.SHUTDOWN: VerificationMechanism(
            id = VerificationMechanismIDs.SHUTDOWN,
            securitization_and_development_latency_years = 0.0
        ),
        VerificationMechanismIDs.INFERENCE_ONLY_PACKAGE: VerificationMechanism(
            id = VerificationMechanismIDs.INFERENCE_ONLY_PACKAGE,
            securitization_and_development_latency_years = simulation_parameters.development_time_of_inference_only_package_years
        ),
        VerificationMechanismIDs.OFF_CHIP_WORKLOAD_MONITORING: VerificationMechanism(
            id = VerificationMechanismIDs.OFF_CHIP_WORKLOAD_MONITORING, 
            securitization_and_development_latency_years = simulation_parameters.development_time_of_off_chip_workload_monitoring_years
        ),
        VerificationMechanismIDs.ON_CHIP_WORKLOAD_MONITORING: VerificationMechanism(
            id = VerificationMechanismIDs.ON_CHIP_WORKLOAD_MONITORING, 
            securitization_and_development_latency_years = simulation_parameters.development_time_of_on_chip_workload_monitoring_years
        ),
    }

def get_available_verification_mechanisms(entity: Entity) -> list[VerificationMechanism]:
    available_mechanisms = []
    for verification_mechanism in get_verification_mechanisms(entity.simulation_parameters).values():
        if entity.verification_capacity.years_spent_developing_or_securing_verification_technology.get(verification_mechanism, 0) >= verification_mechanism.securitization_and_development_latency_years:
            available_mechanisms.append(verification_mechanism)
    return available_mechanisms
