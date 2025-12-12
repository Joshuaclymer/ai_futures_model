from classes.simulations import Updater, Time, WorldState
from classes.simulation_parameters import SimulationParameters

class WorldStateUpdater(Updater):
    simulation_parameters : SimulationParameters
    previous_world_states : dict[Time, WorldState] 
    next_world_state : WorldState

    def get_next_world_state(self) -> WorldState:
        
        # Determine which attacks happen in this time step - TODO

        # self.next_world_state = UpdateAttacksPolitical(
        #     next_world_state=self.next_world_state,
        #     simulation_parameters=self.simulation_parameters,
        #     previous_world_states=self.previous_world_states,
        # ).get_next_world_state()

        # self.next_world_state = UpdateAttacksCyber(
        #     next_world_state=self.next_world_state,
        #     simulation_parameters=self.simulation_parameters,
        #     previous_world_states=self.previous_world_states,
        # ).get_next_world_state()

        # self.next_world_state = UpdateAttacksKinetic(
        #     next_world_state=self.next_world_state,
        #     simulation_parameters=self.simulation_parameters,
        #     previous_world_states=self.previous_world_states,
        # ).get_next_world_state()

        # Update state of assets

        self.next_world_state = UpdateEnergyGenerationAssets(
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        self.next_world_state = UpdateRobotAssets(
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        self.next_world_state = UpdateWeaponAssets(
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        self.next_world_state = UpdateFabAssets(
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        self.next_world_state = UpdateComputeAssets(
            next_world_state=self.next_world_state,
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        self.next_world_state = UpdateDatacenterAssets(
            next_world_state=self.next_world_state,
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        # Update state of software progress

        self.next_world_state = UpdateSoftwareProgress(
            next_world_state=self.next_world_state,
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        # Update state economies and technology levels

        self.next_world_state = UpdateStateEconomies(
            next_world_state=self.next_world_state,
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        self.next_world_state = UpdateTechnologyProgress(
            next_world_state=self.next_world_state,
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        # Update the levels of various risks

        self.next_world_state = UpdateTakeoverRiskLabPowerGrab(
            next_world_state=self.next_world_state,
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        self.next_world_state = UpdateTakeoverRiskMisalignedAI(
            next_world_state=self.next_world_state,
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        # Update perceptions of ai

        self.next_world_state = UpdatePerceptionsBlackProjectExistence(
            next_world_state=self.next_world_state,
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        self.next_world_state = UpdatePerceptionsNatsecImportance(
            next_world_state=self.next_world_state,
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        self.next_world_state = UpdatePerceptionsTakeoverRisks(
            next_world_state=self.next_world_state,
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        # Update policies

        self.next_world_state = UpdatePolicies(
            next_world_state=self.next_world_state,
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
        ).get_next_world_state()

        return self.next_world_state

