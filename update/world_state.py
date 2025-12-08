from classes.simulations import Updator, Time, WorldState
from classes.simulation_parameters import SimulationParameters
from typing import Dict

class WorldStateUpdator(Updator):
    simulation_parameters : SimulationParameters
    previous_world_states : Dict[Time, WorldState] 

    def get_next_world_state(self) -> WorldState:
        self.next_world_state = WorldState(
            coalitions={},
            states={},
            ai_software_developers={},
            ai_policies={}
        )
        self.next_world_state = UpdateAssets(
            simulation_parameters=self.simulation_parameters,
            previous_world_states=self.previous_world_states,
            next_world_state=self.next_world_state
        ).get_next_world_state()

        return self.next_world_state