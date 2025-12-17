from initialize_model_state.world_states import initialize_world_states
from update_model_state.world_state import WorldStateUpdater
from metric_classes.metrics import Metrics
from model_state_classes.world_state import WorldState
from input_parameters.simulation_parameters import ModelParameters, SimulationParameters
from dataclasses import dataclass

@dataclass
class World:
    state: WorldState
    metrics: Metrics # World properties you are interested in tracking but don't affect state evolution

class SimulationRun:
    trajectory: dict["Time", "World"]  # Mapping from Time to WorldState
    simulation_parameters : SimulationParameters

class AIFuturesSimulator:
    distribution_over_simulation_parameters: ModelParameters

    def run_simulation(self, simulation_parameters: SimulationParameters) -> SimulationRun:
        simulation_run = SimulationRun(
            simulation_parameters=simulation_parameters,
            trajectory=initialize_world_states()
        )
        world_state_Updater = WorldStateUpdater(
            simulation_run = simulation_run,
        )

        start_time = max(simulation_run.trajectory.keys()).year + simulation_parameters.year_increment
        end_time = simulation_parameters.simulation_end_year
        increment = simulation_parameters.year_increment

        for time in range(start_time, end_time + increment, increment):
            next_world_state = world_state_Updater.get_next_world_state()
            simulation_run.trajectory[time] = next_world_state
            
        return simulation_run
    
    def run_simulations(self, distribution_over_simulation_parameters: ModelParameters, number_of_simulations: int) -> list[SimulationRun]:
        simulation_runs = []
        for _ in range(number_of_simulations):
            simulation_parameters = distribution_over_simulation_parameters.sample()
            simulation_run = self.run_simulation(simulation_parameters)
            simulation_runs.append(simulation_run)
        return simulation_runs