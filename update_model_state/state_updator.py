@dataclass
class Updater(ABC):
    simulation_run : SimulationRun
    next_world_state : WorldState = None

    def get_next_world_state(self) -> WorldState:
        # Logic to generate the updated world state based on previous states and simulation parameters
        pass