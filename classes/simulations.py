from typing import Dict
from classes.simulation_parameters import DistributionOverSimulationParameters, SimulationParameters
from classes.entities import AISoftwareDeveloper, Coalition, State
from classes.policies import AIPolicy
from abc import ABC
from classes.simulation_parameters import SimulationParameters
from dataclasses import dataclass

class SimulationRun:
    world_states: dict["Time", "WorldState"]  # Mapping from Time to WorldState
    simulation_parameters : SimulationParameters

class Time:
    year : float

class Duration:
    years : float

class WorldState:
    coalitions: Dict[str, Coalition]  # id -> all coalitions of states in the world at this time
    states: Dict[str, State] # id -> all states in the world at this time
    ai_software_developers: Dict[str, AISoftwareDeveloper]  # id -> all AI software developers in the world at this time
    ai_policies: Dict[str, AIPolicy]  # id -> all AI policies in effect at this time

@dataclass
class Updator(ABC):
    simulation_run : SimulationRun
    next_world_state : WorldState = None

    def get_next_world_state(self) -> WorldState:
        # Logic to generate the updated world state based on previous states and simulation parameters
        pass