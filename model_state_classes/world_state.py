from classes.simulation_parameters import DistributionOverSimulationParameters, SimulationParameters
from classes.entities import AISoftwareDeveloper, Coalition, State
from classes.policies import AIPolicy
from abc import ABC
from classes.simulation_parameters import SimulationParameters
from dataclasses import dataclass

class Time:
    year : float

class Duration:
    years : float

@dataclass
class WorldState:
    current_time: Time
    coalitions: dict[str, Coalition]  # id -> all coalitions of states in the world at this time
    states: dict[str, State] # id -> all states in the world at this time
    ai_software_developers: dict[str, AISoftwareDeveloper]  # id -> all AI software developers in the world at this time
    ai_policies: dict[str, AIPolicy]  # id -> all AI policies in effect at this time
