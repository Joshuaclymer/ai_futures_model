from typing import Dict
from classes.entities import State

class Utilities:
    utility_of_status_quo : float
    utility_of_misaligned_AI_takeover : float
    utility_of_foreign_state_AI_takeover : Dict[State, float]
    utility_of_domestic_ai_company_power_grab : float # (not the company in question)