"""
Nation economy classes.
"""

from dataclasses import dataclass


@dataclass
class NationEconomy:
    """Economic state of a nation (placeholder, will improve later)."""
    ai_capex: float = 0.0
    annual_gdp_growth_rate: float = 0.02
    unemployment_rate: float = 0.05
    human_population: int = 0
