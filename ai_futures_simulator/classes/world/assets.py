"""
Asset classes representing physical and compute resources.
All values must be explicitly set during world initialization - no defaults.
"""

from dataclasses import dataclass, field

from classes.tensor_dataclass import TensorDataclass

class Assets():
    pass

@dataclass
class Compute(Assets, TensorDataclass):
    """Represents the stock of a single type of chip."""
    functional_tpp_h100e: float = field(metadata={'is_state': True})
    tpp_h100e_including_attrition: float = field(metadata={'is_state': True})
    watts_per_h100e: float = field(metadata={'is_state': True})
    average_functional_chip_age_years : float = field(metadata={'is_state': True})

@dataclass
class Datacenters(TensorDataclass):
    data_center_capacity_gw: float = field(metadata={'is_state': True})

@dataclass
class Fabs(TensorDataclass):
    monthly_compute_production: Compute = field(metadata={'is_state': True})

@dataclass
class ProcessNode():
    node: float