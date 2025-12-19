"""
Asset classes representing physical and compute resources.
All values must be explicitly set during world initialization - no defaults.
"""

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from classes.world.tensor_dataclass import TensorDataclass

class Assets():
    pass

@dataclass
class Compute(Assets, TensorDataclass):
    """Represents the stock of a single type of chip."""
    all_tpp_h100e: float = field(metadata={'is_state': True})
    functional_tpp_h100e: float = field(metadata={'is_state': True})
    watts_per_h100e: float = field(metadata={'is_state': True})
    average_functional_chip_age_years : float = field(metadata={'is_state': True})

@dataclass
class Datacenters(TensorDataclass):
    data_center_capacity_gw: float = field(metadata={'is_state': True})

@dataclass
class Fabs(TensorDataclass):
    monthly_production_compute: Compute = field(metadata={'is_state': True})

@dataclass
class ProcessNode():
    node: float