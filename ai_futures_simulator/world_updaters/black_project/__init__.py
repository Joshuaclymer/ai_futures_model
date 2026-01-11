"""
Black project existence and initialization.

This module contains:
- initialize_black_project: Factory function to create black projects

The actual updates to black project state are handled by:
- compute/black_compute.py: Fab production, compute stock, attrition
- datacenters_and_energy/update_datacenters_and_energy.py: Datacenter capacity, energy
- perceptions/black_project_perceptions.py: Detection LRs, posterior probability
"""

from world_updaters.black_project.update_black_project import (
    initialize_black_project,
)

__all__ = [
    'initialize_black_project',
]
