"""
World class - the top-level container for all simulation state.

World contains nested entities (coalitions, nations, AI projects, etc.)
along with global state and metrics.
"""

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict

from classes.tensor_dataclass import TensorDataclass
from classes.world.entities import Coalition, Nation, AISoftwareDeveloper, AIBlackProject
from classes.world.policies import AIPolicy
from classes.world.perceptions import Perceptions


@dataclass
class World(TensorDataclass):
    """
    Complete world state at a point in time.

    Contains nested entities with their own state/metric fields.
    State fields have metadata={'is_state': True} and are integrated by the ODE solver.
    All other fields are derived quantities or containers.
    """

    # === Global state ===
    current_time: Tensor = field(metadata={'is_state': True})

    # === Nested entities ===
    coalitions: Dict[str, Coalition]
    nations: Dict[str, Nation]
    ai_software_developers: Dict[str, AISoftwareDeveloper]
    ai_policies: Dict[str, AIPolicy]

    # === Black projects (covert compute infrastructure) ===
    black_projects: Dict[str, AIBlackProject] = field(default_factory=dict)

    # === Entity perceptions (beliefs about other entities) ===
    # Maps entity_id -> Perceptions
    perceptions: Dict[str, Perceptions] = field(default_factory=dict)

    @classmethod
    def zeros(cls, template: 'World' = None) -> 'World':
        """
        Create a zero-initialized World.

        If template is provided, creates zeros with same structure.
        If template is a FlatWorld, returns a FlatWorld.zeros() instead.
        Otherwise creates an empty World.
        """
        if template is not None:
            # Check if template is a FlatWorld
            from classes.flat_world import FlatWorld
            if isinstance(template, FlatWorld):
                return FlatWorld.zeros(template)
            return super().zeros(template)

        # Create empty world with zero time
        return cls(
            current_time=torch.tensor(0.0),
            coalitions={},
            nations={},
            ai_software_developers={},
            ai_policies={},
            black_projects={},
            perceptions={},
        )
