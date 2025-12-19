"""
World class - the top-level container for all simulation state.

World contains nested entities (coalitions, nations, AI projects, etc.)
along with global state and metrics.
"""

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, Optional

from classes.world.tensor_dataclass import TensorDataclass
from classes.world.entities import Coalition, Nation, AISoftwareDeveloper, AIBlackProject
from classes.world.policies import AIPolicy


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

    @classmethod
    def zeros(cls, template: 'World' = None) -> 'World':
        """
        Create a zero-initialized World.

        If template is provided, creates zeros with same structure.
        Otherwise creates an empty World.
        """
        if template is not None:
            return super().zeros(template)

        # Create empty world with zero time
        return cls(
            current_time=torch.tensor(0.0),
            coalitions={},
            nations={},
            ai_software_developers={},
            ai_policies={},
            black_projects={},
        )

    def get_developer(self, developer_id: str) -> AISoftwareDeveloper:
        """Get an AI software developer by ID."""
        return self.ai_software_developers.get(developer_id)

    def get_nation(self, nation_id: str) -> Nation:
        """Get a nation by ID."""
        return self.nations.get(nation_id)

    def get_coalition(self, coalition_id: str) -> Coalition:
        """Get a coalition by ID."""
        return self.coalitions.get(coalition_id)

    def get_black_project(self, project_id: str) -> Optional[AIBlackProject]:
        """Get a black project by ID."""
        return self.black_projects.get(project_id)
