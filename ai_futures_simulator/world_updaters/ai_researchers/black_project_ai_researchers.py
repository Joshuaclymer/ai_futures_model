"""
Black project AI researcher updater.

Updates researcher headcount for AI black projects.
Black projects have their own dedicated researcher allocation specified
in the black project parameters.
"""

import torch
from torch import Tensor
from typing import Optional

from classes.world.world import World
from classes.world.entities import AIBlackProject
from classes.simulation_primitives import WorldUpdater
from parameters.classes import SimulationParameters


class BlackProjectResearcherUpdater(WorldUpdater):
    """
    Updates researchers for AI black projects.

    Black projects have their researcher count specified directly in parameters
    (human_ai_capability_researchers), not derived from nation researchers.
    This updater ensures the black project researcher count is properly maintained.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

    def _get_black_project_researcher_count(self, project: AIBlackProject) -> float:
        """
        Get the researcher count for a black project.

        Black project researchers are specified in the black project parameters.
        """
        researcher_count = getattr(project, 'human_ai_capability_researchers', None)
        if researcher_count is None:
            return 0.0

        if isinstance(researcher_count, Tensor):
            return researcher_count.item()
        return float(researcher_count)

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """
        Update black project researcher metrics.

        For black projects, the researcher count is typically set at initialization
        from black_project_properties.human_ai_capability_researchers and doesn't
        grow over time (unlike nation researchers).

        This updater primarily ensures consistency and could be extended to model
        researcher recruitment dynamics for black projects.
        """
        for project_id, project in world.ai_black_projects.items():
            # Get current researcher count (from initialization/parameters)
            current_researchers = self._get_black_project_researcher_count(project)

            # For now, black project researchers are static (set at initialization)
            # Future enhancement: model researcher recruitment/growth for black projects
            project._set_frozen_field('human_ai_capability_researchers', current_researchers)

        return world
