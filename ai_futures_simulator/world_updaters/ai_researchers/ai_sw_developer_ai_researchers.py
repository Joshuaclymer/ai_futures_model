"""
AI Software Developer researcher updater.

Updates researcher headcount for AI software developers based on their parent nation's researchers.
The fraction of nation researchers available to the largest developer is determined by parameters.
"""

import torch
from torch import Tensor
from typing import Optional

from classes.world.world import World
from classes.world.entities import AISoftwareDeveloper, NamedNations
from classes.simulation_primitives import WorldUpdater
from parameters.classes import SimulationParameters


class AISoftwareDeveloperResearcherUpdater(WorldUpdater):
    """
    Updates researchers for AI software developers based on nation researchers.

    Each AI software developer gets a fraction of their nation's researcher headcount.
    This mirrors the compute allocation pattern.

    This updater should run AFTER NationResearcherUpdater so nation researchers are up to date.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

    def _get_nation_for_developer(self, dev: AISoftwareDeveloper, world: World) -> Optional[str]:
        """
        Get the nation ID for a developer.

        For now, assumes US-based developers. Can be extended to support
        developer-nation mappings.
        """
        # Default to USA for now - could be extended with developer metadata
        return NamedNations.USA

    def _get_fraction_of_nation_researchers(self, dev: AISoftwareDeveloper, nation_id: str) -> float:
        """
        Get the fraction of nation researchers available to this developer.

        Uses parameter: proportion_of_researchers_in_largest_ai_sw_developer
        """
        researcher_params = getattr(self.params, 'ai_researcher_headcount', None)
        if researcher_params is None:
            return 0.05  # Default 5%

        if nation_id == NamedNations.USA:
            return researcher_params.us_researchers.proportion_of_researchers_in_largest_ai_sw_developer
        elif nation_id == NamedNations.PRC:
            return researcher_params.prc_researchers.proportion_of_researchers_in_largest_ai_sw_developer

        return 0.05  # Default fraction

    def _get_nation_researcher_count(self, nation_id: str, world: World) -> float:
        """Get the current researcher count for a nation."""
        if nation_id not in world.nations:
            return 0.0

        nation = world.nations[nation_id]
        researcher_count = getattr(nation, 'ai_researcher_headcount', None)
        if researcher_count is None:
            return 0.0

        if isinstance(researcher_count, Tensor):
            return researcher_count.item()
        return float(researcher_count)

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """
        Update AI software developer researchers from nation researchers.

        For each developer:
        1. Get their nation's researcher count
        2. Apply fraction parameter to get developer's researchers
        3. Update human_ai_capability_researchers
        """
        for dev_id, dev in world.ai_software_developers.items():
            # Get nation for this developer
            nation_id = self._get_nation_for_developer(dev, world)
            if nation_id is None or nation_id not in world.nations:
                continue

            # Get nation's researcher count
            nation_researchers = self._get_nation_researcher_count(nation_id, world)

            # Get fraction of nation researchers for this developer
            fraction = self._get_fraction_of_nation_researchers(dev, nation_id)

            # Compute developer's researcher count
            developer_researchers = nation_researchers * fraction

            # Update developer's researcher count
            dev._set_frozen_field('human_ai_capability_researchers', developer_researchers)

        return world
