"""
AI Researcher-related world updaters.

This module contains:
- AIResearcherUpdater: Combined updater for all researcher-related updates
- NationResearcherUpdater: Updates researcher headcount for nations
- AISoftwareDeveloperResearcherUpdater: Updates researchers for AI software developers
- BlackProjectResearcherUpdater: Updates researchers for black projects
"""

from world_updaters.ai_researchers.update_ai_researchers import AIResearcherUpdater
from world_updaters.ai_researchers.nation_researchers import NationResearcherUpdater
from world_updaters.ai_researchers.ai_sw_developer_ai_researchers import AISoftwareDeveloperResearcherUpdater
from world_updaters.ai_researchers.black_project_ai_researchers import BlackProjectResearcherUpdater

__all__ = [
    'AIResearcherUpdater',
    'NationResearcherUpdater',
    'AISoftwareDeveloperResearcherUpdater',
    'BlackProjectResearcherUpdater',
]
