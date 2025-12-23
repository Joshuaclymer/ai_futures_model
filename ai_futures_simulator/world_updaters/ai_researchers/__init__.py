"""
AI Researcher-related world updaters.

This module contains:
- NationResearcherUpdater: Updates researcher headcount for nations
- AISoftwareDeveloperResearcherUpdater: Updates researchers for AI software developers
- BlackProjectResearcherUpdater: Updates researchers for black projects
"""

from world_updaters.ai_researchers.nation_researchers import NationResearcherUpdater
from world_updaters.ai_researchers.ai_sw_developer_ai_researchers import AISoftwareDeveloperResearcherUpdater
from world_updaters.ai_researchers.black_project_ai_researchers import BlackProjectResearcherUpdater

__all__ = [
    'NationResearcherUpdater',
    'AISoftwareDeveloperResearcherUpdater',
    'BlackProjectResearcherUpdater',
]
