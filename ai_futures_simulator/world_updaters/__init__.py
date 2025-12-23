"""
World updaters module for the AI Futures Simulator.

Each WorldUpdater can contribute to d(state)/dt or directly set state/metric values.
"""

from world_updaters.combined_updater import WorldUpdater, CombinedUpdater
from world_updaters.software_r_and_d import SoftwareRAndD
from world_updaters.compute import ComputeUpdater
from world_updaters.ai_researchers import (
    NationResearcherUpdater,
    AISoftwareDeveloperResearcherUpdater,
    BlackProjectResearcherUpdater,
)
