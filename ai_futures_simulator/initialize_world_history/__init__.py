"""
World initialization module for the AI Futures Simulator.

Initializes the world state at the simulation start year.
Historical trajectory data (progress, research_stock) is computed via calibration
which runs the ProgressModel internally.

Contains initializers for all world state components:
- initialize_world_history.py: Main entry point combining all initializers
- initialize_nations.py: Nation entity initialization
- initialize_ai_software_developers.py: AI developer initialization
- initialize_ai_software_progress.py: AI software progress initialization
"""

from initialize_world_history.initialize_world_history import initialize_world
from initialize_world_history.initialize_nations import initialize_usa
from initialize_world_history.initialize_ai_software_developers.initialize_ai_software_developers import initialize_us_frontier_lab
from initialize_world_history.initialize_ai_software_progress import initialize_ai_software_progress
