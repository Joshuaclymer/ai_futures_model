"""
World updaters module for the AI Futures Simulator.

Each WorldUpdater can contribute to d(state)/dt or directly set state/metric values.

Directory structure:
- compute/
  - update_compute.py: Combined compute updater for nations/developers
  - black_project_compute.py: Compute updater for black projects (fabs, stock, attrition)
- ai_researchers/
  - update_ai_researchers.py: Combined AI researcher updater
- software_r_and_d/
  - update_software_r_and_d.py: Software R&D updater
- black_project/
  - update_black_project.py: Black project initialization (existence only)
- datacenters_and_energy/
  - update_datacenters_and_energy.py: Datacenter/energy updater for black projects
- perceptions/
  - update_perceptions.py: State perceptions updater
  - black_project_perceptions.py: Detection/perception updater for black projects
- combined_updater.py: Top-level combined updater
"""

from world_updaters.combined_updater import CombinedUpdater, FlatCombinedUpdater

__all__ = [
    'CombinedUpdater',
    'FlatCombinedUpdater',
]
