"""
Black project simulation module.

This module provides the API endpoint and supporting functions for running
Monte Carlo simulations of covert black project scenarios.

Module Structure:
- routes.py: Flask route registration
- simulation_runner.py: Monte Carlo simulation execution
- response_builder.py: Main response assembly
- defaults.py: Default parameter extraction
- detection.py: Detection computation functions
- world_data.py: World state extraction
- visualization.py: Chart building functions
- percentile_helpers.py: Statistical helper functions
- reduction_ratios.py: Reduction ratio computations
- utils.py: Utility functions
- response_sections/: Modular response section builders
"""

from .routes import register_black_project_routes
from .simulation_runner import run_black_project_simulations
from .response_builder import extract_black_project_plot_data
from .defaults import get_default_parameters

__all__ = [
    'register_black_project_routes',
    'run_black_project_simulations',
    'extract_black_project_plot_data',
    'get_default_parameters',
]
