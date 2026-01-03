"""Black project simulation endpoint.

This file re-exports from the modular black_project package.
The implementation has been refactored into endpoints/black_project/ for better organization.
"""

from .black_project import register_black_project_routes

__all__ = ['register_black_project_routes']
