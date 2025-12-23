"""API endpoint modules."""

from .simulation import register_simulation_routes
from .sw_progress import register_sw_progress_routes
from .config import register_config_routes
from .black_project import register_black_project_routes
from .black_project_dummy import register_black_project_dummy_routes


def register_all_routes(app):
    """Register all API routes with the Flask app."""
    register_simulation_routes(app)
    register_sw_progress_routes(app)
    register_config_routes(app)
    register_black_project_routes(app)
    register_black_project_dummy_routes(app)
