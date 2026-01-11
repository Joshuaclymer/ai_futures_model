"""API endpoint modules."""

from .run_simulation import register_simulation_routes
from .get_data_for_ai_timelines_and_takeoff_page import register_ai_timelines_routes
from .sampling_config import register_sampling_config_routes
from .parameter_config import register_parameter_config_routes
from .get_data_for_ai_black_projects_page import register_black_project_routes
from .black_project_defaults import register_black_project_defaults_routes


def register_all_routes(app):
    """Register all API routes with the Flask app."""
    register_simulation_routes(app)
    register_ai_timelines_routes(app)
    register_sampling_config_routes(app)
    register_parameter_config_routes(app)
    register_black_project_routes(app)
    register_black_project_defaults_routes(app)
