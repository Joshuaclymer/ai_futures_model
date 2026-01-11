"""Black project defaults endpoint."""

import logging
from flask import jsonify

from ..get_data_for_ai_black_projects_page.defaults import get_default_parameters

logger = logging.getLogger(__name__)


def register_black_project_defaults_routes(app):
    """Register black project defaults route with the Flask app."""

    @app.route('/api/black-project-defaults', methods=['GET'])
    def get_black_project_defaults():
        """
        Get default parameters from YAML configuration.

        Returns parameter values that should be used to initialize
        the frontend parameter sidebar.
        """
        try:
            defaults = get_default_parameters()
            return jsonify({
                'success': True,
                'defaults': defaults,
            })
        except Exception as e:
            logger.exception(f"Error getting defaults: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            }), 500
