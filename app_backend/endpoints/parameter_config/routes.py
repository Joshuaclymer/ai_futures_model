"""Parameter configuration endpoint."""

import logging
import yaml
from flask import jsonify

from api_utils import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


def register_parameter_config_routes(app):
    """Register parameter config route with the Flask app."""

    @app.route('/api/parameter-config', methods=['GET'])
    def get_parameter_config():
        """Return parameter bounds and defaults."""
        try:
            with open(DEFAULT_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)

            return jsonify({
                'success': True,
                'config': config,
            })

        except Exception as e:
            logger.exception(f"Error in get_parameter_config: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            }), 500
