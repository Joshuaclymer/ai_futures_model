"""Configuration endpoints for parameter sampling and defaults."""

import logging
from pathlib import Path
import yaml
from flask import jsonify

from api_utils import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


def register_config_routes(app):
    """Register configuration routes with the Flask app."""

    @app.route('/api/sampling-config', methods=['GET'])
    def get_sampling_config():
        """Return parameter distribution config for Monte Carlo sampling."""
        try:
            # Load the default config
            mc_config_path = Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator" / "parameters" / "default_parameters.yaml"

            if mc_config_path.exists():
                with open(mc_config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # Fall back to modal parameters
                with open(DEFAULT_CONFIG_PATH, 'r') as f:
                    config = yaml.safe_load(f)

            return jsonify({
                'success': True,
                'config': config,
            })

        except Exception as e:
            logger.exception(f"Error in get_sampling_config: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            }), 500

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
