"""Raw simulation endpoint - returns full World objects."""

import logging
from flask import request, jsonify

from api_utils import run_simulation_internal

logger = logging.getLogger(__name__)


def register_simulation_routes(app):
    """Register simulation routes with the Flask app."""

    @app.route('/api/run-simulation', methods=['POST'])
    def run_simulation_raw():
        """
        Run simulation and return raw World objects.

        This endpoint returns the full simulation result including:
        - times: array of time points
        - trajectory: array of World objects (fully serialized)
        - params: the simulation parameters used

        Request body:
        {
            "parameters": {...},  // Optional frontend params
            "time_range": [start_year, end_year]  // Optional, defaults to [2024, 2040]
        }
        """
        try:
            data = request.json or {}
            frontend_params = data.get('parameters', {})
            time_range = data.get('time_range', [2024, 2040])

            logger.info(f"[run-simulation] Received request with time_range: {time_range}")

            # Run simulation using internal function
            result = run_simulation_internal(frontend_params, time_range)

            return jsonify(result)

        except Exception as e:
            logger.exception(f"Error in run_simulation_raw: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            }), 500
