"""Software progress simulation endpoint."""

import logging
from flask import request, jsonify

from api_utils import run_simulation_internal, extract_sw_progress_from_raw

logger = logging.getLogger(__name__)


def register_sw_progress_routes(app):
    """Register software progress routes with the Flask app."""

    @app.route('/api/run-sw-progress-simulation', methods=['POST'])
    def run_sw_progress_simulation():
        """
        Run simulation and return software progress metrics.

        This endpoint runs the simulation internally and extracts
        software progress data from the raw World objects.
        """
        try:
            data = request.json or {}
            frontend_params = data.get('parameters', {})
            time_range = data.get('time_range', [2024, 2040])

            logger.info(f"[run-sw-progress-simulation] Received request with time_range: {time_range}")

            # Run simulation using internal function
            raw_result = run_simulation_internal(frontend_params, time_range)

            if not raw_result.get('success'):
                return jsonify(raw_result)

            # Extract software progress from raw result
            response = extract_sw_progress_from_raw(raw_result)

            return jsonify(response)

        except Exception as e:
            logger.exception(f"Error in run_sw_progress_simulation: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            }), 500
