"""AI Timelines and Takeoff page endpoint routes."""

import logging
from flask import request, jsonify

from api_utils import run_simulation_internal
from .sw_progress_extractor import extract_sw_progress_from_raw

logger = logging.getLogger(__name__)


def register_ai_timelines_routes(app):
    """Register AI timelines and takeoff routes with the Flask app."""

    @app.route('/api/get-data-for-ai-timelines-and-takeoff-page', methods=['POST'])
    def get_data_for_ai_timelines_and_takeoff_page():
        """
        Run simulation and return software progress metrics for the AI Timelines and Takeoff page.

        This endpoint runs the simulation internally and extracts
        software progress data from the raw World objects.
        """
        try:
            data = request.json or {}
            frontend_params = data.get('parameters', {})
            time_range = data.get('time_range', [2024, 2040])

            logger.info(f"[get-data-for-ai-timelines-and-takeoff-page] Received request with time_range: {time_range}")

            # Run simulation using internal function
            raw_result = run_simulation_internal(frontend_params, time_range)

            if not raw_result.get('success'):
                return jsonify(raw_result)

            # Extract software progress from raw result
            response = extract_sw_progress_from_raw(raw_result)

            return jsonify(response)

        except Exception as e:
            logger.exception(f"Error in get_data_for_ai_timelines_and_takeoff_page: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            }), 500
