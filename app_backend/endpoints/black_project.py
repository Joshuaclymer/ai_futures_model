"""Black project simulation endpoint.

Runs 1 central simulation + 10 Monte Carlo simulations and returns
aggregated plot data for visualization.
"""

import logging
from flask import request, jsonify

from api_utils.black_project_simulation import (
    run_black_project_simulations,
    extract_black_project_plot_data,
    get_default_parameters,
)

logger = logging.getLogger(__name__)


def register_black_project_routes(app):
    """Register black project routes with the Flask app."""

    @app.route('/api/run-black-project-simulation', methods=['POST'])
    def run_black_project_simulation():
        """
        Run 1 central + N Monte Carlo simulations and return plot data.

        Request body:
        {
            "parameters": {...},  // Optional frontend params
            "num_simulations": 11,  // Total (1 central + N MC), default: 11
            "time_range": [2027, 2037]  // [agreement_year, end_year]
        }

        Returns:
        {
            "success": true,
            "num_simulations": 11,
            "black_project_model": {...},
            "black_datacenters": {...},
            "initial_stock": {...},
            "rate_of_computation": {...},
            ...
        }
        """
        try:
            data = request.json or {}
            frontend_params = data.get('parameters', {})
            total_simulations = data.get('num_simulations', 11)
            time_range = data.get('time_range', [2027, 2037])

            # Total = 1 central + N MC
            num_mc = max(0, total_simulations - 1)

            logger.info(f"[black-project] Running 1 central + {num_mc} MC simulations")

            # Run simulations
            simulation_results = run_black_project_simulations(
                frontend_params=frontend_params,
                num_mc_simulations=num_mc,
                time_range=time_range,
            )

            # Extract plot data
            plot_data = extract_black_project_plot_data(
                simulation_results=simulation_results,
                frontend_params=frontend_params,
            )

            return jsonify({
                'success': True,
                **plot_data,
            })

        except Exception as e:
            logger.exception(f"Error in run_black_project_simulation: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            }), 500

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
