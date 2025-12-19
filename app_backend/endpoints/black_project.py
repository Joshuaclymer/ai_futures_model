"""Black project simulation endpoint.

Runs Monte Carlo simulations of covert AI projects and returns
aggregated plot data for visualization.
"""

import logging
from flask import request, jsonify

from api_utils.black_project_simulation import run_black_project_simulations, extract_black_project_plot_data

logger = logging.getLogger(__name__)


def register_black_project_routes(app):
    """Register black project routes with the Flask app."""

    @app.route('/api/run-black-project-simulation', methods=['POST'])
    def run_black_project_simulation():
        """
        Run Monte Carlo simulations of a black project and return plot data.

        This endpoint:
        1. Runs N simulations with Monte Carlo sampling for parameter uncertainty
        2. Extracts time series data from each simulation
        3. Aggregates results into percentiles and CCDFs for plotting

        Request body:
        {
            "parameters": {...},  // Optional frontend params
            "num_simulations": 100,  // Number of Monte Carlo runs (default: 100)
            "time_range": [2030, 2050]  // [agreement_year, end_year]
        }

        Returns:
        {
            "success": true,
            "num_simulations": 100,
            "black_project_model": {...},  // Main plot data
            "black_fab": {...},  // Fab-specific data
            "black_datacenters": {...},  // Datacenter-specific data
            "initial_stock": {...},  // Initial stock data
            ...
        }
        """
        try:
            data = request.json or {}
            frontend_params = data.get('parameters', {})
            num_simulations = data.get('num_simulations', 100)
            time_range = data.get('time_range', [2030, 2050])

            logger.info(f"[black-project] Running {num_simulations} simulations")

            # Run simulations
            simulation_results = run_black_project_simulations(
                frontend_params=frontend_params,
                num_simulations=num_simulations,
                time_range=time_range,
            )

            # Extract plot data from simulation results
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
