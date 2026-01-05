"""
Black project simulation endpoint routes.

Runs N Monte Carlo simulations and returns aggregated plot data for visualization.

Supports optional caching: Set USE_CACHE=true in environment to enable.
When enabled and a cached response exists for the default parameters,
the cached response is returned instead of running simulations.
"""

import json
import logging
import os
from pathlib import Path
from flask import request, jsonify

from .simulation_runner import run_black_project_simulations
from .response_builder import extract_black_project_plot_data
from .defaults import get_default_parameters

logger = logging.getLogger(__name__)

# Cache configuration - set USE_CACHE=false in env to disable
USE_CACHE = os.environ.get("USE_CACHE", "true").lower() == "true"
CACHE_DIR = Path(__file__).parent.parent.parent / "cache"


def _load_cached_response() -> dict | None:
    """Load cached response if available."""
    cache_path = CACHE_DIR / "black_project_default.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            logger.info(f"[black-project] Loaded cached response from {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"[black-project] Failed to load cache: {e}")
    return None


def _save_cached_response(data: dict) -> None:
    """Save response to cache for default parameters."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "black_project_default.json"
    try:
        with open(cache_path, "w") as f:
            json.dump(data, f)
        logger.info(f"[black-project] Saved response to cache at {cache_path}")
    except Exception as e:
        logger.warning(f"[black-project] Failed to save cache: {e}")


def _is_default_request(data: dict, defaults: dict) -> bool:
    """Check if request uses default parameters (cacheable)."""
    frontend_params = data.get("parameters", {})
    # If no custom parameters provided, it's a default request
    if not frontend_params:
        return True
    # Check if parameters match defaults (allowing for some tolerance)
    for key, default_val in defaults.items():
        if key in frontend_params:
            param_val = frontend_params[key]
            if isinstance(default_val, (int, float)) and isinstance(param_val, (int, float)):
                if abs(param_val - default_val) > 0.001:
                    return False
            elif param_val != default_val:
                return False
    return True


def register_black_project_routes(app):
    """Register black project routes with the Flask app."""

    @app.route('/api/get-data-for-ai-black-projects-page', methods=['POST'])
    def get_data_for_ai_black_projects_page():
        """
        Run N Monte Carlo simulations and return plot data.

        Request body:
        {
            "parameters": {...},  // Optional frontend params
            "num_simulations": 100,  // Number of Monte Carlo simulations
            "time_range": [2027, 2037]  // [agreement_year, end_year]
        }

        Returns (matching reference API format):
        {
            "num_simulations": 100,
            "prob_fab_built": 0.55,
            "p_project_exists": 0.2,
            "researcher_headcount": 500,
            "black_project_model": {...},
            "black_datacenters": {...},
            "black_fab": {...},
            "initial_black_project": {...},
            "initial_stock": {...}
        }
        """
        try:
            data = request.json or {}
            frontend_params = data.get('parameters', {})
            total_simulations = data.get('num_simulations', 100)
            time_range = data.get('time_range', [2027, 2037])

            # Check for cached response (only for default parameters)
            is_default = False
            if USE_CACHE:
                defaults = get_default_parameters()
                is_default = _is_default_request(data, defaults)
                if is_default:
                    cached = _load_cached_response()
                    if cached:
                        logger.info("[black-project] Returning cached response")
                        return jsonify(cached)
                    else:
                        logger.info("[black-project] No cache found, running simulation")

            logger.info(f"[black-project] Running {total_simulations} Monte Carlo simulations")

            # Run simulations
            simulation_results = run_black_project_simulations(
                frontend_params=frontend_params,
                num_simulations=total_simulations,
                time_range=time_range,
            )

            # Extract plot data
            plot_data = extract_black_project_plot_data(
                simulation_results=simulation_results,
                frontend_params=frontend_params,
            )

            # Save to cache if this was a default request
            if USE_CACHE and is_default:
                _save_cached_response(plot_data)

            # Return data directly (no success wrapper) to match reference API format
            return jsonify(plot_data)

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
