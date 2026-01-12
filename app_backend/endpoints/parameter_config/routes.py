"""Parameter configuration endpoint."""

import logging
import yaml
from flask import jsonify
from typing import Any, Dict, Optional

from api_utils import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


def _get_nested(config: Dict, path: str, default: Any = None) -> Any:
    """Get a value from nested config using dot notation path."""
    keys = path.split('.')
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _flatten_config(config, prefix=''):
    """Flatten nested config into dot-notation keys."""
    result = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and 'value' not in value and 'dist' not in value and 'min' not in value:
            # Nested section, recurse
            result.update(_flatten_config(value, full_key))
        else:
            result[full_key] = value
    return result


def _extract_default(param_value):
    """Extract the default value from a parameter definition."""
    if param_value is None:
        return None
    if isinstance(param_value, dict):
        # Check for explicit value field first
        if 'value' in param_value:
            return param_value['value']
        # For distributions, use modal as the default
        if 'modal' in param_value:
            return param_value['modal']
        # For choice distributions without modal, use first value
        if 'dist' in param_value and param_value['dist'] == 'choice' and 'values' in param_value:
            return param_value['values'][0]
        return None
    # Plain value (number, string, bool)
    return param_value


def _extract_bounds(param_value):
    """Extract min/max bounds from a parameter definition."""
    if not isinstance(param_value, dict):
        return None
    if 'min' in param_value or 'max' in param_value:
        bounds = {}
        if 'min' in param_value:
            bounds['min'] = param_value['min']
        if 'max' in param_value:
            bounds['max'] = param_value['max']
        return bounds
    return None


def _get_param_default(config: Dict, path: str, fallback: Any = None) -> Any:
    """Get the default value for a parameter at the given path."""
    raw_value = _get_nested(config, path)
    if raw_value is None:
        return fallback
    extracted = _extract_default(raw_value)
    return extracted if extracted is not None else fallback


def _is_fab_built(config: Dict, black_project_start_year: float) -> bool:
    """
    Determine if a fab is built based on localization years and minimum process node requirement.
    Uses modal values from the config (not sampled values).
    """
    props = _get_nested(config, 'black_project.properties', {})
    min_node = _get_param_default(config, 'black_project.properties.black_fab_min_process_node', 28.0)

    # Get modal localization years for each node
    localization_years = {
        7: _get_param_default(config, 'black_project.properties.prc_localization_year_7nm', 9999),
        14: _get_param_default(config, 'black_project.properties.prc_localization_year_14nm', 9999),
        28: _get_param_default(config, 'black_project.properties.prc_localization_year_28nm', 9999),
    }

    # Find best available node that is localized by start year
    for node_nm in [7, 14, 28]:
        if localization_years[node_nm] <= black_project_start_year:
            return node_nm <= min_node

    return False


def _extract_black_project_defaults(config: Dict) -> Dict[str, Any]:
    """
    Extract black project defaults from config using modal/value (not sampling).
    Returns camelCase keys matching the frontend expectations.
    """
    black_project_start_year = _get_param_default(config, 'black_project.black_project_start_year', 2029.0)

    return {
        # Simulation settings
        "numSimulations": 100,
        "agreementYear": int(_get_param_default(config, 'policy.ai_slowdown_start_year', 2030)),
        "blackProjectStartYear": int(black_project_start_year),

        # Black project properties
        "workersInCovertProject": _get_param_default(config, 'black_project.properties.total_labor', 11300),
        "fractionOfLaborDevotedToDatacenterConstruction": _get_param_default(config, 'black_project.properties.fraction_of_labor_devoted_to_datacenter_construction', 0.885),
        "fractionOfLaborDevotedToBlackFabConstruction": _get_param_default(config, 'black_project.properties.fraction_of_labor_devoted_to_black_fab_construction', 0.022),
        "fractionOfLaborDevotedToBlackFabOperation": _get_param_default(config, 'black_project.properties.fraction_of_labor_devoted_to_black_fab_operation', 0.049),
        "fractionOfLaborDevotedToAiResearch": _get_param_default(config, 'black_project.properties.fraction_of_labor_devoted_to_ai_research', 0.044),
        "proportionOfInitialChipStockToDivert": _get_param_default(config, 'black_project.properties.fraction_of_initial_compute_stock_to_divert_at_black_project_start', 0.05),
        "fractionOfDatacenterCapacityToDivert": _get_param_default(config, 'black_project.properties.fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start', 0.5),
        "fractionOfLithographyScannersToDivert": _get_param_default(config, 'black_project.properties.fraction_of_lithography_scanners_to_divert_at_black_project_start', 0.10),
        "maxFractionOfTotalNationalEnergyConsumption": _get_param_default(config, 'black_project.properties.max_fraction_of_total_national_energy_consumption', 0.05),
        "buildCovertFab": _is_fab_built(config, black_project_start_year),
        "blackFabMaxProcessNode": str(int(_get_param_default(config, 'black_project.properties.black_fab_min_process_node', 28))),

        # Detection parameters
        "priorOddsOfCovertProject": _get_param_default(config, 'perceptions.black_project_perception_parameters.prior_odds_of_covert_project', 0.111),
        "intelligenceMedianError": _get_param_default(config, 'perceptions.black_project_perception_parameters.intelligence_median_error_in_estimate_of_compute_stock', 0.07),
        "meanDetectionTime100": _get_param_default(config, 'perceptions.black_project_perception_parameters.mean_detection_time_for_100_workers', 6.95),
        "meanDetectionTime1000": _get_param_default(config, 'perceptions.black_project_perception_parameters.mean_detection_time_for_1000_workers', 3.42),
        "varianceDetectionTime": _get_param_default(config, 'perceptions.black_project_perception_parameters.variance_of_detection_time_given_num_workers', 3.88),
        "detectionThreshold": _get_param_default(config, 'perceptions.black_project_perception_parameters.detection_threshold', 100.0),

        # PRC compute
        "totalPrcComputeTppH100eIn2025": _get_param_default(config, 'compute.prc_compute.total_prc_compute_tpp_h100e_in_2025', 100000),
        "annualGrowthRateOfPrcComputeStock": _get_param_default(config, 'compute.prc_compute.annual_growth_rate_of_prc_compute_stock', 2.2),
        "h100SizedChipsPerWafer": _get_param_default(config, 'compute.prc_compute.h100_sized_chips_per_wafer', 28),
        "wafersPerMonthPerLithographyScanner": _get_param_default(config, 'compute.prc_compute.wafers_per_month_per_lithography_scanner', 1000),

        # PRC energy
        "energyEfficiencyOfComputeStockRelativeToStateOfTheArt": _get_param_default(config, 'datacenter_and_energy.prc_energy_consumption.energy_efficiency_of_compute_stock_relative_to_state_of_the_art', 0.20),
        "totalPrcEnergyConsumptionGw": _get_param_default(config, 'datacenter_and_energy.prc_energy_consumption.total_prc_energy_consumption_gw', 1100),
        "dataCenterMwPerYearPerConstructionWorker": _get_param_default(config, 'datacenter_and_energy.prc_energy_consumption.data_center_mw_per_year_per_construction_worker', 1.0),

        # Survival parameters
        "initialAnnualHazardRate": _get_param_default(config, 'compute.survival_rate_parameters.initial_annual_hazard_rate', 0.05),
        "annualHazardRateIncreasePerYear": _get_param_default(config, 'compute.survival_rate_parameters.annual_hazard_rate_increase_per_year', 0.02),

        # Exogenous compute trends
        "transistorDensityScalingExponent": _get_param_default(config, 'compute.exogenous_trends.transistor_density_scaling_exponent', 1.49),
        "stateOfTheArtArchitectureEfficiencyImprovementPerYear": _get_param_default(config, 'compute.exogenous_trends.state_of_the_art_architecture_efficiency_improvement_per_year', 1.23),
        "transistorDensityAtEndOfDennardScaling": _get_param_default(config, 'compute.exogenous_trends.transistor_density_at_end_of_dennard_scaling_m_per_mm2', 10.0),
        "wattsTppDensityExponentBeforeDennard": _get_param_default(config, 'compute.exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended', -1.0),
        "wattsTppDensityExponentAfterDennard": _get_param_default(config, 'compute.exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended', -0.33),
        "stateOfTheArtEnergyEfficiencyImprovementPerYear": _get_param_default(config, 'compute.exogenous_trends.state_of_the_art_energy_efficiency_improvement_per_year', 1.26),
    }


def register_parameter_config_routes(app):
    """Register parameter config route with the Flask app."""

    @app.route('/api/parameter-config', methods=['GET'])
    def get_parameter_config():
        """Return parameter bounds and defaults."""
        try:
            with open(DEFAULT_CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)

            # Flatten the config to dot-notation
            flattened = _flatten_config(config)

            # Extract defaults and bounds
            defaults = {}
            bounds = {}
            for key, value in flattened.items():
                default_value = _extract_default(value)
                if default_value is not None:
                    defaults[key] = default_value

                param_bounds = _extract_bounds(value)
                if param_bounds:
                    bounds[key] = param_bounds

            # Extract model constants from software_r_and_d section
            software_config = config.get('software_r_and_d', {})
            model_constants = {
                'training_compute_reference_year': software_config.get('training_compute_reference_year'),
                'training_compute_reference_ooms': software_config.get('training_compute_reference_ooms'),
                'software_progress_scale_reference_year': software_config.get('software_progress_scale_reference_year'),
                'base_for_software_lom': software_config.get('base_for_software_lom'),
            }

            # Extract black project defaults using modal/value (not sampling)
            black_project_defaults = _extract_black_project_defaults(config)

            return jsonify({
                'success': True,
                'config': config,
                'defaults': defaults,
                'bounds': bounds,
                'model_constants': model_constants,
                'black_project_defaults': black_project_defaults,
            })

        except Exception as e:
            logger.exception(f"Error in get_parameter_config: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
            }), 500
