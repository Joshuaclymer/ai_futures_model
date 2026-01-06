"""
Default parameter extraction for black project simulation.

Loads default parameters from YAML configuration for frontend initialization.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add ai_futures_simulator subdirectory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "ai_futures_simulator"))

from parameters.classes import ModelParameters
from .utils import is_fab_built


def get_default_parameters() -> Dict[str, Any]:
    """Get default parameters from YAML for frontend initialization."""
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "ai_futures_simulator" / "parameters" / "black_project_monte_carlo_parameters.yaml"
    model_params = ModelParameters.from_yaml(config_path)

    # Extract relevant parameters for frontend
    sim_params = model_params.sample()

    bp = sim_params.black_project
    bp_props = bp.black_project_properties if bp else None
    compute = sim_params.compute
    prc_compute = compute.PRCComputeParameters if compute else None
    energy = sim_params.datacenter_and_energy
    prc_energy = energy.prc_energy_consumption if energy else None
    perceptions = sim_params.perceptions
    bp_perceptions = perceptions.black_project_perception_parameters if perceptions else None

    defaults = {
        # Simulation settings
        "numYearsToSimulate": 10,
        "numSimulations": 100,
        "agreementYear": int(sim_params.policy.ai_slowdown_start_year) if sim_params.policy else 2030,
        "blackProjectStartYear": int(bp.black_project_start_year) if bp else 2029,

        # Black project properties
        "workersInCovertProject": bp_props.total_labor if bp_props else 11300,
        "fractionOfLaborDevotedToDatacenterConstruction": bp_props.fraction_of_labor_devoted_to_datacenter_construction if bp_props else 0.885,
        "fractionOfLaborDevotedToBlackFabConstruction": bp_props.fraction_of_labor_devoted_to_black_fab_construction if bp_props else 0.022,
        "fractionOfLaborDevotedToBlackFabOperation": bp_props.fraction_of_labor_devoted_to_black_fab_operation if bp_props else 0.049,
        "fractionOfLaborDevotedToAiResearch": bp_props.fraction_of_labor_devoted_to_ai_research if bp_props else 0.044,
        "proportionOfInitialChipStockToDivert": bp_props.fraction_of_initial_compute_stock_to_divert_at_black_project_start if bp_props else 0.05,
        "fractionOfDatacenterCapacityToDivert": bp_props.fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start if bp_props else 0.5,
        "fractionOfLithographyScannersToDivert": bp_props.fraction_of_lithography_scanners_to_divert_at_black_project_start if bp_props else 0.10,
        "maxFractionOfTotalNationalEnergyConsumption": bp_props.max_fraction_of_total_national_energy_consumption if bp_props else 0.05,
        # build_a_black_fab is derived: True if best available node meets min requirement
        "buildCovertFab": is_fab_built(bp_props, bp.black_project_start_year if bp else 2029) if bp_props else True,
        "blackFabMaxProcessNode": str(int(bp_props.black_fab_min_process_node)) if bp_props else "28",

        # Detection parameters
        "priorOddsOfCovertProject": bp_perceptions.prior_odds_of_covert_project if bp_perceptions else 0.111,
        "intelligenceMedianError": bp_perceptions.intelligence_median_error_in_estimate_of_compute_stock if bp_perceptions else 0.07,
        "meanDetectionTime100": bp_perceptions.mean_detection_time_for_100_workers if bp_perceptions else 6.95,
        "meanDetectionTime1000": bp_perceptions.mean_detection_time_for_1000_workers if bp_perceptions else 3.42,
        "varianceDetectionTime": bp_perceptions.variance_of_detection_time_given_num_workers if bp_perceptions else 3.88,
        "detectionThreshold": bp_perceptions.detection_threshold if bp_perceptions else 100.0,

        # PRC compute
        "totalPrcComputeTppH100eIn2025": prc_compute.total_prc_compute_tpp_h100e_in_2025 if prc_compute else 100000,
        "annualGrowthRateOfPrcComputeStock": prc_compute.annual_growth_rate_of_prc_compute_stock if prc_compute else 2.2,
        "h100SizedChipsPerWafer": prc_compute.h100_sized_chips_per_wafer if prc_compute else 28,
        "wafersPerMonthPerLithographyScanner": prc_compute.wafers_per_month_per_lithography_scanner if prc_compute else 1000,

        # PRC energy
        "energyEfficiencyOfComputeStockRelativeToStateOfTheArt": prc_energy.energy_efficiency_of_compute_stock_relative_to_state_of_the_art if prc_energy else 0.20,
        "totalPrcEnergyConsumptionGw": prc_energy.total_prc_energy_consumption_gw if prc_energy else 1100,
        "dataCenterMwPerYearPerConstructionWorker": prc_energy.data_center_mw_per_year_per_construction_worker if prc_energy else 1.0,

        # Survival parameters
        "initialAnnualHazardRate": compute.survival_rate_parameters.initial_annual_hazard_rate if compute else 0.05,
        "annualHazardRateIncreasePerYear": compute.survival_rate_parameters.annual_hazard_rate_increase_per_year if compute else 0.02,

        # Exogenous compute trends (for Dennard scaling curves)
        "transistorDensityScalingExponent": compute.exogenous_trends.transistor_density_scaling_exponent if compute else 1.49,
        "stateOfTheArtArchitectureEfficiencyImprovementPerYear": compute.exogenous_trends.state_of_the_art_architecture_efficiency_improvement_per_year if compute else 1.23,
        "transistorDensityAtEndOfDennardScaling": compute.exogenous_trends.transistor_density_at_end_of_dennard_scaling_m_per_mm2 if compute else 10.0,
        "wattsTppDensityExponentBeforeDennard": compute.exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended if compute else -1.0,
        "wattsTppDensityExponentAfterDennard": compute.exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended if compute else -0.33,
        "stateOfTheArtEnergyEfficiencyImprovementPerYear": compute.exogenous_trends.state_of_the_art_energy_efficiency_improvement_per_year if compute else 1.26,
    }

    return defaults
