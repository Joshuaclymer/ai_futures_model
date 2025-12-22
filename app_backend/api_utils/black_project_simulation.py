"""
Black project simulation using the actual AIFuturesSimulator.

Runs Monte Carlo simulations and extracts plot data from World trajectories.
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add ai_futures_simulator subdirectory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator"))

from ai_futures_simulator import AIFuturesSimulator
from parameters.simulation_parameters import ModelParameters
from classes.simulation_primitives import SimulationResult
from classes.world.world import World
from classes.world.entities import NamedNations

logger = logging.getLogger(__name__)

# Constants
LIKELIHOOD_RATIO_THRESHOLDS = [1, 2, 4]


def run_black_project_simulations(
    frontend_params: dict,
    num_mc_simulations: int = 10,
    time_range: list = None,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulations using the actual AIFuturesSimulator.

    Returns results dict with:
    - simulation_results: List of SimulationResult objects
    - agreement_year: Start year
    - end_year: End year
    """
    start_time = time.perf_counter()

    # Load model parameters from YAML
    config_path = Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator" / "parameters" / "modal_parameters.yaml"
    logger.info(f"[black-project] Loading config from: {config_path}")

    try:
        model_params = ModelParameters.from_yaml(config_path)
        logger.info("[black-project] Model parameters loaded successfully")
    except Exception as e:
        logger.exception(f"[black-project] Failed to load model parameters: {e}")
        raise

    agreement_year = float(time_range[0]) if time_range else 2027.0
    end_year = float(time_range[1]) if len(time_range) > 1 else 2037.0
    logger.info(f"[black-project] Time range: {agreement_year} to {end_year}")

    # Create simulator
    simulator = AIFuturesSimulator(model_parameters=model_params)

    # Run Monte Carlo simulations
    total_sims = num_mc_simulations + 1  # +1 for central simulation
    logger.info(f"[black-project] Running {total_sims} simulations...")

    simulation_results = simulator.run_simulations(num_simulations=total_sims)

    elapsed = time.perf_counter() - start_time
    logger.info(f"[black-project] Completed {len(simulation_results)} simulations in {elapsed:.2f}s")

    return {
        'simulation_results': simulation_results,
        'agreement_year': agreement_year,
        'end_year': end_year,
    }


def extract_world_data(result: SimulationResult) -> Dict[str, Any]:
    """Extract time series data from a single SimulationResult."""
    trajectory = result.trajectory
    times = result.times.tolist()

    data = {
        'years': times,
        'prc_compute_stock': [],
        'prc_operating_compute': [],
        'black_project': None,
    }

    for world in trajectory:
        # Extract PRC nation data
        prc = world.nations.get(NamedNations.PRC)
        if prc:
            data['prc_compute_stock'].append(
                prc.compute_stock.functional_tpp_h100e if prc.compute_stock else 0.0
            )
            data['prc_operating_compute'].append(prc.operating_compute_tpp_h100e or 0.0)
        else:
            data['prc_compute_stock'].append(0.0)
            data['prc_operating_compute'].append(0.0)

    # Extract black project data if present
    if world.black_projects:
        bp_id = list(world.black_projects.keys())[0]
        bp_data = {
            'id': bp_id,
            'operational_compute': [],
            'total_compute': [],
            'datacenter_capacity_gw': [],
            'fab_is_operational': [],
        }

        for world in trajectory:
            bp = world.black_projects.get(bp_id)
            if bp:
                bp_data['operational_compute'].append(bp.operating_compute_tpp_h100e or 0.0)
                bp_data['total_compute'].append(
                    bp.compute_stock.functional_tpp_h100e if bp.compute_stock else 0.0
                )
                bp_data['datacenter_capacity_gw'].append(
                    bp.datacenters.data_center_capacity_gw if bp.datacenters else 0.0
                )
                bp_data['fab_is_operational'].append(bp.fab_is_operational if hasattr(bp, 'fab_is_operational') else False)
            else:
                bp_data['operational_compute'].append(0.0)
                bp_data['total_compute'].append(0.0)
                bp_data['datacenter_capacity_gw'].append(0.0)
                bp_data['fab_is_operational'].append(False)

        data['black_project'] = bp_data

    return data


def extract_black_project_plot_data(
    simulation_results: Dict[str, Any],
    frontend_params: dict,
) -> Dict[str, Any]:
    """
    Extract plot data from SimulationResult trajectories.
    Returns data formatted for the frontend.
    """
    results: List[SimulationResult] = simulation_results.get('simulation_results', [])
    agreement_year = simulation_results.get('agreement_year', 2027)

    if not results:
        return {"success": False, "error": "No simulation results"}

    # Extract data from all simulations
    all_data = [extract_world_data(r) for r in results]

    # Use first simulation as reference for years
    years = all_data[0]['years'] if all_data else []
    num_sims = len(all_data)

    # Log raw simulation data for debugging
    logger.info("=" * 60)
    logger.info("[RAW SIMULATION RESULTS FROM AIFuturesSimulator]")
    logger.info(f"Number of simulations: {num_sims}")
    logger.info(f"Number of time points: {len(years)}")
    if years:
        logger.info(f"Time range: {years[0]} to {years[-1]}")
    if all_data:
        logger.info(f"First sim PRC compute stock (first 5): {all_data[0]['prc_compute_stock'][:5]}")
        if all_data[0]['black_project']:
            bp = all_data[0]['black_project']
            logger.info(f"First sim black project operational_compute (first 5): {bp['operational_compute'][:5]}")
            logger.info(f"First sim black project total_compute (first 5): {bp['total_compute'][:5]}")
    logger.info("=" * 60)

    # Helper to compute percentiles across simulations
    def get_percentiles(extractor) -> Dict[str, List[float]]:
        try:
            values = np.array([extractor(d) for d in all_data])
            if values.size == 0:
                return {"median": [], "p25": [], "p75": []}
            return {
                "median": np.percentile(values, 50, axis=0).tolist(),
                "p25": np.percentile(values, 25, axis=0).tolist(),
                "p75": np.percentile(values, 75, axis=0).tolist(),
            }
        except Exception as e:
            logger.warning(f"Error computing percentiles: {e}")
            return {"median": [], "p25": [], "p75": []}

    # Helper to compute CCDF
    def compute_ccdf(values: List[float]) -> List[Dict[str, float]]:
        values = [v for v in values if v > 0 and v < float('inf')]
        if not values:
            return []
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        ccdf = []
        seen = set()
        for i, val in enumerate(sorted_vals):
            if val not in seen:
                ccdf.append({"x": float(val), "y": float((n - i) / n)})
                seen.add(val)
        return ccdf

    # Extract black project specific data
    bp_data = [d['black_project'] for d in all_data if d['black_project']]

    # Build response
    response = {
        "num_simulations": num_sims,
        "agreement_year": agreement_year,

        # PRC compute data (from actual World trajectory)
        "prc_compute": {
            "years": years,
            "compute_stock": get_percentiles(lambda d: d['prc_compute_stock']),
            "operating_compute": get_percentiles(lambda d: d['prc_operating_compute']),
        },

        # Main model data
        "black_project_model": {
            "years": years,
            "operational_compute": get_percentiles(lambda d: d['black_project']['operational_compute'] if d['black_project'] else []),
            "total_black_project": get_percentiles(lambda d: d['black_project']['total_compute'] if d['black_project'] else []),
            "datacenter_capacity": get_percentiles(lambda d: d['black_project']['datacenter_capacity_gw'] if d['black_project'] else []),

            # Individual simulation data (placeholders - detection not yet in real simulator)
            "individual_project_time_before_detection": [],
            "individual_project_h100_years_before_detection": [],
            "individual_project_h100e_before_detection": [],

            "time_to_detection_ccdf": {"4": []},
            "h100_years_ccdf": {"4": []},
            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,
        },

        # Datacenter section data
        "black_datacenters": {
            "years": years,
            "datacenter_capacity": get_percentiles(lambda d: d['black_project']['datacenter_capacity_gw'] if d['black_project'] else []),
            "individual_capacity_before_detection": [],
            "individual_time_before_detection": [],
            "capacity_ccdfs": {"4": []},
        },

        # Initial stock section
        "initial_stock": {
            "initial_prc_stock_samples": [d['prc_compute_stock'][0] if d['prc_compute_stock'] else 0 for d in all_data],
            "initial_compute_stock_samples": [
                d['black_project']['total_compute'][0] if d['black_project'] and d['black_project']['total_compute'] else 0
                for d in all_data
            ],
            "diversion_proportion": 0.05,  # TODO: extract from params
        },

        # Fab section
        "black_fab": {
            "years": years,
            "is_operational": {
                "proportion": [],  # TODO: compute from fab_is_operational
            },
            "compute_ccdfs": {"4": []},
            "process_node_by_sim": [],
            "prob_fab_built": 0.0,
        },

        # Rate of computation section
        "rate_of_computation": {
            "years": years,
            "initial_chip_stock_samples": [
                d['black_project']['total_compute'][0] if d['black_project'] and d['black_project']['total_compute'] else 0
                for d in all_data
            ],
            "covert_chip_stock": get_percentiles(lambda d: d['black_project']['total_compute'] if d['black_project'] else []),
            "datacenter_capacity": get_percentiles(lambda d: d['black_project']['datacenter_capacity_gw'] if d['black_project'] else []),
            "operating_chips": get_percentiles(lambda d: d['black_project']['operational_compute'] if d['black_project'] else []),
        },

        # Debug: raw World data for exploration
        "_debug_raw_simulations": {
            "description": "Raw World trajectory data from AIFuturesSimulator",
            "num_simulations": num_sims,
            "num_time_points": len(years),
            "years": years,
            "first_simulation": all_data[0] if all_data else None,
        },
    }

    return response


def get_default_parameters() -> Dict[str, Any]:
    """Get default parameters from YAML for frontend initialization."""
    config_path = Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator" / "parameters" / "modal_parameters.yaml"
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
        "numSimulations": 11,  # 1 central + 10 MC
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
        "buildCovertFab": bp_props.build_a_black_fab if bp_props else True,
        "blackFabMaxProcessNode": str(int(bp_props.black_fab_max_process_node)) if bp_props else "28",

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
    }

    return defaults
