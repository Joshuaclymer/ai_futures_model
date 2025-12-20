"""
Black project simulation and plot data extraction.

Runs Monte Carlo simulations and extracts aggregated data for plotting.
"""

import sys
import time
import math
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add ai_futures_simulator subdirectory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator"))

from parameters.simulation_parameters import ModelParameters
from parameters.black_project_parameters import BlackProjectParameters, BlackProjectProperties
from parameters.compute_parameters import ComputeParameters, PRCComputeParameters
from parameters.energy_consumption_parameters import EnergyConsumptionParameters, PRCEnergyConsumptionParameters
from parameters.perceptions_parameters import BlackProjectPerceptionsParameters
from parameters.policy_parameters import PolicyParameters
from world_updaters.black_project import initialize_black_project
from world_updaters.compute.black_compute import (
    get_black_project_total_labor,
    calculate_survival_rate,
    calculate_concealed_capacity_gw,
    calculate_datacenter_capacity_gw,
    calculate_fab_annual_production_h100e,
    calculate_operating_compute,
)
from classes.world.entities import Nation, NamedNations, AISoftwareDeveloper, ComputeAllocation
from classes.world.assets import Compute, Fabs, Datacenters
from classes.world.software_progress import AISoftwareProgress

logger = logging.getLogger(__name__)

# Constants
LIKELIHOOD_RATIO_THRESHOLDS = [1, 2, 4]
DEFAULT_LR_THRESHOLD = 5
H100_POWER_W = 700.0  # H100 draws ~700W


@dataclass
class BlackProjectSimResult:
    """Result of a single black project simulation."""
    years: List[float]
    operational_compute: List[float]
    total_compute: List[float]
    datacenter_capacity: List[float]
    survival_rate: List[float]
    cumulative_lr: List[float]
    h100_years: List[float]
    detection_time: float
    h100e_at_detection: float
    h100_years_at_detection: float
    fab_built: bool
    fab_is_operational: List[bool]
    fab_production: List[float]


def sample_prc_compute_stock(
    prc_compute_params: PRCComputeParameters,
    agreement_year: float
) -> float:
    """Sample PRC compute stock at agreement year based on growth rate."""
    # Use the median growth rate
    growth_rate = prc_compute_params.annual_growth_rate_of_prc_compute_stock

    # Calculate stock at agreement year
    years_since_2025 = agreement_year - 2025
    stock = prc_compute_params.total_prc_compute_tpp_h100e_in_2025 * (growth_rate ** years_since_2025)

    return stock


def create_dummy_prc_nation(
    compute_stock_h100e: float,
    prc_energy_params: PRCEnergyConsumptionParameters,
    total_energy_gw: float,
) -> Nation:
    """Create a minimal PRC Nation for black project initialization."""
    # Create minimal AISoftwareProgress
    ai_sw_progress = AISoftwareProgress(
        progress=torch.tensor(0.0),
        research_stock=torch.tensor(0.0),
        ai_coding_labor_multiplier=torch.tensor(1.0),
        ai_sw_progress_mult_ref_present_day=torch.tensor(1.0),
        progress_rate=torch.tensor(0.0),
        software_progress_rate=torch.tensor(0.0),
        research_effort=torch.tensor(0.0),
        automation_fraction=torch.tensor(0.0),
        coding_labor=torch.tensor(0.0),
        serial_coding_labor=torch.tensor(0.0),
        ai_research_taste=torch.tensor(0.0),
        ai_research_taste_sd=torch.tensor(0.0),
        aggregate_research_taste=torch.tensor(0.0),
    )

    # Create a minimal AI software developer
    energy_efficiency = prc_energy_params.energy_efficiency_of_compute_stock_relative_to_state_of_the_art
    prc_compute = Compute(
        all_tpp_h100e=compute_stock_h100e,
        functional_tpp_h100e=compute_stock_h100e,
        watts_per_h100e=H100_POWER_W / energy_efficiency,
        average_functional_chip_age_years=0.0,
    )

    compute_allocation = ComputeAllocation(
        fraction_for_ai_r_and_d_inference=0.3,
        fraction_for_ai_r_and_d_training=0.3,
        fraction_for_external_deployment=0.2,
        fraction_for_alignment_research=0.1,
        fraction_for_frontier_training=0.1,
    )

    prc_developer = AISoftwareDeveloper(
        id="prc_developer",
        operating_compute=[prc_compute],
        compute_allocation=compute_allocation,
        human_ai_capability_researchers=1000.0,
        ai_software_progress=ai_sw_progress,
    )
    # Set init=False metrics
    object.__setattr__(prc_developer, 'ai_r_and_d_inference_compute_tpp_h100e', 0.0)
    object.__setattr__(prc_developer, 'ai_r_and_d_training_compute_tpp_h100e', 0.0)
    object.__setattr__(prc_developer, 'external_deployment_compute_tpp_h100e', 0.0)
    object.__setattr__(prc_developer, 'alignment_research_compute_tpp_h100e', 0.0)
    object.__setattr__(prc_developer, 'frontier_training_compute_tpp_h100e', 0.0)

    # Create fab
    fab_production_compute = Compute(
        all_tpp_h100e=0.0,
        functional_tpp_h100e=0.0,
        watts_per_h100e=H100_POWER_W,
        average_functional_chip_age_years=0.0,
    )
    prc_fabs = Fabs(monthly_production_compute=fab_production_compute)

    # Create compute stock
    prc_compute_stock = Compute(
        all_tpp_h100e=compute_stock_h100e,
        functional_tpp_h100e=compute_stock_h100e,
        watts_per_h100e=H100_POWER_W / energy_efficiency,
        average_functional_chip_age_years=2.0,
    )

    # Create datacenters
    total_dc_capacity = total_energy_gw * 0.05  # ~5% of energy for AI
    prc_datacenters = Datacenters(data_center_capacity_gw=total_dc_capacity)

    return Nation(
        id=NamedNations.PRC,
        ai_software_developers=[prc_developer],
        fabs=prc_fabs,
        compute_stock=prc_compute_stock,
        datacenters=prc_datacenters,
        total_energy_consumption_gw=total_energy_gw,
        leading_ai_software_developer=prc_developer,
        operating_compute_tpp_h100e=compute_stock_h100e * 0.8,  # 80% operational
    )


def run_single_simulation(
    agreement_year: float,
    end_year: float,
    black_project_params: BlackProjectParameters,
    compute_params: ComputeParameters,
    energy_params: EnergyConsumptionParameters,
    perception_params: BlackProjectPerceptionsParameters,
    policy_params: PolicyParameters,
    dt: float = 0.1,
) -> BlackProjectSimResult:
    """Run a single black project simulation."""

    # Get nested params
    prc_compute = compute_params.PRCComputeParameters
    prc_energy = energy_params.prc_energy_consumption
    survival_params = compute_params.survival_rate_parameters

    # Sample PRC compute stock
    prc_compute_stock = sample_prc_compute_stock(prc_compute, agreement_year)

    # Generate years array
    years = []
    t = agreement_year
    while t <= end_year:
        years.append(t)
        t += dt

    # Create dummy PRC nation for parent_nation reference
    prc_nation = create_dummy_prc_nation(
        prc_compute_stock,
        prc_energy,
        prc_energy.total_prc_energy_consumption_gw,
    )

    # Initialize black project with new signature
    project, lr_by_year, sampled_detection_time = initialize_black_project(
        project_id="prc_black_project",
        parent_nation=prc_nation,
        black_project_params=black_project_params,
        compute_params=compute_params,
        energy_params=energy_params,
        perception_params=perception_params,
        policy_params=policy_params,
        initial_prc_compute_stock=prc_compute_stock,
        simulation_years=years,
    )

    # Initialize tracking arrays
    operational_compute = []
    total_compute = []
    datacenter_capacity = []
    survival_rate_list = []
    cumulative_lr = []
    h100_years = []
    fab_is_operational_list = []
    fab_production = []

    # Track initial values from compute_stock
    initial_stock = project.compute_stock.all_tpp_h100e if project.compute_stock else 0.0

    # Get hazard parameters from survival rate params
    initial_hazard = survival_params.initial_annual_hazard_rate
    hazard_increase = survival_params.annual_hazard_rate_increase_per_year

    # Get energy efficiency
    energy_efficiency = prc_energy.energy_efficiency_of_compute_stock_relative_to_state_of_the_art
    watts_per_h100e = H100_POWER_W / energy_efficiency

    # Get datacenter construction rate
    gw_per_worker_per_year = black_project_params.datacenter_mw_per_year_per_construction_worker / 1000.0
    construction_rate = gw_per_worker_per_year * project.concealed_datacenter_capacity_construction_labor

    # Detection tracking
    detection_time = float('inf')
    h100e_at_detection = 0.0
    h100_years_at_detection = 0.0
    cumulative_h100_years = 0.0

    # Track fab operational status
    fab_operational = project.fab_is_operational
    fab_operational_year = project.preparation_start_year + project.fab_construction_duration

    # Step through time
    prev_year = None
    for year in years:
        years_since_start = year - agreement_year

        # Update fab operational status
        if not fab_operational and year >= fab_operational_year:
            fab_operational = True

        # Calculate survival rate using hazard model
        survival = calculate_survival_rate(
            years_since_acquisition=years_since_start,
            initial_hazard_rate=initial_hazard,
            hazard_rate_increase_per_year=hazard_increase,
        )

        # Calculate surviving initial compute
        surviving_initial = initial_stock * survival

        # Calculate fab production if operational
        fab_prod_h100e = 0.0
        if fab_operational:
            fab_prod_h100e = calculate_fab_annual_production_h100e(
                fab_wafer_starts_per_month=project.fab_wafer_starts_per_month,
                fab_chips_per_wafer=project.fab_chips_per_wafer,
                fab_h100e_per_chip=project.fab_h100e_per_chip,
                fab_is_operational=True,
            )

        # Cumulative fab production
        if prev_year is not None:
            cumulative_fab = sum(fab_production) * dt * 12  # Convert to annual
        else:
            cumulative_fab = 0.0

        current_stock = surviving_initial + cumulative_fab

        # Calculate datacenter capacity (linear growth)
        concealed = calculate_concealed_capacity_gw(
            current_year=year,
            construction_start_year=project.preparation_start_year,
            construction_rate_gw_per_year=construction_rate,
            max_concealed_capacity_gw=project.concealed_max_total_capacity_gw,
        )

        dc_capacity = calculate_datacenter_capacity_gw(
            unconcealed_capacity_gw=project.unconcealed_datacenter_capacity_diverted_gw,
            concealed_capacity_gw=concealed,
        )

        # Calculate operational compute (limited by datacenter)
        op_compute = calculate_operating_compute(
            functional_compute_h100e=current_stock,
            datacenter_capacity_gw=dc_capacity,
            watts_per_h100e=watts_per_h100e,
        )

        # Calculate H100-years
        if prev_year is not None:
            dt_actual = year - prev_year
            cumulative_h100_years += op_compute * dt_actual

        # Get cumulative LR from pre-computed values
        relative_year = int(years_since_start)
        if lr_by_year and relative_year in lr_by_year:
            lr = lr_by_year[relative_year]
        else:
            lr = 1.0

        # Check detection
        if sampled_detection_time <= years_since_start and detection_time == float('inf'):
            detection_time = years_since_start
            h100e_at_detection = op_compute
            h100_years_at_detection = cumulative_h100_years

        # Store values
        operational_compute.append(op_compute)
        total_compute.append(current_stock)
        datacenter_capacity.append(dc_capacity)
        survival_rate_list.append(survival if initial_stock > 0 else 0.0)
        cumulative_lr.append(lr)
        h100_years.append(cumulative_h100_years)
        fab_is_operational_list.append(fab_operational)
        fab_production.append(fab_prod_h100e)

        prev_year = year

    # Determine if fab was built
    fab_built = black_project_params.black_project_properties.build_a_black_fab

    return BlackProjectSimResult(
        years=years,
        operational_compute=operational_compute,
        total_compute=total_compute,
        datacenter_capacity=datacenter_capacity,
        survival_rate=survival_rate_list,
        cumulative_lr=cumulative_lr,
        h100_years=h100_years,
        detection_time=detection_time,
        h100e_at_detection=h100e_at_detection,
        h100_years_at_detection=h100_years_at_detection,
        fab_built=fab_built,
        fab_is_operational=fab_is_operational_list,
        fab_production=fab_production,
    )


def run_black_project_simulations(
    frontend_params: dict,
    num_simulations: int,
    time_range: list,
) -> List[BlackProjectSimResult]:
    """
    Run multiple Monte Carlo simulations of black projects.

    Args:
        frontend_params: Parameters from frontend
        num_simulations: Number of simulations to run
        time_range: [agreement_year, end_year]

    Returns:
        List of BlackProjectSimResult objects
    """
    start_time = time.perf_counter()

    # Load model parameters
    config_path = Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator" / "parameters" / "modal_parameters.yaml"
    model_params = ModelParameters.from_yaml(config_path)

    agreement_year = float(time_range[0]) if time_range else 2030.0
    end_year = float(time_range[1]) if len(time_range) > 1 else 2050.0

    simulation_results = []

    for i in range(num_simulations):
        try:
            # Sample fresh parameters for each simulation (Monte Carlo)
            sim_params = model_params.sample()

            # Get parameters from sampled params
            black_project_params = sim_params.black_project
            if black_project_params is None:
                logger.warning(f"Simulation {i}: No black project parameters, skipping")
                continue

            compute_params = sim_params.compute
            energy_params = sim_params.energy_consumption
            perception_params = sim_params.perceptions.black_project_perception_parameters
            policy_params = sim_params.policy

            result = run_single_simulation(
                agreement_year=agreement_year,
                end_year=end_year,
                black_project_params=black_project_params,
                compute_params=compute_params,
                energy_params=energy_params,
                perception_params=perception_params,
                policy_params=policy_params,
                dt=0.5,  # 6-month time steps
            )
            simulation_results.append(result)

        except Exception as e:
            logger.warning(f"Simulation {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    elapsed = time.perf_counter() - start_time
    logger.info(f"[black-project] Completed {len(simulation_results)} simulations in {elapsed:.2f}s")

    return simulation_results


def calculate_detection_year(cumulative_lr_list: List[float], years: List[float], threshold: float) -> float:
    """Calculate detection year based on when cumulative LR exceeds threshold."""
    for i, lr in enumerate(cumulative_lr_list):
        if lr >= threshold:
            return years[i]
    return float('inf')


def calculate_h100_years_at_detection(h100_years_list: List[float], years: List[float], detection_year: float) -> float:
    """Get cumulative H100-years at detection time."""
    if detection_year == float('inf'):
        return h100_years_list[-1] if h100_years_list else 0.0
    for i, year in enumerate(years):
        if year >= detection_year:
            return h100_years_list[i]
    return h100_years_list[-1] if h100_years_list else 0.0


def extract_black_project_plot_data(
    simulation_results: List[BlackProjectSimResult],
    frontend_params: dict,
) -> Dict[str, Any]:
    """
    Extract aggregated plot data from simulation results.

    Args:
        simulation_results: List of BlackProjectSimResult objects
        frontend_params: Parameters from frontend

    Returns:
        Dictionary containing all plot data for the black project page,
        matching the structure expected by black_project_frontend.
    """
    if not simulation_results:
        return {"error": "No simulation results"}

    # Get common parameters
    years = simulation_results[0].years
    years_array = np.array(years)
    num_sims = len(simulation_results)
    agreement_year = years[0] if years else 2030.0

    # Collect arrays from all simulations
    all_operational_compute = [r.operational_compute for r in simulation_results]
    all_total_compute = [r.total_compute for r in simulation_results]
    all_datacenter_capacity = [r.datacenter_capacity for r in simulation_results]
    all_survival_rate = [r.survival_rate for r in simulation_results]
    all_cumulative_lr = [r.cumulative_lr for r in simulation_results]
    all_h100_years = [r.h100_years for r in simulation_results]
    all_detection_times = [r.detection_time for r in simulation_results]
    all_h100e_at_detection = [r.h100e_at_detection for r in simulation_results]
    all_h100_years_at_detection = [r.h100_years_at_detection for r in simulation_results]
    all_fab_built = [r.fab_built for r in simulation_results]
    all_fab_is_operational = [r.fab_is_operational for r in simulation_results]
    all_fab_production = [r.fab_production for r in simulation_results]

    def fmt_percentiles(data: list) -> Dict[str, Any]:
        """Calculate percentiles from simulation data."""
        if not data:
            return {"p25": [], "median": [], "p75": [], "individual": []}
        arr = np.array(data)
        if arr.ndim == 1:
            return {
                "p25": float(np.percentile(arr, 25)),
                "median": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "individual": arr.tolist(),
            }
        return {
            "p25": np.percentile(arr, 25, axis=0).tolist(),
            "median": np.percentile(arr, 50, axis=0).tolist(),
            "p75": np.percentile(arr, 75, axis=0).tolist(),
            "individual": [list(row) for row in arr],
        }

    def calculate_ccdf(values: list) -> List[Dict[str, float]]:
        """Calculate CCDF from a list of values."""
        if not values:
            return []
        arr = np.array([v for v in values if v < float('inf') and v > 0])
        if len(arr) == 0:
            return []
        sorted_vals = np.sort(arr)
        n = len(sorted_vals)
        ccdf_points = []
        prev_val = None
        for i, val in enumerate(sorted_vals):
            if val != prev_val:  # Only unique x values
                ccdf_points.append({
                    "x": float(val),
                    "y": float((n - i) / n),
                })
                prev_val = val
        return ccdf_points

    # Calculate CCDFs for multiple thresholds
    def calculate_threshold_ccdfs(metric_func) -> Dict[int, List[Dict[str, float]]]:
        """Calculate CCDFs at multiple LR thresholds."""
        ccdfs = {}
        for threshold in LIKELIHOOD_RATIO_THRESHOLDS:
            values = []
            for i, result in enumerate(simulation_results):
                detection_year = calculate_detection_year(result.cumulative_lr, result.years, threshold)
                metric_value = metric_func(result, detection_year)
                values.append(metric_value)
            ccdfs[threshold] = calculate_ccdf(values)
        return ccdfs

    # Calculate H100-years CCDFs at each threshold
    def get_h100_years_at_threshold(result, detection_year):
        return calculate_h100_years_at_detection(result.h100_years, result.years, detection_year)

    h100_years_ccdfs = calculate_threshold_ccdfs(get_h100_years_at_threshold)

    # Calculate time-to-detection CCDFs at each threshold
    def get_time_to_detection(result, detection_year):
        if detection_year == float('inf'):
            return 100.0  # Large value for "never detected"
        return detection_year - agreement_year

    time_to_detection_ccdfs = calculate_threshold_ccdfs(get_time_to_detection)

    # Calculate average covert compute CCDFs at each threshold
    def get_average_compute(result, detection_year):
        if detection_year == float('inf') or detection_year <= agreement_year:
            return result.operational_compute[-1] if result.operational_compute else 0.0
        h100_years = calculate_h100_years_at_detection(result.h100_years, result.years, detection_year)
        duration = detection_year - agreement_year
        return h100_years / duration if duration > 0 else 0.0

    average_compute_ccdfs = calculate_threshold_ccdfs(get_average_compute)

    # Filter for simulations with fab
    n_with_fab = sum(all_fab_built)
    prob_fab_built = n_with_fab / num_sims if num_sims > 0 else 0.0

    # Calculate median values for dashboard (using default threshold of 4)
    default_threshold = 4
    valid_detection_times = []
    valid_h100_years = []
    valid_h100e = []
    for result in simulation_results:
        det_year = calculate_detection_year(result.cumulative_lr, result.years, default_threshold)
        time_to_det = det_year - agreement_year if det_year < float('inf') else 100.0
        h100_yrs = calculate_h100_years_at_detection(result.h100_years, result.years, det_year)
        valid_detection_times.append(time_to_det)
        valid_h100_years.append(h100_yrs)
        # Get operational compute at detection
        if det_year < float('inf'):
            for i, year in enumerate(result.years):
                if year >= det_year:
                    valid_h100e.append(result.operational_compute[i])
                    break
        else:
            valid_h100e.append(result.operational_compute[-1] if result.operational_compute else 0.0)

    median_detection_time = float(np.median(valid_detection_times)) if valid_detection_times else 0.0
    median_h100_years = float(np.median(valid_h100_years)) if valid_h100_years else 0.0
    median_h100e = float(np.median(valid_h100e)) if valid_h100e else 0.0

    # Calculate posterior probability from cumulative LR
    p_project_exists = frontend_params.get("priorProbability", 0.5)
    prior_odds = p_project_exists / (1 - p_project_exists) if p_project_exists < 1 else float('inf')

    def lr_to_posterior(lr):
        posterior_odds = prior_odds * lr
        return posterior_odds / (1 + posterior_odds) if posterior_odds < float('inf') else 1.0

    posterior_prob_by_sim = []
    for lr_over_time in all_cumulative_lr:
        posterior_prob_by_sim.append([lr_to_posterior(lr) for lr in lr_over_time])

    # Build initial dark compute (assumes initial stock is total - fab production at t=0)
    # For simplicity, use first value of total_compute
    initial_black_compute = []
    for i, result in enumerate(simulation_results):
        # Initial compute is total compute minus fab production
        initial_vals = []
        for j in range(len(result.total_compute)):
            fab_prod = sum(result.fab_production[:j+1]) if result.fab_production else 0.0
            initial_vals.append(max(0, result.total_compute[j] - fab_prod))
        initial_black_compute.append(initial_vals)

    # Build results in format compatible with frontend (matching format_data_for_black_project_plots.py)
    results = {
        "num_simulations": num_sims,
        "prob_fab_built": prob_fab_built,
        "p_project_exists": p_project_exists,
        "researcher_headcount": frontend_params.get("researcherHeadcount", 500),

        "black_project_model": {
            "years": years_array.tolist(),

            # Dashboard values (80th percentile for "median outcome")
            "project_80th_h100_years": [float(np.percentile(valid_h100_years, 80))] if valid_h100_years else [0.0],
            "project_80th_h100e": [float(np.percentile(valid_h100e, 80))] if valid_h100e else [0.0],
            "project_80th_time": [float(np.percentile(valid_detection_times, 80))] if valid_detection_times else [0.0],

            # CCDFs by threshold
            "h100_years_ccdf": h100_years_ccdfs,
            "average_covert_compute_ccdf": average_compute_ccdfs,
            "time_to_detection_ccdf": time_to_detection_ccdfs,
            "ai_rd_reduction_ccdf": {
                "largest_company": {},  # Placeholder - would need largest company model
                "prc": {},
            },
            "chip_production_reduction_ccdf": {},  # Placeholder
            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,

            # Time series data with percentiles
            "h100_years": fmt_percentiles(all_h100_years),
            "cumulative_lr": fmt_percentiles(all_cumulative_lr),
            "initial_black_project": fmt_percentiles(initial_black_compute),
            "black_fab_flow": fmt_percentiles(all_fab_production),
            "black_fab_flow_all_sims": fmt_percentiles(all_fab_production),
            "survival_rate": fmt_percentiles(all_survival_rate),
            "total_black_project": fmt_percentiles(all_total_compute),
            "datacenter_capacity": fmt_percentiles(all_datacenter_capacity),
            "operational_compute": fmt_percentiles(all_operational_compute),

            # LR components (placeholders - would need more detailed tracking)
            "lr_initial_stock": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "lr_diverted_sme": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "lr_other_intel": fmt_percentiles([[lr for lr in row] for row in all_cumulative_lr]),
            "posterior_prob_project": fmt_percentiles(posterior_prob_by_sim),
            "lr_prc_accounting": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "lr_sme_inventory": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "lr_satellite_datacenter": {"individual": [1.0] * num_sims},
            "lr_reported_energy": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "lr_combined_reported_assets": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),

            # Individual simulation data for dashboard
            "individual_project_h100e_before_detection": valid_h100e,
            "individual_project_energy_before_detection": [0.0] * num_sims,  # Placeholder
            "individual_project_time_before_detection": valid_detection_times,
            "individual_project_h100_years_before_detection": valid_h100_years,
        },

        "initial_black_project": {
            "years": years_array.tolist(),
            "h100e": fmt_percentiles(initial_black_compute),
            "survival_rate": fmt_percentiles(all_survival_rate),
            "black_project": fmt_percentiles(initial_black_compute),
        },

        "initial_stock": {
            "initial_prc_stock_samples": [1000000.0] * num_sims,  # Placeholder
            "initial_compute_stock_samples": [r.total_compute[0] if r.total_compute else 0.0 for r in simulation_results],
            "diversion_proportion": frontend_params.get("proportionDiverted", 0.05),
            "lr_prc_accounting_samples": [1.0] * num_sims,  # Placeholder
            "initial_black_project_detection_probs": {f"{t}x": 0.0 for t in LIKELIHOOD_RATIO_THRESHOLDS},
        },

        "black_datacenters": {
            "years": years_array.tolist(),
            "datacenter_capacity": fmt_percentiles(all_datacenter_capacity),
            "energy_by_source": [],  # Placeholder
            "source_labels": [],
            "operational_compute": fmt_percentiles(all_operational_compute),
            "lr_datacenters": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "datacenter_detection_prob": [0.0] * len(years),
            "capacity_ccdfs": {t: calculate_ccdf([dc[-1] for dc in all_datacenter_capacity]) for t in LIKELIHOOD_RATIO_THRESHOLDS},
            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,
            "individual_capacity_before_detection": [dc[-1] for dc in all_datacenter_capacity],
            "individual_time_before_detection": valid_detection_times,
            "prc_capacity_years": list(range(2025, int(agreement_year) + 1)),
            "prc_capacity_gw": {"p25": [], "median": [], "p75": []},
            "prc_capacity_at_agreement_year_gw": 0.0,
            "prc_capacity_at_agreement_year_samples": [],
            "fraction_diverted": frontend_params.get("fractionDatacentersDiverted", 0.01),
        },

        "black_fab": {
            "years": years_array.tolist(),
            "compute_ccdf": calculate_ccdf([sum(fp) for fp in all_fab_production]),
            "compute_ccdfs": {t: calculate_ccdf([sum(fp) for fp in all_fab_production]) for t in LIKELIHOOD_RATIO_THRESHOLDS},
            "op_time_ccdf": [],
            "op_time_ccdfs": {},
            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,
            "lr_inventory": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "lr_procurement": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "lr_other": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "lr_combined": fmt_percentiles(all_cumulative_lr),
            "is_operational": {
                "proportion": np.mean(all_fab_is_operational, axis=0).tolist() if all_fab_is_operational else [],
                "individual": [list(row) for row in all_fab_is_operational],
            },
            "wafer_starts": fmt_percentiles([[0.0] * len(years) for _ in range(num_sims)]),
            "chips_per_wafer": fmt_percentiles([[28.0] * len(years) for _ in range(num_sims)]),
            "architecture_efficiency": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "compute_per_wafer_2022_arch": fmt_percentiles([[0.0] * len(years) for _ in range(num_sims)]),
            "transistor_density": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "watts_per_tpp": fmt_percentiles([[1.0] * len(years) for _ in range(num_sims)]),
            "process_node_by_sim": ["28nm"] * num_sims,
            "architecture_efficiency_at_agreement": 1.0,
            "watts_per_tpp_curve": [],
            "individual_h100e_before_detection": [sum(fp) for fp in all_fab_production],
            "individual_time_before_detection": valid_detection_times,
            "individual_process_node": ["28nm"] * num_sims,
            "individual_energy_before_detection": [0.0] * num_sims,
        },
    }

    return results
