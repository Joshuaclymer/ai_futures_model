"""
Black project simulation and plot data extraction.

Runs Monte Carlo simulations and extracts aggregated data for plotting.
"""

import sys
import time
import math
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add ai_futures_simulator subdirectory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator"))

from parameters.simulation_parameters import ModelParameters
from parameters.black_project_parameters import BlackProjectParameterSet
from parameters.compute_parameters import ComputeParameters
from parameters.energy_consumption_parameters import EnergyConsumptionParameters
from world_updaters.black_project import initialize_black_project
from world_updaters.compute.black_compute import (
    get_compute_stock_h100e,
    get_datacenter_concealed_capacity_gw,
    get_datacenter_total_capacity_gw,
    get_fab_annual_production_h100e,
    get_black_project_total_labor,
    compute_cumulative_likelihood_ratio,
)

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


def sample_prc_compute_stock(compute_params: ComputeParameters, agreement_year: float) -> float:
    """Sample PRC compute stock at agreement year based on growth rate distribution."""
    # Sample growth rate from distribution
    p10 = compute_params.annual_growth_rate_of_prc_compute_stock_p10
    p50 = compute_params.annual_growth_rate_of_prc_compute_stock_p50
    p90 = compute_params.annual_growth_rate_of_prc_compute_stock_p90

    # Sample using log-normal approximation
    log_p10 = math.log(p10)
    log_p50 = math.log(p50)
    log_p90 = math.log(p90)

    # Estimate mu and sigma for log-normal
    mu = log_p50
    sigma = (log_p90 - log_p10) / (2 * 1.28)  # 1.28 is z-score for 80th percentile

    # Sample growth rate
    growth_rate = np.exp(np.random.normal(mu, sigma))

    # Calculate stock at agreement year
    years_since_2025 = agreement_year - 2025
    stock = compute_params.total_prc_compute_stock_in_2025 * (growth_rate ** years_since_2025)

    return stock


def run_single_simulation(
    agreement_year: float,
    end_year: float,
    black_project_params: BlackProjectParameterSet,
    compute_params: ComputeParameters,
    energy_params: EnergyConsumptionParameters,
    dt: float = 0.1,
) -> BlackProjectSimResult:
    """Run a single black project simulation."""

    # Sample PRC compute stock
    prc_compute_stock = sample_prc_compute_stock(compute_params, agreement_year)

    # Generate years array
    years = []
    t = agreement_year
    while t <= end_year:
        years.append(t)
        t += dt

    # Initialize black project
    project = initialize_black_project(
        project_id="prc_black_project",
        ai_slowdown_start_year=agreement_year,
        prc_compute_stock=prc_compute_stock,
        params=black_project_params,
        compute_growth_params=compute_params,
        energy_consumption_params=energy_params,
        sampled_values=None,  # Let it sample internally
        simulation_years=years,
    )

    # Initialize tracking arrays
    operational_compute = []
    total_compute = []
    datacenter_capacity = []
    survival_rate = []
    cumulative_lr = []
    h100_years = []
    fab_is_operational = []
    fab_production = []

    # Track initial values
    initial_stock = get_compute_stock_h100e(project.compute_stock) if project.compute_stock else 0.0

    # Detection tracking
    detection_time = float('inf')
    h100e_at_detection = 0.0
    h100_years_at_detection = 0.0
    cumulative_h100_years = 0.0

    # Step through time
    prev_year = None
    for year in years:
        # Update fab operational status
        if project.fab is not None and not project.fab.is_operational:
            fab_op_year = project.fab.construction_start_year + project.fab.construction_duration
            if year >= fab_op_year:
                project.fab.is_operational = True

        # Calculate compute stock with attrition
        years_since_start = year - agreement_year
        if project.compute_stock is not None:
            # Simple exponential decay model
            hazard = project.compute_stock.initial_hazard_rate + \
                     project.compute_stock.hazard_rate_increase_per_year * years_since_start
            survival = math.exp(-hazard * years_since_start)
            surviving_initial = initial_stock * survival

            # Add fab production if operational
            fab_prod_h100e = 0.0
            if project.fab is not None and project.fab.is_operational:
                fab_prod_h100e = get_fab_annual_production_h100e(project.fab)

            # Cumulative fab production
            if prev_year is not None:
                cumulative_fab = sum(fab_production) * dt * 12  # Convert to annual
            else:
                cumulative_fab = 0.0

            current_stock = surviving_initial + cumulative_fab
        else:
            current_stock = 0.0
            survival = 0.0
            fab_prod_h100e = 0.0

        # Calculate datacenter capacity (linear growth)
        if project.datacenters is not None:
            dc = project.datacenters
            years_since_dc_start = year - dc.construction_start_year
            if years_since_dc_start > 0:
                concealed = min(
                    dc.construction_rate_gw_per_year * years_since_dc_start,
                    dc.max_total_capacity_gw - dc.unconcealed_capacity_gw
                )
            else:
                concealed = 0.0
            dc_capacity = concealed + dc.unconcealed_capacity_gw
        else:
            dc_capacity = 0.0

        # Calculate operational compute (limited by datacenter)
        energy_efficiency = project.compute_stock.energy_efficiency_relative_to_h100 if project.compute_stock else 0.2
        h100e_per_gw = 1e9 / (H100_POWER_W / energy_efficiency)
        max_compute = dc_capacity * h100e_per_gw
        op_compute = min(current_stock, max_compute)

        # Calculate H100-years
        if prev_year is not None:
            dt_actual = year - prev_year
            cumulative_h100_years += op_compute * dt_actual

        # Calculate cumulative LR (simplified)
        # Use the pre-computed lr_by_year if available
        relative_year = int(years_since_start)
        if project.lr_by_year and relative_year in project.lr_by_year:
            lr = project.lr_by_year[relative_year]
        else:
            lr = 1.0

        # Check detection
        if project.sampled_detection_time <= years_since_start and detection_time == float('inf'):
            detection_time = years_since_start
            h100e_at_detection = op_compute
            h100_years_at_detection = cumulative_h100_years

        # Store values
        operational_compute.append(op_compute)
        total_compute.append(current_stock)
        datacenter_capacity.append(dc_capacity)
        survival_rate.append(survival if initial_stock > 0 else 0.0)
        cumulative_lr.append(lr)
        h100_years.append(cumulative_h100_years)
        fab_is_operational.append(project.fab.is_operational if project.fab else False)
        fab_production.append(fab_prod_h100e)

        prev_year = year

    return BlackProjectSimResult(
        years=years,
        operational_compute=operational_compute,
        total_compute=total_compute,
        datacenter_capacity=datacenter_capacity,
        survival_rate=survival_rate,
        cumulative_lr=cumulative_lr,
        h100_years=h100_years,
        detection_time=detection_time,
        h100e_at_detection=h100e_at_detection,
        h100_years_at_detection=h100_years_at_detection,
        fab_built=project.fab is not None,
        fab_is_operational=fab_is_operational,
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

            # Get black project parameters from sampled params
            black_project_params = sim_params.black_project
            if black_project_params is None:
                logger.warning(f"Simulation {i}: No black project parameters, skipping")
                continue

            compute_params = sim_params.compute_growth
            energy_params = sim_params.energy_consumption

            result = run_single_simulation(
                agreement_year=agreement_year,
                end_year=end_year,
                black_project_params=black_project_params,
                compute_params=compute_params,
                energy_params=energy_params,
                dt=0.5,  # 6-month time steps
            )
            simulation_results.append(result)

        except Exception as e:
            logger.warning(f"Simulation {i} failed: {e}")
            continue

    elapsed = time.perf_counter() - start_time
    logger.info(f"[black-project] Completed {len(simulation_results)} simulations in {elapsed:.2f}s")

    return simulation_results


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
        Dictionary containing all plot data for the black project page
    """
    if not simulation_results:
        return {"error": "No simulation results"}

    # Get common parameters
    years = simulation_results[0].years
    years_array = np.array(years)
    num_sims = len(simulation_results)

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
        arr = np.array([v for v in values if v < float('inf')])
        if len(arr) == 0:
            return []
        sorted_vals = np.sort(arr)
        n = len(sorted_vals)
        ccdf_points = []
        for i, val in enumerate(sorted_vals):
            ccdf_points.append({
                "x": float(val),
                "y": float((n - i) / n),
            })
        return ccdf_points

    # Filter for simulations with fab
    n_with_fab = sum(all_fab_built)

    # Calculate median detection time and H100-years
    valid_detection_times = [t for t in all_detection_times if t < float('inf')]
    median_detection_time = float(np.median(valid_detection_times)) if valid_detection_times else 0.0
    valid_h100_years = [h for h in all_h100_years_at_detection if h > 0]
    median_h100_years = float(np.median(valid_h100_years)) if valid_h100_years else 0.0

    # Build results in format compatible with frontend
    results = {
        "num_simulations": num_sims,
        "prob_fab_built": n_with_fab / num_sims if num_sims > 0 else 0.0,
        "p_project_exists": 0.5,
        "researcher_headcount": 500,

        "black_project_model": {
            "years": years_array.tolist(),
            # Match frontend expected field name: total_dark_compute
            "total_dark_compute": fmt_percentiles(all_total_compute),
            "operational_compute": fmt_percentiles(all_operational_compute),
            "total_black_project": fmt_percentiles(all_total_compute),
            "survival_rate": fmt_percentiles(all_survival_rate),
            "cumulative_lr": fmt_percentiles(all_cumulative_lr),
            "h100_years": fmt_percentiles(all_h100_years),
            # Frontend expects h100_years_before_detection with specific structure
            "h100_years_before_detection": {
                "median": median_h100_years,
                "p25": float(np.percentile(valid_h100_years, 25)) if valid_h100_years else 0.0,
                "p75": float(np.percentile(valid_h100_years, 75)) if valid_h100_years else 0.0,
                "individual": all_h100_years_at_detection,
                "ccdf": calculate_ccdf(all_h100_years_at_detection),
            },
            "time_to_detection": {
                "median": median_detection_time,
                "ccdf": calculate_ccdf(valid_detection_times),
            },
            "h100_years_ccdf": {
                1: calculate_ccdf(all_h100_years_at_detection),
            },
            "time_to_detection_ccdf": {
                1: calculate_ccdf(valid_detection_times),
            },
            "posterior_prob_project": fmt_percentiles([[1.0 - (1.0 / (1.0 + lr)) for lr in row] for row in all_cumulative_lr]) if all_cumulative_lr else {"p25": [], "median": [], "p75": []},
            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,
            "individual_project_h100e_before_detection": all_h100e_at_detection,
            "individual_project_time_before_detection": all_detection_times,
            "individual_project_h100_years_before_detection": all_h100_years_at_detection,
        },

        "black_fab": {
            "years": years_array.tolist(),
            "is_operational": {
                "proportion": np.mean(all_fab_is_operational, axis=0).tolist() if all_fab_is_operational else [],
            },
            "fab_built": all_fab_built,
        },

        "black_datacenters": {
            "years": years_array.tolist(),
            "datacenter_capacity": fmt_percentiles(all_datacenter_capacity),
            "capacity_ccdfs": {
                "1": calculate_ccdf([dc[-1] for dc in all_datacenter_capacity]) if all_datacenter_capacity else [],
            },
        },

        "initial_stock": {
            "years": years_array.tolist(),
            "lr_prc_accounting_samples": [1.0] * num_sims,  # Placeholder
        },
    }

    return results
