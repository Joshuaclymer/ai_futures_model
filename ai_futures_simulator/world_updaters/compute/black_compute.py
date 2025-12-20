"""
Black project compute utility functions.

These functions compute derived values from black project assets.
All logic is here rather than in the dataclass definitions to keep
the classes/ directory free of methods.

The detection and likelihood ratio logic mirrors the discrete model
in black_project_backend for consistency.
"""

import math
import numpy as np
from scipy import stats
import torch
from torch import Tensor
from typing import Dict, Tuple, List, Optional, TYPE_CHECKING

from classes.world.assets import Compute, Fabs, Datacenters

if TYPE_CHECKING:
    from classes.world.entities import AIBlackProject
    from parameters.compute_parameters import PRCComputeParameters, ExogenousComputeTrends


# =============================================================================
# FAB METRIC CALCULATIONS
# =============================================================================

def calculate_fab_construction_duration(
    fab_construction_labor: float,
    target_wafer_starts_per_month: float,
    prc_compute_params: "PRCComputeParameters" = None,
    construction_time_for_5k_wafers: float = None,
    construction_time_for_100k_wafers: float = None,
) -> float:
    """
    Calculate fab construction duration based on labor and target capacity.

    Uses log-linear interpolation between reference points.
    """
    # Get construction times from params or use defaults
    if prc_compute_params is not None:
        construction_time_for_5k_wafers = prc_compute_params.construction_time_for_5k_wafers_per_month
        construction_time_for_100k_wafers = prc_compute_params.construction_time_for_100k_wafers_per_month
    else:
        construction_time_for_5k_wafers = construction_time_for_5k_wafers or 1.4
        construction_time_for_100k_wafers = construction_time_for_100k_wafers or 2.41

    if target_wafer_starts_per_month <= 0:
        return 2.0  # Default

    # Log-linear interpolation
    log_5k = math.log(5000)
    log_100k = math.log(100000)
    log_target = math.log(target_wafer_starts_per_month)

    # Linear interpolation in log space
    t = (log_target - log_5k) / (log_100k - log_5k) if log_100k != log_5k else 0.5
    t = max(0, min(1, t))  # Clamp

    return construction_time_for_5k_wafers + t * (construction_time_for_100k_wafers - construction_time_for_5k_wafers)


def calculate_fab_wafer_starts_per_month(
    fab_operating_labor: float,
    fab_number_of_lithography_scanners: float,
    wafers_per_month_per_worker: float = 24.64,
    wafers_per_month_per_scanner: float = 1000.0,
) -> float:
    """Calculate wafer starts per month based on labor and scanner constraints."""
    from_labor = fab_operating_labor * wafers_per_month_per_worker
    from_scanners = fab_number_of_lithography_scanners * wafers_per_month_per_scanner
    return min(from_labor, from_scanners)


def calculate_fab_h100e_per_chip(
    fab_process_node_nm: float,
    year: float,
    exogenous_trends: "ExogenousComputeTrends" = None,
    h100_reference_nm: float = 4.0,
    transistor_density_scaling_exponent: float = None,
    architecture_efficiency_improvement_per_year: float = None,
    h100_release_year: float = 2022.0,
) -> float:
    """Calculate H100-equivalent per chip based on process node and architecture improvements."""
    # Get params from exogenous_trends or use defaults
    if exogenous_trends is not None:
        transistor_density_scaling_exponent = exogenous_trends.transistor_density_scaling_exponent
        architecture_efficiency_improvement_per_year = exogenous_trends.state_of_the_art_architecture_efficiency_improvement_per_year
    else:
        transistor_density_scaling_exponent = transistor_density_scaling_exponent or 1.49
        architecture_efficiency_improvement_per_year = architecture_efficiency_improvement_per_year or 1.23

    # Density ratio relative to H100
    density_ratio = (h100_reference_nm / fab_process_node_nm) ** transistor_density_scaling_exponent

    # Architecture efficiency improvement since H100 release
    years_since_h100 = year - h100_release_year
    arch_efficiency = architecture_efficiency_improvement_per_year ** years_since_h100

    return density_ratio * arch_efficiency


def calculate_fab_watts_per_chip(
    fab_process_node_nm: float,
    exogenous_trends: "ExogenousComputeTrends" = None,
    h100_reference_nm: float = 4.0,
    h100_watts: float = 700.0,
    transistor_density_at_end_of_dennard: float = None,
    watts_per_tpp_exponent_before_dennard: float = None,
    watts_per_tpp_exponent_after_dennard: float = None,
) -> float:
    """Calculate watts per chip based on process node and Dennard scaling."""
    # Get params from exogenous_trends or use defaults
    if exogenous_trends is not None:
        transistor_density_at_end_of_dennard = exogenous_trends.transistor_density_at_end_of_dennard_scaling_m_per_mm2
        watts_per_tpp_exponent_before_dennard = exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended
        watts_per_tpp_exponent_after_dennard = exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended
    else:
        transistor_density_at_end_of_dennard = transistor_density_at_end_of_dennard or 10.0
        watts_per_tpp_exponent_before_dennard = watts_per_tpp_exponent_before_dennard or -1.0
        watts_per_tpp_exponent_after_dennard = watts_per_tpp_exponent_after_dennard or -0.33

    # Calculate transistor density for this node
    # Using inverse relationship: smaller nm = higher density
    density = (h100_reference_nm / fab_process_node_nm) ** 2  # Rough approximation

    if density < transistor_density_at_end_of_dennard:
        # Before Dennard scaling ended
        exponent = watts_per_tpp_exponent_before_dennard
    else:
        # After Dennard scaling ended
        exponent = watts_per_tpp_exponent_after_dennard

    # Scale watts relative to H100
    watts_scaling = (fab_process_node_nm / h100_reference_nm) ** abs(exponent)
    return h100_watts * watts_scaling


def calculate_fab_annual_production_h100e(
    fab_wafer_starts_per_month: float,
    fab_chips_per_wafer: float,
    fab_h100e_per_chip: float,
    fab_is_operational: bool,
) -> float:
    """Calculate annual production in H100e when operational."""
    if not fab_is_operational:
        return 0.0
    return fab_wafer_starts_per_month * fab_chips_per_wafer * fab_h100e_per_chip * 12.0


# =============================================================================
# DATACENTER METRIC CALCULATIONS
# =============================================================================

def calculate_datacenter_capacity_gw(
    unconcealed_capacity_gw: float,
    concealed_capacity_gw: float,
) -> float:
    """Calculate total datacenter capacity."""
    return unconcealed_capacity_gw + concealed_capacity_gw


def calculate_concealed_capacity_gw(
    current_year: float,
    construction_start_year: float,
    construction_rate_gw_per_year: float,
    max_concealed_capacity_gw: float,
) -> float:
    """Calculate concealed datacenter capacity at a given time (linear growth model)."""
    years_since_start = current_year - construction_start_year
    if years_since_start <= 0:
        return 0.0

    return min(
        construction_rate_gw_per_year * years_since_start,
        max_concealed_capacity_gw
    )


def calculate_datacenter_operating_labor(
    datacenter_capacity_gw: float,
    operating_labor_per_gw: float,
) -> float:
    """Calculate operating labor required for datacenter capacity."""
    return datacenter_capacity_gw * operating_labor_per_gw


# =============================================================================
# COMPUTE METRIC CALCULATIONS
# =============================================================================

def calculate_survival_rate(
    years_since_acquisition: float,
    initial_hazard_rate: float,
    hazard_rate_increase_per_year: float,
) -> float:
    """Calculate chip survival rate using Weibull-like hazard model."""
    if years_since_acquisition <= 0:
        return 1.0

    # Cumulative hazard: integral of h(t) = h0 + h1*t
    # H(t) = h0*t + h1*t^2/2
    cumulative_hazard = (
        initial_hazard_rate * years_since_acquisition +
        hazard_rate_increase_per_year * years_since_acquisition ** 2 / 2
    )

    return math.exp(-cumulative_hazard)


def calculate_functional_compute(
    all_compute_h100e: float,
    survival_rate: float,
) -> float:
    """Calculate functional compute after attrition."""
    return all_compute_h100e * survival_rate


def calculate_operating_compute(
    functional_compute_h100e: float,
    datacenter_capacity_gw: float,
    watts_per_h100e: float,
) -> float:
    """Calculate operating compute limited by datacenter capacity."""
    if datacenter_capacity_gw <= 0 or watts_per_h100e <= 0:
        return 0.0

    # Max compute that can be powered
    max_powered = datacenter_capacity_gw * 1e9 / watts_per_h100e

    return min(functional_compute_h100e, max_powered)


# =============================================================================
# LABOR UTILITIES
# =============================================================================

def get_black_project_total_labor(project: "AIBlackProject") -> int:
    """
    Total labor involved in the black project.

    Args:
        project: AIBlackProject instance

    Returns:
        Total labor count
    """
    labor = int(project.human_ai_capability_researchers)
    labor += int(project.concealed_datacenter_capacity_construction_labor)

    # Operating labor depends on datacenter capacity
    if project.datacenters is not None:
        operating_labor = calculate_datacenter_operating_labor(
            project.datacenters.data_center_capacity_gw,
            project.datacenters_operating_labor_per_gw,
        )
        labor += int(operating_labor)

    # Fab labor
    if project.fab_is_operational:
        labor += int(project.fab_operating_labor)
    else:
        labor += int(project.fab_construction_labor)

    return labor


# =============================================================================
# DETECTION UTILITIES
# =============================================================================

def compute_detection_constants(
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float
) -> tuple:
    """
    Compute detection time model constants A and B.

    The mean detection time follows: mu(workers) = A / log10(workers)^B
    """
    x1, mu1 = 100, mean_detection_time_100_workers
    x2, mu2 = 1000, mean_detection_time_1000_workers

    B = np.log(mu1 / mu2) / np.log(np.log10(x2) / np.log10(x1))
    A = mu1 * (np.log10(x1) ** B)

    return A, B


def compute_mean_detection_time(labor: int, A: float, B: float) -> float:
    """Compute mean detection time for a given labor level."""
    if labor <= 1:
        return float('inf')
    return A / (np.log10(labor) ** B)


def sample_detection_time(
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float
) -> float:
    """Sample a detection time from the Gamma distribution for a given labor level."""
    if labor <= 1:
        return float('inf')

    A, B = compute_detection_constants(
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers
    )

    mu = compute_mean_detection_time(labor, A, B)
    k = mu / variance_theta  # shape parameter

    return float(np.random.gamma(k, variance_theta))


def sample_us_estimate_with_error(true_quantity: float, median_error: float) -> float:
    """Sample a US intelligence estimate of a quantity with exponential error distribution."""
    if true_quantity <= 0:
        return 0.0

    k = -np.log(0.5) / median_error
    u = np.random.uniform(0, 1)
    relative_error = -np.log(1 - u) / k
    error_sign = 1 if np.random.random() > 0.5 else -1
    relative_error = error_sign * relative_error
    us_estimate = true_quantity * (1 + relative_error)

    return max(0, us_estimate)


def lr_from_discrepancy_in_us_estimate(
    true_if_project_exists: float,
    true_if_no_project: float,
    us_estimate: float,
    median_error: float
) -> float:
    """Calculate likelihood ratio from discrepancy between US estimate and reported quantity."""
    if true_if_project_exists < 1e-10 or true_if_no_project < 1e-10:
        return 1.0

    error_if_project_exists = abs(us_estimate - true_if_project_exists) / true_if_project_exists
    error_if_no_project = abs(us_estimate - true_if_no_project) / true_if_no_project

    k = -np.log(0.5) / median_error
    p_if_project_exists = k * np.exp(-k * error_if_project_exists)
    p_if_no_project = k * np.exp(-k * error_if_no_project)

    if p_if_no_project > 0:
        lr = p_if_project_exists / p_if_no_project
    else:
        lr = 1e6

    return lr


def compute_lr_from_reported_energy_consumption(
    total_prc_energy_gw: float,
    covert_energy_gw: float,
    us_estimate_energy: float,
    median_error: float
) -> float:
    """Calculate likelihood ratio from energy consumption accounting."""
    if covert_energy_gw < 1e-10 or total_prc_energy_gw < 1e-10:
        return 1.0

    reported_energy = total_prc_energy_gw - covert_energy_gw

    return lr_from_discrepancy_in_us_estimate(
        true_if_project_exists=total_prc_energy_gw,
        true_if_no_project=reported_energy,
        us_estimate=us_estimate_energy,
        median_error=median_error
    )


def compute_lr_from_satellite_detection(
    diverted_capacity_gw: float,
    total_unconcealed_capacity_gw: float,
    us_estimate_capacity: float,
    median_error: float
) -> float:
    """Calculate likelihood ratio from satellite detection of diverted datacenter capacity."""
    if diverted_capacity_gw < 1e-10:
        return 1.0

    reported_capacity = total_unconcealed_capacity_gw - diverted_capacity_gw

    return lr_from_discrepancy_in_us_estimate(
        true_if_project_exists=reported_capacity,
        true_if_no_project=total_unconcealed_capacity_gw,
        us_estimate=us_estimate_capacity,
        median_error=median_error
    )


def compute_lr_from_prc_compute_accounting(
    reported_compute_stock: float,
    diversion_proportion: float,
    us_estimate_compute: float,
    median_error: float
) -> float:
    """Calculate likelihood ratio from PRC compute stock accounting."""
    if reported_compute_stock < 1e-10 or diversion_proportion <= 0:
        return 1.0

    true_stock_if_exists = reported_compute_stock / (1 - diversion_proportion)

    return lr_from_discrepancy_in_us_estimate(
        true_if_project_exists=true_stock_if_exists,
        true_if_no_project=reported_compute_stock,
        us_estimate=us_estimate_compute,
        median_error=median_error
    )


def compute_lr_over_time_vs_num_workers(
    labor_by_year: Dict[float, int],
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float
) -> Tuple[Dict[float, float], float]:
    """
    Calculate likelihood ratios over time accounting for variable labor.

    Returns:
        Tuple of (lr_by_year dict, sampled_detection_time)
    """
    if not labor_by_year:
        return {}, float('inf')

    sorted_years = sorted(labor_by_year.keys())
    A, B = compute_detection_constants(
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers
    )

    # Sample detection time from composite distribution
    # For simplicity, use the average labor level
    avg_labor = sum(labor_by_year.values()) / len(labor_by_year)
    mu = A / (np.log10(max(avg_labor, 2)) ** B)
    k = mu / variance_theta
    time_of_detection = float(np.random.gamma(k, variance_theta))

    # Calculate likelihood ratios for each year
    lr_by_year = {}
    for year in sorted_years:
        if year >= time_of_detection:
            lr_by_year[year] = 100.0
        else:
            labor = labor_by_year[year]
            if labor <= 0:
                lr_by_year[year] = 1.0
                continue

            mu = A / (np.log10(labor) ** B)
            k = mu / variance_theta
            p_not_detected = stats.gamma.sf(year, a=k, scale=variance_theta)
            lr_by_year[year] = max(p_not_detected, 0.001)

    return lr_by_year, time_of_detection


def compute_detection_probability(
    years_since_start: float,
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float
) -> float:
    """Compute the probability of detection by a given time."""
    if labor <= 1 or years_since_start <= 0:
        return 0.0

    A, B = compute_detection_constants(
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers
    )

    mu = compute_mean_detection_time(labor, A, B)
    k = mu / variance_theta

    return float(stats.gamma.cdf(years_since_start, a=k, scale=variance_theta))
