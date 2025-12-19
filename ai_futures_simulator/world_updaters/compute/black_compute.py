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
from typing import Dict, Tuple, List, Optional

from classes.world.assets import BlackFabs, BlackDatacenters, BlackCompute


# =============================================================================
# BLACK FABS UTILITIES
# =============================================================================

def get_fab_operational_year(fab: BlackFabs) -> float:
    """Year when fab becomes operational."""
    return fab.construction_start_year + fab.construction_duration


def get_fab_monthly_production_h100e(fab: BlackFabs) -> float:
    """Monthly production rate in H100e when operational."""
    if not fab.is_operational:
        return 0.0
    return fab.wafer_starts_per_month * fab.chips_per_wafer * fab.h100e_per_chip


def get_fab_annual_production_h100e(fab: BlackFabs) -> float:
    """Annual production rate in H100e when operational."""
    return get_fab_monthly_production_h100e(fab) * 12.0


# =============================================================================
# BLACK DATACENTERS UTILITIES
# =============================================================================

def get_datacenter_concealed_capacity_gw(dc: BlackDatacenters) -> float:
    """Get concealed capacity in GW."""
    return float(torch.exp(dc.log_concealed_capacity_gw).item())


def get_datacenter_total_capacity_gw(dc: BlackDatacenters) -> float:
    """Total covert datacenter capacity (concealed + unconcealed)."""
    return get_datacenter_concealed_capacity_gw(dc) + dc.unconcealed_capacity_gw


def get_datacenter_operating_labor(dc: BlackDatacenters) -> float:
    """Operating labor required for current capacity."""
    return get_datacenter_total_capacity_gw(dc) * dc.operating_labor_per_gw


# =============================================================================
# BLACK COMPUTE UTILITIES
# =============================================================================

def get_compute_stock_h100e(compute: BlackCompute) -> float:
    """Get current compute stock in H100e TPP."""
    return float(torch.exp(compute.log_compute_stock).item())


def get_current_hazard_rate(compute: BlackCompute) -> float:
    """Calculate current hazard rate based on average age."""
    return compute.initial_hazard_rate + compute.hazard_rate_increase_per_year * compute.average_age_years


# =============================================================================
# BLACK PROJECT UTILITIES
# =============================================================================

def get_black_project_total_labor(project) -> int:
    """
    Total labor involved in the black project.

    Args:
        project: AIBlackProject instance

    Returns:
        Total labor count
    """
    labor = project.researcher_headcount
    labor += project.human_datacenter_construction_labor
    if project.datacenters is not None:
        labor += int(get_datacenter_operating_labor(project.datacenters))
    if project.fab is not None and project.fab.is_operational:
        labor += project.human_fab_operating_labor
    else:
        labor += project.human_fab_construction_labor
    return labor


def get_black_project_operational_compute(project) -> float:
    """
    Compute that can actually operate given datacenter capacity.

    Limited by min(compute_stock, datacenter_capacity / energy_per_h100e).

    Args:
        project: AIBlackProject instance

    Returns:
        Operational compute in H100e
    """
    if project.compute_stock is None or project.datacenters is None:
        return 0.0

    stock = get_compute_stock_h100e(project.compute_stock)
    capacity_gw = get_datacenter_total_capacity_gw(project.datacenters)

    # Energy required per H100e (in GW)
    # H100 is ~700W, so H100e TPP needs power proportional to efficiency
    h100_power_gw = 700e-9  # 700W in GW
    power_per_h100e_gw = h100_power_gw / project.compute_stock.energy_efficiency_relative_to_h100

    # Maximum compute that can be powered
    max_powered_compute = capacity_gw / power_per_h100e_gw

    return min(stock, max_powered_compute)


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

    Args:
        mean_detection_time_100_workers: Mean years to detection with 100 workers
        mean_detection_time_1000_workers: Mean years to detection with 1000 workers

    Returns:
        Tuple (A, B) detection constants
    """
    x1, mu1 = 100, mean_detection_time_100_workers
    x2, mu2 = 1000, mean_detection_time_1000_workers

    B = np.log(mu1 / mu2) / np.log(np.log10(x2) / np.log10(x1))
    A = mu1 * (np.log10(x1) ** B)

    return A, B


def compute_mean_detection_time(labor: int, A: float, B: float) -> float:
    """
    Compute mean detection time for a given labor level.

    Args:
        labor: Number of workers
        A: Detection constant A
        B: Detection constant B

    Returns:
        Mean detection time in years
    """
    if labor <= 1:
        return float('inf')
    return A / (np.log10(labor) ** B)


def sample_detection_time(
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float
) -> float:
    """
    Sample a detection time from the Gamma distribution for a given labor level.

    This is used at initialization to sample when detection will occur.
    For variable labor over time, use sample_detection_time_composite instead.

    Args:
        labor: Number of workers at project start
        mean_detection_time_100_workers: Mean years to detection with 100 workers
        mean_detection_time_1000_workers: Mean years to detection with 1000 workers
        variance_theta: Variance (scale) parameter of the Gamma distribution

    Returns:
        Sampled detection time in years since project start
    """
    if labor <= 1:
        return float('inf')

    A, B = compute_detection_constants(
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers
    )

    mu = compute_mean_detection_time(labor, A, B)
    k = mu / variance_theta  # shape parameter

    # Sample from Gamma distribution
    return float(np.random.gamma(k, variance_theta))


def compute_lr_other(
    years_since_start: float,
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float,
    is_detected: bool = False
) -> float:
    """
    Compute the "other intelligence" likelihood ratio based on time and labor.

    Uses the Gamma survival function to compute probability of not being detected
    given the project exists.

    LR = P(not detected by year t | project exists) / P(not detected | no project)
       = Gamma.sf(t, k, theta) / 1.0

    Args:
        years_since_start: Years since project construction started
        labor: Current number of workers
        mean_detection_time_100_workers: Mean years to detection with 100 workers
        mean_detection_time_1000_workers: Mean years to detection with 1000 workers
        variance_theta: Variance (scale) parameter of the Gamma distribution
        is_detected: Whether the project has been detected (triggers LR = 100)

    Returns:
        Likelihood ratio for "other intelligence" sources
    """
    # If detected, return high LR (strong evidence project exists)
    if is_detected:
        return 100.0

    if labor <= 1 or years_since_start <= 0:
        return 1.0

    A, B = compute_detection_constants(
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers
    )

    mu = compute_mean_detection_time(labor, A, B)
    k = mu / variance_theta  # shape parameter

    # Survival function: P(T > t) where T ~ Gamma(k, theta)
    p_not_detected = stats.gamma.sf(years_since_start, a=k, scale=variance_theta)

    # Floor at 0.001 to avoid numerical issues
    return max(p_not_detected, 0.001)


# =============================================================================
# US ESTIMATE AND LIKELIHOOD RATIO UTILITIES
# =============================================================================

def sample_us_estimate_with_error(true_quantity: float, median_error: float) -> float:
    """
    Sample a US intelligence estimate of a quantity with exponential error distribution.

    Uses an exponential distribution for absolute relative error, with the median
    calibrated to the given median_error parameter.

    Args:
        true_quantity: The true value of the quantity being estimated
        median_error: The median absolute relative error in US estimates

    Returns:
        US estimate of the quantity
    """
    if true_quantity <= 0:
        return 0.0

    # Calculate rate parameter k such that P(|error| <= median_error) = 0.5
    k = -np.log(0.5) / median_error

    u = np.random.uniform(0, 1)

    # Invert the CDF: x = -ln(1 - u) / k
    relative_error = -np.log(1 - u) / k

    # Randomly choose direction of error (overestimate or underestimate)
    error_sign = 1 if np.random.random() > 0.5 else -1
    relative_error = error_sign * relative_error

    # Apply error to actual quantity
    us_estimate = true_quantity * (1 + relative_error)

    return max(0, us_estimate)


def lr_from_discrepancy_in_us_estimate(
    true_if_project_exists: float,
    true_if_no_project: float,
    us_estimate: float,
    median_error: float
) -> float:
    """
    Calculate likelihood ratio from discrepancy between US estimate and reported quantity.

    The likelihood ratio is: LR = P(us_estimate | project exists) / P(us_estimate | no project)

    Uses exponential distribution for absolute relative error: pdf(error) = k * e^(-k * |error|)

    Args:
        true_if_project_exists: True value of the quantity if project exists
        true_if_no_project: True value of the quantity if no project exists (reported value)
        us_estimate: The US intelligence estimate (sampled with error)
        median_error: The median absolute relative error in US estimates

    Returns:
        Likelihood ratio (LR > 1 means evidence for project, LR < 1 means evidence against)
    """
    if true_if_project_exists < 1e-10 or true_if_no_project < 1e-10:
        return 1.0  # Neutral evidence if quantities are zero

    # Calculate absolute relative errors under each hypothesis
    error_if_project_exists = abs(us_estimate - true_if_project_exists) / true_if_project_exists
    error_if_no_project = abs(us_estimate - true_if_no_project) / true_if_no_project

    # PDF of absolute error (exponential distribution)
    k = -np.log(0.5) / median_error
    p_if_project_exists = k * np.exp(-k * error_if_project_exists)
    p_if_no_project = k * np.exp(-k * error_if_no_project)

    if p_if_no_project > 0:
        lr = p_if_project_exists / p_if_no_project
    else:
        lr = 1e6  # Very large if no-project scenario makes estimate very unlikely

    return lr


# =============================================================================
# DATACENTER LIKELIHOOD RATIO FUNCTIONS
# =============================================================================

def compute_lr_from_reported_energy_consumption(
    total_prc_energy_gw: float,
    covert_energy_gw: float,
    us_estimate_energy: float,
    median_error: float
) -> float:
    """
    Calculate likelihood ratio from energy consumption accounting.

    The PRC reports total energy consumption excluding covert datacenters.
    The US estimates this and compares to reported values.

    Args:
        total_prc_energy_gw: Total PRC energy consumption in GW
        covert_energy_gw: Energy consumed by covert datacenters in GW
        us_estimate_energy: US estimate of total energy consumption
        median_error: Median error in US energy estimates

    Returns:
        Likelihood ratio from energy consumption evidence
    """
    if covert_energy_gw < 1e-10 or total_prc_energy_gw < 1e-10:
        return 1.0

    # If project exists: true energy = total_prc_energy (but PRC reports less)
    # If no project: true energy = reported energy = total - covert
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
    """
    Calculate likelihood ratio from satellite detection of diverted datacenter capacity.

    The US uses satellite imagery to estimate unconcealed datacenter capacity.
    If some capacity was diverted to the black project, there's a discrepancy.

    Args:
        diverted_capacity_gw: Capacity diverted from unconcealed to black project
        total_unconcealed_capacity_gw: Total unconcealed capacity (including diverted)
        us_estimate_capacity: US satellite estimate of unconcealed capacity
        median_error: Median error in US satellite estimates

    Returns:
        Likelihood ratio from satellite detection evidence
    """
    if diverted_capacity_gw < 1e-10:
        return 1.0

    # If project exists: PRC reports (total - diverted) but true capacity is total
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
    """
    Calculate likelihood ratio from PRC compute stock accounting.

    If compute was diverted, the true PRC stock is higher than reported.

    Args:
        reported_compute_stock: PRC's reported compute stock (after diversion)
        diversion_proportion: Proportion of initial stock diverted to black project
        us_estimate_compute: US estimate of PRC compute stock
        median_error: Median error in US compute estimates

    Returns:
        Likelihood ratio from compute accounting evidence
    """
    if reported_compute_stock < 1e-10 or diversion_proportion <= 0:
        return 1.0

    # If project exists: true stock = reported / (1 - diversion_proportion)
    true_stock_if_exists = reported_compute_stock / (1 - diversion_proportion)

    return lr_from_discrepancy_in_us_estimate(
        true_if_project_exists=true_stock_if_exists,
        true_if_no_project=reported_compute_stock,
        us_estimate=us_estimate_compute,
        median_error=median_error
    )


# =============================================================================
# COMPOSITE DETECTION DISTRIBUTION (matches discrete model)
# =============================================================================

def build_composite_detection_distribution(
    labor_by_year: Dict[float, int],
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float
) -> Tuple[List[float], List[Tuple], float, float, float]:
    """
    Build composite distribution for detection time accounting for variable labor.

    This is the expensive calculation that should be done once and reused.

    Args:
        labor_by_year: Dict mapping relative years to total number of workers
        mean_detection_time_100_workers: Mean detection time for 100 workers
        mean_detection_time_1000_workers: Mean detection time for 1000 workers
        variance_theta: Variance parameter of detection time given num workers

    Returns:
        Tuple of (sorted_years, prob_ranges, A, B, variance_theta)
    """
    if not labor_by_year:
        return ([], [], None, None, variance_theta)

    # Sort years chronologically
    sorted_years = sorted(labor_by_year.keys())

    # Compute detection time constants
    A, B = compute_detection_constants(
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers
    )

    # Build composite distribution by calculating cumulative probabilities
    cumulative_prob = 0.0
    prob_ranges = []

    for i, year in enumerate(sorted_years):
        labor = labor_by_year[year]

        if labor <= 0:
            continue

        # Calculate gamma parameters for this labor level
        mu = A / (np.log10(labor) ** B)
        k = mu / variance_theta

        # Determine the time range for this period
        year_start = year
        year_end = sorted_years[i + 1] if i + 1 < len(sorted_years) else year + 100

        # Calculate probability mass in this interval using CDF differences
        cdf_start = stats.gamma.cdf(year_start, a=k, scale=variance_theta)
        cdf_end = stats.gamma.cdf(year_end, a=k, scale=variance_theta)
        prob_mass = cdf_end - cdf_start

        cum_prob_end = cumulative_prob + prob_mass
        prob_ranges.append((year_start, year_end, labor, k, variance_theta, cumulative_prob, cum_prob_end))
        cumulative_prob = cum_prob_end

    return (sorted_years, prob_ranges, A, B, variance_theta)


def sample_detection_time_from_composite(prob_ranges: List[Tuple]) -> float:
    """
    Sample a detection time from the precomputed composite distribution.

    Args:
        prob_ranges: Precomputed probability ranges from build_composite_detection_distribution

    Returns:
        Sampled detection time in years
    """
    if not prob_ranges:
        return float('inf')

    u = np.random.uniform(0, 1)

    # Find which range the sample falls into
    time_of_detection = float('inf')
    for year_start, year_end, _, k, theta, cum_start, cum_end in prob_ranges:
        if cum_start <= u < cum_end:
            # Map u within this range back to the gamma distribution
            if cum_end > cum_start:
                u_normalized = (u - cum_start) / (cum_end - cum_start)
                cdf_start = stats.gamma.cdf(year_start, a=k, scale=theta)
                cdf_target = cdf_start + u_normalized * (stats.gamma.cdf(year_end, a=k, scale=theta) - cdf_start)
                time_of_detection = stats.gamma.ppf(cdf_target, a=k, scale=theta)
            else:
                time_of_detection = year_start
            break

    return time_of_detection


def compute_lr_over_time_vs_num_workers(
    labor_by_year: Dict[float, int],
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float
) -> Tuple[Dict[float, float], float]:
    """
    Calculate likelihood ratios over time accounting for variable labor.

    This mirrors the discrete model's lr_over_time_vs_num_workers function.

    Args:
        labor_by_year: Dict mapping relative years to total number of workers
        mean_detection_time_100_workers: Mean detection time for 100 workers
        mean_detection_time_1000_workers: Mean detection time for 1000 workers
        variance_theta: Variance parameter of detection time given num workers

    Returns:
        Tuple of (lr_by_year dict, sampled_detection_time)
    """
    if not labor_by_year:
        return {}, float('inf')

    # Build composite distribution
    sorted_years, prob_ranges, A, B, _ = build_composite_detection_distribution(
        labor_by_year=labor_by_year,
        mean_detection_time_100_workers=mean_detection_time_100_workers,
        mean_detection_time_1000_workers=mean_detection_time_1000_workers,
        variance_theta=variance_theta
    )

    # Sample detection time from the composite distribution
    time_of_detection = sample_detection_time_from_composite(prob_ranges)

    # Calculate likelihood ratios for each year
    lr_by_year = {}
    for year in sorted_years:
        if year >= time_of_detection:
            # Detection has occurred by this year
            lr_by_year[year] = 100.0
        else:
            labor = labor_by_year[year]
            if labor <= 0:
                lr_by_year[year] = 1.0
                continue

            # Calculate survival probability using gamma distribution for current labor level
            mu = A / (np.log10(labor) ** B)
            k = mu / variance_theta

            p_not_detected_given_project = stats.gamma.sf(year, a=k, scale=variance_theta)

            # LR = P(evidence | project exists) / P(evidence | no project)
            lr = p_not_detected_given_project / 1.0

            lr_by_year[year] = max(lr, 0.001)  # Floor at 0.001

    return lr_by_year, time_of_detection


# =============================================================================
# CUMULATIVE LIKELIHOOD RATIO
# =============================================================================

def compute_cumulative_likelihood_ratio(
    project,
    current_time: float,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float
) -> float:
    """
    Compute the cumulative detection likelihood ratio for a black project.

    Combines three independent intelligence sources:
    1. LR_inventory: Evidence from missing domestic equipment (constant)
    2. LR_procurement: Evidence from foreign equipment purchases (constant)
    3. LR_other: Evidence from HUMINT/SIGINT/IMINT based on workers and time

    cumulative_LR = LR_inventory × LR_procurement × LR_other

    Args:
        project: AIBlackProject instance
        current_time: Current simulation year
        mean_detection_time_100_workers: Mean years to detection with 100 workers
        mean_detection_time_1000_workers: Mean years to detection with 1000 workers
        variance_theta: Variance parameter for detection time

    Returns:
        Cumulative likelihood ratio
    """
    # Get constant LRs from fab (if it exists)
    lr_inventory = 1.0
    lr_procurement = 1.0
    if project.fab is not None:
        lr_inventory = project.fab.lr_inventory
        lr_procurement = project.fab.lr_procurement

    # Compute time-varying LR_other
    years_since_start = current_time - project.ai_slowdown_start_year

    # Get current labor
    labor = get_black_project_total_labor(project)

    # Check if detected
    is_detected = getattr(project, 'is_detected', False)

    lr_other = compute_lr_other(
        years_since_start=years_since_start,
        labor=labor,
        mean_detection_time_100_workers=mean_detection_time_100_workers,
        mean_detection_time_1000_workers=mean_detection_time_1000_workers,
        variance_theta=variance_theta,
        is_detected=is_detected
    )

    return lr_inventory * lr_procurement * lr_other


def compute_detection_probability(
    years_since_start: float,
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float
) -> float:
    """
    Compute the probability of detection by a given time.

    This is a smooth, differentiable function that can be used for
    backpropagation optimization of project parameters.

    P(detected by time t) = Gamma.cdf(t, k, θ)

    Args:
        years_since_start: Years since project construction started
        labor: Current number of workers
        mean_detection_time_100_workers: Mean years to detection with 100 workers
        mean_detection_time_1000_workers: Mean years to detection with 1000 workers
        variance_theta: Variance (scale) parameter of the Gamma distribution

    Returns:
        Probability of detection by the given time (0 to 1)
    """
    if labor <= 1 or years_since_start <= 0:
        return 0.0

    A, B = compute_detection_constants(
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers
    )

    mu = compute_mean_detection_time(labor, A, B)
    k = mu / variance_theta  # shape parameter

    # CDF: P(T <= t) where T ~ Gamma(k, theta)
    return float(stats.gamma.cdf(years_since_start, a=k, scale=variance_theta))
