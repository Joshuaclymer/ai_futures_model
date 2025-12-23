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
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

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
    construction_workers_per_1000_wafers_per_month: float = None,
    construction_time_multiplier: float = None,
) -> float:
    """
    Calculate fab construction duration based on labor and target capacity.

    Uses a fixed-proportions production function where construction time depends on:
    1. Fab capacity: Larger fabs take longer to build (log-linear relationship)
    2. Construction labor: Fewer workers than required extends construction time
    3. Uncertainty multiplier: Sampled from lognormal (median=1.0, relative_sigma=0.35)

    This matches the discrete model's estimate_construction_duration function.
    """
    # Get construction times and labor requirements from params or use defaults
    if prc_compute_params is not None:
        construction_time_for_5k_wafers = prc_compute_params.construction_time_for_5k_wafers_per_month
        construction_time_for_100k_wafers = prc_compute_params.construction_time_for_100k_wafers_per_month
        construction_time_multiplier = prc_compute_params.fab_construction_time_multiplier
        # Convert from wafers_per_worker to workers_per_1000_wafers
        # wafers_per_worker = 14.1 means we need 14.1 workers per 1000 wafers/month
        construction_workers_per_1000_wafers_per_month = 14.1  # Default from discrete model
    else:
        construction_time_for_5k_wafers = construction_time_for_5k_wafers or 1.4
        construction_time_for_100k_wafers = construction_time_for_100k_wafers or 2.41
        construction_workers_per_1000_wafers_per_month = construction_workers_per_1000_wafers_per_month or 14.1
        construction_time_multiplier = construction_time_multiplier or 1.0

    if target_wafer_starts_per_month <= 0:
        return 2.0  # Default

    # Step 1: Log-linear interpolation for base construction duration given capacity
    # Uses log10 to match discrete model formula: time = slope * log10(wafers) + intercept
    log10_5k = math.log10(5000)
    log10_100k = math.log10(100000)
    log10_target = math.log10(target_wafer_starts_per_month)

    # Calculate slope and intercept for log-linear extrapolation
    slope = (construction_time_for_100k_wafers - construction_time_for_5k_wafers) / (log10_100k - log10_5k)
    intercept = construction_time_for_5k_wafers - slope * log10_5k

    # Calculate base construction duration from capacity
    construction_duration = slope * log10_target + intercept

    # Step 2: Calculate construction labor requirement given wafer capacity
    # The discrete model uses: workers_needed = (workers_per_1000_wafers / 1000) * wafer_capacity
    construction_labor_requirement = (
        construction_workers_per_1000_wafers_per_month / 1000.0
    ) * target_wafer_starts_per_month

    # Step 3: If actual labor < required labor, extend construction duration proportionally
    if fab_construction_labor < construction_labor_requirement and fab_construction_labor > 0:
        construction_duration *= (construction_labor_requirement / fab_construction_labor)

    # Step 4: Apply uncertainty multiplier (sampled from lognormal in monte carlo mode)
    # This matches discrete model which samples from lognormal with relative_sigma=0.35
    construction_duration *= construction_time_multiplier

    return construction_duration


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


def calculate_transistor_density_from_process_node(
    fab_process_node_nm: float,
    exogenous_trends: "ExogenousComputeTrends" = None,
    h100_reference_nm: float = 4.0,
    h100_transistor_density_m_per_mm2: float = 98.28,
    transistor_density_scaling_exponent: float = None,
) -> float:
    """
    Calculate transistor density from process node.

    Uses the formula: density = H100_DENSITY * (process_node / H100_NODE)^(-exponent)

    This matches the discrete model's calculate_transistor_density_from_process_node().

    Args:
        fab_process_node_nm: Process node in nanometers
        exogenous_trends: Optional ExogenousComputeTrends for parameter values
        h100_reference_nm: H100 reference node (default 4nm)
        h100_transistor_density_m_per_mm2: H100 transistor density (default 98.28 M/mm²)
        transistor_density_scaling_exponent: Scaling exponent (default 1.49)

    Returns:
        Transistor density in M/mm²
    """
    if exogenous_trends is not None:
        transistor_density_scaling_exponent = exogenous_trends.transistor_density_scaling_exponent
    else:
        transistor_density_scaling_exponent = transistor_density_scaling_exponent or 1.49

    node_ratio = fab_process_node_nm / h100_reference_nm
    return h100_transistor_density_m_per_mm2 * (node_ratio ** (-transistor_density_scaling_exponent))


def calculate_watts_per_tpp_from_transistor_density(
    transistor_density_m_per_mm2: float,
    exogenous_trends: "ExogenousComputeTrends" = None,
    h100_transistor_density_m_per_mm2: float = 98.28,
    h100_watts_per_tpp: float = 0.326493,
    transistor_density_at_end_of_dennard: float = None,
    watts_per_tpp_exponent_before_dennard: float = None,
    watts_per_tpp_exponent_after_dennard: float = None,
) -> float:
    """
    Calculate watts per TPP from transistor density using Dennard scaling model.

    Uses a piecewise power law with different exponents before/after Dennard scaling ended:
    - Before Dennard: watts_per_tpp scales with density^exponent_before (steeper, ~-1.0)
    - After Dennard: watts_per_tpp scales with density^exponent_after (shallower, ~-0.33)

    The transition point is anchored to the H100 (post-Dennard) and the pre-Dennard
    line is connected to ensure continuity.

    This matches the discrete model's predict_watts_per_tpp_from_transistor_density().

    Args:
        transistor_density_m_per_mm2: Transistor density in M/mm²
        exogenous_trends: Optional ExogenousComputeTrends for parameter values
        h100_transistor_density_m_per_mm2: H100 transistor density (default 98.28)
        h100_watts_per_tpp: H100 watts per TPP (default 0.326493)
        transistor_density_at_end_of_dennard: Density at Dennard transition (default 10.0)
        watts_per_tpp_exponent_before_dennard: Exponent before Dennard ended (default -1.0)
        watts_per_tpp_exponent_after_dennard: Exponent after Dennard ended (default -0.33)

    Returns:
        Watts per TPP for the given transistor density
    """
    if exogenous_trends is not None:
        transistor_density_at_end_of_dennard = exogenous_trends.transistor_density_at_end_of_dennard_scaling_m_per_mm2
        watts_per_tpp_exponent_before_dennard = exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended
        watts_per_tpp_exponent_after_dennard = exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended
    else:
        transistor_density_at_end_of_dennard = transistor_density_at_end_of_dennard or 10.0
        watts_per_tpp_exponent_before_dennard = watts_per_tpp_exponent_before_dennard or -1.0
        watts_per_tpp_exponent_after_dennard = watts_per_tpp_exponent_after_dennard or -0.33

    # Calculate watts_per_tpp at the Dennard transition point using post-Dennard relationship
    transition_density_ratio = transistor_density_at_end_of_dennard / h100_transistor_density_m_per_mm2
    transition_watts_per_tpp = h100_watts_per_tpp * (transition_density_ratio ** watts_per_tpp_exponent_after_dennard)

    if transistor_density_m_per_mm2 < transistor_density_at_end_of_dennard:
        # Before Dennard scaling ended - anchor to transition point
        exponent = watts_per_tpp_exponent_before_dennard
        density_ratio = transistor_density_m_per_mm2 / transistor_density_at_end_of_dennard
        return transition_watts_per_tpp * (density_ratio ** exponent)
    else:
        # After Dennard scaling ended - anchor to H100
        exponent = watts_per_tpp_exponent_after_dennard
        density_ratio = transistor_density_m_per_mm2 / h100_transistor_density_m_per_mm2
        return h100_watts_per_tpp * (density_ratio ** exponent)


def calculate_fab_watts_per_chip(
    fab_process_node_nm: float,
    year: float,
    exogenous_trends: "ExogenousComputeTrends" = None,
    h100_reference_nm: float = 4.0,
    h100_release_year: float = 2022.0,
    h100_transistor_density_m_per_mm2: float = 98.28,
    h100_watts_per_tpp: float = 0.326493,
    h100_tpp_per_chip: float = 2144.0,
    transistor_density_scaling_exponent: float = None,
    architecture_efficiency_improvement_per_year: float = None,
    transistor_density_at_end_of_dennard: float = None,
    watts_per_tpp_exponent_before_dennard: float = None,
    watts_per_tpp_exponent_after_dennard: float = None,
) -> float:
    """
    Calculate watts per chip based on process node and year.

    This matches the discrete model's calculation in get_monthly_production_rate():
    1. Calculate h100e_per_chip (from density ratio and architecture efficiency)
    2. Calculate tpp_per_chip = h100e_per_chip * H100_TPP_PER_CHIP
    3. Calculate transistor_density from process node
    4. Calculate watts_per_tpp from transistor density
    5. watts_per_chip = tpp_per_chip * watts_per_tpp

    Args:
        fab_process_node_nm: Process node in nanometers (e.g., 28 for 28nm)
        year: Current simulation year (for architecture efficiency calculation)
        exogenous_trends: Optional ExogenousComputeTrends for parameter values
        h100_reference_nm: H100 reference node (default 4nm)
        h100_release_year: H100 release year (default 2022)
        h100_transistor_density_m_per_mm2: H100 transistor density (default 98.28)
        h100_watts_per_tpp: H100 watts per TPP (default 0.326493)
        h100_tpp_per_chip: H100 TPP per chip (default 2144.0)
        transistor_density_scaling_exponent: Scaling exponent (default 1.49)
        architecture_efficiency_improvement_per_year: Yearly improvement (default 1.23)
        transistor_density_at_end_of_dennard: Density at Dennard transition (default 10.0)
        watts_per_tpp_exponent_before_dennard: Exponent before Dennard ended (default -1.0)
        watts_per_tpp_exponent_after_dennard: Exponent after Dennard ended (default -0.33)

    Returns:
        Watts per chip for the given process node and year
    """
    # Step 1: Calculate h100e_per_chip (performance relative to H100)
    h100e_per_chip = calculate_fab_h100e_per_chip(
        fab_process_node_nm=fab_process_node_nm,
        year=year,
        exogenous_trends=exogenous_trends,
        h100_reference_nm=h100_reference_nm,
        transistor_density_scaling_exponent=transistor_density_scaling_exponent,
        architecture_efficiency_improvement_per_year=architecture_efficiency_improvement_per_year,
        h100_release_year=h100_release_year,
    )

    # Step 2: Calculate TPP per chip
    tpp_per_chip = h100e_per_chip * h100_tpp_per_chip

    # Step 3: Calculate transistor density from process node
    transistor_density = calculate_transistor_density_from_process_node(
        fab_process_node_nm=fab_process_node_nm,
        exogenous_trends=exogenous_trends,
        h100_reference_nm=h100_reference_nm,
        h100_transistor_density_m_per_mm2=h100_transistor_density_m_per_mm2,
        transistor_density_scaling_exponent=transistor_density_scaling_exponent,
    )

    # Step 4: Calculate watts per TPP
    watts_per_tpp = calculate_watts_per_tpp_from_transistor_density(
        transistor_density_m_per_mm2=transistor_density,
        exogenous_trends=exogenous_trends,
        h100_transistor_density_m_per_mm2=h100_transistor_density_m_per_mm2,
        h100_watts_per_tpp=h100_watts_per_tpp,
        transistor_density_at_end_of_dennard=transistor_density_at_end_of_dennard,
        watts_per_tpp_exponent_before_dennard=watts_per_tpp_exponent_before_dennard,
        watts_per_tpp_exponent_after_dennard=watts_per_tpp_exponent_after_dennard,
    )

    # Step 5: Calculate watts per chip
    return tpp_per_chip * watts_per_tpp


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
    variance: float
) -> float:
    """
    Sample a detection time from the Gamma distribution for a given labor level.

    The Gamma distribution is parameterized by:
    - Mean = k * theta
    - Variance = k * theta^2

    The 'variance' parameter is used directly as the scale parameter (theta).
    This matches the discrete reference model where variance_theta is passed
    directly to scipy.stats.gamma as the scale parameter.
    With k = mu / theta, we get:
    - Mean = k * theta = mu
    - Variance = k * theta^2 = mu * theta
    """
    if labor <= 1:
        return float('inf')

    A, B = compute_detection_constants(
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers
    )

    mu = compute_mean_detection_time(labor, A, B)

    # The 'variance' parameter is used directly as theta (scale parameter)
    # This matches the discrete reference model's parameterization
    theta = variance
    k = mu / theta

    return float(np.random.gamma(k, theta))


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


def compute_lr_from_sme_inventory(
    reported_scanner_count: float,
    diversion_proportion: float,
    us_estimate_scanners: float,
    median_error: float
) -> float:
    """
    Calculate likelihood ratio from SME (lithography scanner) inventory accounting.

    If a covert project exists, some scanners were diverted, so:
    - reported_scanners = true_scanners * (1 - diversion_proportion)
    - true_scanners = reported_scanners / (1 - diversion_proportion)

    Args:
        reported_scanner_count: Number of scanners reported (after diversion)
        diversion_proportion: Fraction of scanners diverted to black project
        us_estimate_scanners: US intelligence estimate of scanner count
        median_error: Median relative error in US estimates (e.g., 0.07 for 7%)

    Returns:
        Likelihood ratio P(evidence | project exists) / P(evidence | no project)
    """
    if reported_scanner_count < 1e-10 or diversion_proportion <= 0:
        return 1.0

    true_scanners_if_exists = reported_scanner_count / (1 - diversion_proportion)

    return lr_from_discrepancy_in_us_estimate(
        true_if_project_exists=true_scanners_if_exists,
        true_if_no_project=reported_scanner_count,
        us_estimate=us_estimate_scanners,
        median_error=median_error
    )


def _build_composite_detection_distribution(
    labor_by_year: Dict[float, int],
    A: float,
    B: float,
    variance_theta: float
) -> List[Tuple]:
    """
    Build composite distribution for detection time accounting for variable labor.

    This matches the discrete reference model's approach which builds probability
    ranges for each year based on varying labor levels.

    Args:
        labor_by_year: Dict mapping year (relative to project start) to labor count
        A, B: Detection time constants
        variance_theta: Scale parameter (theta) for the Gamma distribution

    Returns:
        List of tuples (year_start, year_end, labor, k, theta, cum_start, cum_end)
    """
    if not labor_by_year:
        return []

    sorted_years = sorted(labor_by_year.keys())

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

    return prob_ranges


def _sample_detection_time_from_composite(prob_ranges: List[Tuple]) -> float:
    """
    Sample a detection time from the precomputed composite distribution.

    This matches the discrete reference model's approach.

    Args:
        prob_ranges: Precomputed probability ranges from _build_composite_detection_distribution

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
    variance: float
) -> Tuple[Dict[float, float], float]:
    """
    Calculate likelihood ratios over time accounting for variable labor.

    This function uses the same composite distribution approach as the discrete
    reference model to properly account for varying labor levels over time when
    sampling the detection time.

    Args:
        labor_by_year: Dict mapping year (relative to project start) to labor count
        mean_detection_time_100_workers: Mean detection time for 100 workers
        mean_detection_time_1000_workers: Mean detection time for 1000 workers
        variance: Scale parameter (theta) for the Gamma distribution, used directly

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

    # Build composite distribution that accounts for varying labor
    # This matches the discrete reference model's approach
    prob_ranges = _build_composite_detection_distribution(
        labor_by_year=labor_by_year,
        A=A,
        B=B,
        variance_theta=variance
    )

    # Sample detection time from the composite distribution
    time_of_detection = _sample_detection_time_from_composite(prob_ranges)

    # Calculate likelihood ratios for each year
    # The LR is the survival probability: P(not detected by year | project exists)
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
            k = mu / variance

            # LR = P(evidence | project exists) / P(evidence | no project)
            # P(evidence | no project) = 1.0 (no detection evidence if no project)
            p_not_detected = stats.gamma.sf(year, a=k, scale=variance)
            lr_by_year[year] = max(p_not_detected, 0.001)  # Floor at 0.001 to match discrete

    return lr_by_year, time_of_detection


def compute_detection_probability(
    years_since_start: float,
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance: float
) -> float:
    """
    Compute the probability of detection by a given time.

    Args:
        years_since_start: Time since project started
        labor: Number of workers
        mean_detection_time_100_workers: Mean detection time for 100 workers
        mean_detection_time_1000_workers: Mean detection time for 1000 workers
        variance: Scale parameter (theta) for the Gamma distribution, used directly

    Returns:
        Probability of detection by the given time
    """
    if labor <= 1 or years_since_start <= 0:
        return 0.0

    A, B = compute_detection_constants(
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers
    )

    mu = compute_mean_detection_time(labor, A, B)

    # The 'variance' parameter is used directly as theta (scale parameter)
    theta = variance
    k = mu / theta

    return float(stats.gamma.cdf(years_since_start, a=k, scale=theta))


# =============================================================================
# CONTINUOUS DETECTION MODEL (HAZARD RATE APPROACH)
# =============================================================================

def compute_gamma_hazard_rate(
    t: float,
    k: float,
    theta: float,
    min_hazard: float = 1e-10,
    max_hazard: float = 100.0
) -> float:
    """
    Compute the hazard rate h(t) for a Gamma(k, theta) distribution.

    The hazard rate is h(t) = f(t) / S(t) where:
    - f(t) is the PDF
    - S(t) = 1 - F(t) is the survival function

    For the detection model, this represents the instantaneous rate of detection
    at time t, given that detection has not occurred yet.

    Args:
        t: Time since project start (years)
        k: Shape parameter (k = mu / theta)
        theta: Scale parameter (variance parameter from detection config)
        min_hazard: Minimum hazard rate to return (prevents numerical issues)
        max_hazard: Maximum hazard rate to return (caps extreme values)

    Returns:
        Hazard rate h(t)
    """
    if t <= 0 or k <= 0 or theta <= 0:
        return min_hazard

    # Use scipy's gamma distribution functions
    pdf = stats.gamma.pdf(t, a=k, scale=theta)
    survival = stats.gamma.sf(t, a=k, scale=theta)  # sf = survival function = 1 - cdf

    if survival < 1e-15:
        # If survival is essentially 0, detection is certain
        return max_hazard

    hazard = pdf / survival
    return float(np.clip(hazard, min_hazard, max_hazard))


def compute_worker_detection_hazard_rate(
    t: float,
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance: float
) -> float:
    """
    Compute the instantaneous hazard rate for worker-based detection.

    This is the continuous derivative of the log-likelihood ratio contribution
    from worker-based intelligence detection. The hazard rate h(t) represents
    the rate at which evidence accumulates against the project.

    Args:
        t: Time since project start (years)
        labor: Number of workers in the black project
        mean_detection_time_100_workers: Mean detection time for 100 workers
        mean_detection_time_1000_workers: Mean detection time for 1000 workers
        variance: Scale parameter (theta) for the Gamma distribution, used directly

    Returns:
        Hazard rate h(t) for worker-based detection
    """
    if labor <= 1 or t <= 0:
        return 0.0

    A, B = compute_detection_constants(
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers
    )

    mu = compute_mean_detection_time(labor, A, B)
    if mu <= 0:
        return 0.0

    # The 'variance' parameter is used directly as theta (scale parameter)
    theta = variance
    k = mu / theta

    return compute_gamma_hazard_rate(t, k, theta)


def compute_log_survival_probability(
    t: float,
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance: float
) -> float:
    """
    Compute log(S(t)) where S(t) is the survival probability for worker detection.

    The survival probability S(t) = P(detection time > t) is the probability
    that the project has NOT been detected by time t. The log of this is
    used as the log-likelihood ratio contribution from worker-based evidence.

    Args:
        t: Time since project start (years)
        labor: Number of workers in the black project
        mean_detection_time_100_workers: Mean detection time for 100 workers
        mean_detection_time_1000_workers: Mean detection time for 1000 workers
        variance: Scale parameter (theta) for the Gamma distribution, used directly

    Returns:
        log(S(t)) - this is negative and becomes more negative over time
    """
    if labor <= 1 or t <= 0:
        return 0.0  # log(1) = 0, no evidence accumulation

    A, B = compute_detection_constants(
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers
    )

    mu = compute_mean_detection_time(labor, A, B)
    if mu <= 0:
        return 0.0

    # The 'variance' parameter is used directly as theta (scale parameter)
    theta = variance
    k = mu / theta

    # Survival function S(t) = 1 - CDF(t)
    survival = stats.gamma.sf(t, a=k, scale=theta)

    if survival < 1e-15:
        return -35.0  # log(1e-15) ≈ -34.5, cap to prevent -inf

    return float(np.log(survival))


def compute_cumulative_log_lr(
    t: float,
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance: float,
    static_log_lr: float = 0.0,
    detected: bool = False,
    detection_log_lr: float = np.log(100.0)
) -> float:
    """
    Compute the cumulative log-likelihood ratio at time t.

    The total LR combines:
    1. Static evidence (resource accounting, SME inventory, etc.) - constant
    2. Worker-based evidence - evolves over time based on survival probability
    3. Detection event - if detected, LR jumps to detection_log_lr

    Args:
        t: Time since project start (years)
        labor: Number of workers in the black project
        mean_detection_time_100_workers: Mean detection time for 100 workers
        mean_detection_time_1000_workers: Mean detection time for 1000 workers
        variance: Scale parameter (theta) for the Gamma distribution, used directly
        static_log_lr: Log-LR from static evidence sources (computed at init)
        detected: Whether detection has occurred
        detection_log_lr: Log-LR to assign after detection (default log(100))

    Returns:
        Cumulative log-likelihood ratio
    """
    if detected:
        # After detection, LR is high (evidence strongly supports project exists)
        return static_log_lr + detection_log_lr

    # Before detection, worker evidence is based on survival probability
    # LR_workers = S(t) because no detection = evidence against project
    # As time passes without detection, this becomes more evidence against
    log_survival = compute_log_survival_probability(
        t, labor,
        mean_detection_time_100_workers,
        mean_detection_time_1000_workers,
        variance
    )

    return static_log_lr + log_survival


def compute_posterior_probability(
    cumulative_log_lr: float,
    prior_odds: float
) -> float:
    """
    Compute posterior probability of project existing from log-LR.

    Using Bayes' theorem:
    posterior_odds = prior_odds * likelihood_ratio
    posterior_prob = posterior_odds / (1 + posterior_odds)

    Args:
        cumulative_log_lr: Cumulative log-likelihood ratio
        prior_odds: Prior odds of project existing (odds = p / (1-p))

    Returns:
        Posterior probability P(project exists | evidence)
    """
    # Compute posterior odds in log space to avoid overflow
    log_prior_odds = np.log(prior_odds) if prior_odds > 0 else -35.0
    log_posterior_odds = log_prior_odds + cumulative_log_lr

    # Convert to probability
    # P = odds / (1 + odds) = 1 / (1 + 1/odds) = 1 / (1 + exp(-log_odds))
    # This is the sigmoid function
    if log_posterior_odds > 35:
        return 1.0
    elif log_posterior_odds < -35:
        return 0.0

    posterior_odds = np.exp(log_posterior_odds)
    return float(posterior_odds / (1 + posterior_odds))
