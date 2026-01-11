"""
Black project perceptions world updater.

Contains:
- Detection utility functions (likelihood ratios, detection time sampling, Bayesian updates)
- BlackProjectPerceptionsUpdater: WorldUpdater for black project detection metrics

Updates detection and perception metrics for black projects:
- Likelihood ratios (energy, satellite, worker detection)
- Cumulative likelihood ratio
- Posterior probability of project existence
- Detection status
"""

import math
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, TYPE_CHECKING
from torch import Tensor

from classes.world.world import World
from classes.simulation_primitives import WorldUpdater
from parameters.classes import SimulationParameters
from parameters.classes import BlackProjectParameters
from parameters.classes import BlackProjectPerceptionsParameters
from parameters.classes import DataCenterAndEnergyParameters

if TYPE_CHECKING:
    from classes.world.entities import AIBlackProject


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
    from world_updaters.datacenters_and_energy import calculate_datacenter_operating_labor

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


class BlackProjectPerceptionsUpdater(WorldUpdater):
    """
    Updates detection and perception metrics for black projects.

    Uses continuous detection model based on:
    - Worker-based detection (Gamma distribution hazard rate)
    - Resource accounting evidence (energy, satellite, compute stock)
    - Bayesian updating to compute posterior probability
    """

    def __init__(
        self,
        params: SimulationParameters,
        black_project_params: BlackProjectParameters,
        energy_params: DataCenterAndEnergyParameters,
        perception_params: BlackProjectPerceptionsParameters,
    ):
        super().__init__()
        self.params = params
        self.black_project_params = black_project_params
        self.energy_params = energy_params
        self.perception_params = perception_params

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """
        Compute detection and perception metrics for black projects.

        Updates:
        - lr_reported_energy (dynamic, re-sampled each timestep)
        - lr_other_intel (from precomputed worker detection dict)
        - cumulative_lr (combined likelihood ratio)
        - posterior_prob (Bayesian posterior probability)
        - is_detected (whether detection has occurred)
        - Fab-specific LR metrics (lr_fab_other, lr_fab_combined)
        """
        current_time = t.item() if isinstance(t, Tensor) else float(t)

        if self.perception_params is None:
            return world

        for _, project in world.black_projects.items():
            black_project_start_year = float(
                project.preparation_start_year.item()
                if hasattr(project.preparation_start_year, 'item')
                else project.preparation_start_year
            )
            years_since_project_start = current_time - black_project_start_year

            if years_since_project_start <= 0:
                continue

            # Get total labor for detection calculation
            total_labor = get_black_project_total_labor(project)

            # Get datacenter capacity (computed by BlackProjectDatacenterUpdater)
            total_capacity_gw = project.datacenters.data_center_capacity_gw if project.datacenters else 0.0

            # --- Compute lr_reported_energy (dynamic) ---
            # Re-sample US estimate at each timestep to match reference model
            covert_energy_gw = total_capacity_gw
            if covert_energy_gw > 1e-10 and hasattr(project, 'total_prc_energy_gw'):
                us_estimate_energy_now = sample_us_estimate_with_error(
                    project.total_prc_energy_gw,
                    self.perception_params.intelligence_median_error_in_energy_consumption_estimate_of_datacenter_capacity,
                )
                lr_energy = compute_lr_from_reported_energy_consumption(
                    total_prc_energy_gw=project.total_prc_energy_gw,
                    covert_energy_gw=covert_energy_gw,
                    us_estimate_energy=us_estimate_energy_now,
                    median_error=self.perception_params.intelligence_median_error_in_energy_consumption_estimate_of_datacenter_capacity,
                )
            else:
                lr_energy = 1.0

            # --- Compute static LR components ---
            # lr_sme = lr_inventory * lr_procurement IF fab exists, else 1.0
            has_fab = project.fab_number_of_lithography_scanners > 0
            lr_sme_log = (
                math.log(max(1e-10, project.lr_sme_inventory)) +
                math.log(max(1e-10, project.lr_fab_procurement))
            ) if has_fab else 0.0

            static_log_lr = (
                math.log(max(1e-10, project.lr_prc_accounting)) +
                lr_sme_log +
                math.log(max(1e-10, project.lr_satellite_datacenter)) +
                math.log(max(1e-10, lr_energy))
            )

            # Check if detection has occurred
            detected = years_since_project_start >= project.sampled_detection_time

            # Compute continuous log-LR from worker detection
            log_lr = compute_cumulative_log_lr(
                t=years_since_project_start,
                labor=total_labor,
                mean_detection_time_100_workers=self.perception_params.mean_detection_time_for_100_workers,
                mean_detection_time_1000_workers=self.perception_params.mean_detection_time_for_1000_workers,
                variance=self.perception_params.variance_of_detection_time_given_num_workers,
                static_log_lr=static_log_lr,
                detected=detected,
            )

            # Compute posterior probability
            posterior_prob_val = compute_posterior_probability(
                cumulative_log_lr=log_lr,
                prior_odds=self.perception_params.prior_odds_of_covert_project,
            )

            # Look up worker LR from precomputed dictionary
            relative_year_dc = round(float(years_since_project_start), 1)
            lr_datacenters_dict = getattr(project, 'lr_datacenters_by_year', {})
            worker_lr = lr_datacenters_dict.get(relative_year_dc, 1.0)

            cumulative_lr_val = math.exp(log_lr)

            # Set perception metrics
            project._set_frozen_field('lr_other_intel', worker_lr)
            project._set_frozen_field('cumulative_lr', cumulative_lr_val)
            project._set_frozen_field('posterior_prob', posterior_prob_val)
            project._set_frozen_field('lr_reported_energy', lr_energy)
            project._set_frozen_field('is_detected', detected)

            # --- Fab-specific LR calculations ---
            # Fab construction starts at black_project_start_year
            years_since_fab_construction = round(float(current_time - black_project_start_year), 1)

            if years_since_fab_construction >= 0 and project.fab_number_of_lithography_scanners > 0:
                # Look up fab-specific LR from precomputed dictionary
                lr_fab_other_dict = getattr(project, 'lr_fab_other_by_year', {})
                fab_lr_other = lr_fab_other_dict.get(years_since_fab_construction, 1.0)

                # Fab combined LR = lr_sme_inventory x lr_fab_procurement x lr_fab_other
                fab_lr_combined = project.lr_sme_inventory * project.lr_fab_procurement * fab_lr_other

                project._set_frozen_field('lr_fab_other', fab_lr_other)
                project._set_frozen_field('lr_fab_combined', fab_lr_combined)
            else:
                project._set_frozen_field('lr_fab_other', 1.0)
                project._set_frozen_field('lr_fab_combined', project.lr_sme_inventory * project.lr_fab_procurement)

        return world
