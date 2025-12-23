"""
Compare LIKELIHOOD RATIO DETECTION calculations between the continuous ODE-based model
and the discrete reference model.

This script compares:
- lr_from_discrepancy_in_us_estimate (core formula)
- LR from energy consumption accounting
- LR from satellite detection
- LR from compute stock accounting
- LR from SME (scanner) inventory accounting
- LR from direct observation (workers) - Gamma survival function
"""

import sys
import os
import math
import numpy as np
from scipy import stats
from typing import Dict, List

# Add both project roots to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')
sys.path.insert(0, '/Users/joshuaclymer/github/covert_compute_production_model')


# =============================================================================
# CORE LR FORMULA COMPARISON
# =============================================================================

def discrete_lr_from_discrepancy(
    true_if_project_exists: float,
    true_if_no_project: float,
    us_estimate: float,
    median_error: float
) -> float:
    """
    Discrete model's lr_from_discrepancy_in_us_estimate from util.py.

    Uses exponential distribution for absolute relative error.
    """
    if true_if_project_exists < 1e-10 or true_if_no_project < 1e-10:
        return 1.0

    # Calculate absolute relative errors under each hypothesis
    error_if_project = abs(us_estimate - true_if_project_exists) / true_if_project_exists
    error_if_no_project = abs(us_estimate - true_if_no_project) / true_if_no_project

    # PDF of absolute error (exponential distribution)
    k = -np.log(0.5) / median_error
    p_if_project = k * np.exp(-k * error_if_project)
    p_if_no_project = k * np.exp(-k * error_if_no_project)

    if p_if_no_project > 0:
        lr = p_if_project / p_if_no_project
    else:
        lr = 1e6

    return lr


def continuous_lr_from_discrepancy(
    true_if_project_exists: float,
    true_if_no_project: float,
    us_estimate: float,
    median_error: float
) -> float:
    """
    Continuous model's lr_from_discrepancy_in_us_estimate from black_compute.py.

    Should be identical to discrete model.
    """
    if true_if_project_exists < 1e-10 or true_if_no_project < 1e-10:
        return 1.0

    error_if_project_exists = abs(us_estimate - true_if_project_exists) / true_if_project_exists
    error_if_no_project = abs(us_estimate - true_if_no_project) / true_if_no_project

    k = -np.log(0.5) / median_error
    p_if_project_exists = k * np.exp(-k * error_if_project_exists)
    p_if_no_project_result = k * np.exp(-k * error_if_no_project)

    if p_if_no_project_result > 0:
        lr = p_if_project_exists / p_if_no_project_result
    else:
        lr = 1e6

    return lr


# =============================================================================
# ENERGY CONSUMPTION LR
# =============================================================================

def discrete_lr_from_energy(
    total_prc_energy_gw: float,
    covert_energy_gw: float,
    us_estimate_energy: float,
    median_error: float
) -> float:
    """
    Discrete model's lr_from_reported_energy_consumption from black_datacenters.py.
    """
    if covert_energy_gw < 1e-10 or total_prc_energy_gw < 1e-10:
        return 1.0

    reported_energy = total_prc_energy_gw - covert_energy_gw

    return discrete_lr_from_discrepancy(
        true_if_project_exists=total_prc_energy_gw,
        true_if_no_project=reported_energy,
        us_estimate=us_estimate_energy,
        median_error=median_error
    )


def continuous_lr_from_energy(
    total_prc_energy_gw: float,
    covert_energy_gw: float,
    us_estimate_energy: float,
    median_error: float
) -> float:
    """
    Continuous model's compute_lr_from_reported_energy_consumption from black_compute.py.
    """
    if covert_energy_gw < 1e-10 or total_prc_energy_gw < 1e-10:
        return 1.0

    reported_energy = total_prc_energy_gw - covert_energy_gw

    return continuous_lr_from_discrepancy(
        true_if_project_exists=total_prc_energy_gw,
        true_if_no_project=reported_energy,
        us_estimate=us_estimate_energy,
        median_error=median_error
    )


# =============================================================================
# SATELLITE DETECTION LR
# =============================================================================

def discrete_lr_from_satellite(
    diverted_capacity_gw: float,
    total_unconcealed_capacity_gw: float,
    us_estimate_capacity: float,
    median_error: float
) -> float:
    """
    Discrete model's lr_from_identifying_datacenters_with_satellites.
    """
    if diverted_capacity_gw < 1e-10:
        return 1.0

    reported_capacity = total_unconcealed_capacity_gw - diverted_capacity_gw

    return discrete_lr_from_discrepancy(
        true_if_project_exists=reported_capacity,
        true_if_no_project=total_unconcealed_capacity_gw,
        us_estimate=us_estimate_capacity,
        median_error=median_error
    )


def continuous_lr_from_satellite(
    diverted_capacity_gw: float,
    total_unconcealed_capacity_gw: float,
    us_estimate_capacity: float,
    median_error: float
) -> float:
    """
    Continuous model's compute_lr_from_satellite_detection.
    """
    if diverted_capacity_gw < 1e-10:
        return 1.0

    reported_capacity = total_unconcealed_capacity_gw - diverted_capacity_gw

    return continuous_lr_from_discrepancy(
        true_if_project_exists=reported_capacity,
        true_if_no_project=total_unconcealed_capacity_gw,
        us_estimate=us_estimate_capacity,
        median_error=median_error
    )


# =============================================================================
# COMPUTE STOCK ACCOUNTING LR
# =============================================================================

def discrete_lr_from_compute_accounting(
    reported_compute_stock: float,
    diversion_proportion: float,
    us_estimate_compute: float,
    median_error: float
) -> float:
    """
    Discrete model's lr_from_prc_compute_accounting.
    """
    if reported_compute_stock < 1e-10 or diversion_proportion <= 0:
        return 1.0

    true_stock_if_exists = reported_compute_stock / (1 - diversion_proportion)

    return discrete_lr_from_discrepancy(
        true_if_project_exists=true_stock_if_exists,
        true_if_no_project=reported_compute_stock,
        us_estimate=us_estimate_compute,
        median_error=median_error
    )


def continuous_lr_from_compute_accounting(
    reported_compute_stock: float,
    diversion_proportion: float,
    us_estimate_compute: float,
    median_error: float
) -> float:
    """
    Continuous model's compute_lr_from_prc_compute_accounting.
    """
    if reported_compute_stock < 1e-10 or diversion_proportion <= 0:
        return 1.0

    true_stock_if_exists = reported_compute_stock / (1 - diversion_proportion)

    return continuous_lr_from_discrepancy(
        true_if_project_exists=true_stock_if_exists,
        true_if_no_project=reported_compute_stock,
        us_estimate=us_estimate_compute,
        median_error=median_error
    )


# =============================================================================
# WORKER DETECTION LR (Gamma survival function)
# =============================================================================

def calculate_detection_mean_time(
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
) -> float:
    """
    Calculate mean detection time based on labor count.

    Uses: mu(workers) = A / log10(workers)^B
    """
    if labor <= 0:
        return float('inf')

    # Derive A and B from anchor points
    # mu_100 = A / log10(100)^B = A / 2^B
    # mu_1000 = A / log10(1000)^B = A / 3^B
    # mu_100 / mu_1000 = (3/2)^B
    # B = log(mu_100 / mu_1000) / log(3/2)

    mu_100 = mean_detection_time_100_workers
    mu_1000 = mean_detection_time_1000_workers

    if mu_1000 > 0:
        B = np.log(mu_100 / mu_1000) / np.log(3.0 / 2.0)
    else:
        B = 1.0

    # A = mu_100 * log10(100)^B = mu_100 * 2^B
    A = mu_100 * (2 ** B)

    # Calculate mean detection time for current labor
    log_labor = np.log10(max(labor, 1))
    if log_labor > 0:
        mu = A / (log_labor ** B)
    else:
        mu = float('inf')

    return mu


def discrete_lr_from_workers(
    year_since_start: float,
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float,
    detected: bool = False
) -> float:
    """
    Discrete model's LR from worker detection using Gamma survival function.

    If detected, LR = 100 (high evidence).
    Otherwise, LR = P(not detected by t | project exists) / P(not detected | no project)
                  = Gamma_SF(t) / 1.0
    """
    if detected:
        return 100.0

    if labor <= 0:
        return 1.0

    # Calculate mean detection time
    mu = calculate_detection_mean_time(
        labor=labor,
        mean_detection_time_100_workers=mean_detection_time_100_workers,
        mean_detection_time_1000_workers=mean_detection_time_1000_workers,
    )

    # Gamma distribution parameters: k = mu / theta
    k = mu / variance_theta

    # Survival function: P(T > t) = 1 - CDF(t)
    p_not_detected = stats.gamma.sf(year_since_start, a=k, scale=variance_theta)

    # LR = P(evidence | project exists) / P(evidence | no project)
    # P(no detection by t | no project) = 1 (definitely no detection)
    # P(no detection by t | project exists) = SF(t)
    lr = p_not_detected / 1.0

    return max(lr, 0.001)  # Floor at 0.001


def continuous_lr_from_workers(
    year_since_start: float,
    labor: int,
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float,
    detected: bool = False
) -> float:
    """
    Continuous model's LR from worker detection.
    Should be identical to discrete model.
    """
    if detected:
        return 100.0

    if labor <= 0:
        return 1.0

    # Calculate mean detection time (same formula)
    mu = calculate_detection_mean_time(
        labor=labor,
        mean_detection_time_100_workers=mean_detection_time_100_workers,
        mean_detection_time_1000_workers=mean_detection_time_1000_workers,
    )

    k = mu / variance_theta

    # Survival function
    p_not_detected = stats.gamma.sf(year_since_start, a=k, scale=variance_theta)

    lr = p_not_detected / 1.0

    return max(lr, 0.001)


# =============================================================================
# COMPARISON TESTS
# =============================================================================

def compare_core_lr_formula():
    """Compare the core lr_from_discrepancy_in_us_estimate formula."""
    print("\n" + "=" * 70)
    print("CORE LR FORMULA COMPARISON")
    print("lr_from_discrepancy_in_us_estimate")
    print("=" * 70)

    test_cases = [
        # (true_if_project, true_if_no_project, us_estimate, median_error, description)
        (1100.0, 1000.0, 1050.0, 0.10, "Small diversion (10%), US estimate in between"),
        (1100.0, 1000.0, 1100.0, 0.10, "US estimate matches true if project exists"),
        (1100.0, 1000.0, 1000.0, 0.10, "US estimate matches reported (no project)"),
        (1100.0, 1000.0, 900.0, 0.10, "US estimate below reported"),
        (1200.0, 1000.0, 1100.0, 0.10, "Large diversion (20%), US estimate in between"),
        (1000.0, 1000.0, 1000.0, 0.10, "No diversion, all values equal"),
        (1050.0, 1000.0, 1025.0, 0.07, "Small diversion (5%), smaller median error"),
    ]

    all_aligned = True
    print(f"\n{'Description':<50} {'Discrete':<12} {'Continuous':<12} {'Match':<8}")
    print("-" * 82)

    for true_proj, true_no, us_est, med_err, desc in test_cases:
        d_lr = discrete_lr_from_discrepancy(true_proj, true_no, us_est, med_err)
        c_lr = continuous_lr_from_discrepancy(true_proj, true_no, us_est, med_err)

        match = np.isclose(d_lr, c_lr, rtol=1e-10)
        if not match:
            all_aligned = False

        print(f"{desc:<50} {d_lr:<12.6f} {c_lr:<12.6f} {'✓' if match else '✗':<8}")

    return all_aligned


def compare_energy_lr():
    """Compare energy consumption LR."""
    print("\n" + "=" * 70)
    print("ENERGY CONSUMPTION LR COMPARISON")
    print("=" * 70)

    test_cases = [
        # (total_energy, covert_energy, us_estimate, median_error, description)
        (1100.0, 50.0, 1100.0, 0.10, "US sees full energy (matches true)"),
        (1100.0, 50.0, 1050.0, 0.10, "US sees reported energy (no project case)"),
        (1100.0, 100.0, 1100.0, 0.10, "Larger covert energy, US sees full"),
        (1100.0, 0.001, 1100.0, 0.10, "Near-zero covert energy"),
    ]

    all_aligned = True
    print(f"\n{'Description':<50} {'Discrete':<12} {'Continuous':<12} {'Match':<8}")
    print("-" * 82)

    for total_e, covert_e, us_est, med_err, desc in test_cases:
        d_lr = discrete_lr_from_energy(total_e, covert_e, us_est, med_err)
        c_lr = continuous_lr_from_energy(total_e, covert_e, us_est, med_err)

        match = np.isclose(d_lr, c_lr, rtol=1e-10)
        if not match:
            all_aligned = False

        print(f"{desc:<50} {d_lr:<12.6f} {c_lr:<12.6f} {'✓' if match else '✗':<8}")

    return all_aligned


def compare_satellite_lr():
    """Compare satellite detection LR."""
    print("\n" + "=" * 70)
    print("SATELLITE DETECTION LR COMPARISON")
    print("=" * 70)

    test_cases = [
        # (diverted, total_unconcealed, us_estimate, median_error, description)
        (10.0, 100.0, 90.0, 0.10, "10% diverted, US sees diverted portion"),
        (10.0, 100.0, 100.0, 0.10, "10% diverted, US sees full capacity"),
        (5.0, 100.0, 95.0, 0.10, "5% diverted, US sees diverted portion"),
        (0.001, 100.0, 100.0, 0.10, "Near-zero diversion"),
    ]

    all_aligned = True
    print(f"\n{'Description':<50} {'Discrete':<12} {'Continuous':<12} {'Match':<8}")
    print("-" * 82)

    for diverted, total, us_est, med_err, desc in test_cases:
        d_lr = discrete_lr_from_satellite(diverted, total, us_est, med_err)
        c_lr = continuous_lr_from_satellite(diverted, total, us_est, med_err)

        match = np.isclose(d_lr, c_lr, rtol=1e-10)
        if not match:
            all_aligned = False

        print(f"{desc:<50} {d_lr:<12.6f} {c_lr:<12.6f} {'✓' if match else '✗':<8}")

    return all_aligned


def compare_compute_accounting_lr():
    """Compare compute stock accounting LR."""
    print("\n" + "=" * 70)
    print("COMPUTE STOCK ACCOUNTING LR COMPARISON")
    print("=" * 70)

    test_cases = [
        # (reported_stock, diversion_prop, us_estimate, median_error, description)
        (950000.0, 0.05, 1000000.0, 0.10, "5% diverted, US sees full stock"),
        (950000.0, 0.05, 950000.0, 0.10, "5% diverted, US sees reported"),
        (900000.0, 0.10, 1000000.0, 0.10, "10% diverted, US sees full"),
        (950000.0, 0.05, 975000.0, 0.10, "5% diverted, US in between"),
    ]

    all_aligned = True
    print(f"\n{'Description':<50} {'Discrete':<12} {'Continuous':<12} {'Match':<8}")
    print("-" * 82)

    for reported, div_prop, us_est, med_err, desc in test_cases:
        d_lr = discrete_lr_from_compute_accounting(reported, div_prop, us_est, med_err)
        c_lr = continuous_lr_from_compute_accounting(reported, div_prop, us_est, med_err)

        match = np.isclose(d_lr, c_lr, rtol=1e-10)
        if not match:
            all_aligned = False

        print(f"{desc:<50} {d_lr:<12.6f} {c_lr:<12.6f} {'✓' if match else '✗':<8}")

    return all_aligned


def compare_worker_detection_lr():
    """Compare worker-based detection LR."""
    print("\n" + "=" * 70)
    print("WORKER DETECTION LR COMPARISON (Gamma Survival)")
    print("=" * 70)

    # Detection parameters
    mean_100 = 6.95
    mean_1000 = 3.42
    variance = 3.88

    test_cases = [
        # (years_since_start, labor, detected, description)
        (0.0, 1000, False, "Year 0, 1000 workers"),
        (1.0, 1000, False, "Year 1, 1000 workers"),
        (2.0, 1000, False, "Year 2, 1000 workers"),
        (5.0, 1000, False, "Year 5, 1000 workers"),
        (1.0, 100, False, "Year 1, 100 workers"),
        (5.0, 100, False, "Year 5, 100 workers"),
        (1.0, 10000, False, "Year 1, 10000 workers"),
        (3.0, 10000, False, "Year 3, 10000 workers"),
        (2.0, 500, True, "Year 2, detected"),
    ]

    all_aligned = True
    print(f"\n{'Description':<35} {'Discrete':<12} {'Continuous':<12} {'Match':<8}")
    print("-" * 67)

    for years, labor, detected, desc in test_cases:
        d_lr = discrete_lr_from_workers(years, labor, mean_100, mean_1000, variance, detected)
        c_lr = continuous_lr_from_workers(years, labor, mean_100, mean_1000, variance, detected)

        match = np.isclose(d_lr, c_lr, rtol=1e-10)
        if not match:
            all_aligned = False

        print(f"{desc:<35} {d_lr:<12.6f} {c_lr:<12.6f} {'✓' if match else '✗':<8}")

    return all_aligned


def compare_worker_lr_over_time():
    """Compare worker LR evolution over time."""
    print("\n" + "=" * 70)
    print("WORKER LR EVOLUTION OVER TIME")
    print("=" * 70)

    mean_100 = 6.95
    mean_1000 = 3.42
    variance = 3.88
    labor = 1000

    print(f"\nParameters: labor={labor}, mean_100={mean_100}, mean_1000={mean_1000}")

    print(f"\n{'Year':<8} {'Discrete':<12} {'Continuous':<12} {'Match':<8}")
    print("-" * 40)

    all_aligned = True
    for year in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        d_lr = discrete_lr_from_workers(year, labor, mean_100, mean_1000, variance, False)
        c_lr = continuous_lr_from_workers(year, labor, mean_100, mean_1000, variance, False)

        match = np.isclose(d_lr, c_lr, rtol=1e-10)
        if not match:
            all_aligned = False

        print(f"{year:<8} {d_lr:<12.6f} {c_lr:<12.6f} {'✓' if match else '✗':<8}")

    return all_aligned


def main():
    """Run all LR comparison tests."""
    print("=" * 70)
    print("LIKELIHOOD RATIO DETECTION COMPARISON")
    print("Discrete Model vs Continuous Model")
    print("=" * 70)

    results = {
        'Core LR Formula': compare_core_lr_formula(),
        'Energy Consumption LR': compare_energy_lr(),
        'Satellite Detection LR': compare_satellite_lr(),
        'Compute Stock Accounting LR': compare_compute_accounting_lr(),
        'Worker Detection LR': compare_worker_detection_lr(),
        'Worker LR Over Time': compare_worker_lr_over_time(),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_aligned = all(results.values())

    for name, aligned in results.items():
        status = "✓ ALIGNED" if aligned else "✗ MISALIGNED"
        print(f"  {name}: {status}")

    print("\n" + "-" * 70)
    if all_aligned:
        print("✓ ALL LIKELIHOOD RATIO CALCULATIONS ALIGNED")
    else:
        print("✗ SOME LIKELIHOOD RATIO CALCULATIONS MISALIGNED")
    print("-" * 70)

    return all_aligned


if __name__ == "__main__":
    main()
