"""
Simple comparison of likelihood ratio calculation between discrete and continuous models.
Focuses specifically on the lr_over_time_vs_num_workers function.
"""

import sys
import os

# Add both project roots to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')
sys.path.insert(0, '/Users/joshuaclymer/github/covert_compute_production_model')

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import discrete model's lr_over_time_vs_num_workers
from black_project_backend.util import (
    lr_over_time_vs_num_workers as discrete_lr_over_time,
    _cache as discrete_cache,
    build_composite_detection_distribution,
    sample_detection_time_from_composite,
)

# Import continuous model's lr_over_time_vs_num_workers
from world_updaters.compute.black_compute import (
    compute_lr_over_time_vs_num_workers as continuous_lr_over_time,
    compute_detection_constants,
)


def test_lr_functions_comparison(
    labor_by_year=None,
    mean_detection_time_100=6.95,
    mean_detection_time_1000=3.42,
    variance=3.88,
    num_sims=200,
    test_name="Standard Test"
):
    """Compare the LR calculation functions directly."""

    if labor_by_year is None:
        # Create labor profile (typical black project)
        # Start with construction labor only, then add operating labor
        labor_by_year = {
            0: 10500,   # Year 0: construction + researcher
            1: 11000,   # Year 1: add some operating labor
            2: 11500,   # Year 2: more operating labor
            3: 12000,   # Year 3
            4: 12500,   # Year 4
            5: 13000,   # Year 5
            6: 13500,   # Year 6
            7: 14000,   # Year 7
        }

    print("="*70)
    print(f"LIKELIHOOD RATIO COMPARISON: {test_name}")
    print("="*70)
    print(f"\nParameters:")
    print(f"  mean_detection_time_100: {mean_detection_time_100}")
    print(f"  mean_detection_time_1000: {mean_detection_time_1000}")
    print(f"  variance (theta): {variance}")
    print(f"\nLabor by year: {labor_by_year}")

    # Run multiple simulations
    num_sims = 200

    discrete_results = {year: [] for year in labor_by_year.keys()}
    continuous_results = {year: [] for year in labor_by_year.keys()}
    discrete_detection_times = []
    continuous_detection_times = []

    print(f"\nRunning {num_sims} simulations...")

    for i in range(num_sims):
        # Clear discrete cache
        discrete_cache.clear()
        np.random.seed(1000 + i)

        # Run discrete model
        discrete_lr_by_year = discrete_lr_over_time(
            labor_by_year=labor_by_year,
            mean_detection_time_100_workers=mean_detection_time_100,
            mean_detection_time_1000_workers=mean_detection_time_1000,
            variance_theta=variance,
        )

        # Extract detection time from discrete model (year where LR jumps to 100)
        discrete_det_time = float('inf')
        for year in sorted(discrete_lr_by_year.keys()):
            if discrete_lr_by_year[year] >= 99.0:
                discrete_det_time = year
                break
        discrete_detection_times.append(discrete_det_time)

        for year in labor_by_year.keys():
            discrete_results[year].append(discrete_lr_by_year.get(year, 1.0))

        # Reset seed for continuous model
        np.random.seed(1000 + i)

        # Run continuous model
        continuous_lr_by_year, continuous_det_time = continuous_lr_over_time(
            labor_by_year=labor_by_year,
            mean_detection_time_100_workers=mean_detection_time_100,
            mean_detection_time_1000_workers=mean_detection_time_1000,
            variance=variance,
        )

        continuous_detection_times.append(continuous_det_time)

        for year in labor_by_year.keys():
            continuous_results[year].append(continuous_lr_by_year.get(year, 1.0))

    # Analyze results
    print("\n" + "="*70)
    print("LR BY YEAR COMPARISON")
    print("="*70)

    for year in sorted(labor_by_year.keys()):
        discrete_vals = discrete_results[year]
        continuous_vals = continuous_results[year]

        print(f"\nYear {year}:")
        print(f"  Discrete:   median={np.median(discrete_vals):.4f}, "
              f"mean={np.mean(discrete_vals):.4f}, "
              f"std={np.std(discrete_vals):.4f}")
        print(f"  Continuous: median={np.median(continuous_vals):.4f}, "
              f"mean={np.mean(continuous_vals):.4f}, "
              f"std={np.std(continuous_vals):.4f}")

        # Count detections
        discrete_detections = sum(1 for x in discrete_vals if x >= 99.0)
        continuous_detections = sum(1 for x in continuous_vals if x >= 99.0)
        print(f"  Detections: discrete={discrete_detections}, continuous={continuous_detections}")

    # Detection time analysis
    print("\n" + "="*70)
    print("DETECTION TIME COMPARISON")
    print("="*70)

    finite_discrete = [t for t in discrete_detection_times if t < float('inf')]
    finite_continuous = [t for t in continuous_detection_times if t < float('inf')]

    print(f"\nDiscrete detections: {len(finite_discrete)}/{num_sims}")
    if finite_discrete:
        print(f"  Median: {np.median(finite_discrete):.2f}")
        print(f"  Mean: {np.mean(finite_discrete):.2f}")
        print(f"  Std: {np.std(finite_discrete):.2f}")

    print(f"\nContinuous detections: {len(finite_continuous)}/{num_sims}")
    if finite_continuous:
        print(f"  Median: {np.median(finite_continuous):.2f}")
        print(f"  Mean: {np.mean(finite_continuous):.2f}")
        print(f"  Std: {np.std(finite_continuous):.2f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: LR over time (median)
    ax1 = axes[0, 0]
    years = sorted(labor_by_year.keys())
    discrete_medians = [np.median(discrete_results[y]) for y in years]
    continuous_medians = [np.median(continuous_results[y]) for y in years]
    ax1.plot(years, discrete_medians, 'b-o', label='Discrete (median)', linewidth=2)
    ax1.plot(years, continuous_medians, 'g--s', label='Continuous (median)', linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Likelihood Ratio')
    ax1.set_title('LR Over Time (Median)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: LR over time (log scale)
    ax2 = axes[0, 1]
    ax2.semilogy(years, discrete_medians, 'b-o', label='Discrete (median)', linewidth=2)
    ax2.semilogy(years, continuous_medians, 'g--s', label='Continuous (median)', linewidth=2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Likelihood Ratio (log)')
    ax2.set_title('LR Over Time (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Detection time distribution
    ax3 = axes[1, 0]
    if finite_discrete:
        ax3.hist(finite_discrete, bins=20, alpha=0.5, label='Discrete', color='blue')
    if finite_continuous:
        ax3.hist(finite_continuous, bins=20, alpha=0.5, label='Continuous', color='green')
    ax3.set_xlabel('Detection Time (years)')
    ax3.set_ylabel('Count')
    ax3.set_title('Detection Time Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: LR at specific years (box plot)
    ax4 = axes[1, 1]
    selected_years = [0, 2, 4, 6]
    positions = range(len(selected_years))
    for i, year in enumerate(selected_years):
        d_vals = [v for v in discrete_results[year] if v < 99]
        c_vals = [v for v in continuous_results[year] if v < 99]
        if d_vals:
            bp1 = ax4.boxplot([d_vals], positions=[i-0.15], widths=0.25, patch_artist=True)
            bp1['boxes'][0].set_facecolor('lightblue')
        if c_vals:
            bp2 = ax4.boxplot([c_vals], positions=[i+0.15], widths=0.25, patch_artist=True)
            bp2['boxes'][0].set_facecolor('lightgreen')
    ax4.set_xticks(positions)
    ax4.set_xticklabels([f'Year {y}' for y in selected_years])
    ax4.set_ylabel('LR (excluding detections)')
    ax4.set_title('LR Distribution by Year')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/lr_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return discrete_results, continuous_results


def analyze_lr_formula_difference():
    """Analyze the difference in how LR is calculated."""

    print("\n" + "="*70)
    print("LR FORMULA ANALYSIS")
    print("="*70)

    mean_detection_time_100 = 6.95
    mean_detection_time_1000 = 3.42
    variance = 3.88

    # Calculate detection constants (same in both models)
    A, B = compute_detection_constants(mean_detection_time_100, mean_detection_time_1000)
    print(f"\nDetection constants: A={A:.4f}, B={B:.4f}")

    labor = 10500
    mu = A / (np.log10(labor) ** B)
    k = mu / variance
    print(f"\nFor labor={labor}:")
    print(f"  mu (mean detection time) = {mu:.4f}")
    print(f"  k (gamma shape) = {k:.4f}")
    print(f"  theta (gamma scale) = {variance:.4f}")

    # Calculate survival function at different times
    print(f"\nSurvival function S(t) = 1 - CDF(t):")
    for t in [0, 1, 2, 3, 4, 5, 6, 7]:
        survival = stats.gamma.sf(t, a=k, scale=variance)
        print(f"  Year {t}: S({t}) = {survival:.6f}")

    print("\nKEY INSIGHT:")
    print("  Discrete model: LR = P(not detected by year | project exists) = S(year)")
    print("  This represents: 'probability of this specific observation pattern'")
    print("  Given project exists AND not detected, the observation is 'not detected by year t'")
    print("  The LR is P(this observation | exists) / P(this observation | not exists)")
    print("  Since P(not detected | not exists) = 1.0 (always), LR = S(t)")


def run_all_tests():
    """Run comprehensive alignment tests."""

    # Test 1: Standard case
    print("\n" + "="*80)
    print("TEST 1: STANDARD CONFIGURATION")
    print("="*80)
    test_lr_functions_comparison(test_name="Standard Configuration")

    # Test 2: Lower labor (harder to detect)
    print("\n" + "="*80)
    print("TEST 2: LOWER LABOR (HARDER TO DETECT)")
    print("="*80)
    low_labor = {i: 1000 + i * 100 for i in range(8)}
    test_lr_functions_comparison(
        labor_by_year=low_labor,
        test_name="Low Labor Configuration"
    )

    # Test 3: Very high labor (easy to detect)
    print("\n" + "="*80)
    print("TEST 3: HIGH LABOR (EASY TO DETECT)")
    print("="*80)
    high_labor = {i: 50000 + i * 5000 for i in range(8)}
    test_lr_functions_comparison(
        labor_by_year=high_labor,
        test_name="High Labor Configuration"
    )

    # Test 4: Variable labor (large swings)
    print("\n" + "="*80)
    print("TEST 4: HIGHLY VARIABLE LABOR")
    print("="*80)
    variable_labor = {
        0: 5000,
        1: 15000,
        2: 8000,
        3: 20000,
        4: 10000,
        5: 25000,
        6: 12000,
        7: 30000,
    }
    test_lr_functions_comparison(
        labor_by_year=variable_labor,
        test_name="Variable Labor Configuration"
    )

    # Test 5: Different variance parameter
    print("\n" + "="*80)
    print("TEST 5: DIFFERENT VARIANCE PARAMETER")
    print("="*80)
    test_lr_functions_comparison(
        variance=1.5,
        test_name="Lower Variance (theta=1.5)"
    )

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
    print("\nSUMMARY: If all LR BY YEAR values match between discrete and continuous")
    print("models across all tests, the alignment is successful.")


if __name__ == "__main__":
    # Run full test suite
    run_all_tests()
    analyze_lr_formula_difference()
