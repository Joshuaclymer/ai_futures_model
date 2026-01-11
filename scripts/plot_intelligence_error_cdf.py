"""
Plot empirical and fitted CDF of intelligence error in estimating PRC compute stock.

The empirical data comes from historical US intelligence estimates vs ground truth values.
The fitted distribution is a Laplace (double-exponential) distribution where:
- The absolute relative error follows Exponential(k) with k = ln(2) / median_error
- The sign is randomly +/- with 50% probability

For the CDF of absolute error: F(x) = 1 - exp(-k*x) = 1 - 2^(-x/median_error)
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import numpy as np
import matplotlib.pyplot as plt

# Historical intelligence estimates vs ground truth
# From app_frontend/.../HistoricalCharts.tsx
estimates = np.array([700, 800, 900, 300, 1000, 50, 800, 441, 18, 1000, 600, 428, 287, 311, 208])
ground_truths = np.array([610, 280, 847, 0, 1308, 60, 819, 499, 5, 1027, 661, 348, 308, 248, 287])

# Compute empirical relative errors (absolute value)
# Only include cases where ground truth > 0
mask = ground_truths > 0
empirical_errors = np.abs((estimates[mask] - ground_truths[mask]) / ground_truths[mask])

# Sort for empirical CDF
empirical_errors_sorted = np.sort(empirical_errors)
n = len(empirical_errors_sorted)
empirical_cdf = np.arange(1, n + 1) / n

# Stated error bars data - these are the ranges intelligence agencies reported
# From app_frontend/.../HistoricalCharts.tsx
stated_error_bars = [
    {"min": 150, "max": 160},   # Nuclear Warheads 1984
    {"min": 140, "max": 157},   # Nuclear Warheads 1999
    {"min": 225, "max": 300},   # Nuclear Warheads 1984
    {"min": 60, "max": 80},     # Pakistani nuclear weapons 1999
    {"min": 25, "max": 35},     # Fissile material 1994
    {"min": 30, "max": 50},     # Fissile material 2007
    {"min": 17, "max": 33},     # Fissile material 1994
    {"min": 335, "max": 400},   # Fissile material 1998
    {"min": 330, "max": 580},   # Fissile material 1996
    {"min": 240, "max": 395},   # Fissile material 2000
    {"min": 10, "max": 25},     # ICBM launchers 1961
    {"min": 10, "max": 25},     # ICBM launchers 1961
    {"min": 105, "max": 120},   # ICBM launchers 1963
    {"min": 200, "max": 240},   # ICBM launchers 1964
    {"min": 180, "max": 190},   # Intercontinental missiles 2019
    {"min": 200, "max": 300},   # Intercontinental missiles 2025
]

# Compute stated error margins as half-width / midpoint = (max - min) / 2 / midpoint
# This represents the ± error bar (half the confidence interval)
# Simplifies to: (max - min) / (max + min)
stated_errors = np.array([(d["max"] - d["min"]) / (d["max"] + d["min"]) for d in stated_error_bars])

# Sort for empirical CDF
stated_errors_sorted = np.sort(stated_errors)
n_stated = len(stated_errors_sorted)
stated_cdf = np.arange(1, n_stated + 1) / n_stated

# Parameters for fitted distributions
median_error_empirical = np.median(empirical_errors)  # Actual empirical median
median_error_stated = np.median(stated_errors)  # Stated error median

# Fitted CDF for exponential distribution on absolute error
# CDF: F(x) = 1 - exp(-k*x) where k = ln(2) / median_error
# At x = median_error, F(median_error) = 0.5
x_fitted = np.linspace(0, max(empirical_errors_sorted.max(), stated_errors_sorted.max(), 1.0), 500)

def exponential_cdf(x, median_error):
    """CDF of exponential distribution parameterized by median."""
    k = np.log(2) / median_error
    return 1 - np.exp(-k * x)

fitted_cdf_empirical = exponential_cdf(x_fitted, median_error_empirical)
fitted_cdf_stated = exponential_cdf(x_fitted, median_error_stated)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Empirical errors (estimate vs ground truth)
ax.step(empirical_errors_sorted, empirical_cdf, where='post',
        linewidth=2, color='steelblue', label='Empirical error (estimate vs truth)')
ax.scatter(empirical_errors_sorted, empirical_cdf, s=30, color='steelblue', zorder=5)
ax.plot(x_fitted, fitted_cdf_empirical, '--', linewidth=2, color='steelblue', alpha=0.7,
        label=f'Fitted (median={median_error_empirical:.0%})')

# Stated error margins
ax.step(stated_errors_sorted, stated_cdf, where='post',
        linewidth=2, color='darkorange', label='Stated error margins')
ax.scatter(stated_errors_sorted, stated_cdf, s=30, color='darkorange', zorder=5)
ax.plot(x_fitted, fitted_cdf_stated, '--', linewidth=2, color='darkorange', alpha=0.7,
        label=f'Fitted (median={median_error_stated:.0%})')

# Mark the median line
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

# Labels and formatting
ax.set_xlabel('Relative Error', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('CDF of Intelligence Estimation Error\n(US estimates of adversary weapons/equipment stockpiles)', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, max(empirical_errors_sorted.max() * 1.1, stated_errors_sorted.max() * 1.1, 1.0))
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/intelligence_error_cdf.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also display summary statistics
print("\n=== Summary Statistics ===")
print(f"\nEmpirical errors (estimate vs ground truth):")
print(f"  Number of data points: {n}")
print(f"  Median error: {np.median(empirical_errors):.1%}")
print(f"  Mean error: {np.mean(empirical_errors):.1%}")
print(f"  Max error: {np.max(empirical_errors):.1%}")

print(f"\nStated error margins:")
print(f"  Number of data points: {n_stated}")
print(f"  Median: {np.median(stated_errors):.1%}")
print(f"  Mean: {np.mean(stated_errors):.1%}")
print(f"  Max: {np.max(stated_errors):.1%}")
