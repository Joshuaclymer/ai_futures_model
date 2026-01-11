"""
Plot the fitted CDF for intelligence error in estimating PRC compute stock.

The distribution is exponential with median error = 5%.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# Parameter for compute stock estimation error
median_error = 0.05  # 5% median error

# Fitted CDF for exponential distribution
# CDF: F(x) = 1 - exp(-k*x) where k = ln(2) / median_error
x_fitted = np.linspace(0, 1.0, 500)

def exponential_cdf(x, median_error):
    """CDF of exponential distribution parameterized by median."""
    k = np.log(2) / median_error
    return 1 - np.exp(-k * x)

fitted_cdf = exponential_cdf(x_fitted, median_error)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Fitted CDF
ax.plot(x_fitted, fitted_cdf, '-', linewidth=2.5, color='steelblue',
        label=f'Exponential CDF (median={median_error:.0%})')

# Mark the median point
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=median_error, color='steelblue', linestyle=':', alpha=0.5)
ax.scatter([median_error], [0.5], s=80, color='steelblue', zorder=5)

# Add percentile annotations
for p in [0.25, 0.75, 0.90, 0.95]:
    x_p = -np.log(1 - p) * median_error / np.log(2)
    ax.plot([x_p, x_p], [0, p], 'k:', alpha=0.3)
    ax.plot([0, x_p], [p, p], 'k:', alpha=0.3)
    ax.annotate(f'{p:.0%}: {x_p:.1%}', xy=(x_p, p), xytext=(x_p + 0.02, p),
                fontsize=9, va='center')

# Labels and formatting
ax.set_xlabel('Absolute Relative Error', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('CDF of Intelligence Error in Estimating PRC Compute Stock', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/compute_error_cdf.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Display key percentiles
print("\n=== Key Percentiles ===")
for p in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    x_p = -np.log(1 - p) * median_error / np.log(2)
    print(f"  {p:>3.0%} of errors are below {x_p:.1%}")
