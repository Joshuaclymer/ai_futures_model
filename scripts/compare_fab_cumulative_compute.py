"""
Compare cumulative fab compute production between discrete and continuous models.

This script compares the TOTAL compute produced by black fabs over time,
without the pymetalog dependency issues. It uses deterministic parameters
to isolate the mathematical differences between the models.

Key differences to investigate:
1. Production rate calculation (should be aligned)
2. Attrition model (discrete: cohort-based, continuous: average-age ODE)
3. Cumulative compute tracking
"""

import sys
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')

# Import from continuous model
from world_updaters.compute.black_compute import (
    calculate_fab_construction_duration,
    calculate_fab_wafer_starts_per_month,
    calculate_fab_h100e_per_chip,
    calculate_fab_annual_production_h100e,
)
from world_updaters.compute.chip_survival import (
    calculate_survival_rate,
    calculate_compute_derivative,
    calculate_average_age_derivative,
    calculate_hazard_rate,
)


class DiscreteModelSimulator:
    """Simulates the discrete model's fab production and attrition."""

    def __init__(
        self,
        wafer_starts_per_month: float,
        chips_per_wafer: float,
        h100e_per_chip: float,
        initial_hazard_rate: float,
        hazard_rate_increase_per_year: float,
        fab_construction_duration: float,
        preparation_start_year: float,
    ):
        self.wafer_starts_per_month = wafer_starts_per_month
        self.chips_per_wafer = chips_per_wafer
        self.h100e_per_chip = h100e_per_chip
        self.initial_hazard_rate = initial_hazard_rate
        self.hazard_rate_increase_per_year = hazard_rate_increase_per_year
        self.fab_construction_duration = fab_construction_duration
        self.preparation_start_year = preparation_start_year
        self.fab_operational_year = preparation_start_year + fab_construction_duration

        # Monthly production rate in H100e
        self.monthly_production_h100e = wafer_starts_per_month * chips_per_wafer * h100e_per_chip

        # Track compute added by year (cohorts)
        self.compute_by_year = {}  # year -> h100e added that year

    def is_operational(self, year: float) -> bool:
        return year >= self.fab_operational_year

    def run_simulation(self, years: list, dt: float) -> dict:
        """Run simulation and return results."""
        results = {
            'years': [],
            'fab_monthly_production': [],
            'fab_cumulative_raw': [],  # Without attrition
            'fab_cumulative_surviving': [],  # With attrition
            'total_surviving': [],
        }

        cumulative_raw = 0.0

        for year in years:
            results['years'].append(year)

            if self.is_operational(year):
                # Add production for this timestep
                production_this_step = self.monthly_production_h100e * 12 * dt

                # Store by year for cohort-based survival calculation
                if year not in self.compute_by_year:
                    self.compute_by_year[year] = 0.0
                self.compute_by_year[year] += production_this_step

                cumulative_raw += production_this_step
                results['fab_monthly_production'].append(self.monthly_production_h100e)
            else:
                results['fab_monthly_production'].append(0.0)

            results['fab_cumulative_raw'].append(cumulative_raw)

            # Calculate surviving compute using cohort-based survival
            surviving = self._calculate_surviving_compute(year)
            results['fab_cumulative_surviving'].append(surviving)
            results['total_surviving'].append(surviving)

        return results

    def _calculate_surviving_compute(self, current_year: float) -> float:
        """Calculate surviving fab compute at current_year using cohort model."""
        total_surviving = 0.0

        for year_added, h100e_added in self.compute_by_year.items():
            years_of_life = current_year - year_added
            if years_of_life < 0:
                continue

            # Cumulative hazard: H(t) = h0*t + h1*t^2/2
            cumulative_hazard = (
                self.initial_hazard_rate * years_of_life +
                self.hazard_rate_increase_per_year * years_of_life**2 / 2
            )
            survival_rate = math.exp(-cumulative_hazard)
            total_surviving += h100e_added * survival_rate

        return total_surviving


class ContinuousModelSimulator:
    """Simulates the continuous ODE model's fab production and attrition."""

    def __init__(
        self,
        wafer_starts_per_month: float,
        chips_per_wafer: float,
        h100e_per_chip: float,
        initial_hazard_rate: float,
        hazard_rate_increase_per_year: float,
        fab_construction_duration: float,
        preparation_start_year: float,
    ):
        self.wafer_starts_per_month = wafer_starts_per_month
        self.chips_per_wafer = chips_per_wafer
        self.h100e_per_chip = h100e_per_chip
        self.initial_hazard_rate = initial_hazard_rate
        self.hazard_rate_increase_per_year = hazard_rate_increase_per_year
        self.fab_construction_duration = fab_construction_duration
        self.preparation_start_year = preparation_start_year
        self.fab_operational_year = preparation_start_year + fab_construction_duration

        # Annual production rate in H100e
        self.annual_production_h100e = wafer_starts_per_month * chips_per_wafer * h100e_per_chip * 12.0

    def is_operational(self, year: float) -> bool:
        return year >= self.fab_operational_year

    def run_simulation(self, years: list, dt: float) -> dict:
        """Run simulation using ODE integration and return results."""
        results = {
            'years': [],
            'fab_monthly_production': [],
            'fab_cumulative_raw': [],  # Integral of production rate
            'fab_cumulative_surviving': [],  # ODE state variable
            'fab_average_age': [],
            'total_surviving': [],
        }

        # ODE state variables
        C_fab = 0.0  # Functional fab compute
        a_fab = 0.0  # Average age of fab chips
        cumulative_raw = 0.0

        for year in years:
            results['years'].append(year)

            if self.is_operational(year):
                F = self.annual_production_h100e  # Production rate

                # Compute derivatives
                dC_dt = calculate_compute_derivative(
                    functional_compute=C_fab,
                    average_age=a_fab,
                    production_rate=F,
                    initial_hazard_rate=self.initial_hazard_rate,
                    hazard_rate_increase_per_year=self.hazard_rate_increase_per_year,
                )

                da_dt = calculate_average_age_derivative(
                    functional_compute=C_fab,
                    average_age=a_fab,
                    production_rate=F,
                )

                # Euler integration
                C_fab += dC_dt * dt
                a_fab += da_dt * dt

                # Ensure non-negative
                C_fab = max(0.0, C_fab)
                a_fab = max(0.0, a_fab)

                cumulative_raw += F * dt
                results['fab_monthly_production'].append(F / 12.0)
            else:
                results['fab_monthly_production'].append(0.0)

            results['fab_cumulative_raw'].append(cumulative_raw)
            results['fab_cumulative_surviving'].append(C_fab)
            results['fab_average_age'].append(a_fab)
            results['total_surviving'].append(C_fab)

        return results


def run_comparison():
    """Compare discrete and continuous models."""
    print("=" * 70)
    print("COMPARING BLACK FAB CUMULATIVE COMPUTE PRODUCTION")
    print("Discrete (cohort-based) vs Continuous (average-age ODE)")
    print("=" * 70)

    # Common parameters (deterministic)
    wafer_starts_per_month = 5000.0  # wafers/month
    chips_per_wafer = 28.0  # chips per wafer
    h100e_per_chip = 0.15  # H100e per chip (28nm node at ~2030)
    initial_hazard_rate = 0.05  # 5% per year
    hazard_rate_increase_per_year = 0.02  # 2% per year increase
    fab_construction_duration = 2.5  # years
    preparation_start_year = 2029.0

    # Simulation parameters
    start_year = 2029.0
    end_year = 2040.0
    dt = 0.25  # quarterly timesteps

    years = np.arange(start_year, end_year + dt, dt).tolist()

    print(f"\nParameters:")
    print(f"  wafer_starts_per_month: {wafer_starts_per_month}")
    print(f"  chips_per_wafer: {chips_per_wafer}")
    print(f"  h100e_per_chip: {h100e_per_chip:.4f}")
    print(f"  Monthly production: {wafer_starts_per_month * chips_per_wafer * h100e_per_chip:.1f} H100e/month")
    print(f"  Annual production: {wafer_starts_per_month * chips_per_wafer * h100e_per_chip * 12:.1f} H100e/year")
    print(f"  initial_hazard_rate: {initial_hazard_rate}")
    print(f"  hazard_rate_increase_per_year: {hazard_rate_increase_per_year}")
    print(f"  fab_construction_duration: {fab_construction_duration} years")
    print(f"  fab_operational_year: {preparation_start_year + fab_construction_duration}")
    print(f"  dt: {dt} years")

    # Run discrete model
    discrete = DiscreteModelSimulator(
        wafer_starts_per_month=wafer_starts_per_month,
        chips_per_wafer=chips_per_wafer,
        h100e_per_chip=h100e_per_chip,
        initial_hazard_rate=initial_hazard_rate,
        hazard_rate_increase_per_year=hazard_rate_increase_per_year,
        fab_construction_duration=fab_construction_duration,
        preparation_start_year=preparation_start_year,
    )
    discrete_results = discrete.run_simulation(years, dt)

    # Run continuous model
    continuous = ContinuousModelSimulator(
        wafer_starts_per_month=wafer_starts_per_month,
        chips_per_wafer=chips_per_wafer,
        h100e_per_chip=h100e_per_chip,
        initial_hazard_rate=initial_hazard_rate,
        hazard_rate_increase_per_year=hazard_rate_increase_per_year,
        fab_construction_duration=fab_construction_duration,
        preparation_start_year=preparation_start_year,
    )
    continuous_results = continuous.run_simulation(years, dt)

    # Print comparison at key years
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    key_years = [2031.5, 2033, 2035, 2037, 2040]

    print("\n--- Raw Cumulative Production (no attrition) ---")
    print(f"{'Year':<10} {'Discrete':<15} {'Continuous':<15} {'Diff %':<10}")
    for key_year in key_years:
        idx = min(range(len(years)), key=lambda i: abs(years[i] - key_year))
        d_raw = discrete_results['fab_cumulative_raw'][idx]
        c_raw = continuous_results['fab_cumulative_raw'][idx]
        diff_pct = 100 * (c_raw - d_raw) / d_raw if d_raw > 0 else 0
        print(f"{years[idx]:<10.1f} {d_raw:<15.1f} {c_raw:<15.1f} {diff_pct:<10.2f}")

    print("\n--- Surviving Cumulative Production (with attrition) ---")
    print(f"{'Year':<10} {'Discrete':<15} {'Continuous':<15} {'Diff %':<10} {'Cont Avg Age':<12}")
    for key_year in key_years:
        idx = min(range(len(years)), key=lambda i: abs(years[i] - key_year))
        d_surv = discrete_results['fab_cumulative_surviving'][idx]
        c_surv = continuous_results['fab_cumulative_surviving'][idx]
        c_age = continuous_results['fab_average_age'][idx]
        diff_pct = 100 * (c_surv - d_surv) / d_surv if d_surv > 0 else 0
        print(f"{years[idx]:<10.1f} {d_surv:<15.1f} {c_surv:<15.1f} {diff_pct:<10.2f} {c_age:<12.2f}")

    # Calculate steady-state average age for constant production
    # At steady state: ā_ss = C/F (from dā/dt = 0)
    # The hazard rate at steady state is h(ā_ss)
    print("\n--- Steady State Analysis ---")
    F = wafer_starts_per_month * chips_per_wafer * h100e_per_chip * 12.0
    # At steady state, dC/dt = 0 means F = h(ā)*C, so C = F/h(ā)
    # And dā/dt = 0 means ā = C/F
    # Combined: ā = 1/h(ā) = 1/(h0 + h1*ā)
    # Solving: ā*(h0 + h1*ā) = 1
    # h1*ā² + h0*ā - 1 = 0
    # ā = (-h0 + sqrt(h0² + 4*h1)) / (2*h1)
    h0, h1 = initial_hazard_rate, hazard_rate_increase_per_year
    discriminant = h0**2 + 4*h1
    a_ss = (-h0 + math.sqrt(discriminant)) / (2*h1)
    C_ss = F / (h0 + h1*a_ss)
    print(f"  Steady-state average age: {a_ss:.2f} years")
    print(f"  Steady-state compute stock: {C_ss:.1f} H100e")
    print(f"  Steady-state hazard rate: {h0 + h1*a_ss:.4f}")

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Raw cumulative production
    ax1 = axes[0, 0]
    ax1.plot(discrete_results['years'], discrete_results['fab_cumulative_raw'], 'b-', label='Discrete', linewidth=2)
    ax1.plot(continuous_results['years'], continuous_results['fab_cumulative_raw'], 'r--', label='Continuous', linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Cumulative H100e')
    ax1.set_title('Raw Cumulative Fab Production (no attrition)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=preparation_start_year + fab_construction_duration, color='g', linestyle=':', label='Fab operational')

    # Plot 2: Surviving cumulative production
    ax2 = axes[0, 1]
    ax2.plot(discrete_results['years'], discrete_results['fab_cumulative_surviving'], 'b-', label='Discrete (cohort)', linewidth=2)
    ax2.plot(continuous_results['years'], continuous_results['fab_cumulative_surviving'], 'r--', label='Continuous (ODE)', linewidth=2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Surviving H100e')
    ax2.set_title('Surviving Fab Compute (with attrition)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=preparation_start_year + fab_construction_duration, color='g', linestyle=':')
    ax2.axhline(y=C_ss, color='purple', linestyle=':', alpha=0.5, label=f'Steady state: {C_ss:.0f}')

    # Plot 3: Percentage difference
    ax3 = axes[1, 0]
    pct_diff = []
    for i in range(len(years)):
        d_val = discrete_results['fab_cumulative_surviving'][i]
        c_val = continuous_results['fab_cumulative_surviving'][i]
        if d_val > 100:  # Only compute % diff when values are meaningful
            pct_diff.append(100 * (c_val - d_val) / d_val)
        else:
            pct_diff.append(0)
    ax3.plot(years, pct_diff, 'g-', linewidth=2)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('% Difference')
    ax3.set_title('Continuous vs Discrete % Difference (surviving)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Plot 4: Average age (continuous only)
    ax4 = axes[1, 1]
    ax4.plot(continuous_results['years'], continuous_results['fab_average_age'], 'r-', linewidth=2)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Average Age (years)')
    ax4.set_title('Continuous Model: Average Chip Age')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=a_ss, color='purple', linestyle=':', alpha=0.5, label=f'Steady state: {a_ss:.1f}y')
    ax4.legend()

    plt.tight_layout()
    output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/fab_cumulative_compute_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    final_idx = len(years) - 1
    d_final = discrete_results['fab_cumulative_surviving'][final_idx]
    c_final = continuous_results['fab_cumulative_surviving'][final_idx]
    final_diff = 100 * (c_final - d_final) / d_final if d_final > 0 else 0

    print(f"\nFinal year ({years[final_idx]}):")
    print(f"  Discrete surviving: {d_final:.1f} H100e")
    print(f"  Continuous surviving: {c_final:.1f} H100e")
    print(f"  Difference: {final_diff:.2f}%")

    if abs(final_diff) > 5:
        print("\n⚠️  SIGNIFICANT DIFFERENCE DETECTED")
        print("  The continuous ODE model diverges from the discrete cohort model.")
        print("  This may indicate a mathematical discrepancy that needs fixing.")
    else:
        print("\n✓ Models are well-aligned (difference < 5%)")

    return discrete_results, continuous_results


if __name__ == "__main__":
    run_comparison()
