"""
Compare cumulative fab compute production between models with ALIGNED parameters.

This script compares the discrete reference model values from the API against
the continuous ODE model, using matched parameters to identify core algorithm
differences.

Key alignment points from reference API response:
- wafer_starts: 9175 wafers/month (median)
- architecture_efficiency: 5.24 (= 1.23^8 for year 2030)
- transistor_density: 0.055 (28nm process)
- chips_per_wafer: 28
- h100e_per_chip = 0.055 * 5.24 = 0.288
- Monthly production = 9175 * 28 * 0.288 = ~74K H100e/month
- black_fab_flow: cumulative fab production (with attrition)
"""

import sys
import json
import urllib.request
import numpy as np
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai_futures_simulator"))

from world_updaters.compute.chip_survival import (
    calculate_compute_derivative,
    calculate_average_age_derivative,
)


def call_reference_api(num_samples: int = 50, start_year: int = 2029, total_labor: int = 11300, timeout: int = 180):
    """Call the reference model API and return results."""
    print(f"Calling reference API...")
    url = 'https://dark-compute.onrender.com/run_simulation'
    data = json.dumps({
        'num_samples': num_samples,
        'start_year': start_year,
        'total_labor': total_labor,
    }).encode()

    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode())
            elapsed = time.time() - start_time
            print(f"  API response received in {elapsed:.1f}s")
            return result
    except Exception as e:
        print(f"  ERROR calling API: {e}")
        return None


class ContinuousODEModel:
    """Simulates the continuous ODE model's fab production with attrition."""

    def __init__(
        self,
        wafer_starts_per_month: float,
        chips_per_wafer: float,
        h100e_per_chip: float,
        initial_hazard_rate: float,
        hazard_rate_increase_per_year: float,
        fab_operational_year: float,
    ):
        self.wafer_starts_per_month = wafer_starts_per_month
        self.chips_per_wafer = chips_per_wafer
        self.h100e_per_chip = h100e_per_chip
        self.initial_hazard_rate = initial_hazard_rate
        self.hazard_rate_increase_per_year = hazard_rate_increase_per_year
        self.fab_operational_year = fab_operational_year

        # Annual production rate in H100e
        self.annual_production_h100e = wafer_starts_per_month * chips_per_wafer * h100e_per_chip * 12.0

    def is_operational(self, year: float) -> bool:
        return year >= self.fab_operational_year

    def run_simulation(self, years: list, dt: float) -> dict:
        """Run simulation using ODE integration and return results."""
        results = {
            'years': [],
            'fab_cumulative_raw': [],  # Integral of production rate (no attrition)
            'fab_cumulative_surviving': [],  # ODE state variable (with attrition)
            'fab_average_age': [],
        }

        # ODE state variables
        C_fab = 0.0  # Functional fab compute
        a_fab = 0.0  # Average age of fab chips
        cumulative_raw = 0.0

        for year in years:
            results['years'].append(year)

            if self.is_operational(year):
                F = self.annual_production_h100e  # Production rate

                # Compute derivatives using the actual survival module
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
            else:
                # Fab not operational - still apply attrition to existing compute
                if C_fab > 0:
                    dC_dt = calculate_compute_derivative(
                        functional_compute=C_fab,
                        average_age=a_fab,
                        production_rate=0.0,  # No production
                        initial_hazard_rate=self.initial_hazard_rate,
                        hazard_rate_increase_per_year=self.hazard_rate_increase_per_year,
                    )
                    da_dt = calculate_average_age_derivative(
                        functional_compute=C_fab,
                        average_age=a_fab,
                        production_rate=0.0,
                    )
                    C_fab += dC_dt * dt
                    a_fab += da_dt * dt
                    C_fab = max(0.0, C_fab)
                    a_fab = max(0.0, a_fab)

            results['fab_cumulative_raw'].append(cumulative_raw)
            results['fab_cumulative_surviving'].append(C_fab)
            results['fab_average_age'].append(a_fab)

        return results


class DiscreteCohortModel:
    """Simulates the discrete cohort model (matches reference model logic)."""

    def __init__(
        self,
        wafer_starts_per_month: float,
        chips_per_wafer: float,
        h100e_per_chip: float,
        initial_hazard_rate: float,
        hazard_rate_increase_per_year: float,
        fab_operational_year: float,
    ):
        self.wafer_starts_per_month = wafer_starts_per_month
        self.chips_per_wafer = chips_per_wafer
        self.h100e_per_chip = h100e_per_chip
        self.initial_hazard_rate = initial_hazard_rate
        self.hazard_rate_increase_per_year = hazard_rate_increase_per_year
        self.fab_operational_year = fab_operational_year

        # Monthly production rate in H100e
        self.monthly_production_h100e = wafer_starts_per_month * chips_per_wafer * h100e_per_chip

        # Track compute added by year (cohorts)
        self.compute_by_year = {}

    def is_operational(self, year: float) -> bool:
        return year >= self.fab_operational_year

    def run_simulation(self, years: list, dt: float) -> dict:
        """Run simulation and return results."""
        results = {
            'years': [],
            'fab_cumulative_raw': [],  # Without attrition
            'fab_cumulative_surviving': [],  # With cohort-based attrition
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

            results['fab_cumulative_raw'].append(cumulative_raw)

            # Calculate surviving compute using cohort-based survival
            surviving = self._calculate_surviving_compute(year)
            results['fab_cumulative_surviving'].append(surviving)

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


def main():
    print("=" * 70)
    print("ALIGNED COMPARISON: FAB CUMULATIVE COMPUTE PRODUCTION")
    print("=" * 70)

    # Get reference API data
    ref_data = call_reference_api(num_samples=100, start_year=2029, total_labor=11300)

    if not ref_data:
        print("Failed to get reference data")
        return

    # Extract reference fab data
    black_fab = ref_data.get('black_fab', {})
    black_project_model = ref_data.get('black_project_model', {})

    # Get reference years and fab flow
    ref_years = black_project_model.get('years', [])
    ref_fab_flow = black_project_model.get('black_fab_flow', {}).get('median', [])

    # Get reference parameters
    ref_wafer_starts = black_fab.get('wafer_starts', {}).get('median', [0])[0]
    ref_transistor_density = black_fab.get('transistor_density', {}).get('median', [0])[0]
    ref_arch_eff = black_fab.get('architecture_efficiency', {}).get('median', [0])[0]
    ref_chips_per_wafer = black_fab.get('chips_per_wafer', {}).get('median', [0])[0]
    ref_is_operational = black_fab.get('is_operational', {}).get('proportion', [])

    print(f"\nReference model parameters (from API):")
    print(f"  wafer_starts: {ref_wafer_starts:.1f} wafers/month")
    print(f"  transistor_density: {ref_transistor_density:.4f}")
    print(f"  architecture_efficiency: {ref_arch_eff:.4f}")
    print(f"  chips_per_wafer: {ref_chips_per_wafer:.0f}")

    # Calculate h100e per chip
    h100e_per_chip = ref_transistor_density * ref_arch_eff
    print(f"  h100e_per_chip: {h100e_per_chip:.4f}")

    # Calculate production rate
    monthly_production = ref_wafer_starts * ref_chips_per_wafer * h100e_per_chip
    annual_production = monthly_production * 12
    print(f"  monthly_production: {monthly_production:.0f} H100e/month")
    print(f"  annual_production: {annual_production:.0f} H100e/year")

    # Determine when fab becomes operational from reference data
    fab_operational_idx = next((i for i, p in enumerate(ref_is_operational) if p > 0), len(ref_is_operational) // 3)
    fab_operational_year = ref_years[fab_operational_idx] if fab_operational_idx < len(ref_years) else 2032.0
    print(f"  fab_operational_year (from API): ~{fab_operational_year:.1f}")

    # Reference survival parameters (from black_project_model)
    ref_survival = black_project_model.get('survival_rate', {}).get('median', [])
    if ref_survival and len(ref_survival) > 10:
        # Estimate hazard rates from survival curve
        # S(t) = exp(-h0*t - h1*t^2/2)
        # At t=1: S(1) = exp(-h0 - h1/2)
        # Approximate using reference values
        print(f"  survival_rate at year 0: {ref_survival[0]:.4f}")
        print(f"  survival_rate at year 5: {ref_survival[min(50, len(ref_survival)-1)]:.4f}")

    # Use reference-aligned parameters
    wafer_starts = ref_wafer_starts
    chips_per_wafer = int(ref_chips_per_wafer)
    h100e_per_chip = ref_transistor_density * ref_arch_eff

    # Survival parameters - use typical values (these are sampled in the reference model)
    initial_hazard_rate = 0.05  # 5% per year base rate
    hazard_rate_increase = 0.02  # 2% per year increase

    # Simulation parameters
    dt = 0.1  # Match reference API time step
    years = ref_years.copy()

    print(f"\n  Simulation using dt={dt} years")
    print(f"  Time range: {years[0]} to {years[-1]}")
    print(f"  Number of time points: {len(years)}")

    # Run continuous ODE model
    ode_model = ContinuousODEModel(
        wafer_starts_per_month=wafer_starts,
        chips_per_wafer=chips_per_wafer,
        h100e_per_chip=h100e_per_chip,
        initial_hazard_rate=initial_hazard_rate,
        hazard_rate_increase_per_year=hazard_rate_increase,
        fab_operational_year=fab_operational_year,
    )
    ode_results = ode_model.run_simulation(years, dt)

    # Run discrete cohort model (should match reference logic)
    cohort_model = DiscreteCohortModel(
        wafer_starts_per_month=wafer_starts,
        chips_per_wafer=chips_per_wafer,
        h100e_per_chip=h100e_per_chip,
        initial_hazard_rate=initial_hazard_rate,
        hazard_rate_increase_per_year=hazard_rate_increase,
        fab_operational_year=fab_operational_year,
    )
    cohort_results = cohort_model.run_simulation(years, dt)

    # Print comparison at key years
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    key_indices = [0, len(years)//4, len(years)//2, 3*len(years)//4, len(years)-1]

    print("\n--- Raw Cumulative Production (no attrition) ---")
    print(f"{'Year':<10} {'ODE':<15} {'Cohort':<15}")
    for idx in key_indices:
        if idx < len(years):
            ode_raw = ode_results['fab_cumulative_raw'][idx]
            cohort_raw = cohort_results['fab_cumulative_raw'][idx]
            print(f"{years[idx]:<10.1f} {ode_raw:<15,.0f} {cohort_raw:<15,.0f}")

    print("\n--- Surviving Cumulative Production (with attrition) ---")
    print(f"{'Year':<10} {'ODE':<15} {'Cohort':<15} {'Reference':<15} {'ODE vs Ref %':<12}")
    for idx in key_indices:
        if idx < len(years) and idx < len(ref_fab_flow):
            ode_surv = ode_results['fab_cumulative_surviving'][idx]
            cohort_surv = cohort_results['fab_cumulative_surviving'][idx]
            ref_surv = ref_fab_flow[idx]
            diff_pct = 100 * (ode_surv - ref_surv) / ref_surv if ref_surv > 100 else 0
            print(f"{years[idx]:<10.1f} {ode_surv:<15,.0f} {cohort_surv:<15,.0f} {ref_surv:<15,.0f} {diff_pct:<12.1f}")

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Raw cumulative (ODE vs Cohort)
    ax1 = axes[0, 0]
    ax1.plot(years, ode_results['fab_cumulative_raw'], 'b-', label='ODE Model', linewidth=2)
    ax1.plot(years, cohort_results['fab_cumulative_raw'], 'r--', label='Cohort Model', linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Cumulative H100e')
    ax1.set_title('Raw Cumulative Production (no attrition)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=fab_operational_year, color='g', linestyle=':', alpha=0.7, label='Fab operational')

    # Plot 2: Surviving (all three models)
    ax2 = axes[0, 1]
    ax2.plot(years, ode_results['fab_cumulative_surviving'], 'b-', label='ODE Model', linewidth=2)
    ax2.plot(years, cohort_results['fab_cumulative_surviving'], 'r--', label='Cohort Model', linewidth=2)
    if ref_fab_flow:
        ax2.plot(ref_years[:len(ref_fab_flow)], ref_fab_flow, 'g-', label='Reference API', linewidth=2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Surviving H100e')
    ax2.set_title('Surviving Fab Compute (with attrition)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=fab_operational_year, color='purple', linestyle=':', alpha=0.7)

    # Plot 3: Percentage difference (ODE vs Reference)
    ax3 = axes[1, 0]
    if ref_fab_flow:
        pct_diff_ref = []
        for i in range(min(len(years), len(ref_fab_flow))):
            ode_val = ode_results['fab_cumulative_surviving'][i]
            ref_val = ref_fab_flow[i]
            if ref_val > 100:
                pct_diff_ref.append(100 * (ode_val - ref_val) / ref_val)
            else:
                pct_diff_ref.append(0)
        ax3.plot(years[:len(pct_diff_ref)], pct_diff_ref, 'g-', linewidth=2, label='ODE vs Reference')

    pct_diff_cohort = []
    for i in range(len(years)):
        ode_val = ode_results['fab_cumulative_surviving'][i]
        cohort_val = cohort_results['fab_cumulative_surviving'][i]
        if cohort_val > 100:
            pct_diff_cohort.append(100 * (ode_val - cohort_val) / cohort_val)
        else:
            pct_diff_cohort.append(0)
    ax3.plot(years, pct_diff_cohort, 'r--', linewidth=2, label='ODE vs Cohort')

    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('% Difference')
    ax3.set_title('Percentage Difference')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Average age (ODE model)
    ax4 = axes[1, 1]
    ax4.plot(years, ode_results['fab_average_age'], 'b-', linewidth=2)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Average Age (years)')
    ax4.set_title('ODE Model: Average Chip Age')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/fab_aligned_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    final_idx = len(years) - 1
    if final_idx < len(ref_fab_flow):
        ode_final = ode_results['fab_cumulative_surviving'][final_idx]
        cohort_final = cohort_results['fab_cumulative_surviving'][final_idx]
        ref_final = ref_fab_flow[final_idx]

        print(f"\nFinal year ({years[final_idx]:.1f}):")
        print(f"  ODE model: {ode_final:,.0f} H100e")
        print(f"  Cohort model: {cohort_final:,.0f} H100e")
        print(f"  Reference API: {ref_final:,.0f} H100e")
        print(f"  ODE vs Reference: {100 * (ode_final - ref_final) / ref_final:.1f}%")
        print(f"  ODE vs Cohort: {100 * (ode_final - cohort_final) / cohort_final:.1f}%")

        if abs(100 * (ode_final - ref_final) / ref_final) > 10:
            print("\n⚠️  SIGNIFICANT DIFFERENCE between ODE and Reference")
            print("  This may indicate:")
            print("  1. Different attrition models (cohort vs average-age ODE)")
            print("  2. Different hazard rate parameters")
            print("  3. Different production rate calculations")
        else:
            print("\n✓ ODE model is reasonably aligned with Reference (<10% difference)")


if __name__ == "__main__":
    main()
