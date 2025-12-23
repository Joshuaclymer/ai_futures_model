"""
Compare fab production with CORRECTED parameters based on reference API survival rates.

Key observation: Reference API shows 92.7% survival at year 5, but my default hazard
parameters (h0=0.05, h1=0.02) give only ~60% survival. Need to use correct parameters.

Solving for hazard rates from survival data:
  survival = 0.9271 at t=5
  H(5) = -ln(0.9271) = 0.0757
  If H(t) = h0*t + h1*t²/2:
  5*h0 + 12.5*h1 = 0.0757

The reference model uses different (lower) hazard rates.
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
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai_futures_simulator"))

from world_updaters.compute.chip_survival import (
    calculate_compute_derivative,
    calculate_average_age_derivative,
)


def call_reference_api(num_samples: int = 100, start_year: int = 2029, total_labor: int = 11300, timeout: int = 180):
    """Call the reference model API and return results."""
    print(f"Calling reference API...")
    url = 'https://dark-compute.onrender.com/run_simulation'
    data = json.dumps({
        'num_samples': num_samples,
        'start_year': start_year,
        'total_labor': total_labor,
    }).encode()

    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode())
            print(f"  API response received")
            return result
    except Exception as e:
        print(f"  ERROR calling API: {e}")
        return None


def estimate_hazard_params_from_survival(survival_data: List[float], years: List[float]) -> Tuple[float, float]:
    """
    Estimate hazard rate parameters from survival curve data.

    The survival model is: S(t) = exp(-h0*t - h1*t²/2)
    Taking log: -ln(S(t)) = h0*t + h1*t²/2 = H(t)

    Fit h0 and h1 using least squares on the cumulative hazard.
    """
    if not survival_data or len(survival_data) < 5:
        return 0.01, 0.002  # Default values

    # Calculate cumulative hazard at each time point
    H_data = []
    t_data = []
    t2_data = []

    for i, (s, t) in enumerate(zip(survival_data, years)):
        if s > 0.01 and s < 0.99 and t > years[0]:  # Valid range
            H = -math.log(s)
            t_rel = t - years[0]
            H_data.append(H)
            t_data.append(t_rel)
            t2_data.append(t_rel**2 / 2)

    if len(H_data) < 3:
        return 0.01, 0.002

    # Least squares fit: H = h0*t + h1*t²/2
    # Using numpy for simple linear regression
    A = np.column_stack([t_data, t2_data])
    H = np.array(H_data)

    # Solve A @ [h0, h1]^T = H using least squares
    result, residuals, rank, s = np.linalg.lstsq(A, H, rcond=None)
    h0, h1 = result

    # Ensure non-negative
    h0 = max(0.001, h0)
    h1 = max(0.0, h1)

    return h0, h1


def find_fab_operational_year(is_operational: List[float], years: List[float]) -> float:
    """Find the year when fab first becomes operational (proportion > 0.5)."""
    for i, prop in enumerate(is_operational):
        if prop >= 0.5:
            return years[i] if i < len(years) else years[-1]
    # If never reaches 0.5, find first non-zero
    for i, prop in enumerate(is_operational):
        if prop > 0.1:
            return years[i] if i < len(years) else years[-1]
    return years[len(years)//3]  # Default to 1/3 through


class ContinuousODEModel:
    """Simulates the continuous ODE model with given parameters."""

    def __init__(
        self,
        annual_production_h100e: float,
        initial_hazard_rate: float,
        hazard_rate_increase_per_year: float,
        fab_operational_year: float,
    ):
        self.annual_production_h100e = annual_production_h100e
        self.initial_hazard_rate = initial_hazard_rate
        self.hazard_rate_increase_per_year = hazard_rate_increase_per_year
        self.fab_operational_year = fab_operational_year

    def run_simulation(self, years: list, dt: float) -> dict:
        """Run simulation using ODE integration."""
        results = {
            'years': [],
            'fab_cumulative_raw': [],
            'fab_cumulative_surviving': [],
            'fab_average_age': [],
        }

        C_fab = 0.0
        a_fab = 0.0
        cumulative_raw = 0.0

        for year in years:
            results['years'].append(year)

            is_operational = year >= self.fab_operational_year

            if is_operational:
                F = self.annual_production_h100e
                cumulative_raw += F * dt
            else:
                F = 0.0

            # Always compute derivatives (for attrition of existing stock)
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

            C_fab += dC_dt * dt
            a_fab += da_dt * dt
            C_fab = max(0.0, C_fab)
            a_fab = max(0.0, a_fab)

            results['fab_cumulative_raw'].append(cumulative_raw)
            results['fab_cumulative_surviving'].append(C_fab)
            results['fab_average_age'].append(a_fab)

        return results


class DiscreteCohortModel:
    """Simulates the discrete cohort model."""

    def __init__(
        self,
        annual_production_h100e: float,
        initial_hazard_rate: float,
        hazard_rate_increase_per_year: float,
        fab_operational_year: float,
    ):
        self.annual_production_h100e = annual_production_h100e
        self.initial_hazard_rate = initial_hazard_rate
        self.hazard_rate_increase_per_year = hazard_rate_increase_per_year
        self.fab_operational_year = fab_operational_year
        self.compute_by_year = {}

    def run_simulation(self, years: list, dt: float) -> dict:
        """Run simulation with cohort-based attrition."""
        results = {
            'years': [],
            'fab_cumulative_raw': [],
            'fab_cumulative_surviving': [],
        }

        cumulative_raw = 0.0

        for year in years:
            results['years'].append(year)

            if year >= self.fab_operational_year:
                production_this_step = self.annual_production_h100e * dt
                if year not in self.compute_by_year:
                    self.compute_by_year[year] = 0.0
                self.compute_by_year[year] += production_this_step
                cumulative_raw += production_this_step

            results['fab_cumulative_raw'].append(cumulative_raw)

            # Cohort-based survival
            total_surviving = 0.0
            for year_added, h100e_added in self.compute_by_year.items():
                years_of_life = year - year_added
                if years_of_life >= 0:
                    cumulative_hazard = (
                        self.initial_hazard_rate * years_of_life +
                        self.hazard_rate_increase_per_year * years_of_life**2 / 2
                    )
                    survival_rate = math.exp(-cumulative_hazard)
                    total_surviving += h100e_added * survival_rate

            results['fab_cumulative_surviving'].append(total_surviving)

        return results


def main():
    print("=" * 70)
    print("CORRECTED COMPARISON: FAB CUMULATIVE COMPUTE PRODUCTION")
    print("=" * 70)

    # Get reference API data
    ref_data = call_reference_api(num_samples=100, start_year=2029, total_labor=11300)

    if not ref_data:
        print("Failed to get reference data")
        return

    # Extract reference data
    black_fab = ref_data.get('black_fab', {})
    black_project_model = ref_data.get('black_project_model', {})

    ref_years = black_project_model.get('years', [])
    ref_fab_flow = black_project_model.get('black_fab_flow', {}).get('median', [])
    ref_survival = black_project_model.get('survival_rate', {}).get('median', [])
    ref_is_operational = black_fab.get('is_operational', {}).get('proportion', [])

    # Get production parameters
    ref_wafer_starts = black_fab.get('wafer_starts', {}).get('median', [0])[0]
    ref_transistor_density = black_fab.get('transistor_density', {}).get('median', [0])[0]
    ref_arch_eff = black_fab.get('architecture_efficiency', {}).get('median', [0])[0]
    ref_chips_per_wafer = black_fab.get('chips_per_wafer', {}).get('median', [0])[0]

    h100e_per_chip = ref_transistor_density * ref_arch_eff
    annual_production = ref_wafer_starts * ref_chips_per_wafer * h100e_per_chip * 12

    print(f"\nReference model parameters:")
    print(f"  wafer_starts: {ref_wafer_starts:.1f} wafers/month")
    print(f"  h100e_per_chip: {h100e_per_chip:.4f}")
    print(f"  annual_production: {annual_production:,.0f} H100e/year")

    # Estimate hazard parameters from reference survival curve
    h0_est, h1_est = estimate_hazard_params_from_survival(ref_survival, ref_years)
    print(f"\nEstimated hazard parameters from survival curve:")
    print(f"  h0 (initial hazard rate): {h0_est:.4f}")
    print(f"  h1 (hazard rate increase/year): {h1_est:.6f}")

    # Verify by computing survival at year 5
    t = 5.0
    H_5 = h0_est * t + h1_est * t**2 / 2
    survival_5 = math.exp(-H_5)
    print(f"  Predicted survival at year 5: {survival_5:.4f}")
    if len(ref_survival) > 50:
        print(f"  Actual survival at year 5: {ref_survival[50]:.4f}")

    # Find fab operational year
    fab_operational_year = find_fab_operational_year(ref_is_operational, ref_years)
    print(f"\nFab operational year (from API): {fab_operational_year:.2f}")

    # Check: Where does reference fab_flow start being significant?
    for i, flow in enumerate(ref_fab_flow):
        if flow > 100000:  # More than 100K H100e
            print(f"  Reference fab_flow > 100K at year {ref_years[i]:.1f}: {flow:,.0f}")
            break

    # Simulation parameters
    dt = ref_years[1] - ref_years[0] if len(ref_years) > 1 else 0.1
    years = ref_years.copy()

    print(f"\n  dt: {dt:.2f} years")
    print(f"  Time range: {years[0]:.1f} to {years[-1]:.1f}")

    # Run models with estimated hazard parameters
    ode_model = ContinuousODEModel(
        annual_production_h100e=annual_production,
        initial_hazard_rate=h0_est,
        hazard_rate_increase_per_year=h1_est,
        fab_operational_year=fab_operational_year,
    )
    ode_results = ode_model.run_simulation(years, dt)

    cohort_model = DiscreteCohortModel(
        annual_production_h100e=annual_production,
        initial_hazard_rate=h0_est,
        hazard_rate_increase_per_year=h1_est,
        fab_operational_year=fab_operational_year,
    )
    cohort_results = cohort_model.run_simulation(years, dt)

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON (with corrected hazard parameters)")
    print("=" * 70)

    key_indices = [0, len(years)//4, len(years)//2, 3*len(years)//4, len(years)-1]

    print("\n--- Surviving Cumulative Production ---")
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

    # Plot 1: All three models - surviving production
    ax1 = axes[0, 0]
    ax1.plot(years, ode_results['fab_cumulative_surviving'], 'b-', label='ODE Model', linewidth=2)
    ax1.plot(years, cohort_results['fab_cumulative_surviving'], 'r--', label='Cohort Model', linewidth=2)
    if ref_fab_flow:
        ax1.plot(ref_years[:len(ref_fab_flow)], ref_fab_flow, 'g-', label='Reference API', linewidth=2)
    ax1.axvline(x=fab_operational_year, color='purple', linestyle=':', alpha=0.7, label=f'Fab operational ({fab_operational_year:.1f})')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Surviving H100e')
    ax1.set_title('Surviving Fab Compute (with corrected hazard params)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(6,6))

    # Plot 2: Percentage difference
    ax2 = axes[0, 1]
    if ref_fab_flow:
        pct_diff = []
        for i in range(min(len(years), len(ref_fab_flow))):
            ode_val = ode_results['fab_cumulative_surviving'][i]
            ref_val = ref_fab_flow[i]
            if ref_val > 1000:
                pct_diff.append(100 * (ode_val - ref_val) / ref_val)
            else:
                pct_diff.append(None)
        # Filter out None values for plotting
        valid_years = [years[i] for i in range(len(pct_diff)) if pct_diff[i] is not None]
        valid_pct = [p for p in pct_diff if p is not None]
        ax2.plot(valid_years, valid_pct, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axhline(y=5, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=-5, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('% Difference (ODE - Reference)')
    ax2.set_title('Percentage Difference')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Reference survival rate vs model prediction
    ax3 = axes[1, 0]
    if ref_survival:
        ax3.plot(ref_years[:len(ref_survival)], ref_survival, 'g-', label='Reference API', linewidth=2)
    # Predict survival using estimated h0, h1
    predicted_survival = []
    for year in ref_years:
        t = year - ref_years[0]
        H = h0_est * t + h1_est * t**2 / 2
        predicted_survival.append(math.exp(-H))
    ax3.plot(ref_years, predicted_survival, 'b--', label='Model (fitted h0, h1)', linewidth=2)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Survival Rate')
    ax3.set_title('Survival Rate Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Fab operational proportion
    ax4 = axes[1, 1]
    if ref_is_operational:
        ax4.plot(ref_years[:len(ref_is_operational)], ref_is_operational, 'g-', label='Reference API', linewidth=2)
    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax4.axvline(x=fab_operational_year, color='purple', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Proportion Operational')
    ax4.set_title('Fab Operational Status')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/fab_corrected_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    final_idx = len(years) - 1
    if final_idx < len(ref_fab_flow):
        ode_final = ode_results['fab_cumulative_surviving'][final_idx]
        ref_final = ref_fab_flow[final_idx]
        diff_pct = 100 * (ode_final - ref_final) / ref_final if ref_final > 0 else 0

        print(f"\nFinal year ({years[final_idx]:.1f}):")
        print(f"  ODE model: {ode_final:,.0f} H100e")
        print(f"  Reference API: {ref_final:,.0f} H100e")
        print(f"  Difference: {diff_pct:.1f}%")

        if abs(diff_pct) < 5:
            print("\n✓ Models are well-aligned (<5% difference)")
        elif abs(diff_pct) < 10:
            print("\n✓ Models are reasonably aligned (<10% difference)")
        else:
            print("\n⚠️  Significant difference detected")
            print("   Possible causes:")
            print("   1. Different fab operational timing logic")
            print("   2. Reference model has stochastic variation not captured in median")
            print("   3. Different production calculation")


if __name__ == "__main__":
    main()
