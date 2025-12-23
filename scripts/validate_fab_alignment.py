"""
Final validation: Compare fab compute production using YAML parameters.

This script validates that the continuous ODE model's fab production aligns
with the discrete reference model when using the same parameters.

Current modal_parameters.yaml values:
- initial_annual_hazard_rate: 0.01
- annual_hazard_rate_increase_per_year: 0.0035
- hazard_rate_multiplier: lognormal(ci80=[0.1, 6.0])
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai_futures_simulator"))

from world_updaters.compute.chip_survival import (
    calculate_compute_derivative,
    calculate_average_age_derivative,
)


def call_reference_api(num_samples: int = 100, start_year: int = 2029, total_labor: int = 11300, timeout: int = 180):
    """Call the reference model API."""
    url = 'https://dark-compute.onrender.com/run_simulation'
    data = json.dumps({
        'num_samples': num_samples,
        'start_year': start_year,
        'total_labor': total_labor,
    }).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"API error: {e}")
        return None


def run_ode_simulation(years, dt, annual_production, h0, h1, fab_operational_year):
    """Run ODE simulation for fab compute production."""
    results = {'years': [], 'raw': [], 'surviving': []}

    C_fab = 0.0  # Functional compute
    a_fab = 0.0  # Average age
    cumulative_raw = 0.0

    for year in years:
        results['years'].append(year)

        is_operational = year >= fab_operational_year
        F = annual_production if is_operational else 0.0

        if is_operational:
            cumulative_raw += F * dt

        # ODE integration
        dC_dt = calculate_compute_derivative(C_fab, a_fab, F, h0, h1)
        da_dt = calculate_average_age_derivative(C_fab, a_fab, F)

        C_fab = max(0.0, C_fab + dC_dt * dt)
        a_fab = max(0.0, a_fab + da_dt * dt)

        results['raw'].append(cumulative_raw)
        results['surviving'].append(C_fab)

    return results


def run_cohort_simulation(years, dt, annual_production, h0, h1, fab_operational_year):
    """Run cohort-based simulation (matches discrete reference model)."""
    results = {'years': [], 'raw': [], 'surviving': []}
    compute_by_year = {}
    cumulative_raw = 0.0

    for year in years:
        results['years'].append(year)

        if year >= fab_operational_year:
            production = annual_production * dt
            compute_by_year[year] = compute_by_year.get(year, 0) + production
            cumulative_raw += production

        results['raw'].append(cumulative_raw)

        # Cohort-based survival
        surviving = sum(
            h100e * math.exp(-(h0 * (year - y) + h1 * (year - y)**2 / 2))
            for y, h100e in compute_by_year.items()
            if year >= y
        )
        results['surviving'].append(surviving)

    return results


def main():
    print("=" * 70)
    print("VALIDATION: FAB COMPUTE PRODUCTION ALIGNMENT")
    print("=" * 70)

    # Get reference API data
    print("\n1. Fetching reference API data...")
    ref_data = call_reference_api()
    if not ref_data:
        print("   Failed to get reference data")
        return

    # Extract reference parameters and results
    black_fab = ref_data.get('black_fab', {})
    black_project_model = ref_data.get('black_project_model', {})

    ref_years = black_project_model.get('years', [])
    ref_fab_flow = black_project_model.get('black_fab_flow', {}).get('median', [])
    ref_survival = black_project_model.get('survival_rate', {}).get('median', [])
    ref_is_operational = black_fab.get('is_operational', {}).get('proportion', [])

    # Production parameters
    wafer_starts = black_fab.get('wafer_starts', {}).get('median', [0])[0]
    transistor_density = black_fab.get('transistor_density', {}).get('median', [0])[0]
    arch_eff = black_fab.get('architecture_efficiency', {}).get('median', [0])[0]
    chips_per_wafer = black_fab.get('chips_per_wafer', {}).get('median', [0])[0]

    h100e_per_chip = transistor_density * arch_eff
    annual_production = wafer_starts * chips_per_wafer * h100e_per_chip * 12

    # Find fab operational year (50% threshold)
    fab_op_idx = next((i for i, p in enumerate(ref_is_operational) if p >= 0.5), len(ref_is_operational)//3)
    fab_operational_year = ref_years[fab_op_idx] if fab_op_idx < len(ref_years) else 2032.0

    print(f"\n2. Reference model parameters:")
    print(f"   wafer_starts: {wafer_starts:.0f} wafers/month")
    print(f"   h100e_per_chip: {h100e_per_chip:.4f}")
    print(f"   annual_production: {annual_production:,.0f} H100e/year")
    print(f"   fab_operational_year: {fab_operational_year:.1f}")

    # Estimate hazard params from reference survival curve
    H_data, t_data = [], []
    for i, (s, t) in enumerate(zip(ref_survival, ref_years)):
        if 0.01 < s < 0.99 and t > ref_years[0]:
            H_data.append(-math.log(s))
            t_data.append(t - ref_years[0])
    if H_data:
        A = np.column_stack([t_data, [t**2/2 for t in t_data]])
        h0_est, h1_est = np.linalg.lstsq(A, H_data, rcond=None)[0]
        h0_est = max(0.001, h0_est)
        h1_est = max(0.0, h1_est)
    else:
        h0_est, h1_est = 0.01, 0.0035

    # YAML parameters
    h0_yaml = 0.01
    h1_yaml = 0.0035

    print(f"\n3. Hazard rate parameters:")
    print(f"   Estimated from reference: h0={h0_est:.4f}, h1={h1_est:.6f}")
    print(f"   YAML defaults: h0={h0_yaml:.4f}, h1={h1_yaml:.6f}")

    # Simulation
    dt = ref_years[1] - ref_years[0] if len(ref_years) > 1 else 0.1

    print(f"\n4. Running simulations...")

    # Run with estimated params (should match reference closely)
    ode_est = run_ode_simulation(ref_years, dt, annual_production, h0_est, h1_est, fab_operational_year)
    cohort_est = run_cohort_simulation(ref_years, dt, annual_production, h0_est, h1_est, fab_operational_year)

    # Run with YAML params
    ode_yaml = run_ode_simulation(ref_years, dt, annual_production, h0_yaml, h1_yaml, fab_operational_year)
    cohort_yaml = run_cohort_simulation(ref_years, dt, annual_production, h0_yaml, h1_yaml, fab_operational_year)

    # Results comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    key_indices = [0, len(ref_years)//4, len(ref_years)//2, 3*len(ref_years)//4, len(ref_years)-1]

    print("\n5. Surviving fab compute (with estimated hazard params):")
    print(f"{'Year':<10} {'ODE':<15} {'Cohort':<15} {'Reference':<15} {'ODE/Ref %':<12}")
    for idx in key_indices:
        if idx < len(ref_years) and idx < len(ref_fab_flow):
            year = ref_years[idx]
            ode = ode_est['surviving'][idx]
            cohort = cohort_est['surviving'][idx]
            ref = ref_fab_flow[idx]
            diff = 100 * (ode - ref) / ref if ref > 100 else 0
            print(f"{year:<10.1f} {ode:<15,.0f} {cohort:<15,.0f} {ref:<15,.0f} {diff:<12.1f}")

    print("\n6. Surviving fab compute (with YAML hazard params):")
    print(f"{'Year':<10} {'ODE':<15} {'Cohort':<15} {'Reference':<15} {'ODE/Ref %':<12}")
    for idx in key_indices:
        if idx < len(ref_years) and idx < len(ref_fab_flow):
            year = ref_years[idx]
            ode = ode_yaml['surviving'][idx]
            cohort = cohort_yaml['surviving'][idx]
            ref = ref_fab_flow[idx]
            diff = 100 * (ode - ref) / ref if ref > 100 else 0
            print(f"{year:<10.1f} {ode:<15,.0f} {cohort:<15,.0f} {ref:<15,.0f} {diff:<12.1f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All models comparison (estimated params)
    ax1 = axes[0, 0]
    ax1.plot(ref_years, ode_est['surviving'], 'b-', label='ODE (estimated h)', linewidth=2)
    ax1.plot(ref_years, cohort_est['surviving'], 'b--', label='Cohort (estimated h)', linewidth=1.5, alpha=0.7)
    ax1.plot(ref_years[:len(ref_fab_flow)], ref_fab_flow, 'g-', label='Reference API', linewidth=2)
    ax1.axvline(x=fab_operational_year, color='purple', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Surviving H100e')
    ax1.set_title('With Estimated Hazard Params')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(6,6))

    # Plot 2: All models comparison (YAML params)
    ax2 = axes[0, 1]
    ax2.plot(ref_years, ode_yaml['surviving'], 'r-', label='ODE (YAML h)', linewidth=2)
    ax2.plot(ref_years, cohort_yaml['surviving'], 'r--', label='Cohort (YAML h)', linewidth=1.5, alpha=0.7)
    ax2.plot(ref_years[:len(ref_fab_flow)], ref_fab_flow, 'g-', label='Reference API', linewidth=2)
    ax2.axvline(x=fab_operational_year, color='purple', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Surviving H100e')
    ax2.set_title('With YAML Hazard Params')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(6,6))

    # Plot 3: ODE vs Cohort difference
    ax3 = axes[1, 0]
    ode_cohort_diff_est = [100 * (o - c) / c if c > 100 else 0 for o, c in zip(ode_est['surviving'], cohort_est['surviving'])]
    ode_cohort_diff_yaml = [100 * (o - c) / c if c > 100 else 0 for o, c in zip(ode_yaml['surviving'], cohort_yaml['surviving'])]
    ax3.plot(ref_years, ode_cohort_diff_est, 'b-', label='Estimated h params', linewidth=2)
    ax3.plot(ref_years, ode_cohort_diff_yaml, 'r-', label='YAML h params', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('% Difference')
    ax3.set_title('ODE vs Cohort Model Difference')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: ODE vs Reference difference
    ax4 = axes[1, 1]
    ode_ref_diff_est = [100 * (ode_est['surviving'][i] - ref_fab_flow[i]) / ref_fab_flow[i]
                        if i < len(ref_fab_flow) and ref_fab_flow[i] > 1000 else 0
                        for i in range(len(ref_years))]
    ode_ref_diff_yaml = [100 * (ode_yaml['surviving'][i] - ref_fab_flow[i]) / ref_fab_flow[i]
                         if i < len(ref_fab_flow) and ref_fab_flow[i] > 1000 else 0
                         for i in range(len(ref_years))]
    ax4.plot(ref_years, ode_ref_diff_est, 'b-', label='Estimated h params', linewidth=2)
    ax4.plot(ref_years, ode_ref_diff_yaml, 'r-', label='YAML h params', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.5)
    ax4.axhline(y=-10, color='orange', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('% Difference')
    ax4.set_title('ODE vs Reference API Difference')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/fab_validation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n7. Plot saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    final_idx = len(ref_years) - 1
    if final_idx < len(ref_fab_flow):
        ode_final_est = ode_est['surviving'][final_idx]
        ode_final_yaml = ode_yaml['surviving'][final_idx]
        ref_final = ref_fab_flow[final_idx]

        diff_est = 100 * (ode_final_est - ref_final) / ref_final
        diff_yaml = 100 * (ode_final_yaml - ref_final) / ref_final

        print(f"\nFinal year ({ref_years[final_idx]:.1f}):")
        print(f"   Reference API: {ref_final:,.0f} H100e")
        print(f"   ODE (estimated h): {ode_final_est:,.0f} H100e ({diff_est:+.1f}%)")
        print(f"   ODE (YAML h): {ode_final_yaml:,.0f} H100e ({diff_yaml:+.1f}%)")

        print(f"\nAlignment assessment:")
        if abs(diff_est) < 10:
            print(f"   ✓ ODE with estimated h is well-aligned (<10% diff)")
        else:
            print(f"   ⚠ ODE with estimated h has {abs(diff_est):.1f}% diff")

        if abs(diff_yaml) < 15:
            print(f"   ✓ ODE with YAML h is reasonably aligned (<15% diff)")
        else:
            print(f"   ⚠ ODE with YAML h has {abs(diff_yaml):.1f}% diff")

        # ODE vs Cohort alignment
        cohort_final_yaml = cohort_yaml['surviving'][final_idx]
        ode_cohort_diff = 100 * (ode_final_yaml - cohort_final_yaml) / cohort_final_yaml
        print(f"\n   ODE vs Cohort (same params): {ode_cohort_diff:+.2f}%")
        print(f"   (This validates ODE approximation accuracy)")


if __name__ == "__main__":
    main()
