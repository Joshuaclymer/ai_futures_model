"""
Comprehensive alignment check: Compare all key metrics against reference API.

This script compares the continuous ODE model outputs against the reference API
without needing the discrete model's pymetalog dependency.

Key metrics to check:
1. Initial black project compute (surviving over time)
2. Fab production (already validated in validate_fab_alignment.py)
3. Total black project compute
4. Datacenter capacity
5. Survival rates
6. Operating compute (powered by datacenter)
"""

import sys
import json
import urllib.request
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai_futures_simulator"))

from world_updaters.compute.chip_survival import (
    calculate_survival_rate,
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


def estimate_hazard_params(survival_data: List[float], years: List[float]) -> Tuple[float, float]:
    """Estimate h0 and h1 from survival curve."""
    H_data, t_data = [], []
    for s, t in zip(survival_data, years):
        if 0.01 < s < 0.99 and t > years[0]:
            H_data.append(-math.log(s))
            t_data.append(t - years[0])
    if len(H_data) < 3:
        return 0.01, 0.0035
    A = np.column_stack([t_data, [t**2/2 for t in t_data]])
    result = np.linalg.lstsq(A, H_data, rcond=None)[0]
    return max(0.001, result[0]), max(0.0, result[1])


def simulate_initial_compute(years, h0, h1, initial_compute):
    """Simulate initial diverted compute with attrition (single cohort)."""
    results = []
    start_year = years[0]
    for year in years:
        t = year - start_year
        survival = math.exp(-(h0 * t + h1 * t**2 / 2))
        results.append(initial_compute * survival)
    return results


def simulate_fab_compute_ode(years, dt, annual_production, h0, h1, fab_operational_year):
    """Simulate fab compute using ODE model."""
    results = []
    C_fab = 0.0
    a_fab = 0.0
    for year in years:
        F = annual_production if year >= fab_operational_year else 0.0
        dC = calculate_compute_derivative(C_fab, a_fab, F, h0, h1)
        da = calculate_average_age_derivative(C_fab, a_fab, F)
        C_fab = max(0.0, C_fab + dC * dt)
        a_fab = max(0.0, a_fab + da * dt)
        results.append(C_fab)
    return results


def simulate_datacenter_capacity(years, construction_rate_gw_per_year, unconcealed_fraction, start_year):
    """Simulate datacenter capacity growth."""
    results = []
    for year in years:
        t = year - start_year
        if t < 0:
            t = 0
        concealed = construction_rate_gw_per_year * t
        unconcealed = concealed * unconcealed_fraction / (1 - unconcealed_fraction) if unconcealed_fraction < 1 else 0
        total = concealed + unconcealed
        results.append(total)
    return results


def check_metric_alignment(name, continuous, reference, threshold=0.15):
    """Check if metric is aligned within threshold."""
    if not reference or not continuous:
        return None, None, "NO_DATA"

    # Find last valid comparison point
    for i in range(min(len(continuous), len(reference)) - 1, -1, -1):
        c, r = continuous[i], reference[i]
        if r > 100:  # Meaningful value
            diff = (c - r) / r
            status = "✓ ALIGNED" if abs(diff) < threshold else "⚠ MISALIGNED"
            return c, r, f"{status} ({diff*100:+.1f}%)"
    return None, None, "NO_VALID_DATA"


def main():
    print("=" * 70)
    print("COMPREHENSIVE METRICS ALIGNMENT CHECK")
    print("=" * 70)

    # Get reference API data
    print("\n1. Fetching reference API data...")
    ref_data = call_reference_api(num_samples=100)
    if not ref_data:
        print("   Failed to get reference data")
        return

    # Extract data
    black_fab = ref_data.get('black_fab', {})
    black_project_model = ref_data.get('black_project_model', {})
    black_datacenters = ref_data.get('black_datacenters', {})
    initial_stock = ref_data.get('initial_stock', {})

    ref_years = black_project_model.get('years', [])
    dt = ref_years[1] - ref_years[0] if len(ref_years) > 1 else 0.1

    # Reference metrics
    ref_initial_bp = black_project_model.get('initial_black_project', {}).get('median', [])
    ref_fab_flow = black_project_model.get('black_fab_flow', {}).get('median', [])
    ref_total_bp = black_project_model.get('total_black_project', {}).get('median', [])
    ref_survival = black_project_model.get('survival_rate', {}).get('median', [])
    ref_dc_capacity = black_datacenters.get('datacenter_capacity', {}).get('median', [])
    ref_operational = black_project_model.get('operational_compute', {}).get('median', [])

    # Parameters from reference
    wafer_starts = black_fab.get('wafer_starts', {}).get('median', [0])[0]
    transistor_density = black_fab.get('transistor_density', {}).get('median', [0])[0]
    arch_eff = black_fab.get('architecture_efficiency', {}).get('median', [0])[0]
    chips_per_wafer = black_fab.get('chips_per_wafer', {}).get('median', [0])[0]
    ref_is_operational = black_fab.get('is_operational', {}).get('proportion', [])

    h100e_per_chip = transistor_density * arch_eff
    annual_production = wafer_starts * chips_per_wafer * h100e_per_chip * 12

    # Estimate hazard parameters from reference survival
    h0, h1 = estimate_hazard_params(ref_survival, ref_years)

    # Find fab operational year
    fab_op_idx = next((i for i, p in enumerate(ref_is_operational) if p >= 0.5), len(ref_is_operational)//3)
    fab_operational_year = ref_years[fab_op_idx] if fab_op_idx < len(ref_years) else 2032.0

    # Initial compute (from reference)
    initial_compute = ref_initial_bp[0] if ref_initial_bp else 250000

    print(f"\n2. Reference parameters:")
    print(f"   Years: {ref_years[0]:.1f} to {ref_years[-1]:.1f}")
    print(f"   Time step: {dt:.2f} years")
    print(f"   Hazard params: h0={h0:.4f}, h1={h1:.6f}")
    print(f"   Fab operational: {fab_operational_year:.1f}")
    print(f"   Annual fab production: {annual_production:,.0f} H100e/year")
    print(f"   Initial black project: {initial_compute:,.0f} H100e")

    # Run continuous model simulations
    print("\n3. Running continuous model simulations...")

    # Initial black project compute (single cohort with attrition)
    cont_initial_bp = simulate_initial_compute(ref_years, h0, h1, initial_compute)

    # Fab compute (ODE model)
    cont_fab = simulate_fab_compute_ode(ref_years, dt, annual_production, h0, h1, fab_operational_year)

    # Total black project = initial + fab
    cont_total_bp = [i + f for i, f in zip(cont_initial_bp, cont_fab)]

    # Datacenter capacity (linear growth model)
    # Reference shows ~1.3 GW at start, ~10 GW at end over 7 years
    # construction_rate ≈ (10 - 1.3) / 7 ≈ 1.24 GW/year
    dc_start = ref_dc_capacity[0] if ref_dc_capacity else 1.3
    dc_end = ref_dc_capacity[-1] if ref_dc_capacity else 10.0
    construction_rate = (dc_end - dc_start) / (ref_years[-1] - ref_years[0])
    cont_dc_capacity = [dc_start + construction_rate * (y - ref_years[0]) for y in ref_years]

    # Operating compute (limited by datacenter capacity)
    # Assume 700W per H100e
    watts_per_h100e = 700
    cont_operational = []
    for total, dc in zip(cont_total_bp, cont_dc_capacity):
        max_powered = dc * 1e9 / watts_per_h100e if dc > 0 else 0
        cont_operational.append(min(total, max_powered))

    # Survival rate comparison (direct)
    cont_survival = [math.exp(-(h0 * (y - ref_years[0]) + h1 * (y - ref_years[0])**2 / 2)) for y in ref_years]

    # Results comparison
    print("\n" + "=" * 70)
    print("ALIGNMENT RESULTS")
    print("=" * 70)

    metrics = [
        ("Initial Black Project (surviving)", cont_initial_bp, ref_initial_bp),
        ("Fab Production (cumulative)", cont_fab, ref_fab_flow),
        ("Total Black Project", cont_total_bp, ref_total_bp),
        ("Datacenter Capacity (GW)", cont_dc_capacity, ref_dc_capacity),
        ("Survival Rate", cont_survival, ref_survival),
    ]

    print(f"\n{'Metric':<35} {'Continuous':<15} {'Reference':<15} {'Status':<20}")
    print("-" * 85)

    alignment_results = {}
    for name, cont, ref in metrics:
        c_val, r_val, status = check_metric_alignment(name, cont, ref)
        if c_val is not None:
            if "GW" in name or "Rate" in name:
                print(f"{name:<35} {c_val:<15.2f} {r_val:<15.2f} {status:<20}")
            else:
                print(f"{name:<35} {c_val:<15,.0f} {r_val:<15,.0f} {status:<20}")
        else:
            print(f"{name:<35} {'N/A':<15} {'N/A':<15} {status:<20}")
        alignment_results[name] = status

    # Detailed year-by-year comparison for key metrics
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON AT KEY TIME POINTS")
    print("=" * 70)

    key_indices = [0, len(ref_years)//4, len(ref_years)//2, 3*len(ref_years)//4, len(ref_years)-1]

    print("\n--- Total Black Project Compute ---")
    print(f"{'Year':<10} {'Continuous':<15} {'Reference':<15} {'Diff %':<12}")
    for idx in key_indices:
        if idx < len(ref_years) and idx < len(ref_total_bp):
            c = cont_total_bp[idx]
            r = ref_total_bp[idx]
            diff = 100 * (c - r) / r if r > 100 else 0
            print(f"{ref_years[idx]:<10.1f} {c:<15,.0f} {r:<15,.0f} {diff:<12.1f}")

    print("\n--- Initial Black Project (with attrition) ---")
    print(f"{'Year':<10} {'Continuous':<15} {'Reference':<15} {'Diff %':<12}")
    for idx in key_indices:
        if idx < len(ref_years) and idx < len(ref_initial_bp):
            c = cont_initial_bp[idx]
            r = ref_initial_bp[idx]
            diff = 100 * (c - r) / r if r > 100 else 0
            print(f"{ref_years[idx]:<10.1f} {c:<15,.0f} {r:<15,.0f} {diff:<12.1f}")

    print("\n--- Datacenter Capacity (GW) ---")
    print(f"{'Year':<10} {'Continuous':<15} {'Reference':<15} {'Diff %':<12}")
    for idx in key_indices:
        if idx < len(ref_years) and idx < len(ref_dc_capacity):
            c = cont_dc_capacity[idx]
            r = ref_dc_capacity[idx]
            diff = 100 * (c - r) / r if r > 0.1 else 0
            print(f"{ref_years[idx]:<10.1f} {c:<15.2f} {r:<15.2f} {diff:<12.1f}")

    # Create summary plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Initial black project
    ax1 = axes[0, 0]
    ax1.plot(ref_years, cont_initial_bp, 'b-', label='Continuous', linewidth=2)
    ax1.plot(ref_years[:len(ref_initial_bp)], ref_initial_bp, 'g--', label='Reference', linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('H100e')
    ax1.set_title('Initial Black Project (surviving)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(5,5))

    # Plot 2: Fab production
    ax2 = axes[0, 1]
    ax2.plot(ref_years, cont_fab, 'b-', label='Continuous', linewidth=2)
    ax2.plot(ref_years[:len(ref_fab_flow)], ref_fab_flow, 'g--', label='Reference', linewidth=2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('H100e')
    ax2.set_title('Fab Production (cumulative)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(6,6))

    # Plot 3: Total black project
    ax3 = axes[0, 2]
    ax3.plot(ref_years, cont_total_bp, 'b-', label='Continuous', linewidth=2)
    ax3.plot(ref_years[:len(ref_total_bp)], ref_total_bp, 'g--', label='Reference', linewidth=2)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('H100e')
    ax3.set_title('Total Black Project')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(6,6))

    # Plot 4: Datacenter capacity
    ax4 = axes[1, 0]
    ax4.plot(ref_years, cont_dc_capacity, 'b-', label='Continuous', linewidth=2)
    ax4.plot(ref_years[:len(ref_dc_capacity)], ref_dc_capacity, 'g--', label='Reference', linewidth=2)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('GW')
    ax4.set_title('Datacenter Capacity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Survival rate
    ax5 = axes[1, 1]
    ax5.plot(ref_years, cont_survival, 'b-', label='Continuous', linewidth=2)
    ax5.plot(ref_years[:len(ref_survival)], ref_survival, 'g--', label='Reference', linewidth=2)
    ax5.set_xlabel('Year')
    ax5.set_ylabel('Survival Rate')
    ax5.set_title('Survival Rate')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Percentage differences
    ax6 = axes[1, 2]
    # Total BP diff
    total_diff = [100 * (c - r) / r if r > 1000 else 0 for c, r in zip(cont_total_bp, ref_total_bp[:len(cont_total_bp)])]
    ax6.plot(ref_years[:len(total_diff)], total_diff, 'b-', label='Total BP', linewidth=2)
    # Initial BP diff
    initial_diff = [100 * (c - r) / r if r > 1000 else 0 for c, r in zip(cont_initial_bp, ref_initial_bp[:len(cont_initial_bp)])]
    ax6.plot(ref_years[:len(initial_diff)], initial_diff, 'r-', label='Initial BP', linewidth=2)
    ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax6.axhline(y=10, color='orange', linestyle='--', alpha=0.5)
    ax6.axhline(y=-10, color='orange', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Year')
    ax6.set_ylabel('% Difference')
    ax6.set_title('Percentage Differences')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/all_metrics_alignment.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n4. Plot saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    aligned = sum(1 for s in alignment_results.values() if "ALIGNED" in s and "MIS" not in s)
    total = len(alignment_results)
    print(f"\nMetrics aligned: {aligned}/{total}")

    if aligned == total:
        print("\n✓ ALL METRICS ARE ALIGNED")
    else:
        misaligned = [k for k, v in alignment_results.items() if "MISALIGNED" in v]
        if misaligned:
            print(f"\n⚠ Metrics needing attention: {', '.join(misaligned)}")


if __name__ == "__main__":
    main()
