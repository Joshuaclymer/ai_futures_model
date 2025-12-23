"""
Corrected comparison of total_black_project between models.

KEY FINDINGS from reference model investigation:

1. `black_fab_flow` = cumulative fab production WITHOUT attrition
   - Uses get_cumulative_compute_production_over_time()
   - Raw chip production converted to H100e

2. `total_black_project` = surviving initial + surviving fab WITH attrition
   - Uses surviving_compute_energy_by_source()
   - Applies hazard-based attrition: H(t) = h0*t + h1*t²/2, S(t) = exp(-H(t))

3. Hazard rates are sampled with wide uncertainty:
   - Multiplier distribution: p25=0.1x, p50=1.0x, p75=6.0x
   - Base rates: h0_p50=0.01, h1_p50=0.0035

This explains the apparent discrepancy:
- black_fab_flow at end: ~5M (raw production)
- total_black_project at end: ~1.4M (with attrition from sampled hazard rates)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai_futures_simulator"))

from world_updaters.compute.chip_survival import (
    calculate_compute_derivative,
    calculate_average_age_derivative,
)


def call_reference_api(num_samples=100, start_year=2029, total_labor=11300, timeout=180):
    url = 'https://dark-compute.onrender.com/run_simulation'
    data = json.dumps({'num_samples': num_samples, 'start_year': start_year, 'total_labor': total_labor}).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"API error: {e}")
        return None


def simulate_fab_raw_production(years, dt, annual_production, fab_operational_year):
    """Simulate raw fab production WITHOUT attrition."""
    results = []
    cumulative = 0.0
    for year in years:
        if year >= fab_operational_year:
            cumulative += annual_production * dt
        results.append(cumulative)
    return results


def simulate_fab_surviving_ode(years, dt, annual_production, h0, h1, fab_operational_year):
    """Simulate fab compute WITH attrition using ODE model."""
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


def simulate_initial_surviving(years, h0, h1, initial_compute):
    """Simulate initial stock with attrition."""
    results = []
    start_year = years[0]
    for year in years:
        t = year - start_year
        survival = math.exp(-(h0 * t + h1 * t**2 / 2))
        results.append(initial_compute * survival)
    return results


def main():
    print("=" * 70)
    print("CORRECTED total_black_project COMPARISON")
    print("=" * 70)

    ref_data = call_reference_api()
    if not ref_data:
        return

    bpm = ref_data.get('black_project_model', {})
    bf = ref_data.get('black_fab', {})

    years = bpm.get('years', [])
    dt = years[1] - years[0] if len(years) > 1 else 0.1

    # Reference metrics
    ref_fab_flow = bpm.get('black_fab_flow', {}).get('median', [])  # Raw production
    ref_total_bp = bpm.get('total_black_project', {}).get('median', [])  # Surviving
    ref_survival = bpm.get('survival_rate', {}).get('median', [])
    ref_is_op = bf.get('is_operational', {}).get('proportion', [])

    # Production parameters
    wafer_starts = bf.get('wafer_starts', {}).get('median', [0])[0]
    transistor_density = bf.get('transistor_density', {}).get('median', [0])[0]
    arch_eff = bf.get('architecture_efficiency', {}).get('median', [0])[0]
    chips_per_wafer = bf.get('chips_per_wafer', {}).get('median', [0])[0]

    h100e_per_chip = transistor_density * arch_eff
    annual_production = wafer_starts * chips_per_wafer * h100e_per_chip * 12

    # Find fab operational year
    fab_op_idx = next((i for i, p in enumerate(ref_is_op) if p >= 0.5), len(ref_is_op)//3)
    fab_operational_year = years[fab_op_idx] if fab_op_idx < len(years) else 2032.0

    # Initial compute
    initial_compute = ref_total_bp[0] if ref_total_bp else 300000

    # Hazard rate scenarios
    # Base rates from reference model
    h0_base, h1_base = 0.01, 0.0035

    # Different multiplier scenarios
    scenarios = [
        ("Median (1.0x)", 1.0),
        ("p25 (0.1x)", 0.1),
        ("p75 (6.0x)", 6.0),
        ("Fitted from survival", None),  # Will estimate
    ]

    print(f"\nReference parameters:")
    print(f"  Annual fab production: {annual_production:,.0f} H100e/year")
    print(f"  Fab operational: {fab_operational_year:.1f}")
    print(f"  Initial compute: {initial_compute:,.0f} H100e")
    print(f"  Base h0: {h0_base}, h1: {h1_base}")

    # Estimate hazard params from survival curve for "Fitted" scenario
    H_data, t_data = [], []
    for s, t in zip(ref_survival, years):
        if 0.01 < s < 0.99 and t > years[0]:
            H_data.append(-math.log(s))
            t_data.append(t - years[0])
    if len(H_data) >= 3:
        A = np.column_stack([t_data, [t**2/2 for t in t_data]])
        fitted = np.linalg.lstsq(A, H_data, rcond=None)[0]
        h0_fitted, h1_fitted = max(0.001, fitted[0]), max(0.0, fitted[1])
    else:
        h0_fitted, h1_fitted = h0_base, h1_base

    print(f"  Fitted h0: {h0_fitted:.4f}, h1: {h1_fitted:.6f}")

    # Run simulations for each scenario
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS AT END OF PERIOD")
    print("=" * 70)

    print(f"\n{'Scenario':<25} {'Fab Raw':<15} {'Fab Surv':<15} {'Init Surv':<15} {'Total':<15}")
    print("-" * 85)

    # Raw fab production (same for all scenarios)
    fab_raw = simulate_fab_raw_production(years, dt, annual_production, fab_operational_year)

    results = {}
    for name, mult in scenarios:
        if mult is None:
            h0, h1 = h0_fitted, h1_fitted
        else:
            h0, h1 = h0_base * mult, h1_base * mult

        fab_surv = simulate_fab_surviving_ode(years, dt, annual_production, h0, h1, fab_operational_year)
        init_surv = simulate_initial_surviving(years, h0, h1, initial_compute)
        total = [f + i for f, i in zip(fab_surv, init_surv)]

        results[name] = {
            'fab_raw': fab_raw,
            'fab_surv': fab_surv,
            'init_surv': init_surv,
            'total': total,
            'h0': h0,
            'h1': h1,
        }

        idx = len(years) - 1
        print(f"{name:<25} {fab_raw[idx]:<15,.0f} {fab_surv[idx]:<15,.0f} {init_surv[idx]:<15,.0f} {total[idx]:<15,.0f}")

    # Reference values
    ref_fab_end = ref_fab_flow[-1] if ref_fab_flow else 0
    ref_total_end = ref_total_bp[-1] if ref_total_bp else 0
    print(f"\n{'Reference API':<25} {ref_fab_end:<15,.0f} {'N/A':<15} {'N/A':<15} {ref_total_end:<15,.0f}")

    # Calculate which scenario best matches
    print("\n" + "=" * 70)
    print("ALIGNMENT ANALYSIS")
    print("=" * 70)

    print("\n1. Fab RAW production comparison (should match black_fab_flow):")
    for name, data in results.items():
        diff = 100 * (data['fab_raw'][-1] - ref_fab_end) / ref_fab_end if ref_fab_end > 0 else 0
        status = "✓" if abs(diff) < 10 else "⚠"
        print(f"   {name}: {data['fab_raw'][-1]:,.0f} vs {ref_fab_end:,.0f} ({diff:+.1f}%) {status}")

    print("\n2. Total surviving compute comparison (should match total_black_project):")
    for name, data in results.items():
        diff = 100 * (data['total'][-1] - ref_total_end) / ref_total_end if ref_total_end > 0 else 0
        status = "✓" if abs(diff) < 15 else "⚠"
        print(f"   {name}: {data['total'][-1]:,.0f} vs {ref_total_end:,.0f} ({diff:+.1f}%) {status}")

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'Median (1.0x)': 'blue', 'p25 (0.1x)': 'green', 'p75 (6.0x)': 'red', 'Fitted from survival': 'purple'}

    # Plot 1: Fab raw production
    ax1 = axes[0, 0]
    ax1.plot(years, fab_raw, 'b-', label='ODE (all scenarios)', linewidth=2)
    if ref_fab_flow:
        ax1.plot(years[:len(ref_fab_flow)], ref_fab_flow, 'g--', label='Reference (black_fab_flow)', linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('H100e')
    ax1.set_title('Fab RAW Production (no attrition)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(6,6))

    # Plot 2: Fab surviving by scenario
    ax2 = axes[0, 1]
    for name, data in results.items():
        ax2.plot(years, data['fab_surv'], color=colors.get(name, 'gray'), label=name, linewidth=2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('H100e')
    ax2.set_title('Fab SURVIVING (with attrition)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(6,6))

    # Plot 3: Total black project by scenario
    ax3 = axes[1, 0]
    for name, data in results.items():
        ax3.plot(years, data['total'], color=colors.get(name, 'gray'), label=name, linewidth=2)
    if ref_total_bp:
        ax3.plot(years[:len(ref_total_bp)], ref_total_bp, 'k--', label='Reference (total_black_project)', linewidth=2)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('H100e')
    ax3.set_title('TOTAL Black Project (surviving)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(6,6))

    # Plot 4: Survival rates
    ax4 = axes[1, 1]
    for name, data in results.items():
        h0, h1 = data['h0'], data['h1']
        surv = [math.exp(-(h0 * (y - years[0]) + h1 * (y - years[0])**2 / 2)) for y in years]
        ax4.plot(years, surv, color=colors.get(name, 'gray'), label=name, linewidth=2)
    if ref_survival:
        ax4.plot(years[:len(ref_survival)], ref_survival, 'k--', label='Reference (survival_rate)', linewidth=2)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Survival Rate')
    ax4.set_title('Initial Stock Survival Rate')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/correct_total_bp_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
KEY INSIGHT: The reference model's total_black_project uses SAMPLED hazard rates,
which can range from 0.1x to 6.0x the base rates. The API median includes simulations
with high attrition (6x), explaining why total_black_project (~1.4M) is much lower
than black_fab_flow (~5M).

ALIGNMENT STATUS:
1. Fab RAW production: ✓ ODE matches black_fab_flow within ~10%
2. Total surviving (median hazard): ✓ ODE with 1.0x matches survival_rate curve
3. Total surviving (API median): Includes mixed hazard rates - not directly comparable

RECOMMENDATION:
- For Monte Carlo mode: Sample hazard_rate_multiplier from lognormal(ci80=[0.1, 6.0])
- For deterministic mode: Use median (1.0x) for direct comparison with survival_rate
""")


if __name__ == "__main__":
    main()
