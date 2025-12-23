"""
Analyze reference API metrics to understand their definitions.

The goal is to understand what each metric represents and why there are
apparent inconsistencies between metrics.
"""

import sys
import json
import urllib.request
import numpy as np
from typing import Dict, Any


def call_reference_api(num_samples: int = 50, start_year: int = 2029, total_labor: int = 11300, timeout: int = 180):
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


def main():
    print("=" * 70)
    print("REFERENCE API METRICS ANALYSIS")
    print("=" * 70)

    ref_data = call_reference_api()
    if not ref_data:
        return

    bpm = ref_data.get('black_project_model', {})
    bf = ref_data.get('black_fab', {})
    bd = ref_data.get('black_datacenters', {})
    ibp = ref_data.get('initial_black_project', {})
    ist = ref_data.get('initial_stock', {})

    years = bpm.get('years', [])
    n = len(years)

    print(f"\nSimulation period: {years[0]:.1f} to {years[-1]:.1f} ({n} time points)")

    # Extract key metrics
    metrics = {
        # Black project model metrics
        'total_black_project': bpm.get('total_black_project', {}).get('median', []),
        'initial_black_project_bpm': bpm.get('initial_black_project', {}).get('median', []),
        'black_fab_flow': bpm.get('black_fab_flow', {}).get('median', []),
        'operational_compute_bpm': bpm.get('operational_compute', {}).get('median', []),
        'survival_rate': bpm.get('survival_rate', {}).get('median', []),

        # Black datacenters metrics
        'datacenter_capacity': bd.get('datacenter_capacity', {}).get('median', []),
        'operational_compute_bd': bd.get('operational_compute', {}).get('median', []),

        # Initial black project metrics
        'black_project_ibp': ibp.get('black_project', {}).get('median', []),
        'h100e_ibp': ibp.get('h100e', {}).get('median', []),
        'survival_rate_ibp': ibp.get('survival_rate', {}).get('median', []),
    }

    # Key time indices
    t0 = 0
    t_mid = n // 2
    t_end = n - 1

    print("\n" + "=" * 70)
    print("METRIC VALUES AT KEY TIME POINTS")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'t=0':<15} {'t=mid':<15} {'t=end':<15}")
    print("-" * 80)

    for name, values in metrics.items():
        if values and len(values) > t_end:
            v0 = values[t0]
            v_mid = values[t_mid]
            v_end = values[t_end]
            if v_end > 1000:
                print(f"{name:<35} {v0:<15,.0f} {v_mid:<15,.0f} {v_end:<15,.0f}")
            else:
                print(f"{name:<35} {v0:<15.4f} {v_mid:<15.4f} {v_end:<15.4f}")

    # Analysis: Check relationships
    print("\n" + "=" * 70)
    print("METRIC RELATIONSHIP ANALYSIS")
    print("=" * 70)

    total_bp = metrics['total_black_project']
    initial_bp = metrics['initial_black_project_bpm']
    fab_flow = metrics['black_fab_flow']
    survival = metrics['survival_rate']

    if total_bp and initial_bp:
        print("\n1. Is total_black_project == initial_black_project_bpm?")
        same = all(abs(t - i) < 1 for t, i in zip(total_bp[:10], initial_bp[:10]))
        print(f"   Answer: {'YES' if same else 'NO'}")
        if same:
            print("   → These two metrics are the SAME in the reference model")
            print("   → 'initial_black_project' likely means 'the black project that started from initial diversion'")
            print("   → NOT 'just the initial diverted compute'")

    if total_bp and fab_flow:
        print("\n2. Does total_black_project include fab production?")
        # If total = initial_surviving + fab_surviving, then total should grow as fab_flow grows
        total_growth = total_bp[t_end] - total_bp[t0]
        fab_end = fab_flow[t_end] if fab_flow else 0
        print(f"   Total growth from t=0 to t=end: {total_growth:,.0f} H100e")
        print(f"   Fab flow at t=end: {fab_end:,.0f} H100e")
        if fab_end > 0:
            ratio = total_growth / fab_end
            print(f"   Ratio (total_growth / fab_flow): {ratio:.2%}")
            if ratio < 0.5:
                print("   → Total growth is MUCH LESS than fab production")
                print("   → Suggests heavy attrition OR total_black_project doesn't include fab")

    if survival:
        print("\n3. Survival rate analysis:")
        print(f"   Survival at t=0: {survival[t0]:.4f}")
        print(f"   Survival at t=mid: {survival[t_mid]:.4f}")
        print(f"   Survival at t=end: {survival[t_end]:.4f}")

        # Calculate implied hazard rate
        if survival[t_end] > 0.01:
            t_elapsed = years[t_end] - years[t0]
            H_end = -np.log(survival[t_end])
            avg_hazard = H_end / t_elapsed
            print(f"   Implied average hazard rate: {avg_hazard:.4f} per year")

    # Check for "black_project_ibp" vs "total_black_project"
    bp_ibp = metrics['black_project_ibp']
    if bp_ibp and total_bp:
        print("\n4. initial_black_project.black_project vs total_black_project:")
        # Note: bp_ibp appears to be in kH100e based on values
        print(f"   ibp.black_project at t=0: {bp_ibp[t0]:.2f}")
        print(f"   total_black_project at t=0: {total_bp[t0]:,.0f}")
        ratio = total_bp[t0] / bp_ibp[t0] if bp_ibp[t0] > 0 else 0
        print(f"   Ratio: {ratio:.1f}")
        if abs(ratio - 1000) < 100:
            print("   → ibp.black_project is in kH100e (x1000)")

    # Check h100e_ibp (fab production in initial_black_project section)
    h100e_ibp = metrics['h100e_ibp']
    if h100e_ibp:
        print("\n5. initial_black_project.h100e (likely fab compute):")
        print(f"   At t=0: {h100e_ibp[t0]:.2f}")
        print(f"   At t=mid: {h100e_ibp[t_mid]:.2f}")
        print(f"   At t=end: {h100e_ibp[t_end]:.2f}")
        if fab_flow and fab_flow[t_end] > 0:
            ratio = h100e_ibp[t_end] / (fab_flow[t_end] / 1000) if fab_flow[t_end] > 0 else 0
            print(f"   Ratio to fab_flow/1000: {ratio:.2f}")

    # Compare black_fab_flow vs black_fab_flow_all_sims
    fab_flow_all = bpm.get('black_fab_flow_all_sims', {}).get('median', [])
    if fab_flow_all and fab_flow:
        print("\n6. black_fab_flow vs black_fab_flow_all_sims:")
        print(f"   black_fab_flow at t=end: {fab_flow[t_end]:,.0f}")
        print(f"   black_fab_flow_all_sims at t=end: {fab_flow_all[t_end]:,.0f}")
        print("   → 'all_sims' includes sims where fab wasn't built (zeros)")
        print("   → 'black_fab_flow' is only for sims where fab WAS built")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    print("""
1. 'total_black_project' and 'initial_black_project' appear to be the SAME metric
   - This naming is confusing but 'initial' likely means 'originating from initial diversion'
   - NOT 'the initial stock before attrition'

2. 'black_fab_flow' is cumulative fab production (for sims where fab was built)
   - 'black_fab_flow_all_sims' is the same but averaged across ALL sims

3. The survival_rate applies to initial stock (single cohort model)
   - Fab-produced compute has different survival characteristics
   - The ODE average-age model captures this differently

4. For alignment purposes:
   - Fab production: compare ODE fab_compute vs black_fab_flow ✓ (already aligned)
   - Initial stock: verify attrition model matches survival_rate
   - Total compute: need to understand how ref model combines initial + fab
""")


if __name__ == "__main__":
    main()
