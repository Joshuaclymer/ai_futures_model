"""
Compare cumulative fab compute production between continuous model and reference API.

This script:
1. Calls the reference model API at https://dark-compute.onrender.com/run_simulation
2. Runs the continuous ODE model simulation
3. Compares fab compute production metrics over time

Focus is specifically on fab-produced compute, not initial diverted compute.
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
)


def call_reference_api(
    num_samples: int = 50,
    start_year: int = 2029,
    total_labor: int = 11300,
    timeout: int = 180,
) -> Dict[str, Any]:
    """
    Call the reference model API and return results.

    Returns dict with simulation results from the discrete model.
    """
    print(f"Calling reference API at https://dark-compute.onrender.com/run_simulation")
    print(f"  num_samples={num_samples}, start_year={start_year}, total_labor={total_labor}")

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


def run_continuous_simulation(
    start_year: float = 2029.0,
    end_year: float = 2040.0,
    total_labor: int = 11300,
    num_simulations: int = 50,
    dt: float = 0.25,
) -> Dict[str, Any]:
    """
    Run the continuous ODE model simulation.

    Returns dict with simulation results for comparison.
    """
    print(f"\nRunning continuous ODE model simulation")
    print(f"  start_year={start_year}, end_year={end_year}, num_simulations={num_simulations}")

    # Parameters matching reference model defaults
    # Labor allocation fractions (from modal_parameters.yaml)
    frac_dc_construction = 0.885
    frac_fab_construction = 0.022
    frac_fab_operation = 0.049
    frac_ai_research = 0.044

    # Compute labor values
    dc_construction_labor = total_labor * frac_dc_construction
    fab_construction_labor = total_labor * frac_fab_construction
    fab_operating_labor = total_labor * frac_fab_operation
    ai_research_labor = total_labor * frac_ai_research

    # Fab parameters
    chips_per_wafer = 28.0
    wafers_per_month_per_worker = 24.64
    wafers_per_month_per_scanner = 1000.0
    process_node_nm = 28.0  # Default process node
    transistor_density_exponent = 1.49
    architecture_efficiency_improvement = 1.23

    # Localization and scanner parameters
    # Default: 5 scanners devoted to fab
    num_scanners = 5.0

    # Construction duration parameters
    construction_time_5k = 1.4
    construction_time_100k = 2.41
    construction_workers_per_1000_wafers = 14.1

    # Survival/attrition parameters (for fab-produced compute)
    initial_hazard_rate = 0.05
    hazard_rate_increase = 0.02

    # Calculate derived values
    wafer_starts = calculate_fab_wafer_starts_per_month(
        fab_operating_labor=fab_operating_labor,
        fab_number_of_lithography_scanners=num_scanners,
        wafers_per_month_per_worker=wafers_per_month_per_worker,
        wafers_per_month_per_scanner=wafers_per_month_per_scanner,
    )

    h100e_per_chip = calculate_fab_h100e_per_chip(
        fab_process_node_nm=process_node_nm,
        year=start_year,  # Fixed at construction start
        h100_reference_nm=4.0,
        transistor_density_scaling_exponent=transistor_density_exponent,
        architecture_efficiency_improvement_per_year=architecture_efficiency_improvement,
    )

    fab_construction_duration = calculate_fab_construction_duration(
        fab_construction_labor=fab_construction_labor,
        target_wafer_starts_per_month=wafer_starts,
        construction_time_for_5k_wafers=construction_time_5k,
        construction_time_for_100k_wafers=construction_time_100k,
        construction_workers_per_1000_wafers_per_month=construction_workers_per_1000_wafers,
    )

    annual_production = calculate_fab_annual_production_h100e(
        fab_wafer_starts_per_month=wafer_starts,
        fab_chips_per_wafer=chips_per_wafer,
        fab_h100e_per_chip=h100e_per_chip,
        fab_is_operational=True,
    )

    fab_operational_year = start_year + fab_construction_duration

    print(f"  Derived values:")
    print(f"    wafer_starts: {wafer_starts:.1f} wafers/month")
    print(f"    h100e_per_chip: {h100e_per_chip:.4f}")
    print(f"    fab_construction_duration: {fab_construction_duration:.2f} years")
    print(f"    fab_operational_year: {fab_operational_year:.2f}")
    print(f"    annual_production: {annual_production:.1f} H100e/year")

    # Run simulation
    years = np.arange(start_year, end_year + dt, dt).tolist()

    results = {
        'years': years,
        'years_since_start': [y - start_year for y in years],
        'fab_cumulative_raw': [],  # No attrition
        'fab_cumulative_surviving': [],  # With ODE attrition
        'fab_monthly_production': [],
        'fab_is_operational': [],
    }

    # ODE state variables
    C_fab = 0.0  # Functional fab compute
    a_fab = 0.0  # Average age of fab chips
    cumulative_raw = 0.0

    for year in years:
        if year >= fab_operational_year:
            F = annual_production

            # Compute derivatives
            dC_dt = calculate_compute_derivative(
                functional_compute=C_fab,
                average_age=a_fab,
                production_rate=F,
                initial_hazard_rate=initial_hazard_rate,
                hazard_rate_increase_per_year=hazard_rate_increase,
            )

            da_dt = calculate_average_age_derivative(
                functional_compute=C_fab,
                average_age=a_fab,
                production_rate=F,
            )

            # Euler integration
            C_fab += dC_dt * dt
            a_fab += da_dt * dt
            C_fab = max(0.0, C_fab)
            a_fab = max(0.0, a_fab)

            cumulative_raw += F * dt
            results['fab_monthly_production'].append(F / 12.0)
            results['fab_is_operational'].append(True)
        else:
            results['fab_monthly_production'].append(0.0)
            results['fab_is_operational'].append(False)

        results['fab_cumulative_raw'].append(cumulative_raw)
        results['fab_cumulative_surviving'].append(C_fab)

    return results


def compare_results(reference_data: Dict, continuous_data: Dict) -> None:
    """Compare results from reference API and continuous model."""

    print("\n" + "=" * 70)
    print("COMPARISON: REFERENCE API vs CONTINUOUS MODEL")
    print("=" * 70)

    # Extract relevant data from reference API response
    # The reference API returns data in a specific format - we need to parse it
    if reference_data is None:
        print("ERROR: Reference API data not available")
        return

    # Print structure of reference data for debugging
    print("\nReference API response keys:", list(reference_data.keys()))

    # The reference API returns percentile data for various metrics
    # Look for fab-related data
    black_project = reference_data.get('black_project_model', {})
    covert_fab = reference_data.get('covert_fab', {})

    if not black_project and not covert_fab:
        print("WARNING: No black project or covert fab data in reference response")
        return

    # Extract years array
    ref_years = black_project.get('years', [])
    if not ref_years:
        ref_years = reference_data.get('years', [])

    if not ref_years:
        print("WARNING: No years data found in reference response")
        return

    print(f"\nReference model years: {ref_years[:5]}...{ref_years[-3:]}")
    print(f"Continuous model years: {continuous_data['years'][:5]}...{continuous_data['years'][-3:]}")

    # Look for fab production data in reference response
    # Try different possible keys
    fab_prod_ref = None
    fab_prod_key = None

    # Check covert_fab section first
    if covert_fab:
        time_series = covert_fab.get('time_series_data', {})
        if time_series:
            h100e_flow = time_series.get('h100e_flow', {})
            if h100e_flow:
                fab_prod_ref = h100e_flow.get('median', [])
                fab_prod_key = 'covert_fab.time_series_data.h100e_flow.median'

    # Try black_project_model
    if not fab_prod_ref:
        bp_model = reference_data.get('black_project_model', {})
        if bp_model:
            # Look for fab-related fields
            for key in ['fab_cumulative_production_h100e', 'fab_production', 'covert_compute']:
                if key in bp_model:
                    data = bp_model[key]
                    if isinstance(data, dict):
                        fab_prod_ref = data.get('median', [])
                    else:
                        fab_prod_ref = data
                    fab_prod_key = f'black_project_model.{key}'
                    break

    # Try total_black_project as fallback
    if not fab_prod_ref:
        total_bp = black_project.get('total_black_project', {})
        if total_bp:
            fab_prod_ref = total_bp.get('median', [])
            fab_prod_key = 'black_project_model.total_black_project.median'

    if fab_prod_ref:
        print(f"\nFound fab production data at: {fab_prod_key}")
        print(f"Reference fab production (first 5): {fab_prod_ref[:5]}")
    else:
        print("\nWARNING: Could not find fab production data in reference response")
        print("Available keys in black_project_model:", list(black_project.keys()))
        print("Available keys in covert_fab:", list(covert_fab.keys()) if covert_fab else "None")

    # Compare at aligned time points
    cont_years = continuous_data['years']
    cont_fab_surviving = continuous_data['fab_cumulative_surviving']

    print("\n--- Continuous Model Fab Production ---")
    for i, year in enumerate(cont_years):
        if year in [2031.0, 2032.0, 2033.0, 2035.0, 2037.0, 2040.0]:
            print(f"  Year {year}: raw={continuous_data['fab_cumulative_raw'][i]:.1f}, surviving={cont_fab_surviving[i]:.1f} H100e")

    if fab_prod_ref:
        print("\n--- Reference Model Fab Production (from API) ---")
        for i, year in enumerate(ref_years):
            if year in [2031.0, 2032.0, 2033.0, 2035.0, 2037.0, 2040.0] and i < len(fab_prod_ref):
                print(f"  Year {year}: {fab_prod_ref[i]:.1f} H100e")

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Continuous model fab production
    ax1 = axes[0, 0]
    ax1.plot(cont_years, continuous_data['fab_cumulative_raw'], 'b-', label='Raw production', linewidth=2)
    ax1.plot(cont_years, cont_fab_surviving, 'r--', label='Surviving (with attrition)', linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Cumulative H100e')
    ax1.set_title('Continuous Model: Fab Cumulative Production')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reference model (if available)
    ax2 = axes[0, 1]
    if fab_prod_ref and len(fab_prod_ref) > 0:
        ax2.plot(ref_years[:len(fab_prod_ref)], fab_prod_ref, 'g-', label='Reference API', linewidth=2)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative H100e')
        ax2.set_title('Reference Model: Fab Production')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Reference data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Reference Model: Fab Production')

    # Plot 3: Comparison (if both available)
    ax3 = axes[1, 0]
    if fab_prod_ref and len(fab_prod_ref) > 0:
        # Align time points
        common_years = []
        cont_values = []
        ref_values = []

        for i, year in enumerate(cont_years):
            if year in ref_years:
                ref_idx = ref_years.index(year)
                if ref_idx < len(fab_prod_ref):
                    common_years.append(year)
                    cont_values.append(cont_fab_surviving[i])
                    ref_values.append(fab_prod_ref[ref_idx])

        if common_years:
            ax3.plot(common_years, cont_values, 'r--', label='Continuous (surviving)', linewidth=2)
            ax3.plot(common_years, ref_values, 'g-', label='Reference API', linewidth=2)
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Cumulative H100e')
            ax3.set_title('Comparison: Continuous vs Reference')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No common time points', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'Reference data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Comparison')

    # Plot 4: Percentage difference
    ax4 = axes[1, 1]
    if fab_prod_ref and common_years:
        pct_diff = []
        for c, r in zip(cont_values, ref_values):
            if r > 100:
                pct_diff.append(100 * (c - r) / r)
            else:
                pct_diff.append(0)
        ax4.plot(common_years, pct_diff, 'purple', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Year')
        ax4.set_ylabel('% Difference')
        ax4.set_title('(Continuous - Reference) / Reference %')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Cannot compute difference', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Percentage Difference')

    plt.tight_layout()
    output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/fab_vs_reference_api_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    print("=" * 70)
    print("COMPARING FAB COMPUTE: CONTINUOUS MODEL vs REFERENCE API")
    print("=" * 70)

    # Call reference API
    reference_data = call_reference_api(
        num_samples=30,
        start_year=2029,
        total_labor=11300,
        timeout=180,
    )

    # Run continuous model
    continuous_data = run_continuous_simulation(
        start_year=2029.0,
        end_year=2040.0,
        total_labor=11300,
        num_simulations=30,
        dt=0.25,
    )

    # Compare results
    compare_results(reference_data, continuous_data)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
