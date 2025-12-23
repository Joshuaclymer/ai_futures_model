#!/usr/bin/env python
"""
Direct comparison of ai_futures_simulator vs black_project_backend.

This script calls the underlying simulation functions of both Flask APIs
and compares their outputs to identify differences.

Usage:
    python scripts/direct_model_comparison.py
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add paths for both projects
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai_futures_simulator"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app_backend"))
sys.path.insert(0, "/Users/joshuaclymer/github/covert_compute_production_model")


def run_continuous_model(time_range=None, num_simulations=11):
    """
    Run the continuous model (ai_futures_simulator) via app_backend functions.

    Returns the same data structure that the /api/run-black-project-simulation endpoint returns.
    """
    print("=" * 60)
    print("RUNNING CONTINUOUS MODEL (ai_futures_simulator via app_backend)")
    print("=" * 60)

    from api_utils.black_project_simulation import (
        run_black_project_simulations,
        extract_black_project_plot_data,
    )

    time_range = time_range or [2029, 2037]

    # Run simulations
    simulation_results = run_black_project_simulations(
        frontend_params={},
        num_mc_simulations=num_simulations - 1,  # -1 because it adds 1 for central
        time_range=time_range,
    )

    # Extract plot data
    plot_data = extract_black_project_plot_data(
        simulation_results=simulation_results,
        frontend_params={},
    )

    return plot_data


def run_discrete_model(time_range=None, num_simulations=11):
    """
    Run the discrete model (black_project_backend) via its functions.

    Returns the same data structure that the /run_simulation endpoint returns.
    """
    print("\n" + "=" * 60)
    print("RUNNING DISCRETE MODEL (black_project_backend)")
    print("=" * 60)

    from black_project_backend.model import Model
    from black_project_backend.black_project_parameters import (
        ModelParameters,
        SimulationSettings,
        BlackProjectProperties,
        BlackProjectParameters,
    )
    from black_project_backend.format_data_for_black_project_plots import extract_plot_data

    time_range = time_range or [2029, 2037]
    agreement_year = time_range[0]
    end_year = time_range[1]

    # Create parameters matching the continuous model's defaults
    params = ModelParameters(
        simulation_settings=SimulationSettings(
            agreement_start_year=agreement_year,
            num_years_to_simulate=end_year - agreement_year,
            time_step_years=0.1,
            num_simulations=num_simulations,
        ),
        black_project_properties=BlackProjectProperties(
            run_a_black_project=True,
            build_a_black_fab=True,  # Match continuous model default
        ),
        black_project_parameters=BlackProjectParameters(),
    )

    # Run simulations
    model = Model(params)
    model.run_simulations(num_simulations=num_simulations)

    # Extract plot data
    plot_data = extract_plot_data(model, params)

    return plot_data


def compare_time_series(name, continuous_data, discrete_data, years_key='years'):
    """Compare time series data between models."""
    print(f"\n--- {name} ---")

    c_years = continuous_data.get(years_key, [])
    d_years = discrete_data.get(years_key, [])

    if not c_years or not d_years:
        print(f"  Missing years data")
        return

    # Get median values
    c_median = continuous_data.get('median', [])
    d_median = discrete_data.get('median', [])

    if not c_median or not d_median:
        print(f"  Missing median data")
        return

    print(f"{'Year':<10} {'Continuous':<15} {'Discrete':<15} {'Diff %':<10}")
    print("-" * 50)

    # Sample at a few key points
    sample_indices = [0, len(c_years)//4, len(c_years)//2, 3*len(c_years)//4, -1]

    for idx in sample_indices:
        if idx >= len(c_years) or idx >= len(c_median):
            continue
        if idx == -1:
            idx = len(c_years) - 1

        c_year = c_years[idx]
        c_val = c_median[idx]

        # Find closest discrete year
        d_idx = min(range(len(d_years)), key=lambda i: abs(d_years[i] - c_year))
        d_year = d_years[d_idx]
        d_val = d_median[d_idx] if d_idx < len(d_median) else 0

        if d_val != 0:
            diff_pct = (c_val - d_val) / abs(d_val) * 100
        else:
            diff_pct = float('inf') if c_val != 0 else 0

        print(f"{c_year:<10.2f} {c_val:<15.2f} {d_val:<15.2f} {diff_pct:<10.1f}%")


def compare_results(continuous, discrete):
    """Compare outputs from both models."""
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    # Compare black project model data
    c_bp = continuous.get('black_project_model', {})
    d_bp = discrete.get('black_project_model', {})

    if c_bp and d_bp:
        # Operational compute
        c_op = c_bp.get('operational_compute', {})
        d_op = d_bp.get('operational_compute', {})
        if c_op and d_op:
            c_op['years'] = c_bp.get('years', [])
            d_op['years'] = d_bp.get('years', [])
            compare_time_series("Operational Compute (H100e)", c_op, d_op)

        # Total black project compute
        c_total = c_bp.get('total_black_project', {})
        d_total = d_bp.get('total_black_project', {})
        if c_total and d_total:
            c_total['years'] = c_bp.get('years', [])
            d_total['years'] = d_bp.get('years', [])
            compare_time_series("Total Black Project Compute (H100e)", c_total, d_total)

        # Datacenter capacity
        c_dc = c_bp.get('datacenter_capacity', {})
        d_dc = d_bp.get('datacenter_capacity', {})
        if c_dc and d_dc:
            c_dc['years'] = c_bp.get('years', [])
            d_dc['years'] = d_bp.get('years', [])
            compare_time_series("Datacenter Capacity (GW)", c_dc, d_dc)

    # Compare datacenter specific data
    c_datacenter = continuous.get('black_datacenters', {})
    d_datacenter = discrete.get('black_datacenters', {})

    if c_datacenter and d_datacenter:
        c_cap = c_datacenter.get('datacenter_capacity', {})
        d_cap = d_datacenter.get('datacenter_capacity', {})
        if c_cap and d_cap:
            c_cap['years'] = c_datacenter.get('years', [])
            d_cap['years'] = d_datacenter.get('years', [])
            compare_time_series("Black Datacenter Capacity (detailed)", c_cap, d_cap)

    # Compare initial stock
    c_init = continuous.get('initial_stock', {})
    d_init = discrete.get('initial_stock', {})

    if c_init and d_init:
        print("\n--- Initial Stock Samples ---")
        c_samples = c_init.get('initial_compute_stock_samples', [])
        d_samples = d_init.get('initial_compute_stock_samples', [])

        if c_samples and d_samples:
            c_median = np.median(c_samples)
            d_median = np.median(d_samples)
            diff = (c_median - d_median) / d_median * 100 if d_median != 0 else float('inf')
            print(f"  Continuous median: {c_median:.2f} H100e")
            print(f"  Discrete median:   {d_median:.2f} H100e")
            print(f"  Difference:        {diff:.1f}%")

    # Compare covert fab data
    c_fab = continuous.get('covert_fab', {})
    d_fab = discrete.get('covert_fab', {}) or discrete.get('black_fab', {})

    if c_fab and d_fab:
        print("\n--- Covert Fab ---")
        c_dashboard = c_fab.get('dashboard', {})
        d_dashboard = d_fab.get('dashboard', {})

        if c_dashboard:
            print(f"  Continuous production: {c_dashboard.get('production', 'N/A')}")
        if d_dashboard:
            print(f"  Discrete production:   {d_dashboard.get('production', 'N/A')}")

    # Compare rate of computation
    c_rate = continuous.get('rate_of_computation', {})
    d_rate = discrete.get('rate_of_computation', {})

    if c_rate and d_rate:
        c_chip = c_rate.get('covert_chip_stock', {})
        d_chip = d_rate.get('covert_chip_stock', {})
        if c_chip and d_chip:
            c_chip['years'] = c_rate.get('years', [])
            d_chip['years'] = d_rate.get('years', [])
            compare_time_series("Covert Chip Stock", c_chip, d_chip)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Calculate overall difference metrics
    if c_bp and d_bp:
        c_total_median = c_bp.get('total_black_project', {}).get('median', [])
        d_total_median = d_bp.get('total_black_project', {}).get('median', [])

        if c_total_median and d_total_median:
            c_final = c_total_median[-1]
            d_final = d_total_median[-1]
            final_diff = abs(c_final - d_final) / d_final * 100 if d_final != 0 else float('inf')

            print(f"\nFinal total black project compute:")
            print(f"  Continuous: {c_final:,.0f} H100e")
            print(f"  Discrete:   {d_final:,.0f} H100e")
            print(f"  Difference: {final_diff:.1f}%")

            if final_diff > 50:
                print("\n⚠️  WARNING: Models differ by more than 50%!")
                print("   This indicates significant bugs in the continuous model.")


def save_results_for_debugging(continuous, discrete):
    """Save raw results to JSON for manual inspection."""
    output_dir = Path(__file__).resolve().parent.parent / "scripts" / "comparison_output"
    output_dir.mkdir(exist_ok=True)

    # Save continuous results
    with open(output_dir / "continuous_results.json", 'w') as f:
        json.dump(continuous, f, indent=2, default=str)

    # Save discrete results
    with open(output_dir / "discrete_results.json", 'w') as f:
        json.dump(discrete, f, indent=2, default=str)

    print(f"\nRaw results saved to: {output_dir}")


if __name__ == "__main__":
    print("=" * 70)
    print("DIRECT MODEL COMPARISON")
    print("ai_futures_simulator (continuous) vs black_project_backend (discrete)")
    print("=" * 70)

    # Use same time range for both
    time_range = [2029, 2037]
    num_simulations = 11

    try:
        continuous_results = run_continuous_model(time_range=time_range, num_simulations=num_simulations)
    except Exception as e:
        print(f"\nContinuous model failed: {e}")
        import traceback
        traceback.print_exc()
        continuous_results = {}

    try:
        discrete_results = run_discrete_model(time_range=time_range, num_simulations=num_simulations)
    except Exception as e:
        print(f"\nDiscrete model failed: {e}")
        import traceback
        traceback.print_exc()
        discrete_results = {}

    if continuous_results and discrete_results:
        compare_results(continuous_results, discrete_results)
        save_results_for_debugging(continuous_results, discrete_results)
    else:
        print("\n❌ Could not compare - one or both models failed to run")
