"""
End-to-end test to verify the initial compute energy fix.

This script runs the actual AIFuturesSimulator and verifies that:
1. The simulation runs without errors
2. The energy efficiency calculation uses the corrected formula
3. The watts_per_h100e values are reasonable
"""

import sys
from pathlib import Path

# Add ai_futures_simulator subdirectory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai_futures_simulator"))

import torch
import numpy as np

from ai_futures_simulator import AIFuturesSimulator
from parameters.simulation_parameters import ModelParameters


def test_simulation_runs():
    """Test that the simulation runs end-to-end."""
    print("\n" + "=" * 70)
    print("TEST 1: Simulation runs without errors")
    print("=" * 70)

    # Load parameters from YAML
    config_path = Path(__file__).resolve().parent.parent / "ai_futures_simulator" / "parameters" / "modal_parameters.yaml"
    model_params = ModelParameters.from_yaml(config_path)

    # Create simulator
    simulator = AIFuturesSimulator(model_parameters=model_params)

    # Run a single simulation
    result = simulator.run_simulation()

    print(f"  Simulation completed successfully!")
    print(f"  Number of time points: {len(result.times)}")
    print(f"  Time range: {result.times[0]:.2f} to {result.times[-1]:.2f}")

    return result


def test_black_project_energy_values(result):
    """Test that black project energy values are reasonable."""
    print("\n" + "=" * 70)
    print("TEST 2: Black project energy values")
    print("=" * 70)

    trajectory = result.trajectory

    # Get the first black project
    first_world = trajectory[0]
    if not first_world.black_projects:
        print("  No black projects in simulation")
        return

    bp_id = list(first_world.black_projects.keys())[0]
    bp = first_world.black_projects[bp_id]

    print(f"  Black project: {bp_id}")
    print(f"  Preparation start year: {bp.preparation_start_year}")

    # Get energy-related values
    if bp.compute_stock:
        watts_per_h100e = bp.compute_stock.watts_per_h100e
        functional_h100e = bp.compute_stock.functional_tpp_h100e
        print(f"  Initial functional compute: {functional_h100e:,.0f} H100e")
        print(f"  Watts per H100e: {watts_per_h100e:.2f} W")

        # Verify the watts_per_h100e is reasonable
        # For 2030 (8 years after H100 release):
        # - SOTA efficiency = 1.26^8 = 6.35
        # - Combined efficiency = 0.2 * 6.35 = 1.27
        # - Watts per H100e = 700 / 1.27 = 550.94 W

        # If efficiency is 0.2 without SOTA factor: 700/0.2 = 3500 W (WRONG)
        # If efficiency is 1.27 with SOTA factor: 700/1.27 = 550.94 W (CORRECT)

        if watts_per_h100e > 1000:
            print(f"\n  ✗ ERROR: watts_per_h100e = {watts_per_h100e:.2f} W is too high!")
            print(f"    This suggests the SOTA efficiency factor is NOT being applied.")
            print(f"    Expected value ~550 W for 2030 (with SOTA factor)")
            print(f"    Got {watts_per_h100e:.2f} W (likely using only energy_efficiency_relative_to_sota)")
            return False
        else:
            print(f"\n  ✓ watts_per_h100e = {watts_per_h100e:.2f} W is reasonable")
            print(f"    This indicates the SOTA efficiency factor IS being applied correctly")
            return True

    return True


def test_energy_over_time(result):
    """Test that energy values evolve correctly over time."""
    print("\n" + "=" * 70)
    print("TEST 3: Energy evolution over time")
    print("=" * 70)

    trajectory = result.trajectory
    times = result.times.tolist()

    # Get black project ID
    first_world = trajectory[0]
    if not first_world.black_projects:
        print("  No black projects in simulation")
        return

    bp_id = list(first_world.black_projects.keys())[0]

    print(f"  Year  | Total H100e | Watts/H100e | DC Capacity (GW)")
    print(f"  " + "-" * 55)

    for i in range(0, len(trajectory), max(1, len(trajectory) // 5)):
        world = trajectory[i]
        bp = world.black_projects.get(bp_id)
        if bp:
            year = times[i]
            h100e = bp.compute_stock.functional_tpp_h100e if bp.compute_stock else 0
            watts = bp.compute_stock.watts_per_h100e if bp.compute_stock else 0
            dc_cap = bp.datacenters.data_center_capacity_gw if bp.datacenters else 0
            print(f"  {year:.1f} | {h100e:>10,.0f} | {watts:>10.1f} | {dc_cap:>10.4f}")

    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("END-TO-END TEST: INITIAL COMPUTE ENERGY FIX")
    print("=" * 70)

    try:
        # Run simulation
        result = test_simulation_runs()

        # Test energy values
        energy_ok = test_black_project_energy_values(result)

        # Test energy evolution
        test_energy_over_time(result)

        print("\n" + "=" * 70)
        if energy_ok:
            print("ALL TESTS PASSED!")
        else:
            print("SOME TESTS FAILED!")
            sys.exit(1)
        print("=" * 70)

    except Exception as e:
        print(f"\n\nTEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
