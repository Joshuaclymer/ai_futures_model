#!/usr/bin/env python
"""
Compare the new AI Futures Simulator output with the old ai_takeoff_model.

This script runs both models with equivalent default parameters and compares
the progress trajectories to verify consistency.
"""

import sys
from pathlib import Path

# Add the old model to path
OLD_MODEL_PATH = Path(__file__).resolve().parents[1] / "old" / "update_model_state" / "update_ai_software_r_and_d" / "ai_takeoff_model"
sys.path.insert(0, str(OLD_MODEL_PATH))

import numpy as np
import pandas as pd

# Import from old model
from progress_model import ProgressModel, Parameters, TimeSeriesData
import model_config as cfg

# Import from new model
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_futures_simulator import AIFuturesSimulator
from parameters.simulation_parameters import ModelParameters


def run_old_model():
    """Run the old ai_takeoff_model with default parameters using the full ProgressModel."""
    print("=" * 60)
    print("RUNNING OLD MODEL (ai_takeoff_model)")
    print("=" * 60)

    # Use the new_simulator_default.csv (matches our modal_parameters.yaml)
    csv_path = OLD_MODEL_PATH / "inputs" / "new_simulator_default.csv"

    # Load time series data using pandas
    df = pd.read_csv(csv_path)
    time_series = TimeSeriesData(
        time=df['time'].values,
        L_HUMAN=df['L_HUMAN'].values,
        inference_compute=df['inference_compute'].values,
        experiment_compute=df['experiment_compute'].values,
        training_compute_growth_rate=df['training_compute_growth_rate'].values
    )

    # Create parameters with defaults
    params = Parameters()
    params.progress_at_aa = 10.0

    # Print key parameters from old model defaults
    print(f"\nOld Model Parameters (from model_config.py):")
    print(f"  r_software: {cfg.DEFAULT_PARAMETERS.get('r_software', 'N/A')}")
    print(f"  rho_coding_labor: {cfg.DEFAULT_PARAMETERS.get('rho_coding_labor', 'N/A')}")
    print(f"  parallel_penalty: {cfg.DEFAULT_PARAMETERS.get('parallel_penalty', 'N/A')}")
    print(f"  present_day: {cfg.DEFAULT_PARAMETERS.get('present_day', 'N/A')}")
    print(f"  present_horizon: {cfg.DEFAULT_PARAMETERS.get('present_horizon', 'N/A')}")
    print(f"  constant_training_compute_growth_rate: {cfg.DEFAULT_PARAMETERS.get('constant_training_compute_growth_rate', 'N/A')}")
    print(f"  slowdown_year: {cfg.DEFAULT_PARAMETERS.get('slowdown_year', 'N/A')}")
    print(f"  post_slowdown_training_compute_growth_rate: {cfg.DEFAULT_PARAMETERS.get('post_slowdown_training_compute_growth_rate', 'N/A')}")

    # Create and run the full model
    model = ProgressModel(params, time_series)
    time_range = [2012.0, 2050.0]
    times, progress_values, research_stock_values = model.compute_progress_trajectory(
        time_range, initial_progress=0.0
    )

    print(f"\nProgress trajectory:")
    print(f"{'Year':<10} {'Progress (OOMs)':<20} {'Progress Rate':<20}")
    print("-" * 50)

    results = []
    times_to_check = [2026.0, 2028.0, 2030.0, 2032.0, 2035.0, 2040.0]

    for target_time in times_to_check:
        # Find closest time in trajectory
        idx = np.argmin(np.abs(times - target_time))
        actual_t = times[idx]
        progress = progress_values[idx]

        # Compute progress rate by finite difference
        if idx > 0:
            dt = times[idx] - times[idx-1]
            dp = progress_values[idx] - progress_values[idx-1]
            rate = dp / dt if dt > 0 else 0.0
        else:
            rate = 0.0

        print(f"{actual_t:<10.2f} {progress:<20.4f} {rate:<20.4f}")
        results.append({
            'time': actual_t,
            'progress': progress,
            'rate': rate,
        })

    return results


def run_new_model():
    """Run the new AI Futures Simulator with default parameters."""
    print("\n" + "=" * 60)
    print("RUNNING NEW MODEL (AIFuturesSimulator)")
    print("=" * 60)

    # Load parameters from YAML
    config_path = Path(__file__).resolve().parents[1] / "parameters" / "modal_parameters.yaml"
    model_params = ModelParameters.from_yaml(config_path)

    # Create and run simulator
    simulator = AIFuturesSimulator(model_parameters=model_params)
    result = simulator.run_simulation()

    params = result.params
    print(f"\nNew Model Parameters (from modal_parameters.yaml):")
    print(f"  r_software: {params.software_r_and_d.r_software}")
    print(f"  rho_coding_labor: {params.software_r_and_d.rho_coding_labor}")
    print(f"  parallel_penalty: {params.software_r_and_d.parallel_penalty}")
    print(f"  present_day: {params.software_r_and_d.present_day}")
    print(f"  present_horizon: {params.software_r_and_d.present_horizon}")
    print(f"  us_frontier_project_compute_growth_rate: {params.compute_growth.us_frontier_project_compute_growth_rate}")
    print(f"  slowdown_year: {params.compute_growth.slowdown_year}")
    print(f"  post_slowdown_training_compute_growth_rate: {params.compute_growth.post_slowdown_training_compute_growth_rate}")

    # Extract progress at key time points
    times_to_check = [2026.0, 2028.0, 2030.0, 2032.0, 2035.0, 2040.0]

    print(f"\nProgress trajectory:")
    print(f"{'Year':<10} {'Progress (OOMs)':<20} {'Progress Rate':<20}")
    print("-" * 50)

    results = []
    for target_t in times_to_check:
        # Find closest time in trajectory
        for i, t in enumerate(result.times):
            if t.item() >= target_t:
                world = result.trajectory[i]
                dev = list(world.ai_software_developers.values())[0]
                progress = dev.ai_software_progress.progress.item()
                rate = dev.ai_software_progress.progress_rate.item()
                actual_t = t.item()
                print(f"{actual_t:<10.2f} {progress:<20.4f} {rate:<20.4f}")
                results.append({
                    'time': actual_t,
                    'progress': progress,
                    'rate': rate,
                })
                break

    return results


def compare_results(old_results, new_results):
    """Compare results between old and new models."""
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if not old_results or not new_results:
        print("Cannot compare - one of the models failed to produce results")
        return None, None

    print(f"\n{'Year':<10} {'Old Progress':<15} {'New Progress':<15} {'Diff':<15} {'% Diff':<10}")
    print("-" * 65)

    max_diff = 0
    max_pct_diff = 0

    for old, new in zip(old_results, new_results):
        diff = new['progress'] - old['progress']
        pct_diff = (diff / old['progress'] * 100) if old['progress'] != 0 else 0

        max_diff = max(max_diff, abs(diff))
        max_pct_diff = max(max_pct_diff, abs(pct_diff))

        print(f"{old['time']:<10.1f} {old['progress']:<15.4f} {new['progress']:<15.4f} {diff:<15.4f} {pct_diff:<10.2f}%")

    print(f"\nMax absolute difference: {max_diff:.4f} OOMs")
    print(f"Max percentage difference: {max_pct_diff:.2f}%")

    if max_pct_diff < 5:
        print("\n✓ Results are within 5% - models are consistent!")
    else:
        print("\n✗ Results differ by more than 5% - investigation needed!")

    return max_diff, max_pct_diff


if __name__ == "__main__":
    print("Model Comparison: Old ai_takeoff_model vs New AIFuturesSimulator")
    print("=" * 70)
    print("\nBoth models now use equivalent input data:")
    print("  Old model: inputs/new_simulator_default.csv")
    print("  New model: initialize_world_history/.../largest_ai_developer.csv")
    print("\nParameters and input data should now match.\n")

    try:
        old_results = run_old_model()
    except Exception as e:
        print(f"\nOld model failed with error: {e}")
        import traceback
        traceback.print_exc()
        old_results = []

    try:
        new_results = run_new_model()
    except Exception as e:
        print(f"\nNew model failed with error: {e}")
        import traceback
        traceback.print_exc()
        new_results = []

    if old_results and new_results:
        compare_results(old_results, new_results)
