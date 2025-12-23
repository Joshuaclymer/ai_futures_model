#!/usr/bin/env python
"""
Compare the new AI Futures Simulator output with the old ai_takeoff_model.

This script runs both models with equivalent default parameters and compares
the progress trajectories to verify consistency.

Also compares detection model results with the reference at https://dark-compute.onrender.com/
"""

import sys
from pathlib import Path
import json
import requests

import numpy as np
import pandas as pd

# Define paths
OLD_MODEL_PATH = Path(__file__).resolve().parents[1] / "old" / "update_model_state" / "update_ai_software_r_and_d" / "ai_takeoff_model"
NEW_MODEL_PATH = Path(__file__).resolve().parents[1]

# IMPORTANT: Import order matters due to module caching.
# We import the new model FIRST (before adding old model path) since
# the new model's software_r_and_d.py depends on the new progress_model.
# After new model is imported, we can safely add old model path and import its modules.

# First, import new model (adds its own progress_model path internally)
sys.path.insert(0, str(NEW_MODEL_PATH))
from ai_futures_simulator import AIFuturesSimulator
from parameters.simulation_parameters import ModelParameters

# Now add old model to path for the comparison functions
# Store reference to new progress_model before it gets shadowed
import progress_model as new_progress_model
sys.path.insert(0, str(OLD_MODEL_PATH))

# Import old model modules (these will shadow the new ones in sys.modules for direct imports)
# Use importlib to force loading from old path
import importlib.util
spec = importlib.util.spec_from_file_location("old_progress_model", OLD_MODEL_PATH / "progress_model.py")
old_progress_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_progress_model)

ProgressModel = old_progress_model.ProgressModel
Parameters = old_progress_model.Parameters
TimeSeriesData = old_progress_model.TimeSeriesData

spec_cfg = importlib.util.spec_from_file_location("old_model_config", OLD_MODEL_PATH / "model_config.py")
cfg = importlib.util.module_from_spec(spec_cfg)
spec_cfg.loader.exec_module(cfg)


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
    if hasattr(params, 'compute') and params.compute is not None:
        compute = params.compute
        if hasattr(compute, 'us_frontier_project_compute_growth_rate'):
            print(f"  us_frontier_project_compute_growth_rate: {compute.us_frontier_project_compute_growth_rate}")
        if hasattr(compute, 'slowdown_year'):
            print(f"  slowdown_year: {compute.slowdown_year}")
        if hasattr(compute, 'post_slowdown_training_compute_growth_rate'):
            print(f"  post_slowdown_training_compute_growth_rate: {compute.post_slowdown_training_compute_growth_rate}")
    else:
        print(f"  (compute parameters not set)")

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


# =============================================================================
# DETECTION MODEL COMPARISON
# =============================================================================

REFERENCE_URL = "https://dark-compute.onrender.com"
LOCAL_BACKEND_URL = "http://localhost:5329"


def fetch_reference_detection_results():
    """Fetch detection results from the reference model."""
    print("\n" + "=" * 60)
    print("FETCHING REFERENCE MODEL RESULTS")
    print("=" * 60)
    print(f"URL: {REFERENCE_URL}")

    try:
        # Try to fetch simulation results from reference
        response = requests.post(
            f"{REFERENCE_URL}/run_simulation",
            json={"num_simulations": 20},
            timeout=120
        )
        if response.status_code == 200:
            data = response.json()
            print("✓ Successfully fetched reference results")
            return data
        else:
            print(f"✗ Reference API returned status {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"✗ Could not connect to reference: {e}")
        return None


def fetch_local_detection_results():
    """Fetch detection results from our local implementation."""
    print("\n" + "=" * 60)
    print("FETCHING LOCAL MODEL RESULTS")
    print("=" * 60)
    print(f"URL: {LOCAL_BACKEND_URL}")

    try:
        response = requests.post(
            f"{LOCAL_BACKEND_URL}/api/black-project",
            json={
                "numSimulations": 20,
                "timeRange": [2029, 2039]
            },
            timeout=180
        )
        if response.status_code == 200:
            data = response.json()
            print("✓ Successfully fetched local results")
            return data
        else:
            print(f"✗ Local API returned status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"✗ Could not connect to local backend: {e}")
        return None


def run_local_detection_model():
    """Run detection model directly (without backend)."""
    print("\n" + "=" * 60)
    print("RUNNING LOCAL DETECTION MODEL DIRECTLY")
    print("=" * 60)

    sys.path.insert(0, str(NEW_MODEL_PATH))

    from ai_futures_simulator import AIFuturesSimulator
    from parameters.simulation_parameters import ModelParameters
    from world_updaters.compute.black_compute import (
        compute_detection_constants,
        compute_mean_detection_time,
    )

    # Load parameters
    config_path = NEW_MODEL_PATH / "parameters" / "modal_parameters.yaml"
    model_params = ModelParameters.from_yaml(config_path)

    # Print detection parameters
    params = model_params.sample()
    bp_perception = params.perceptions.black_project_perception_parameters
    bp_props = params.black_project.black_project_properties

    print("\nDetection Parameters:")
    print(f"  mean_detection_time_for_100_workers: {bp_perception.mean_detection_time_for_100_workers}")
    print(f"  mean_detection_time_for_1000_workers: {bp_perception.mean_detection_time_for_1000_workers}")
    print(f"  variance_of_detection_time: {bp_perception.variance_of_detection_time_given_num_workers}")
    print(f"  prior_odds: {bp_perception.prior_odds_of_covert_project}")
    print(f"  total_labor: {bp_props.total_labor}")

    # Compute expected detection time
    A, B = compute_detection_constants(
        bp_perception.mean_detection_time_for_100_workers,
        bp_perception.mean_detection_time_for_1000_workers
    )
    mu = compute_mean_detection_time(int(bp_props.total_labor), A, B)
    variance = bp_perception.variance_of_detection_time_given_num_workers

    # Correct Gamma parameterization: theta = variance/mean, k = mean^2/variance
    theta = variance / mu
    k = (mu ** 2) / variance

    print(f"\nExpected Detection Time (Gamma distribution):")
    print(f"  Mean: {mu:.4f} years")
    print(f"  Variance: {variance:.4f}")
    print(f"  Shape k: {k:.4f}")
    print(f"  Scale theta: {theta:.4f}")

    # Run simulations
    print(f"\nRunning 20 simulations...")
    simulator = AIFuturesSimulator(model_parameters=model_params)
    results = simulator.run_simulations(num_simulations=20)

    # Extract detection times
    detection_times = []
    h100_years = []

    for result in results:
        world = result.trajectory[-1]
        if world.black_projects:
            bp_id = list(world.black_projects.keys())[0]
            bp = world.black_projects[bp_id]

            det_time = bp.sampled_detection_time
            detection_times.append(det_time)

            # Compute H100-years before detection
            cumulative_h100_years = 0.0
            for i, t in enumerate(result.times):
                t_val = t.item()
                years_since_start = t_val - bp.preparation_start_year
                if years_since_start >= det_time:
                    break
                w = result.trajectory[i]
                bp_state = w.black_projects.get(bp_id)
                if bp_state:
                    if i > 0:
                        dt = t_val - result.times[i-1].item()
                        cumulative_h100_years += bp_state.operating_compute_tpp_h100e * dt

            h100_years.append(cumulative_h100_years)

    # Print results
    print(f"\nDetection Results ({len(detection_times)} simulations):")
    print(f"  Detection time - Min: {np.min(detection_times):.4f}, Median: {np.median(detection_times):.4f}, Mean: {np.mean(detection_times):.4f}, Max: {np.max(detection_times):.4f}")
    print(f"  H100-years - Min: {np.min(h100_years):.0f}, Median: {np.median(h100_years):.0f}, Mean: {np.mean(h100_years):.0f}, Max: {np.max(h100_years):.0f}")

    return {
        'detection_times': detection_times,
        'h100_years': h100_years,
        'mean_detection_time': mu,
        'shape_k': k,
    }


def compare_detection_results(local_results, reference_results=None):
    """Compare detection results between local and reference models."""
    print("\n" + "=" * 60)
    print("DETECTION MODEL COMPARISON")
    print("=" * 60)

    if local_results is None:
        print("✗ No local results to compare")
        return

    # Reference values from https://dark-compute.onrender.com/
    # These are approximate values observed from the reference UI
    reference_values = {
        'expected_detection_time_median': 0.7,  # ~0.7 years shown in reference
        'expected_h100_years_median': 100000,   # ~100K H100-years
        'expected_h100e_at_detection': 0,       # ~0 H100e produced before detection
    }

    print("\nReference Model Expected Values (from dark-compute.onrender.com):")
    print(f"  Detection time (median): ~{reference_values['expected_detection_time_median']} years")
    print(f"  H100-years (median): ~{reference_values['expected_h100_years_median']:,.0f}")

    local_det_times = local_results.get('detection_times', [])
    local_h100_years = local_results.get('h100_years', [])

    if local_det_times:
        local_median_det = np.median(local_det_times)
        local_mean_det = np.mean(local_det_times)

        print("\nLocal Model Results:")
        print(f"  Detection time (median): {local_median_det:.4f} years")
        print(f"  Detection time (mean): {local_mean_det:.4f} years")

        if local_h100_years:
            local_median_h100 = np.median(local_h100_years)
            print(f"  H100-years (median): {local_median_h100:,.0f}")

        # Compare
        print("\nComparison:")
        det_ratio = local_median_det / reference_values['expected_detection_time_median']
        print(f"  Detection time ratio (local/reference): {det_ratio:.2f}x")

        if det_ratio < 0.5:
            print("  → Local detection is FASTER than reference (shorter detection times)")
        elif det_ratio > 2.0:
            print("  → Local detection is SLOWER than reference (longer detection times)")
        else:
            print("  → Detection times are within reasonable range of reference")

        # Note about distribution
        print("\nNote: The Gamma distribution with k < 1 is highly skewed.")
        print(f"  Our model: k = {local_results.get('shape_k', 'N/A'):.4f}")
        print("  Median << Mean for such distributions.")
        print("  Most detections happen quickly, but some take much longer.")

    return local_results


def run_detection_comparison():
    """Run complete detection model comparison."""
    print("\n" + "=" * 70)
    print("DETECTION MODEL COMPARISON: Local vs Reference")
    print("=" * 70)

    # Run local model directly
    local_results = run_local_detection_model()

    # Try to fetch reference results (may fail if server is down)
    # reference_results = fetch_reference_detection_results()

    # Compare
    compare_detection_results(local_results, None)

    return local_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare AI models")
    parser.add_argument("--detection", action="store_true", help="Compare detection models only")
    parser.add_argument("--progress", action="store_true", help="Compare progress models only")
    args = parser.parse_args()

    if args.detection:
        # Detection model comparison only
        run_detection_comparison()
    elif args.progress:
        # Original progress model comparison
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
    else:
        # Run both comparisons
        print("=" * 70)
        print("RUNNING ALL COMPARISONS")
        print("=" * 70)

        # Detection comparison
        print("\n\n>>> DETECTION MODEL COMPARISON <<<")
        run_detection_comparison()

        # Progress comparison (optional - may fail if old model not available)
        print("\n\n>>> PROGRESS MODEL COMPARISON <<<")
        try:
            old_results = run_old_model()
            new_results = run_new_model()
            if old_results and new_results:
                compare_results(old_results, new_results)
        except Exception as e:
            print(f"Progress comparison skipped: {e}")
