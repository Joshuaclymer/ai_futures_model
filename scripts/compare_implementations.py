#!/usr/bin/env python3
"""
Compare the ODE-based ai_futures_simulator with the reference ProgressModel implementation.

This script runs both implementations with the same parameters and compares their outputs
to ensure they produce similar results.
"""

import sys
import os
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pandas as pd

# Add paths for imports
ROOT = Path(__file__).resolve().parent.parent
AI_FUTURES_CALCULATOR_PATH = ROOT / "new_version_of_takeoff_model" / "ai-futures-calculator"
sys.path.insert(0, str(ROOT / "ai_futures_simulator"))
sys.path.insert(0, str(AI_FUTURES_CALCULATOR_PATH))
sys.path.insert(0, str(ROOT / "app_backend"))

# Suppress excessive logging
import logging
logging.disable(logging.WARNING)


@contextmanager
def working_directory(path):
    """Context manager to temporarily change working directory."""
    old_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)

def run_reference_model(time_range, initial_progress=0.0):
    """Run the reference ProgressModel implementation."""
    from progress_model import ProgressModel
    from progress_model.parameters import Parameters
    from progress_model.types import TimeSeriesData
    import model_config as cfg

    # Load time series data
    csv_path = ROOT / "new_version_of_takeoff_model" / "ai-futures-calculator" / "input_data.csv"
    df = pd.read_csv(csv_path)

    time_series = TimeSeriesData(
        time=df['time'].values,
        L_HUMAN=df['L_HUMAN'].values,
        inference_compute=df['inference_compute'].values,
        experiment_compute=df['experiment_compute'].values,
        training_compute_growth_rate=df['training_compute_growth_rate'].values
    )

    # Create parameters with defaults from model_config
    # Filter to only include params that exist in the Parameters dataclass
    default_params = {k: v for k, v in cfg.DEFAULT_PARAMETERS.items()
                      if k in Parameters.__dataclass_fields__}

    # Set progress_at_aa to 100.0 if None (this is what the ODE simulator uses)
    if default_params.get('progress_at_aa') is None:
        default_params['progress_at_aa'] = 100.0

    params = Parameters(**default_params)

    # Create and run model
    # Run from ai-futures-calculator directory so benchmark_results.yaml can be found
    model = ProgressModel(params, time_series)
    with working_directory(AI_FUTURES_CALCULATOR_PATH):
        times, progress_values, research_stock_values = model.compute_progress_trajectory(
            time_range,
            initial_progress
        )

    # Extract results
    results = model.results

    return {
        'times': np.array(times),
        'progress': np.array(progress_values),
        'research_stock': np.array(research_stock_values),
        'horizon_lengths': np.array(results.get('horizon_lengths', [])),
        'automation_fractions': np.array(results.get('automation_fraction', [])),  # Note: singular key
        'serial_coding_labor_multipliers': np.array(results.get('serial_coding_labor_multipliers', [])),
        'experiment_capacity': np.array(results.get('experiment_capacity', [])),
        'software_efficiency': np.array(results.get('software_efficiency', [])),
        'ai_research_tastes': np.array(results.get('ai_research_taste', [])),  # Note: singular key
        'research_efforts': np.array(results.get('research_efforts', [])),
        'software_progress_rates': np.array(results.get('software_progress_rates', [])),
        'ai_coding_labor_multipliers': np.array(results.get('ai_coding_labor_multipliers', [])),
        'progress_rates': np.array(results.get('progress_rates', [])),  # Overall progress rate
        'aggregate_research_taste': np.array(results.get('aggregate_research_taste', [])),
        'serial_coding_labors': np.array(results.get('serial_coding_labors', [])),
        'coding_labors': np.array(results.get('coding_labors', [])),
        'human_labor_contributions': np.array(results.get('human_labor_contributions', [])),
        'params': params,
    }


def run_ode_simulator(time_range, initial_progress=0.0):
    """Run the ODE-based AIFuturesSimulator implementation."""
    from api_utils.simulation import run_simulation_internal, extract_sw_progress_from_raw

    # Run simulation
    params = {}  # Use defaults
    raw_result = run_simulation_internal(params, time_range)
    sw_result = extract_sw_progress_from_raw(raw_result)

    time_series = sw_result['time_series']

    # Extract arrays
    times = np.array([p['year'] for p in time_series])
    progress = np.array([p.get('progress', 0) or 0 for p in time_series])
    research_stock = np.array([p.get('researchStock', 0) or 0 for p in time_series])
    horizon_lengths = np.array([p.get('horizonLength', 0) or 0 for p in time_series])
    automation_fractions = np.array([p.get('automationFraction', 0) or 0 for p in time_series])
    serial_coding_labor_multipliers = np.array([p.get('serialCodingLaborMultiplier', 1) or 1 for p in time_series])
    experiment_capacity = np.array([p.get('experimentCapacity', 0) or 0 for p in time_series])
    software_efficiency = np.array([p.get('softwareEfficiency', 0) or 0 for p in time_series])
    ai_research_tastes = np.array([p.get('aiResearchTaste', 0) or 0 for p in time_series])
    research_efforts = np.array([p.get('researchEffort', 0) or 0 for p in time_series])
    software_progress_rates = np.array([p.get('softwareProgressRate', 0) or 0 for p in time_series])
    ai_coding_labor_multipliers = np.array([p.get('aiCodingLaborMultiplier', 1) or 1 for p in time_series])
    aggregate_research_taste = np.array([p.get('aggregateResearchTaste', 1) or 1 for p in time_series])
    serial_coding_labors = np.array([p.get('serialCodingLabor', 0) or 0 for p in time_series])
    coding_labors = np.array([p.get('codingLabor', 0) or 0 for p in time_series])
    human_labor_contributions = np.array([p.get('humanLabor', 0) or 0 for p in time_series])

    return {
        'times': times,
        'progress': progress,
        'research_stock': research_stock,
        'horizon_lengths': horizon_lengths,
        'automation_fractions': automation_fractions,
        'serial_coding_labor_multipliers': serial_coding_labor_multipliers,
        'experiment_capacity': experiment_capacity,
        'software_efficiency': software_efficiency,
        'ai_research_tastes': ai_research_tastes,
        'research_efforts': research_efforts,
        'software_progress_rates': software_progress_rates,
        'ai_coding_labor_multipliers': ai_coding_labor_multipliers,
        'aggregate_research_taste': aggregate_research_taste,
        'serial_coding_labors': serial_coding_labors,
        'coding_labors': coding_labors,
        'human_labor_contributions': human_labor_contributions,
        'progress_rates': np.array([]),  # Not computed in ODE yet
        'params': raw_result.get('params', {}),
    }


def compare_arrays(name, ref_arr, ode_arr, ref_times, ode_times, tolerance_pct=10.0):
    """Compare two arrays and report differences."""
    # Interpolate ODE values to reference times for comparison
    if len(ref_arr) == 0 or len(ode_arr) == 0:
        print(f"  {name}: SKIP (empty array)")
        return None

    # Interpolate ODE to reference times
    ode_interp = np.interp(ref_times, ode_times, ode_arr)

    # Compute relative errors (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs(ref_arr - ode_interp) / np.maximum(np.abs(ref_arr), 1e-10)
        rel_error = np.where(np.isfinite(rel_error), rel_error, 0)

    max_rel_error = np.max(rel_error) * 100
    mean_rel_error = np.mean(rel_error) * 100

    # Sample values at key points
    indices = [0, len(ref_arr)//4, len(ref_arr)//2, 3*len(ref_arr)//4, -1]

    status = "OK" if max_rel_error < tolerance_pct else "DIFFERS"
    print(f"\n  {name}: {status}")
    print(f"    Max relative error: {max_rel_error:.1f}%")
    print(f"    Mean relative error: {mean_rel_error:.1f}%")
    print(f"    Sample values (year, ref, ode):")
    for i in indices:
        if i < len(ref_times):
            t = ref_times[i]
            r = ref_arr[i]
            o = ode_interp[i]
            print(f"      {t:.1f}: ref={r:.4g}, ode={o:.4g}")

    return {
        'name': name,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'status': status,
    }


def compare_parameters(ref_params, ode_params):
    """Compare key parameters between implementations."""
    print("\n" + "="*60)
    print("PARAMETER COMPARISON")
    print("="*60)

    # Key parameters to compare
    key_params = [
        ('rho_coding_labor', 'rho_coding_labor'),
        ('parallel_penalty', 'parallel_penalty'),
        ('r_software', 'r_software'),
        ('present_day', 'present_day'),
        ('present_horizon', 'present_horizon'),
        ('present_doubling_time', 'present_doubling_time'),
        ('doubling_difficulty_growth_factor', 'doubling_difficulty_growth_factor'),
        ('ai_research_taste_slope', 'ai_research_taste_slope'),
        ('software_progress_rate_at_reference_year', 'software_progress_rate_at_reference_year'),
        ('swe_multiplier_at_present_day', 'swe_multiplier_at_present_day'),
    ]

    for ref_name, ode_name in key_params:
        ref_val = getattr(ref_params, ref_name, None) if hasattr(ref_params, ref_name) else None

        # For ODE params, they might be nested in software_r_and_d
        if isinstance(ode_params, dict):
            sw_params = ode_params.get('software_r_and_d', {})
            ode_val = sw_params.get(ode_name)
        else:
            ode_val = getattr(ode_params, ode_name, None) if hasattr(ode_params, ode_name) else None

        match = "MATCH" if ref_val == ode_val else "DIFFER"
        print(f"  {ref_name}: ref={ref_val}, ode={ode_val} [{match}]")


def main():
    print("="*60)
    print("COMPARING REFERENCE vs ODE IMPLEMENTATIONS")
    print("="*60)

    # Time range for comparison - must start from 2012 like the reference app
    # This is required for proper calibration of r_software and other parameters
    time_range = [2012.0, 2035.0]
    initial_progress = 0.0

    print(f"\nTime range: {time_range}")
    print(f"Initial progress: {initial_progress}")

    # Run reference model
    print("\n" + "-"*40)
    print("Running REFERENCE model (ProgressModel)...")
    print("-"*40)
    try:
        ref_results = run_reference_model(time_range, initial_progress)
        print(f"  Generated {len(ref_results['times'])} time points")
        print(f"  Time range: {ref_results['times'][0]:.2f} - {ref_results['times'][-1]:.2f}")
        print(f"  Final progress: {ref_results['progress'][-1]:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run ODE simulator
    print("\n" + "-"*40)
    print("Running ODE model (AIFuturesSimulator)...")
    print("-"*40)
    try:
        ode_results = run_ode_simulator(time_range, initial_progress)
        print(f"  Generated {len(ode_results['times'])} time points")
        print(f"  Time range: {ode_results['times'][0]:.2f} - {ode_results['times'][-1]:.2f}")
        print(f"  Final progress: {ode_results['progress'][-1]:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare parameters
    compare_parameters(ref_results['params'], ode_results['params'])

    # Compare output arrays
    print("\n" + "="*60)
    print("OUTPUT COMPARISON")
    print("="*60)

    metrics = [
        'progress',
        'research_stock',
        'horizon_lengths',
        'automation_fractions',
        'experiment_capacity',
        'research_efforts',
        'aggregate_research_taste',
        'serial_coding_labors',
        'coding_labors',
        'human_labor_contributions',
        'serial_coding_labor_multipliers',
        'ai_coding_labor_multipliers',
        'software_progress_rates',
        'ai_research_tastes',
    ]

    comparisons = []
    for metric in metrics:
        ref_arr = ref_results.get(metric, np.array([]))
        ode_arr = ode_results.get(metric, np.array([]))

        result = compare_arrays(
            metric,
            ref_arr,
            ode_arr,
            ref_results['times'],
            ode_results['times'],
            tolerance_pct=20.0  # 20% tolerance
        )
        if result:
            comparisons.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    ok_count = sum(1 for c in comparisons if c['status'] == 'OK')
    total_count = len(comparisons)

    print(f"\n  Metrics matching within tolerance: {ok_count}/{total_count}")

    if ok_count < total_count:
        print("\n  Metrics with significant differences:")
        for c in comparisons:
            if c['status'] != 'OK':
                print(f"    - {c['name']}: {c['max_rel_error']:.1f}% max error")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
