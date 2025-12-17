#!/usr/bin/env python3
"""
Test that ProgressModelIncremental produces identical results to ProgressModel.
"""

import numpy as np
import pandas as pd
import copy


def test_incremental_vs_full():
    """Test that incremental model matches full model."""

    # Import here to avoid module-level initialization issues
    from progress_model import ProgressModel, Parameters, TimeSeriesData
    from progress_model_incremental import ProgressModelIncremental
    import model_config as cfg

    def load_ts(csv_path: str = "input_data.csv"):
        """Load time series from CSV."""
        df = pd.read_csv(csv_path)
        return TimeSeriesData(
            time=df['time'].values,
            L_HUMAN=df['L_HUMAN'].values,
            inference_compute=df['inference_compute'].values,
            experiment_compute=df['experiment_compute'].values,
            training_compute_growth_rate=df['training_compute_growth_rate'].values
        )

    print("Loading time series data...")
    time_series = load_ts()

    # Set time range
    time_range = [2018.0, 2035.0]

    # Create parameters for full model (deep copy to avoid shared state)
    # Use linear automation interpolation to avoid a bug in logistic mode with upper anchor = 1.0
    params_dict = copy.deepcopy(cfg.DEFAULT_PARAMETERS)
    params_dict['automation_interp_type'] = 'linear'
    params = Parameters(**{k: v for k, v in params_dict.items() if k in Parameters.__dataclass_fields__})

    # Create full model
    print("Creating full ProgressModel...")
    full_model = ProgressModel(params, time_series)

    # Run full model
    print("Running full model trajectory...")
    times_full, progress_full, rs_full = full_model.compute_progress_trajectory(time_range, initial_progress=0.0)

    # Create incremental model from the calibrated full model
    # This ensures both models use identical calibration
    print("Creating incremental model from calibrated ProgressModel...")
    incr_model = ProgressModelIncremental.from_progress_model(full_model, time_range, initial_progress=0.0)

    # Step through incrementally using same time points
    print(f"Stepping through {len(times_full)} time points incrementally...")

    incr_progress = [incr_model.progress]
    incr_rs = [incr_model.research_stock]

    for i, t in enumerate(times_full[1:], 1):
        # Get inputs at this time from time series (use same time_series as full model)
        human_labor = np.interp(t, time_series.time, time_series.L_HUMAN)
        inference_compute = np.interp(t, time_series.time, time_series.inference_compute)
        experiment_compute = np.interp(t, time_series.time, time_series.experiment_compute)

        # Log-interpolate for exponential trends
        if np.all(time_series.inference_compute > 0):
            log_ic = np.log(time_series.inference_compute)
            inference_compute = np.exp(np.interp(t, time_series.time, log_ic))
        if np.all(time_series.experiment_compute > 0):
            log_ec = np.log(time_series.experiment_compute)
            experiment_compute = np.exp(np.interp(t, time_series.time, log_ec))
        if np.all(time_series.L_HUMAN > 0):
            log_lh = np.log(time_series.L_HUMAN)
            human_labor = np.exp(np.interp(t, time_series.time, log_lh))

        training_rate = np.interp(t, time_series.time, time_series.training_compute_growth_rate)

        # Increment
        incr_model.increment(t, human_labor, inference_compute, experiment_compute, training_rate)

        incr_progress.append(incr_model.progress)
        incr_rs.append(incr_model.research_stock)

    # Compare results
    incr_progress = np.array(incr_progress)
    incr_rs = np.array(incr_rs)

    # Compute differences
    progress_diff = np.abs(progress_full - incr_progress)
    rs_diff = np.abs(rs_full - incr_rs)

    max_progress_diff = np.max(progress_diff)
    max_rs_diff = np.max(rs_diff)
    mean_progress_diff = np.mean(progress_diff)
    mean_rs_diff = np.mean(rs_diff)

    # Also compute relative differences
    rel_progress_diff = progress_diff / (np.abs(progress_full) + 1e-10)
    rel_rs_diff = rs_diff / (np.abs(rs_full) + 1e-10)

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nProgress trajectory comparison:")
    print(f"  Full model final progress:        {progress_full[-1]:.6f}")
    print(f"  Incremental model final progress: {incr_progress[-1]:.6f}")
    print(f"  Max absolute difference:          {max_progress_diff:.6e}")
    print(f"  Mean absolute difference:         {mean_progress_diff:.6e}")
    print(f"  Max relative difference:          {np.max(rel_progress_diff):.6e}")

    print(f"\nResearch stock comparison:")
    print(f"  Full model final RS:        {rs_full[-1]:.6f}")
    print(f"  Incremental model final RS: {incr_rs[-1]:.6f}")
    print(f"  Max absolute difference:    {max_rs_diff:.6e}")
    print(f"  Mean absolute difference:   {mean_rs_diff:.6e}")
    print(f"  Max relative difference:    {np.max(rel_rs_diff):.6e}")

    # Check if differences are acceptable (within numerical tolerance)
    tolerance = 0.01  # 1% relative tolerance
    progress_ok = np.max(rel_progress_diff) < tolerance
    rs_ok = np.max(rel_rs_diff) < tolerance

    print(f"\n{'PASS' if progress_ok else 'FAIL'}: Progress trajectory within {tolerance*100}% tolerance")
    print(f"{'PASS' if rs_ok else 'FAIL'}: Research stock within {tolerance*100}% tolerance")

    # Sample comparison at specific times
    print("\n" + "-"*60)
    print("Sample values at specific times:")
    print("-"*60)
    for check_year in [2020.0, 2025.0, 2030.0, 2035.0]:
        if check_year <= times_full[-1]:
            idx = np.searchsorted(times_full, check_year)
            if idx < len(times_full):
                print(f"\nYear {check_year}:")
                print(f"  Full progress:   {progress_full[idx]:.6f}")
                print(f"  Incr progress:   {incr_progress[idx]:.6f}")
                print(f"  Difference:      {progress_diff[idx]:.6e}")

    return progress_ok and rs_ok


if __name__ == "__main__":
    success = test_incremental_vs_full()
    exit(0 if success else 1)
