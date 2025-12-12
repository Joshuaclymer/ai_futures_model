#!/usr/bin/env python3
"""
Test script that compares ProgressModel and ProgressModelIncremental trajectories.
Plots AI R&D speedups vs time side by side.
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model_config as cfg
from progress_model import ProgressModel, Parameters, TimeSeriesData
from progress_model_incremental import ProgressModelIncremental


def load_time_series(csv_path: str = "input_data.csv") -> TimeSeriesData:
    """Load time series from CSV."""
    df = pd.read_csv(csv_path)
    return TimeSeriesData(
        time=df['time'].values,
        L_HUMAN=df['L_HUMAN'].values,
        inference_compute=df['inference_compute'].values,
        experiment_compute=df['experiment_compute'].values,
        training_compute_growth_rate=df['training_compute_growth_rate'].values
    )


def run_comparison():
    """Run both models and compare trajectories."""
    print("Loading time series data...")
    time_series = load_time_series()

    # Set time range
    time_range = [2018.0, 2035.0]

    # Create parameters (use linear interpolation to avoid logistic bug)
    print("Creating parameters...")
    params_dict = copy.deepcopy(cfg.DEFAULT_PARAMETERS)
    params_dict['automation_interp_type'] = 'linear'
    params = Parameters(**{k: v for k, v in params_dict.items() if k in Parameters.__dataclass_fields__})

    # Create and run full model
    print("Creating full ProgressModel...")
    full_model = ProgressModel(params, time_series)

    print("Running full model trajectory...")
    times_full, progress_full, rs_full = full_model.compute_progress_trajectory(time_range, initial_progress=0.0)

    # Get full model metrics
    full_results = full_model.results

    # Use ai_sw_progress_mult_ref_present_day for comparison
    # This is the software progress multiplier referenced to present day resources
    ai_rnd_speedups_full = np.array(full_results.get('ai_sw_progress_mult_ref_present_day', np.ones_like(times_full)))

    # Also get research efforts for alternative comparison
    full_research_efforts = np.array(full_results.get('research_efforts', []))
    full_human_only_research_efforts = np.array(full_results.get('human_only_research_efforts', []))

    # Create incremental model using the SAME calibrated params from the full model
    # This ensures both models use identical calibration
    print("Creating incremental model with full model's calibrated params...")

    # Get initial research stock from full model's trajectory
    initial_rs = rs_full[0]

    incr_model = ProgressModelIncremental(
        params=full_model.params,  # Use the calibrated params from full model
        initial_time_series=time_series,
        initial_progress=0.0,
        time_range=time_range,
        skip_calibration=True  # Skip calibration since params are already calibrated
    )
    # Set initial research stock from full model
    incr_model._initial_research_stock = initial_rs
    incr_model._state.research_stock = initial_rs
    # Copy horizon trajectory from full model
    incr_model.horizon_trajectory = full_model.horizon_trajectory
    # Copy anchor stats from full model
    incr_model._anchor_stats = full_model.human_only_results['anchor_stats']
    # Set present_day baseline values from full model's human_only_results
    incr_model._present_day_sw_progress_rate = full_model.human_only_results['anchor_stats']['sw_progress_rate']
    incr_model._present_day_research_stock = full_model.human_only_results['anchor_stats']['research_stock']
    incr_model._present_day_human_labor = full_model.human_only_results['anchor_stats']['human_labor']
    incr_model._present_day_experiment_compute = full_model.human_only_results['anchor_stats']['experiment_compute']
    # Mark as calibrated
    incr_model._calibrated = True

    # Step through incrementally
    print(f"Stepping through {len(times_full)} time points incrementally...")

    incr_times = [incr_model.time]
    incr_progress = [incr_model.progress]
    incr_rs = [incr_model.research_stock]
    incr_speedups = []

    # Compute absolute training compute from growth rate
    # Start with an arbitrary base value and integrate the growth rate
    # training_compute(t) = base * 10^(integral of growth_rate dt)
    base_training_compute = 1e24  # Arbitrary base value at t0

    def get_training_compute(t, t0, base_tc):
        """Compute training compute at time t by integrating growth rate from t0."""
        # Integrate growth rate from t0 to t
        # For simplicity, use trapezoidal integration on the time series
        mask = (time_series.time >= t0) & (time_series.time <= t)
        if not np.any(mask):
            return base_tc
        times_subset = time_series.time[mask]
        rates_subset = time_series.training_compute_growth_rate[mask]
        if len(times_subset) < 2:
            return base_tc
        # Integrate: integral of rate dt gives OOMs change
        ooms_change = np.trapz(rates_subset, times_subset)
        # Also add contribution from t0 to first point and last point to t
        if times_subset[0] > t0:
            rate_at_t0 = np.interp(t0, time_series.time, time_series.training_compute_growth_rate)
            ooms_change += rate_at_t0 * (times_subset[0] - t0)
        if times_subset[-1] < t:
            rate_at_t = np.interp(t, time_series.time, time_series.training_compute_growth_rate)
            ooms_change += rate_at_t * (t - times_subset[-1])
        return base_tc * (10 ** ooms_change)

    # Get initial metrics
    t0 = times_full[0]
    human_labor_0 = np.exp(np.interp(t0, time_series.time, np.log(time_series.L_HUMAN)))
    inference_compute_0 = np.exp(np.interp(t0, time_series.time, np.log(time_series.inference_compute)))
    experiment_compute_0 = np.exp(np.interp(t0, time_series.time, np.log(time_series.experiment_compute)))
    training_compute_0 = base_training_compute

    # Store initial inputs in the model's history
    incr_model._input_history.append((t0, human_labor_0, inference_compute_0, experiment_compute_0, training_compute_0))

    training_rate_0 = np.interp(t0, time_series.time, time_series.training_compute_growth_rate)
    metrics_0 = incr_model.get_metrics(human_labor_0, inference_compute_0, experiment_compute_0, training_rate_0)

    # Use ai_sw_progress_mult_ref_present_day for comparison
    incr_speedups.append(metrics_0.ai_sw_progress_mult_ref_present_day)

    for i, t in enumerate(times_full[1:], 1):
        # Get inputs at current time
        human_labor_t = np.exp(np.interp(t, time_series.time, np.log(time_series.L_HUMAN)))
        inference_compute_t = np.exp(np.interp(t, time_series.time, np.log(time_series.inference_compute)))
        experiment_compute_t = np.exp(np.interp(t, time_series.time, np.log(time_series.experiment_compute)))
        training_compute_t = get_training_compute(t, t0, base_training_compute)

        # increment() now takes training_compute directly (not growth rate)
        incr_model.increment(t, human_labor_t, inference_compute_t, experiment_compute_t, training_compute_t)

        incr_times.append(incr_model.time)
        incr_progress.append(incr_model.progress)
        incr_rs.append(incr_model.research_stock)

        # Get metrics for AI R&D speedup (still needs growth rate for metrics calculation)
        training_rate_t = np.interp(t, time_series.time, time_series.training_compute_growth_rate)
        metrics = incr_model.get_metrics(human_labor_t, inference_compute_t, experiment_compute_t, training_rate_t)

        # Use ai_sw_progress_mult_ref_present_day for comparison
        incr_speedups.append(metrics.ai_sw_progress_mult_ref_present_day)

    incr_times = np.array(incr_times)
    incr_progress = np.array(incr_progress)
    incr_rs = np.array(incr_rs)
    incr_speedups = np.array(incr_speedups)

    # Create comparison plots
    print("Creating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: AI SW Progress Multiplier (ref present day)
    ax1 = axes[0, 0]
    ax1.plot(times_full, ai_rnd_speedups_full, 'b-', linewidth=2, label='Full Model')
    ax1.plot(incr_times, incr_speedups, 'r--', linewidth=2, label='Incremental Model')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('AI SW Progress Mult (x)')
    ax1.set_title('AI SW Progress Multiplier (ref present day) vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Progress
    ax2 = axes[0, 1]
    ax2.plot(times_full, progress_full, 'b-', linewidth=2, label='Full Model')
    ax2.plot(incr_times, incr_progress, 'r--', linewidth=2, label='Incremental Model')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cumulative Progress')
    ax2.set_title('Progress vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Research Stock
    ax3 = axes[1, 0]
    ax3.plot(times_full, rs_full, 'b-', linewidth=2, label='Full Model')
    ax3.plot(incr_times, incr_rs, 'r--', linewidth=2, label='Incremental Model')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Research Stock')
    ax3.set_title('Research Stock vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Plot 4: Differences
    ax4 = axes[1, 1]
    progress_diff = np.abs(progress_full - incr_progress)
    rel_progress_diff = progress_diff / (np.abs(progress_full) + 1e-10) * 100
    ax4.plot(times_full, rel_progress_diff, 'g-', linewidth=2, label='Progress Diff (%)')

    speedup_diff = np.abs(ai_rnd_speedups_full - incr_speedups)
    rel_speedup_diff = speedup_diff / (np.abs(ai_rnd_speedups_full) + 1e-10) * 100
    ax4.plot(times_full, rel_speedup_diff, 'm-', linewidth=2, label='Speedup Diff (%)')

    ax4.set_xlabel('Year')
    ax4.set_ylabel('Relative Difference (%)')
    ax4.set_title('Model Differences')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('incremental_model_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved plot to incremental_model_comparison.png")

    # Print summary statistics
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    print(f"\nProgress trajectory:")
    print(f"  Full model final:        {progress_full[-1]:.4f}")
    print(f"  Incremental final:       {incr_progress[-1]:.4f}")
    print(f"  Max relative diff:       {np.max(rel_progress_diff):.4f}%")
    print(f"  Mean relative diff:      {np.mean(rel_progress_diff):.4f}%")

    print(f"\nAI SW Progress Multipliers (ref present day):")
    print(f"  Full model final:        {ai_rnd_speedups_full[-1]:.4f}x")
    print(f"  Incremental final:       {incr_speedups[-1]:.4f}x")
    print(f"  Max relative diff:       {np.max(rel_speedup_diff):.4f}%")
    print(f"  Mean relative diff:      {np.mean(rel_speedup_diff):.4f}%")

    print(f"\nResearch Stock:")
    print(f"  Full model final:        {rs_full[-1]:.6e}")
    print(f"  Incremental final:       {incr_rs[-1]:.6e}")

    # plt.show()  # Commented out to avoid hanging

    return fig


if __name__ == "__main__":
    run_comparison()
