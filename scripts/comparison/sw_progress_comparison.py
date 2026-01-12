#!/usr/bin/env python3
"""
Compare modal software progress trajectories between:
1. Original ai-futures-calculator repo's ProgressModel
2. Current ai_futures_simulator's progress model

This script runs both models with their default (modal) parameters
and compares the resulting software progress trajectories.

Since both repos use the same ProgressModel code (the simulator imports from
software_r_and_d/progress_model which is structurally identical to the original),
this comparison verifies that the implementations are functionally equivalent.

Expected result: 0% difference when using identical parameters and data.

Usage:
    python -m scripts.comparison.sw_progress_comparison [--plot] [--time-range START END]
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import logging

# Path configuration
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AI_FUTURES_CALCULATOR_PATH = Path.home() / "github" / "ai-futures-calculator"


def suppress_logging():
    """Suppress verbose logging from progress_model."""
    # Set root logging level higher to suppress INFO messages
    logging.basicConfig(level=logging.ERROR)
    # Also set specific loggers
    for logger_name in [
        'progress_model',
        'progress_model._impl',
        'progress_model.integration',
        'progress_model.parameters',
        'progress_model.taste_distribution',
        'progress_model.automation_model',
        'progress_model.metrics_computation',
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def run_original_progress_model(time_range: tuple = (2017.0, 2100.0), quiet: bool = True):
    """
    Run the original ai-futures-calculator ProgressModel with default parameters.

    Returns:
        dict with keys: 'times', 'progress', 'sw_progress_rates', 'research_stock'
    """
    # Suppress logging before import if quiet mode
    if quiet:
        suppress_logging()

    # Add original repo to path
    if str(AI_FUTURES_CALCULATOR_PATH) not in sys.path:
        sys.path.insert(0, str(AI_FUTURES_CALCULATOR_PATH))

    # Import from original repo
    from progress_model import ProgressModel, Parameters, load_time_series_data

    # Load default time series data
    input_csv = AI_FUTURES_CALCULATOR_PATH / "input_data.csv"
    time_series = load_time_series_data(str(input_csv))

    # Create model with default parameters
    params = Parameters()
    model = ProgressModel(params, time_series)

    # Run trajectory computation
    times, progress_values, research_stock_values = model.compute_progress_trajectory(
        list(time_range), initial_progress=0.0
    )

    # Extract results
    results = model.results

    return {
        'times': np.array(times),
        'progress': np.array(progress_values),
        'research_stock': np.array(research_stock_values),
        'sw_progress_rates': np.array(results.get('sw_progress_rates', [])),
        'progress_rates': np.array(results.get('progress_rates', [])),
        'automation_fractions': np.array(results.get('automation_fractions', [])),
        'ai_research_tastes': np.array(results.get('ai_research_tastes', [])),
        'horizon_lengths': np.array(results.get('horizon_lengths', [])),
    }


def run_current_progress_model(time_range: tuple = (2017.0, 2100.0), quiet: bool = True):
    """
    Run the current ai_futures_simulator's ProgressModel with default parameters.

    Returns:
        dict with keys: 'times', 'progress', 'sw_progress_rates', 'research_stock'
    """
    # Suppress logging before import if quiet mode
    if quiet:
        suppress_logging()

    # Add current repo paths
    if str(REPO_ROOT / "ai_futures_simulator") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "ai_futures_simulator"))

    software_r_and_d_dir = REPO_ROOT / "ai_futures_simulator" / "world_updaters" / "software_r_and_d"
    if str(software_r_and_d_dir) not in sys.path:
        sys.path.insert(0, str(software_r_and_d_dir))

    # Import from current repo
    from progress_model import ProgressModel, Parameters, TimeSeriesData
    import pandas as pd

    # Load historical time series data (same as used for calibration)
    csv_path = REPO_ROOT / "ai_futures_simulator" / "parameters" / "historical_calibration_data.csv"

    # For comparison, we need to load the same full time series as the original
    # Our historical_calibration_data.csv only has 2017-2026, so we need to use
    # the original's input_data.csv format for fair comparison
    original_csv = AI_FUTURES_CALCULATOR_PATH / "input_data.csv"
    df = pd.read_csv(original_csv)

    time_series = TimeSeriesData(
        time=df['time'].values,
        L_HUMAN=df['L_HUMAN'].values,
        inference_compute=df['inference_compute'].values,
        experiment_compute=df['experiment_compute'].values,
        training_compute=df['training_compute'].values,
    )

    # Create model with default parameters
    params = Parameters()
    model = ProgressModel(params, time_series)

    # Run trajectory computation
    times, progress_values, research_stock_values = model.compute_progress_trajectory(
        list(time_range), initial_progress=0.0
    )

    # Extract results
    results = model.results

    return {
        'times': np.array(times),
        'progress': np.array(progress_values),
        'research_stock': np.array(research_stock_values),
        'sw_progress_rates': np.array(results.get('sw_progress_rates', [])),
        'progress_rates': np.array(results.get('progress_rates', [])),
        'automation_fractions': np.array(results.get('automation_fractions', [])),
        'ai_research_tastes': np.array(results.get('ai_research_tastes', [])),
        'horizon_lengths': np.array(results.get('horizon_lengths', [])),
    }


def run_full_simulator(time_range: tuple = (2026.0, 2050.0), quiet: bool = True):
    """
    Run the full AIFuturesSimulator and extract sw_progress metrics.

    This tests what the actual simulation produces, not just the ProgressModel.

    Returns:
        dict with keys: 'times', 'progress', 'sw_progress_rates', etc.
    """
    if quiet:
        suppress_logging()

    # Add current repo paths
    if str(REPO_ROOT / "ai_futures_simulator") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "ai_futures_simulator"))

    from ai_futures_simulator import AIFuturesSimulator
    from parameters.model_parameters import ModelParameters

    # Load default parameters
    config_path = REPO_ROOT / "ai_futures_simulator" / "parameters" / "default_parameters.yaml"
    model_params = ModelParameters.from_yaml(config_path)

    # Override time range
    model_params.params.settings.simulation_start_year = int(time_range[0])
    model_params.params.settings.simulation_end_year = float(time_range[1])
    n_years = time_range[1] - time_range[0]
    model_params.params.settings.n_eval_points = int(n_years * 10) + 1

    # Create simulator and run
    simulator = AIFuturesSimulator(model_parameters=model_params)
    result = simulator.run_modal_simulation()

    # Extract sw_progress data from trajectory
    times = []
    progress_values = []
    sw_progress_rates = []
    ai_sw_progress_mult = []
    horizon_lengths = []

    for i, world in enumerate(result.trajectory):
        t = result.times[i].item()
        times.append(t)

        # Get AI software developer data
        for dev_name, dev in world.ai_software_developers.items():
            if hasattr(dev, 'ai_software_progress') and dev.ai_software_progress is not None:
                sp = dev.ai_software_progress

                # Extract progress
                if hasattr(sp, 'progress') and sp.progress is not None:
                    progress_values.append(float(sp.progress.item() if hasattr(sp.progress, 'item') else sp.progress))
                else:
                    progress_values.append(np.nan)

                # Extract sw_progress_rate
                if hasattr(sp, 'sw_progress_rate') and sp.sw_progress_rate is not None:
                    sw_progress_rates.append(float(sp.sw_progress_rate.item() if hasattr(sp.sw_progress_rate, 'item') else sp.sw_progress_rate))
                else:
                    sw_progress_rates.append(np.nan)

                # Extract ai_sw_progress_mult_ref_present_day (the uplift)
                if hasattr(sp, 'ai_sw_progress_mult_ref_present_day') and sp.ai_sw_progress_mult_ref_present_day is not None:
                    ai_sw_progress_mult.append(float(sp.ai_sw_progress_mult_ref_present_day.item() if hasattr(sp.ai_sw_progress_mult_ref_present_day, 'item') else sp.ai_sw_progress_mult_ref_present_day))
                else:
                    ai_sw_progress_mult.append(np.nan)

                # Extract horizon_length
                if hasattr(sp, 'horizon_length') and sp.horizon_length is not None:
                    horizon_lengths.append(float(sp.horizon_length.item() if hasattr(sp.horizon_length, 'item') else sp.horizon_length))
                else:
                    horizon_lengths.append(np.nan)

                break  # Only get first developer

    return {
        'times': np.array(times),
        'progress': np.array(progress_values),
        'sw_progress_rates': np.array(sw_progress_rates),
        'ai_sw_progress_mult': np.array(ai_sw_progress_mult),
        'horizon_lengths': np.array(horizon_lengths),
    }


def compare_trajectories(original: dict, current: dict) -> dict:
    """
    Compare two trajectory results and compute differences.

    Returns:
        dict with comparison metrics
    """
    metrics = {}

    # Find common time range
    t_min = max(original['times'].min(), current['times'].min())
    t_max = min(original['times'].max(), current['times'].max())

    # Create common time grid
    common_times = np.linspace(t_min, t_max, 500)

    # Interpolate both to common grid for comparison
    def interpolate_to_common(data, times, common_times):
        if len(data) == 0:
            return np.full_like(common_times, np.nan)
        return np.interp(common_times, times, data)

    for key in ['progress', 'sw_progress_rates', 'progress_rates', 'research_stock',
                'automation_fractions', 'ai_research_tastes', 'horizon_lengths']:
        orig_data = original.get(key, np.array([]))
        curr_data = current.get(key, np.array([]))

        if len(orig_data) == 0 or len(curr_data) == 0:
            continue

        orig_interp = interpolate_to_common(orig_data, original['times'], common_times)
        curr_interp = interpolate_to_common(curr_data, current['times'], common_times)

        # Compute differences
        abs_diff = curr_interp - orig_interp

        # Percent difference (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_diff = np.where(
                np.abs(orig_interp) > 1e-10,
                100 * (curr_interp - orig_interp) / orig_interp,
                np.nan
            )

        metrics[key] = {
            'mean_abs_diff': np.nanmean(np.abs(abs_diff)),
            'max_abs_diff': np.nanmax(np.abs(abs_diff)),
            'mean_pct_diff': np.nanmean(np.abs(pct_diff)),
            'max_pct_diff': np.nanmax(np.abs(pct_diff)),
        }

    metrics['common_times'] = common_times

    return metrics


def print_comparison_summary(metrics: dict, original: dict, current: dict):
    """Print a summary of the comparison results."""
    print("\n" + "=" * 70)
    print("Software Progress Trajectory Comparison")
    print("Original: ai-futures-calculator (default parameters)")
    print("Current:  ai_futures_simulator (default parameters)")
    print("=" * 70)

    # Print time range info
    print(f"\nOriginal time range: {original['times'].min():.1f} - {original['times'].max():.1f}")
    print(f"Current time range:  {current['times'].min():.1f} - {current['times'].max():.1f}")
    print(f"Data points: original={len(original['times'])}, current={len(current['times'])}")

    # Print key values at specific years
    years_to_check = [2026, 2030, 2035, 2040, 2050]
    print("\n" + "-" * 70)
    print("Progress values at key years:")
    print("-" * 70)
    print(f"{'Year':>8} {'Original':>15} {'Current':>15} {'Diff (%)':>15}")
    print("-" * 70)

    for year in years_to_check:
        orig_val = np.interp(year, original['times'], original['progress'])
        curr_val = np.interp(year, current['times'], current['progress'])
        pct_diff = 100 * (curr_val - orig_val) / orig_val if abs(orig_val) > 1e-10 else np.nan
        print(f"{year:>8} {orig_val:>15.4f} {curr_val:>15.4f} {pct_diff:>14.2f}%")

    # Print metric summary
    print("\n" + "-" * 70)
    print("Metric Comparison Summary:")
    print("-" * 70)
    print(f"{'Metric':<25} {'Mean Abs Diff':>15} {'Mean Pct Diff':>15}")
    print("-" * 70)

    for key in ['progress', 'sw_progress_rates', 'automation_fractions',
                'ai_research_tastes', 'horizon_lengths']:
        if key not in metrics:
            continue
        m = metrics[key]
        print(f"{key:<25} {m['mean_abs_diff']:>15.4f} {m['mean_pct_diff']:>14.2f}%")

    print("-" * 70)


def plot_comparison(original: dict, current: dict, save_path: str = None):
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    plots = [
        ('progress', 'Cumulative Progress'),
        ('sw_progress_rates', 'Software Progress Rate'),
        ('progress_rates', 'Overall Progress Rate'),
        ('automation_fractions', 'Automation Fraction'),
        ('ai_research_tastes', 'AI Research Taste'),
        ('horizon_lengths', 'Horizon Length (minutes)'),
    ]

    for idx, (key, title) in enumerate(plots):
        ax = axes.flat[idx]

        orig_data = original.get(key, np.array([]))
        curr_data = current.get(key, np.array([]))

        if len(orig_data) > 0:
            ax.plot(original['times'], orig_data, 'b-', label='Original', alpha=0.8)
        if len(curr_data) > 0:
            ax.plot(current['times'], curr_data, 'r--', label='Current', alpha=0.8)

        ax.set_xlabel('Year')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def print_full_simulator_summary(simulator_result: dict, progress_model_result: dict):
    """Print comparison of full simulator vs ProgressModel."""
    print("\n" + "=" * 70)
    print("Full Simulator vs ProgressModel Comparison")
    print("=" * 70)

    sim_times = simulator_result['times']
    sim_progress = simulator_result['progress']
    sim_uplift = simulator_result['ai_sw_progress_mult']

    pm_times = progress_model_result['times']
    pm_progress = progress_model_result['progress']

    # Compare progress values at key years
    years_to_check = [2026, 2030, 2035, 2040, 2050]
    print("\nProgress comparison (Full Sim vs ProgressModel):")
    print("-" * 70)
    print(f"{'Year':>8} {'Full Sim':>15} {'ProgressModel':>15} {'Diff (%)':>15}")
    print("-" * 70)

    for year in years_to_check:
        if year > sim_times.max() or year > pm_times.max():
            continue
        sim_val = np.interp(year, sim_times, sim_progress)
        pm_val = np.interp(year, pm_times, pm_progress)
        pct_diff = 100 * (sim_val - pm_val) / pm_val if abs(pm_val) > 1e-10 else np.nan
        print(f"{year:>8} {sim_val:>15.4f} {pm_val:>15.4f} {pct_diff:>14.2f}%")

    # Show AI SW Progress Mult values
    print("\n" + "-" * 70)
    print("AI Software R&D Uplift (ai_sw_progress_mult_ref_present_day):")
    print("-" * 70)
    print(f"{'Year':>8} {'Full Sim Uplift':>20}")
    print("-" * 70)

    for year in years_to_check:
        if year > sim_times.max():
            continue
        uplift_val = np.interp(year, sim_times, sim_uplift)
        print(f"{year:>8} {uplift_val:>20.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare software progress trajectories between repos'
    )
    parser.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Generate comparison plots'
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Save plot to file (implies --plot)'
    )
    parser.add_argument(
        '--time-range',
        nargs=2,
        type=float,
        default=[2017.0, 2100.0],
        metavar=('START', 'END'),
        help='Time range for simulation (default: 2017 2100)'
    )
    parser.add_argument(
        '--full-sim',
        action='store_true',
        help='Also run full AIFuturesSimulator and compare'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose logging from progress models'
    )

    args = parser.parse_args()
    time_range = tuple(args.time_range)
    quiet = not args.verbose

    # Suppress logging early, before any imports from progress_model
    if quiet:
        suppress_logging()

    print("Running original ai-futures-calculator ProgressModel...")
    original = run_original_progress_model(time_range, quiet=quiet)
    print(f"  Computed {len(original['times'])} time points")

    print("\nRunning current ai_futures_simulator ProgressModel...")
    current = run_current_progress_model(time_range, quiet=quiet)
    print(f"  Computed {len(current['times'])} time points")

    print("\nComparing trajectories...")
    metrics = compare_trajectories(original, current)

    print_comparison_summary(metrics, original, current)

    # Also run full simulator if requested
    if args.full_sim:
        print("\n" + "=" * 70)
        print("Running full AIFuturesSimulator...")
        print("=" * 70)
        # Full simulator starts from 2026
        sim_result = run_full_simulator(time_range=(2026.0, min(2100.0, time_range[1])), quiet=quiet)
        print(f"  Computed {len(sim_result['times'])} time points")

        print_full_simulator_summary(sim_result, original)

    if args.plot or args.save_plot:
        plot_comparison(original, current, save_path=args.save_plot)

    return 0


if __name__ == '__main__':
    sys.exit(main())
