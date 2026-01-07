#!/usr/bin/env python3
"""
Compare effective compute and training compute curves between:
1. ai_futures_simulator (current model)
2. new_version_of_takeoff_model (reference model)

This script runs both models with comparable parameters and plots the results.
"""

import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add paths for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ai_futures_simulator"))
sys.path.insert(0, str(REPO_ROOT / "new_version_of_takeoff_model" / "ai-futures-calculator"))

# Import current model
from ai_futures_simulator import AIFuturesSimulator
from parameters.model_parameters import ModelParameters

# Import reference model
from progress_model import ProgressModel, Parameters as RefParameters, TimeSeriesData
import model_config as cfg


def load_time_series_from_csv(csv_path):
    """Load time series data from a given CSV file path"""
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    time = np.array([float(row['time']) for row in data])
    L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
    inference_compute = np.array([float(row['inference_compute']) for row in data])
    experiment_compute = np.array([float(row['experiment_compute']) for row in data])
    training_compute_growth_rate = np.array([float(row['training_compute_growth_rate']) for row in data])
    return TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate)


def run_current_model(start_year=2024, end_year=2040):
    """Run the ai_futures_simulator and extract compute curves."""
    config_path = REPO_ROOT / "ai_futures_simulator" / "parameters" / "default_parameters.yaml"
    model_params = ModelParameters.from_yaml(config_path)

    # The model starts at 2026 (needs historical data to initialize)
    actual_start = max(start_year, 2026)
    model_params.settings['simulation_start_year'] = actual_start
    model_params.settings['simulation_end_year'] = end_year
    model_params.settings['n_eval_points'] = int((end_year - actual_start) * 10) + 1

    simulator = AIFuturesSimulator(model_parameters=model_params)
    result = simulator.run_simulation()

    times = result.times.numpy()

    # Extract data from trajectory
    progress_values = []
    training_compute_growth_rates = []
    frontier_training_compute = []

    for world in result.trajectory:
        dev = world.ai_software_developers.get('us_frontier_lab')
        if dev:
            prog = dev.ai_software_progress
            progress_values.append(prog.progress if prog else 0.0)
            training_compute_growth_rates.append(dev.training_compute_growth_rate or 0.0)
            frontier_training_compute.append(dev.frontier_training_compute_tpp_h100e or 0.0)
        else:
            progress_values.append(0.0)
            training_compute_growth_rates.append(0.0)
            frontier_training_compute.append(0.0)

    progress_values = np.array(progress_values)
    training_compute_growth_rates = np.array(training_compute_growth_rates)
    frontier_training_compute = np.array(frontier_training_compute)

    # Convert frontier_training_compute to FLOP OOMs
    # The reference model normalizes training_compute to 26.54 OOMs at 2025.13
    # and integrates growth rate from there. At 2026 with 0.59 OOMs/year growth,
    # reference model shows ~27.07 OOMs.
    # frontier_training at 2026 ≈ 86,249 H100e, so:
    # offset = 27.07 - log10(86249) = 27.07 - 4.94 = 22.13
    H100E_TPP_TO_FLOP_OOM_OFFSET = 22.13
    training_compute = np.where(
        frontier_training_compute > 0,
        np.log10(frontier_training_compute) + H100E_TPP_TO_FLOP_OOM_OFFSET,
        0
    )

    # NO NORMALIZATION - use raw values
    # Software efficiency = progress - (training_compute - training_compute[0])
    initial_progress = progress_values[0] if len(progress_values) > 0 else 0.0
    initial_tc = training_compute[0] if len(training_compute) > 0 else 0.0
    tc_delta = training_compute - initial_tc
    software_efficiency = progress_values - initial_progress - tc_delta

    # Effective compute = training_compute + software_efficiency
    effective_compute = training_compute + software_efficiency

    return {
        'times': times,
        'progress': progress_values,
        'training_compute': training_compute,
        'software_efficiency': software_efficiency,
        'effective_compute': effective_compute,
        'training_compute_growth_rate': training_compute_growth_rates,
    }


def run_reference_model(start_year=2024, end_year=2040):
    """Run the reference model (new_version_of_takeoff_model)."""
    # Load default input data
    input_csv = REPO_ROOT / "new_version_of_takeoff_model" / "ai-futures-calculator" / "input_data.csv"
    data = load_time_series_from_csv(str(input_csv))

    # Create default parameters (these match the reference model's defaults)
    params = RefParameters()

    # Create model
    model = ProgressModel(params, data)

    # Run the computation for the full time range (this populates model.results)
    # The model handles initialization based on present_day settings
    # initial_progress=0.0 is the default starting point
    time_range = [2020.0, min(end_year, data.time[-1])]
    initial_progress = 0.0
    model.compute_progress_trajectory(time_range, initial_progress)

    # Get results from the model's results dictionary
    results = model.results
    times = np.array(results['times'])

    # Filter to requested time range
    mask = (times >= start_year) & (times <= end_year)
    times = times[mask]

    # Get training compute growth rate from the original data, interpolated to model times
    tc_growth_rate = np.interp(np.array(results['times']), data.time, data.training_compute_growth_rate)

    return {
        'times': times,
        'progress': np.array(results['progress'])[mask],
        'training_compute': np.array(results['training_compute'])[mask],
        'software_efficiency': np.array(results['software_efficiency'])[mask],
        'effective_compute': np.array(results['effective_compute'])[mask],
        'training_compute_growth_rate': tc_growth_rate[mask],
    }


def plot_comparison(current, reference, output_path=None):
    """Plot comparison of compute curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Effective Compute
    ax1 = axes[0, 0]
    ax1.plot(current['times'], current['effective_compute'], 'b-', label='Current Model', linewidth=2)
    ax1.plot(reference['times'], reference['effective_compute'], 'r--', label='Reference Model', linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Effective Compute (OOMs)')
    ax1.set_title('Effective Compute Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=cfg.TRAINING_COMPUTE_REFERENCE_YEAR, color='gray', linestyle=':', alpha=0.5, label=f'Ref Year ({cfg.TRAINING_COMPUTE_REFERENCE_YEAR})')
    ax1.axhline(y=cfg.TRAINING_COMPUTE_REFERENCE_OOMS, color='gray', linestyle=':', alpha=0.5)

    # Plot 2: Training Compute
    ax2 = axes[0, 1]
    ax2.plot(current['times'], current['training_compute'], 'b-', label='Current Model', linewidth=2)
    ax2.plot(reference['times'], reference['training_compute'], 'r--', label='Reference Model', linewidth=2)
    if 'frontier_training_ooms' in current:
        ax2.plot(current['times'], current['frontier_training_ooms'], 'g:', label='Current (frontier_training)', linewidth=1.5)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Training Compute (OOMs)')
    ax2.set_title('Training Compute Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=cfg.TRAINING_COMPUTE_REFERENCE_YEAR, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=cfg.TRAINING_COMPUTE_REFERENCE_OOMS, color='gray', linestyle=':', alpha=0.5)

    # Plot 3: Software Efficiency
    ax3 = axes[1, 0]
    ax3.plot(current['times'], current['software_efficiency'], 'b-', label='Current Model', linewidth=2)
    ax3.plot(reference['times'], reference['software_efficiency'], 'r--', label='Reference Model', linewidth=2)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Software Efficiency (OOMs)')
    ax3.set_title('Software Efficiency Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=cfg.TRAINING_COMPUTE_REFERENCE_YEAR, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # Plot 4: Progress
    ax4 = axes[1, 1]
    ax4.plot(current['times'], current['progress'], 'b-', label='Current Model', linewidth=2)
    ax4.plot(reference['times'], reference['progress'], 'r--', label='Reference Model', linewidth=2)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Progress (OOMs)')
    ax4.set_title('Progress Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    # Close the figure to avoid blocking
    plt.close(fig)


def print_summary(current, reference):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    ref_year = cfg.TRAINING_COMPUTE_REFERENCE_YEAR

    # Values at reference year
    current_eff_at_ref = float(np.interp(ref_year, current['times'], current['effective_compute']))
    ref_eff_at_ref = float(np.interp(ref_year, reference['times'], reference['effective_compute']))

    current_tc_at_ref = float(np.interp(ref_year, current['times'], current['training_compute']))
    ref_tc_at_ref = float(np.interp(ref_year, reference['times'], reference['training_compute']))

    print(f"\nAt reference year {ref_year}:")
    print(f"  Effective Compute:")
    print(f"    Current Model:   {current_eff_at_ref:.2f} OOMs")
    print(f"    Reference Model: {ref_eff_at_ref:.2f} OOMs")
    print(f"    Expected:        {cfg.TRAINING_COMPUTE_REFERENCE_OOMS:.2f} OOMs")

    print(f"\n  Training Compute:")
    print(f"    Current Model:   {current_tc_at_ref:.2f} OOMs")
    print(f"    Reference Model: {ref_tc_at_ref:.2f} OOMs")
    print(f"    Expected:        {cfg.TRAINING_COMPUTE_REFERENCE_OOMS:.2f} OOMs")

    # Values at start and end
    print(f"\nAt start ({current['times'][0]:.1f}):")
    print(f"  Current effective_compute:   {current['effective_compute'][0]:.2f} OOMs")
    print(f"  Reference effective_compute: {reference['effective_compute'][0]:.2f} OOMs")

    print(f"\nAt end ({current['times'][-1]:.1f}):")
    print(f"  Current effective_compute:   {current['effective_compute'][-1]:.2f} OOMs")
    print(f"  Reference effective_compute: {reference['effective_compute'][-1]:.2f} OOMs")

    # Training compute growth rates
    print(f"\nTraining Compute Growth Rate (at start):")
    print(f"  Current Model:   {current['training_compute_growth_rate'][0]:.4f} OOMs/year")
    print(f"  Reference Model: {reference['training_compute_growth_rate'][0]:.4f} OOMs/year")


def main():
    print("Running comparison of compute curves...")
    print("="*60)

    # Use time range that works for both models
    # Current model starts at 2026, reference model has data from 2012-2050
    start_year = 2026
    end_year = 2040

    print(f"\nTime range: {start_year} to {end_year}")

    print("\n1. Running current model (ai_futures_simulator)...")
    try:
        current = run_current_model(start_year, end_year)
        print(f"   Got {len(current['times'])} time points")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n2. Running reference model (new_version_of_takeoff_model)...")
    try:
        reference = run_reference_model(start_year, end_year)
        print(f"   Got {len(reference['times'])} time points")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    print_summary(current, reference)

    output_path = REPO_ROOT / "scripts" / "comparison_output" / "compute_curves_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_comparison(current, reference, output_path)


if __name__ == "__main__":
    main()
