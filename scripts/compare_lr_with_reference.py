"""
Compare likelihood ratio outputs between local continuous model and reference API.
Fetches reference data from dark-compute.onrender.com and compares with local calculations.

This script runs the FULL simulation to get cumulative LR that includes all components:
- LR from PRC compute accounting
- LR from SME inventory
- LR from satellite detection
- LR from reported energy consumption
- LR from worker-based detection (survival probability / detection event)
"""

import sys
import os
from pathlib import Path

# Add project paths
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')
sys.path.insert(0, '/Users/joshuaclymer/github/covert_compute_production_model')

import json
import urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Import local simulator
from ai_futures_simulator import AIFuturesSimulator
from parameters.simulation_parameters import ModelParameters


def fetch_reference_data():
    """Fetch cached results from reference API."""
    url = 'https://dark-compute.onrender.com/get_default_results'
    req = urllib.request.Request(url, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching reference data: {e}")
        return None


def extract_reference_lr_data(ref_data):
    """Extract LR-related data from reference API response."""
    bp_model = ref_data.get('black_project_model', {})

    return {
        'years': bp_model.get('years', []),
        'cumulative_lr': bp_model.get('cumulative_lr', {}),
        'lr_other_intel': bp_model.get('lr_other_intel', {}),
        'lr_prc_accounting': bp_model.get('lr_prc_accounting', {}),
        'lr_sme_inventory': bp_model.get('lr_sme_inventory', {}),
        'lr_reported_energy': bp_model.get('lr_reported_energy', {}),
        'detection_times': bp_model.get('individual_project_time_before_detection', []),
        'time_to_detection_ccdf': bp_model.get('time_to_detection_ccdf', {}),
        'h100_years_ccdf': bp_model.get('h100_years_ccdf', {}),
        'num_simulations': ref_data.get('num_simulations', 0),
    }


def run_full_simulations(num_sims=100):
    """
    Run full simulations using AIFuturesSimulator to get cumulative LR
    that includes ALL evidence sources.
    """
    config_path = Path(__file__).resolve().parent.parent / "ai_futures_simulator" / "parameters" / "modal_parameters.yaml"
    model_params = ModelParameters.from_yaml(config_path)

    # Results storage
    results = {
        'cumulative_lr': [],  # List of trajectories, each trajectory is list of LR values
        'lr_other_intel': [],
        'lr_prc_accounting': [],
        'lr_sme_inventory': [],
        'lr_reported_energy': [],
        'detection_times': [],
        'years': None,
    }

    print(f"Running {num_sims} full simulations...")

    for i in range(num_sims):
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{num_sims} simulations...")

        # Set seed for reproducibility
        np.random.seed(1000 + i)

        # Create fresh simulator for each run (to get different random samples)
        simulator = AIFuturesSimulator(model_parameters=model_params)
        result = simulator.run_simulation()

        trajectory = result.trajectory
        times = result.times.tolist()

        # Get black project data from trajectory
        first_world = trajectory[0]
        if not first_world.black_projects:
            continue

        bp_id = list(first_world.black_projects.keys())[0]

        # Extract LR values over time
        cumulative_lr_traj = []
        lr_other_traj = []
        lr_prc_traj = []
        lr_sme_traj = []
        lr_energy_traj = []
        detection_time = None

        # Get black project start year
        bp_start = first_world.black_projects[bp_id].preparation_start_year

        for j, world in enumerate(trajectory):
            bp = world.black_projects.get(bp_id)
            if bp:
                cumulative_lr_traj.append(bp.cumulative_lr)
                lr_other_traj.append(bp.lr_other_intel)
                lr_prc_traj.append(bp.lr_prc_accounting)
                lr_sme_traj.append(bp.lr_sme_inventory)
                lr_energy_traj.append(bp.lr_reported_energy)

                # Check for detection (cumulative LR > threshold, e.g., 4)
                if detection_time is None and bp.cumulative_lr >= 4.0:
                    detection_time = times[j] - bp_start

        results['cumulative_lr'].append(cumulative_lr_traj)
        results['lr_other_intel'].append(lr_other_traj)
        results['lr_prc_accounting'].append(lr_prc_traj)
        results['lr_sme_inventory'].append(lr_sme_traj)
        results['lr_reported_energy'].append(lr_energy_traj)
        results['detection_times'].append(detection_time if detection_time is not None else float('inf'))

        if results['years'] is None:
            # Store years relative to black project start
            results['years'] = [t - bp_start for t in times]

    results['num_simulations'] = len(results['cumulative_lr'])
    print(f"Completed {results['num_simulations']} simulations")

    return results


def compute_percentiles_over_time(trajectories, years_local, years_target):
    """
    Compute median, p25, p75 for trajectories at target year points.

    Args:
        trajectories: List of trajectories (each is list of values)
        years_local: List of years corresponding to trajectory indices
        years_target: List of target years to compute percentiles for

    Returns:
        Dict with 'median', 'p25', 'p75' lists
    """
    if not trajectories or not years_local:
        return {'median': [], 'p25': [], 'p75': []}

    years_local = np.array(years_local)
    medians = []
    p25s = []
    p75s = []

    for target_year in years_target:
        # Find closest index in local years
        idx = np.argmin(np.abs(years_local - target_year))

        # Collect values at this index from all trajectories
        values = []
        for traj in trajectories:
            if idx < len(traj):
                values.append(traj[idx])

        if values:
            medians.append(float(np.median(values)))
            p25s.append(float(np.percentile(values, 25)))
            p75s.append(float(np.percentile(values, 75)))
        else:
            medians.append(1.0)
            p25s.append(1.0)
            p75s.append(1.0)

    return {'median': medians, 'p25': p25s, 'p75': p75s}


def compute_ccdf(values):
    """Compute CCDF from values list."""
    values = [v for v in values if v is not None and v < float('inf')]
    if not values:
        return []

    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    ccdf = []
    seen = set()

    for i, val in enumerate(sorted_vals):
        if val not in seen:
            ccdf.append({'x': float(val), 'y': float((n - i) / n)})
            seen.add(val)

    return ccdf


def compare_lr_outputs(local_data, ref_lr_data):
    """Compare LR outputs between local and reference models."""

    print("="*70)
    print("LIKELIHOOD RATIO COMPARISON: LOCAL vs REFERENCE")
    print("="*70)

    # Define target years for comparison (relative to project start)
    target_years = list(range(8))  # Years 0-7

    # Compute local percentiles at target years
    local_lr_stats = compute_percentiles_over_time(
        local_data['cumulative_lr'],
        local_data['years'],
        target_years
    )
    local_other_stats = compute_percentiles_over_time(
        local_data['lr_other_intel'],
        local_data['years'],
        target_years
    )

    # Reference data
    ref_lr = ref_lr_data['cumulative_lr']
    ref_other = ref_lr_data['lr_other_intel']

    # Compare detection time distributions
    print("\n--- DETECTION TIME DISTRIBUTION ---")

    local_det_times = [t for t in local_data['detection_times'] if t < float('inf')]
    ref_det_times = ref_lr_data['detection_times']
    ref_det_finite = [t for t in ref_det_times if t is not None and t < float('inf')]

    print(f"\nLocal Model ({local_data['num_simulations']} sims):")
    print(f"  Detections (LR>=4): {len(local_det_times)}/{local_data['num_simulations']}")
    if local_det_times:
        print(f"  Median: {np.median(local_det_times):.2f} years")
        print(f"  Mean: {np.mean(local_det_times):.2f} years")
        print(f"  P25/P75: {np.percentile(local_det_times, 25):.2f} / {np.percentile(local_det_times, 75):.2f}")

    print(f"\nReference Model ({ref_lr_data['num_simulations']} sims):")
    print(f"  Detections: {len(ref_det_finite)}/{len(ref_det_times)}")
    if ref_det_finite:
        print(f"  Median: {np.median(ref_det_finite):.2f} years")
        print(f"  Mean: {np.mean(ref_det_finite):.2f} years")
        print(f"  P25/P75: {np.percentile(ref_det_finite, 25):.2f} / {np.percentile(ref_det_finite, 75):.2f}")

    # Compare cumulative LR over time
    print("\n--- CUMULATIVE LR OVER TIME (All Evidence Sources) ---")

    ref_median = ref_lr.get('median', [])

    print(f"\nYear-by-year comparison (median cumulative LR):")
    for i, year in enumerate(target_years):
        local_val = local_lr_stats['median'][i] if i < len(local_lr_stats['median']) else None
        ref_val = ref_median[i] if i < len(ref_median) else None

        if local_val is not None:
            print(f"  Year {year}: Local={local_val:.4f}", end="")
            if ref_val is not None:
                diff_pct = abs(local_val - ref_val) / max(ref_val, 0.001) * 100 if ref_val > 0 else 0
                print(f", Reference={ref_val:.4f}, Diff={diff_pct:.1f}%")
            else:
                print(" (no reference data)")
        else:
            print(f"  Year {year}: (no local data)")

    # Compare worker-based LR (lr_other_intel)
    print("\n--- WORKER-BASED LR (lr_other_intel) ---")
    ref_other_median = ref_other.get('median', [])

    print(f"\nYear-by-year comparison (median worker-based LR):")
    for i, year in enumerate(target_years):
        local_val = local_other_stats['median'][i] if i < len(local_other_stats['median']) else None
        ref_val = ref_other_median[i] if i < len(ref_other_median) else None

        if local_val is not None:
            print(f"  Year {year}: Local={local_val:.4f}", end="")
            if ref_val is not None:
                diff_pct = abs(local_val - ref_val) / max(ref_val, 0.001) * 100 if ref_val > 0 else 0
                print(f", Reference={ref_val:.4f}, Diff={diff_pct:.1f}%")
            else:
                print(" (no reference data)")
        else:
            print(f"  Year {year}: (no local data)")

    # Check static LR components
    print("\n--- STATIC LR COMPONENTS (first simulation) ---")
    if local_data['lr_prc_accounting'] and local_data['lr_prc_accounting'][0]:
        print(f"  LR PRC Accounting: {local_data['lr_prc_accounting'][0][0]:.4f}")
    if local_data['lr_sme_inventory'] and local_data['lr_sme_inventory'][0]:
        print(f"  LR SME Inventory: {local_data['lr_sme_inventory'][0][0]:.4f}")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cumulative LR over time
    ax1 = axes[0, 0]
    ax1.plot(target_years, local_lr_stats['median'], 'b-', linewidth=2, label='Local (median)')
    ax1.fill_between(target_years, local_lr_stats['p25'], local_lr_stats['p75'], alpha=0.3, color='blue')

    if ref_median:
        ax1.plot(target_years[:len(ref_median)], ref_median[:len(target_years)], 'g--', linewidth=2, label='Reference (median)')
        ref_p25 = ref_lr.get('p25', [])
        ref_p75 = ref_lr.get('p75', [])
        if ref_p25 and ref_p75:
            ax1.fill_between(target_years[:len(ref_p25)], ref_p25[:len(target_years)], ref_p75[:len(target_years)], alpha=0.3, color='green')

    ax1.axhline(y=4.0, color='r', linestyle=':', label='Detection threshold (LR=4)')
    ax1.set_xlabel('Years since project start')
    ax1.set_ylabel('Cumulative Likelihood Ratio')
    ax1.set_title('Cumulative LR Over Time (All Evidence Sources)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Detection time distribution
    ax2 = axes[0, 1]
    bins = np.linspace(0, 10, 25)
    if local_det_times:
        ax2.hist(local_det_times, bins=bins, alpha=0.5, label='Local', color='blue', density=True)
    if ref_det_finite:
        ax2.hist(ref_det_finite, bins=bins, alpha=0.5, label='Reference', color='green', density=True)
    ax2.set_xlabel('Detection Time (years)')
    ax2.set_ylabel('Density')
    ax2.set_title('Detection Time Distribution (LR >= 4)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Worker-based LR over time
    ax3 = axes[1, 0]
    ax3.plot(target_years, local_other_stats['median'], 'b-', linewidth=2, label='Local (median)')
    ax3.fill_between(target_years, local_other_stats['p25'], local_other_stats['p75'], alpha=0.3, color='blue')

    if ref_other_median:
        ax3.plot(target_years[:len(ref_other_median)], ref_other_median[:len(target_years)], 'g--', linewidth=2, label='Reference (median)')

    ax3.set_xlabel('Years since project start')
    ax3.set_ylabel('Worker-based LR')
    ax3.set_title('Worker-based LR (lr_other_intel)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Time to detection CCDF comparison
    ax4 = axes[1, 1]
    local_ccdf = compute_ccdf(local_det_times)
    if local_ccdf:
        ax4.plot([p['x'] for p in local_ccdf], [p['y'] for p in local_ccdf],
                 'b-', linewidth=2, label='Local')

    # Plot reference CCDF (threshold 4)
    ref_ccdf = ref_lr_data['time_to_detection_ccdf'].get('4', [])
    if ref_ccdf:
        ax4.plot([p['x'] for p in ref_ccdf], [p['y'] for p in ref_ccdf],
                 'g--', linewidth=2, label='Reference (LR=4)')

    ax4.set_xlabel('Time to Detection (years)')
    ax4.set_ylabel('P(Time > x)')
    ax4.set_title('Time to Detection CCDF')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/lr_reference_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return local_det_times, ref_det_finite


def main():
    print("Fetching reference data from dark-compute.onrender.com...")
    ref_data = fetch_reference_data()

    if not ref_data:
        print("Failed to fetch reference data. Exiting.")
        return

    ref_lr_data = extract_reference_lr_data(ref_data)
    print(f"Reference data: {ref_lr_data['num_simulations']} simulations, {len(ref_lr_data['years'])} time points")

    print("\nRunning local simulations (using full AIFuturesSimulator)...")
    local_data = run_full_simulations(num_sims=100)

    # Compare outputs
    compare_lr_outputs(local_data, ref_lr_data)

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print("\nNote: The local model now uses FULL simulation with ALL LR components:")
    print("  - LR from PRC compute accounting")
    print("  - LR from SME inventory")
    print("  - LR from satellite detection")
    print("  - LR from reported energy consumption")
    print("  - LR from worker-based detection")


if __name__ == "__main__":
    main()
