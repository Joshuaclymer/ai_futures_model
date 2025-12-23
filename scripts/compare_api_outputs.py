"""
Compare API outputs between local backend and reference model.
Focuses on likelihood ratio and detection probability metrics.
"""

import json
import urllib.request
import urllib.error
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

def fetch_local_api(num_simulations=50):
    """Fetch simulation results from local API."""
    url = 'http://localhost:8000/api/run-black-project-simulation'
    data = json.dumps({
        'num_simulations': num_simulations,
        'time_range': [2030, 2037],  # Same as reference model defaults
        'parameters': {}
    }).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.loads(response.read().decode())
    except urllib.error.URLError as e:
        print(f"Error connecting to local API: {e}")
        return None


def fetch_reference_api(num_samples=50):
    """Fetch simulation results from reference model API (using cached default results)."""
    url = 'https://dark-compute.onrender.com/get_default_results'
    req = urllib.request.Request(url, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.loads(response.read().decode())
    except urllib.error.URLError as e:
        print(f"Error connecting to reference API: {e}")
        return None


def compare_detection_data(local_data, reference_data):
    """Compare detection-related data between local and reference models."""

    print("="*70)
    print("COMPARING LOCAL VS REFERENCE MODEL")
    print("="*70)

    # Extract local detection times
    local_bp = local_data.get('black_project_model', {})
    local_detection_times = local_bp.get('individual_project_time_before_detection', [])

    # Extract local cumulative LR over time
    local_detection = local_data.get('detection_likelihood', {})
    local_lr_combined = local_detection.get('combined_evidence', {})
    local_years = local_lr_combined.get('years', [])
    local_lr_median = local_lr_combined.get('median', [])
    local_lr_p25 = local_lr_combined.get('p25', [])
    local_lr_p75 = local_lr_combined.get('p75', [])

    # Extract local direct evidence (worker-based LR)
    local_direct = local_detection.get('direct_evidence', {})
    local_direct_median = local_direct.get('median', [])

    # Extract reference data
    ref_sims = reference_data.get('simulations', [])
    ref_years = reference_data.get('years', [])

    # Collect reference LR data across simulations
    ref_lr_by_year = {}
    ref_detection_times = []

    for sim in ref_sims:
        bp = sim.get('prc_black_project', {})
        lr_over_time = bp.get('cumulative_lr_over_time', [])

        # Extract detection time (year where LR first hits 100)
        detection_time = float('inf')
        for i, lr in enumerate(lr_over_time):
            if lr >= 99.0 and i < len(ref_years):
                detection_time = ref_years[i] - ref_years[0]  # Relative to start
                break
        ref_detection_times.append(detection_time)

        # Store LR by year
        for i, year in enumerate(ref_years):
            rel_year = year - ref_years[0]
            if rel_year not in ref_lr_by_year:
                ref_lr_by_year[rel_year] = []
            if i < len(lr_over_time):
                ref_lr_by_year[rel_year].append(lr_over_time[i])

    # Compare detection times
    print("\n--- DETECTION TIME DISTRIBUTION ---")
    finite_local = [t for t in local_detection_times if t < float('inf') and t is not None]
    finite_ref = [t for t in ref_detection_times if t < float('inf')]

    print(f"\nLocal Model:")
    print(f"  Detections: {len(finite_local)}/{len(local_detection_times)}")
    if finite_local:
        print(f"  Median: {np.median(finite_local):.2f} years")
        print(f"  Mean: {np.mean(finite_local):.2f} years")
        print(f"  Std: {np.std(finite_local):.2f}")

    print(f"\nReference Model:")
    print(f"  Detections: {len(finite_ref)}/{len(ref_detection_times)}")
    if finite_ref:
        print(f"  Median: {np.median(finite_ref):.2f} years")
        print(f"  Mean: {np.mean(finite_ref):.2f} years")
        print(f"  Std: {np.std(finite_ref):.2f}")

    # Compare LR by year
    print("\n--- LIKELIHOOD RATIO BY YEAR ---")
    for year in sorted(ref_lr_by_year.keys())[:8]:
        ref_vals = ref_lr_by_year[year]
        ref_median = np.median(ref_vals)
        ref_mean = np.mean(ref_vals)

        # Find corresponding local year
        local_idx = None
        for i, y in enumerate(local_years):
            if abs(y - (ref_years[0] + year)) < 0.1:
                local_idx = i
                break

        print(f"\nYear {year}:")
        if local_idx is not None and local_idx < len(local_lr_median):
            print(f"  Local:     median={local_lr_median[local_idx]:.4f}")
        else:
            print(f"  Local:     (no data)")
        print(f"  Reference: median={ref_median:.4f}, mean={ref_mean:.4f}")

        # Count detections (LR >= 99)
        ref_detections = sum(1 for v in ref_vals if v >= 99.0)
        print(f"  Reference detections: {ref_detections}/{len(ref_vals)}")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: LR over time comparison
    ax1 = axes[0, 0]
    if local_years and local_lr_median:
        rel_local_years = [y - local_years[0] for y in local_years]
        ax1.plot(rel_local_years, local_lr_median, 'b-', linewidth=2, label='Local (median)')
        if local_lr_p25 and local_lr_p75:
            ax1.fill_between(rel_local_years, local_lr_p25, local_lr_p75, alpha=0.3, color='blue')

    # Plot reference median
    ref_median_by_year = []
    ref_years_sorted = sorted(ref_lr_by_year.keys())
    for year in ref_years_sorted:
        ref_median_by_year.append(np.median(ref_lr_by_year[year]))

    if ref_years_sorted and ref_median_by_year:
        ax1.plot(ref_years_sorted, ref_median_by_year, 'g--', linewidth=2, label='Reference (median)')

    ax1.set_xlabel('Years since agreement')
    ax1.set_ylabel('Cumulative Likelihood Ratio')
    ax1.set_title('Cumulative LR Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Detection time distribution
    ax2 = axes[0, 1]
    bins = np.linspace(0, 8, 20)
    if finite_local:
        ax2.hist(finite_local, bins=bins, alpha=0.5, label='Local', color='blue', density=True)
    if finite_ref:
        ax2.hist(finite_ref, bins=bins, alpha=0.5, label='Reference', color='green', density=True)
    ax2.set_xlabel('Detection Time (years)')
    ax2.set_ylabel('Density')
    ax2.set_title('Detection Time Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Local direct evidence
    ax3 = axes[1, 0]
    if local_years and local_direct_median:
        rel_local_years = [y - local_years[0] for y in local_years]
        ax3.plot(rel_local_years, local_direct_median, 'b-', linewidth=2, label='Local Direct Evidence')
    ax3.set_xlabel('Years since agreement')
    ax3.set_ylabel('LR (Direct Evidence)')
    ax3.set_title('Worker-Based Detection LR (Local)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Plot 4: Reference direct observation LR
    ax4 = axes[1, 1]
    # Extract direct observation from reference
    ref_direct_by_year = {}
    for sim in ref_sims:
        bp = sim.get('prc_black_project', {})
        lr_direct = bp.get('lr_direct_observation_over_time', [])
        for i, year in enumerate(ref_years):
            rel_year = year - ref_years[0]
            if rel_year not in ref_direct_by_year:
                ref_direct_by_year[rel_year] = []
            if i < len(lr_direct):
                ref_direct_by_year[rel_year].append(lr_direct[i])

    ref_direct_median_by_year = []
    for year in ref_years_sorted:
        if year in ref_direct_by_year:
            ref_direct_median_by_year.append(np.median(ref_direct_by_year[year]))
        else:
            ref_direct_median_by_year.append(1.0)

    if ref_years_sorted and ref_direct_median_by_year:
        ax4.plot(ref_years_sorted, ref_direct_median_by_year, 'g-', linewidth=2, label='Reference Direct Observation')
    ax4.set_xlabel('Years since agreement')
    ax4.set_ylabel('LR (Direct Observation)')
    ax4.set_title('Worker-Based Detection LR (Reference)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    plt.tight_layout()
    output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/api_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return {
        'local_detection_times': finite_local,
        'ref_detection_times': finite_ref,
        'local_lr_median': local_lr_median,
        'ref_lr_by_year': ref_lr_by_year,
    }


def main():
    print("Fetching data from reference model API...")
    reference_data = fetch_reference_api(num_samples=50)

    if not reference_data:
        print("Failed to fetch reference data. Exiting.")
        return

    print(f"Reference data received: {len(reference_data.get('simulations', []))} simulations")

    print("\nFetching data from local API...")
    local_data = fetch_local_api(num_simulations=50)

    if not local_data:
        print("Failed to fetch local data. Is the backend running?")
        print("Start with: cd app_backend && python app.py")
        return

    print(f"Local data received: {local_data.get('num_simulations', 0)} simulations")

    # Compare the data
    results = compare_detection_data(local_data, reference_data)

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
