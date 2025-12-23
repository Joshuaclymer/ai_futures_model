"""
Diagnose detection time discrepancy between local and reference models.

This script compares the parameters and distributions used for detection time sampling.
"""

import sys
from pathlib import Path

sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')
sys.path.insert(0, '/Users/joshuaclymer/github/covert_compute_production_model')

import numpy as np
from scipy import stats
import json
import urllib.request

# Import from local model
from world_updaters.compute.black_compute import (
    compute_detection_constants,
    compute_lr_over_time_vs_num_workers,
)

# Import from reference model
from black_project_backend.util import (
    build_composite_detection_distribution,
    sample_detection_time_from_composite,
)


def fetch_reference_params():
    """Fetch reference model parameters from API."""
    url = 'https://dark-compute.onrender.com/get_default_results'
    req = urllib.request.Request(url, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode())
            return data.get('parameters', {})
    except Exception as e:
        print(f"Error fetching reference parameters: {e}")
        return None


def run_local_simulation_and_extract_params():
    """Run local simulation and extract detection parameters."""
    from ai_futures_simulator import AIFuturesSimulator
    from parameters.simulation_parameters import ModelParameters

    config_path = Path(__file__).resolve().parent.parent / "ai_futures_simulator" / "parameters" / "modal_parameters.yaml"
    model_params = ModelParameters.from_yaml(config_path)

    # Sample parameters
    params = model_params.sample()

    # Extract perception parameters
    perception = params.perceptions.black_project_perception_parameters

    print("\n--- LOCAL MODEL PARAMETERS ---")
    print(f"  mean_detection_time_100_workers: {perception.mean_detection_time_for_100_workers}")
    print(f"  mean_detection_time_1000_workers: {perception.mean_detection_time_for_1000_workers}")
    print(f"  variance_of_detection_time: {perception.variance_of_detection_time_given_num_workers}")

    # Run simulation to get labor profile
    np.random.seed(42)
    simulator = AIFuturesSimulator(model_parameters=model_params)
    result = simulator.run_simulation()

    # Get black project labor profile
    trajectory = result.trajectory
    times = result.times.tolist()

    first_world = trajectory[0]
    if first_world.black_projects:
        bp_id = list(first_world.black_projects.keys())[0]
        bp = first_world.black_projects[bp_id]
        bp_start = bp.preparation_start_year

        print(f"\n  Black project start year: {bp_start}")
        print(f"  Initial total labor: {bp.total_labor}")
        print(f"  Construction labor: {bp.concealed_datacenter_capacity_construction_labor}")
        print(f"  AI research labor: {bp.ai_research_labor}")

        # Extract labor over time
        labor_by_year = {}
        for j, world in enumerate(trajectory):
            bp = world.black_projects.get(bp_id)
            if bp:
                rel_year = round(times[j] - bp_start, 1)
                if rel_year >= 0 and rel_year not in labor_by_year:
                    labor_by_year[rel_year] = int(bp.total_labor)

        print(f"\n  Labor profile (first 8 years):")
        for year in sorted(labor_by_year.keys())[:8]:
            print(f"    Year {year}: {labor_by_year[year]} workers")

        return {
            'mean_100': perception.mean_detection_time_for_100_workers,
            'mean_1000': perception.mean_detection_time_for_1000_workers,
            'variance': perception.variance_of_detection_time_given_num_workers,
            'labor_by_year': labor_by_year,
            'bp_start': bp_start,
        }

    return None


def compare_detection_time_distributions():
    """Compare detection time distributions between models."""

    print("="*70)
    print("DETECTION TIME DISTRIBUTION DIAGNOSIS")
    print("="*70)

    # Get local parameters
    local_params = run_local_simulation_and_extract_params()

    if not local_params:
        print("Failed to get local parameters")
        return

    # Reference model default parameters
    ref_params = {
        'mean_100': 6.95,
        'mean_1000': 3.42,
        'variance': 3.88,
    }

    print("\n--- REFERENCE MODEL DEFAULT PARAMETERS ---")
    print(f"  mean_detection_time_100_workers: {ref_params['mean_100']}")
    print(f"  mean_detection_time_1000_workers: {ref_params['mean_1000']}")
    print(f"  variance_of_detection_time: {ref_params['variance']}")

    # Compare A and B constants
    print("\n--- DETECTION CONSTANTS (A, B) ---")

    local_A, local_B = compute_detection_constants(local_params['mean_100'], local_params['mean_1000'])
    print(f"  Local:     A={local_A:.4f}, B={local_B:.4f}")

    # Reference calculation
    x1, mu1 = 100, ref_params['mean_100']
    x2, mu2 = 1000, ref_params['mean_1000']
    ref_B = np.log(mu1 / mu2) / np.log(np.log10(x2) / np.log10(x1))
    ref_A = mu1 * (np.log10(x1) ** ref_B)
    print(f"  Reference: A={ref_A:.4f}, B={ref_B:.4f}")

    # Compare mean detection times for typical labor levels
    print("\n--- MEAN DETECTION TIME BY LABOR LEVEL ---")
    for labor in [100, 500, 1000, 5000, 10000, 11000]:
        local_mu = local_A / (np.log10(labor) ** local_B)
        ref_mu = ref_A / (np.log10(labor) ** ref_B)
        print(f"  Labor {labor:5d}: Local={local_mu:.2f} years, Reference={ref_mu:.2f} years")

    # Create standard labor profile matching reference model
    ref_labor_by_year = {
        0: 10500,
        1: 11000,
        2: 11500,
        3: 12000,
        4: 12500,
        5: 13000,
        6: 13500,
        7: 14000,
    }

    print("\n--- DETECTION TIME SAMPLING COMPARISON ---")
    print("Using reference labor profile (10500 workers at year 0, growing)")

    # Sample detection times using both approaches
    num_samples = 1000

    # Local model sampling
    local_det_times = []
    for i in range(num_samples):
        np.random.seed(i)
        _, det_time = compute_lr_over_time_vs_num_workers(
            labor_by_year=ref_labor_by_year,
            mean_detection_time_100_workers=local_params['mean_100'],
            mean_detection_time_1000_workers=local_params['mean_1000'],
            variance=local_params['variance'],
        )
        local_det_times.append(det_time if det_time < float('inf') else 100)

    # Reference model sampling (using same parameters as local)
    ref_det_times_same_params = []
    sorted_years, prob_ranges, A, B, variance = build_composite_detection_distribution(
        labor_by_year=ref_labor_by_year,
        mean_detection_time_100_workers=local_params['mean_100'],
        mean_detection_time_1000_workers=local_params['mean_1000'],
        variance_theta=local_params['variance'],
    )
    for i in range(num_samples):
        np.random.seed(i)
        det_time = sample_detection_time_from_composite(prob_ranges)
        ref_det_times_same_params.append(det_time if det_time < float('inf') else 100)

    # Reference model sampling (using reference default parameters)
    ref_det_times_ref_params = []
    sorted_years, prob_ranges, A, B, variance = build_composite_detection_distribution(
        labor_by_year=ref_labor_by_year,
        mean_detection_time_100_workers=ref_params['mean_100'],
        mean_detection_time_1000_workers=ref_params['mean_1000'],
        variance_theta=ref_params['variance'],
    )
    for i in range(num_samples):
        np.random.seed(i)
        det_time = sample_detection_time_from_composite(prob_ranges)
        ref_det_times_ref_params.append(det_time if det_time < float('inf') else 100)

    print(f"\nLocal model (local params):")
    print(f"  Median: {np.median(local_det_times):.2f} years")
    print(f"  Mean: {np.mean(local_det_times):.2f} years")
    print(f"  P25/P75: {np.percentile(local_det_times, 25):.2f} / {np.percentile(local_det_times, 75):.2f}")

    print(f"\nReference sampling (same params as local):")
    print(f"  Median: {np.median(ref_det_times_same_params):.2f} years")
    print(f"  Mean: {np.mean(ref_det_times_same_params):.2f} years")
    print(f"  P25/P75: {np.percentile(ref_det_times_same_params, 25):.2f} / {np.percentile(ref_det_times_same_params, 75):.2f}")

    print(f"\nReference sampling (reference default params):")
    print(f"  Median: {np.median(ref_det_times_ref_params):.2f} years")
    print(f"  Mean: {np.mean(ref_det_times_ref_params):.2f} years")
    print(f"  P25/P75: {np.percentile(ref_det_times_ref_params, 25):.2f} / {np.percentile(ref_det_times_ref_params, 75):.2f}")

    # Check if parameters differ
    print("\n--- PARAMETER DIFFERENCE ANALYSIS ---")
    if abs(local_params['mean_100'] - ref_params['mean_100']) > 0.01:
        print(f"  ⚠ mean_detection_time_100_workers differs!")
        print(f"    Local: {local_params['mean_100']}, Reference: {ref_params['mean_100']}")
    if abs(local_params['mean_1000'] - ref_params['mean_1000']) > 0.01:
        print(f"  ⚠ mean_detection_time_1000_workers differs!")
        print(f"    Local: {local_params['mean_1000']}, Reference: {ref_params['mean_1000']}")
    if abs(local_params['variance'] - ref_params['variance']) > 0.01:
        print(f"  ⚠ variance_of_detection_time differs!")
        print(f"    Local: {local_params['variance']}, Reference: {ref_params['variance']}")

    # Use actual local labor profile
    print("\n--- USING ACTUAL LOCAL LABOR PROFILE ---")
    local_labor = {int(k): v for k, v in local_params['labor_by_year'].items() if k <= 7}
    print(f"Local labor profile: {local_labor}")

    local_det_with_local_labor = []
    for i in range(num_samples):
        np.random.seed(i)
        _, det_time = compute_lr_over_time_vs_num_workers(
            labor_by_year=local_labor,
            mean_detection_time_100_workers=local_params['mean_100'],
            mean_detection_time_1000_workers=local_params['mean_1000'],
            variance=local_params['variance'],
        )
        local_det_with_local_labor.append(det_time if det_time < float('inf') else 100)

    print(f"\nLocal model with actual local labor profile:")
    print(f"  Median: {np.median(local_det_with_local_labor):.2f} years")
    print(f"  Mean: {np.mean(local_det_with_local_labor):.2f} years")
    print(f"  P25/P75: {np.percentile(local_det_with_local_labor, 25):.2f} / {np.percentile(local_det_with_local_labor, 75):.2f}")


if __name__ == "__main__":
    compare_detection_time_distributions()
