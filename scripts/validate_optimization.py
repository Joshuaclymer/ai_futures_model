#!/usr/bin/env python3
"""
Validate optimization results by ablation analysis.

For each optimized parameter, this script perturbs it in both directions
and checks if the objective decreases - validating that we're at a local maximum.

Usage:
    python scripts/validate_optimization.py [options]

Options:
    --results-file FILE    Path to optimization results JSON (default: scripts/optimization_results.json)
    --num-simulations N    Simulations per evaluation (default: 100)
    --perturbation-pct P   Perturbation percentage (default: 10.0)
    --verbose              Print detailed progress
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Add paths for imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "ai_futures_simulator"))
sys.path.insert(0, str(repo_root / "app_backend"))

from ai_futures_simulator import AIFuturesSimulator
from parameters.model_parameters import ModelParameters
from endpoints.black_project.world_data import extract_world_data


# Parameter bounds (must match optimize_black_project_properties.py)
PARAM_BOUNDS = {
    "total_labor": (1000, 50000),
    "fraction_of_labor_devoted_to_datacenter_construction": (0.0, 1.0),
    "fraction_of_labor_devoted_to_black_fab_construction": (0.0, 1.0),
    "fraction_of_labor_devoted_to_black_fab_operation": (0.0, 1.0),
    "fraction_of_labor_devoted_to_ai_research": (0.0, 1.0),
    "fraction_of_initial_compute_stock_to_divert_at_black_project_start": (0.0, 0.5),
    "fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start": (0.0, 0.2),
    "fraction_of_lithography_scanners_to_divert_at_black_project_start": (0.0, 0.5),
    "max_fraction_of_total_national_energy_consumption": (0.01, 0.20),
    "years_before_black_project_start_to_begin_datacenter_construction": (0.0, 5.0),
}

LABOR_FRACTION_PARAMS = [
    "fraction_of_labor_devoted_to_datacenter_construction",
    "fraction_of_labor_devoted_to_black_fab_construction",
    "fraction_of_labor_devoted_to_black_fab_operation",
    "fraction_of_labor_devoted_to_ai_research",
]


def normalize_labor_fractions(params: Dict[str, float]) -> Dict[str, float]:
    """Normalize labor fractions to sum to 1.0 if they exceed it."""
    params = params.copy()
    total = sum(params[p] for p in LABOR_FRACTION_PARAMS)
    if total > 1.0:
        for p in LABOR_FRACTION_PARAMS:
            params[p] = params[p] / total
    return params


def run_simulations_with_params(
    params_dict: Dict[str, float],
    base_model_params: ModelParameters,
    num_simulations: int = 100,
    ai_slowdown_start_year: float = 2030.0,
    end_year: float = 2037.0,
) -> List[Dict[str, Any]]:
    """Run simulations with the given black project parameters."""
    model_params = copy.deepcopy(base_model_params)

    if model_params.black_project is None:
        model_params.black_project = {}
    if 'properties' not in model_params.black_project:
        model_params.black_project['properties'] = {}

    for param_name, value in params_dict.items():
        model_params.black_project['properties'][param_name] = value

    model_params.software_r_and_d['update_software_progress'] = False

    start_year = model_params.settings.get('simulation_start_year', 2026)
    model_params.settings['simulation_end_year'] = end_year
    n_years = end_year - start_year
    model_params.settings['n_eval_points'] = int(n_years * 10) + 1

    simulator = AIFuturesSimulator(model_parameters=model_params)
    simulation_results = simulator.run_simulations(num_simulations=num_simulations)

    world_data_list = []
    for result in simulation_results:
        world_data = extract_world_data(result)
        world_data_list.append(world_data)

    return world_data_list


def compute_objective(
    world_data_list: List[Dict[str, Any]],
    ai_slowdown_start_year: float = 2030.0,
) -> float:
    """Compute median of max operational compute across simulations."""
    max_operational_computes = []

    for world_data in world_data_list:
        bp_data = world_data.get('black_project')
        if bp_data is None:
            max_operational_computes.append(0.0)
            continue

        years = world_data.get('years', [])
        operational_compute = bp_data.get('operational_compute', [])

        if not operational_compute:
            max_operational_computes.append(0.0)
            continue

        max_compute = 0.0
        for i, year in enumerate(years):
            if year >= ai_slowdown_start_year and i < len(operational_compute):
                compute = operational_compute[i]
                if compute > max_compute:
                    max_compute = compute

        max_operational_computes.append(max_compute)

    if not max_operational_computes:
        return 0.0

    return float(np.median(max_operational_computes))


def evaluate_params(
    params: Dict[str, float],
    base_model_params: ModelParameters,
    num_simulations: int,
    verbose: bool = False,
) -> float:
    """Evaluate objective for given parameters."""
    params = normalize_labor_fractions(params)
    world_data_list = run_simulations_with_params(
        params, base_model_params, num_simulations
    )
    return compute_objective(world_data_list)


def ablate_parameter(
    base_params: Dict[str, float],
    param_name: str,
    perturbation_pct: float,
    base_model_params: ModelParameters,
    num_simulations: int,
    verbose: bool = False,
) -> Tuple[float, float, float, float, float]:
    """
    Ablate a single parameter by perturbing it in both directions.

    Returns: (base_value, minus_value, minus_obj, plus_value, plus_obj)
    """
    base_value = base_params[param_name]
    lb, ub = PARAM_BOUNDS[param_name]

    # Calculate perturbation amount
    param_range = ub - lb
    delta = param_range * (perturbation_pct / 100.0)

    # Ensure we don't go out of bounds
    minus_value = max(lb, base_value - delta)
    plus_value = min(ub, base_value + delta)

    # Evaluate minus perturbation
    minus_params = base_params.copy()
    minus_params[param_name] = minus_value
    minus_obj = evaluate_params(minus_params, base_model_params, num_simulations, verbose)

    # Evaluate plus perturbation
    plus_params = base_params.copy()
    plus_params[param_name] = plus_value
    plus_obj = evaluate_params(plus_params, base_model_params, num_simulations, verbose)

    return base_value, minus_value, minus_obj, plus_value, plus_obj


def run_ablation(
    optimized_params: Dict[str, float],
    base_objective: float,
    base_model_params: ModelParameters,
    num_simulations: int,
    perturbation_pct: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run full ablation analysis on all parameters.

    Returns dict with ablation results for each parameter.
    """
    results = {}

    for param_name in optimized_params.keys():
        if verbose:
            print(f"\n  Ablating {param_name}...")

        base_val, minus_val, minus_obj, plus_val, plus_obj = ablate_parameter(
            optimized_params, param_name, perturbation_pct,
            base_model_params, num_simulations, verbose
        )

        # Check if this is a local maximum for this parameter
        is_local_max = (minus_obj <= base_objective) and (plus_obj <= base_objective)

        results[param_name] = {
            "base_value": base_val,
            "base_objective": base_objective,
            "minus_value": minus_val,
            "minus_objective": minus_obj,
            "minus_change": minus_obj - base_objective,
            "plus_value": plus_val,
            "plus_objective": plus_obj,
            "plus_change": plus_obj - base_objective,
            "is_local_max": is_local_max,
        }

        if verbose:
            status = "✓ LOCAL MAX" if is_local_max else "✗ NOT LOCAL MAX"
            print(f"    {param_name}: {status}")
            print(f"      Base: {base_val:.4f} -> {base_objective/1000:.0f} K H100e")
            print(f"      Minus ({minus_val:.4f}): {minus_obj/1000:.0f} K H100e (Δ={minus_obj-base_objective:+.0f})")
            print(f"      Plus ({plus_val:.4f}): {plus_obj/1000:.0f} K H100e (Δ={plus_obj-base_objective:+.0f})")

    return results


def print_summary(results: Dict[str, Any], base_objective: float):
    """Print summary of ablation results."""
    print("\n" + "=" * 80)
    print("ABLATION VALIDATION SUMMARY")
    print("=" * 80)

    local_max_count = sum(1 for r in results.values() if r["is_local_max"])
    total_params = len(results)

    print(f"\nBase objective: {base_objective/1000:.0f} K H100e")
    print(f"Parameters at local maximum: {local_max_count}/{total_params}")

    if local_max_count == total_params:
        print("\n✓ VALIDATION PASSED: All parameters are at local maximum!")
    else:
        print("\n✗ VALIDATION FAILED: Some parameters are NOT at local maximum")
        print("\nParameters that could be improved:")
        for param_name, data in results.items():
            if not data["is_local_max"]:
                if data["minus_change"] > 0:
                    print(f"  - {param_name}: decrease by {(data['base_value'] - data['minus_value']):.4f} "
                          f"could improve by {data['minus_change']/1000:.0f} K H100e")
                if data["plus_change"] > 0:
                    print(f"  - {param_name}: increase by {(data['plus_value'] - data['base_value']):.4f} "
                          f"could improve by {data['plus_change']/1000:.0f} K H100e")

    print("\n" + "-" * 80)
    print("DETAILED RESULTS")
    print("-" * 80)

    for param_name, data in results.items():
        status = "✓" if data["is_local_max"] else "✗"
        print(f"\n{status} {param_name}:")
        print(f"    Optimized value: {data['base_value']:.4f}")
        print(f"    -{perturbation_pct}%: {data['minus_value']:.4f} -> {data['minus_objective']/1000:.0f} K H100e (Δ={data['minus_change']/1000:+.1f} K)")
        print(f"    +{perturbation_pct}%: {data['plus_value']:.4f} -> {data['plus_objective']/1000:.0f} K H100e (Δ={data['plus_change']/1000:+.1f} K)")


def main():
    global perturbation_pct

    parser = argparse.ArgumentParser(
        description="Validate optimization results by ablation analysis"
    )
    parser.add_argument(
        "--results-file", "-r",
        type=str,
        default=str(repo_root / "scripts" / "optimization_results.json"),
        help="Path to optimization results JSON"
    )
    parser.add_argument(
        "--num-simulations", "-n",
        type=int,
        default=100,
        help="Number of simulations per evaluation (default: 100)"
    )
    parser.add_argument(
        "--perturbation-pct", "-p",
        type=float,
        default=10.0,
        help="Perturbation percentage of parameter range (default: 10.0)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path for ablation results (default: scripts/ablation_results.json)"
    )
    args = parser.parse_args()

    perturbation_pct = args.perturbation_pct

    # Load optimization results
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        opt_results = json.load(f)

    optimized_params = opt_results["best_params"]

    print("=" * 80)
    print("ABLATION VALIDATION")
    print("=" * 80)
    print(f"\nLoaded optimization results from: {results_path}")
    print(f"Perturbation: ±{args.perturbation_pct}% of parameter range")
    print(f"Simulations per evaluation: {args.num_simulations}")

    # Load base model parameters
    config_path = repo_root / "ai_futures_simulator" / "parameters" / "default_parameters.yaml"
    base_model_params = ModelParameters.from_yaml(config_path)

    # First, re-evaluate the base objective with more simulations for accuracy
    print("\nRe-evaluating base objective...")
    start_time = time.time()

    base_objective = evaluate_params(
        optimized_params, base_model_params, args.num_simulations, args.verbose
    )
    print(f"Base objective: {base_objective/1000:.0f} K H100e")

    # Run ablation
    print(f"\nRunning ablation analysis ({len(optimized_params)} parameters)...")

    ablation_results = run_ablation(
        optimized_params, base_objective, base_model_params,
        args.num_simulations, args.perturbation_pct, args.verbose
    )

    elapsed = time.time() - start_time
    print(f"\nAblation complete in {elapsed:.1f}s")

    # Print summary
    print_summary(ablation_results, base_objective)

    # Save results
    output_path = Path(args.output) if args.output else repo_root / "scripts" / "ablation_results.json"
    output_data = {
        "optimized_params": optimized_params,
        "base_objective": base_objective,
        "perturbation_pct": args.perturbation_pct,
        "num_simulations": args.num_simulations,
        "ablation_results": ablation_results,
        "elapsed_time_seconds": elapsed,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
