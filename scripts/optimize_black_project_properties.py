#!/usr/bin/env python3
"""
Optimize black project properties to maximize median covert computation.

Uses a hybrid optimization approach combining:
1. Differential Evolution (discrete/global search)
2. Local refinement with L-BFGS-B (gradient-based)

The simulation is partially differentiable and partially discrete, so this
hybrid approach helps find good solutions in the mixed search space.

Usage:
    python scripts/optimize_black_project_properties.py [options]

Options:
    --num-simulations N    Simulations per evaluation (default: 50)
    --max-iterations N     Max DE iterations (default: 100)
    --population-size N    DE population size (default: 15)
    --local-refinement     Run local optimization after DE (default: True)
    --seed S               Random seed (default: 42)
    --verbose              Print detailed progress
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.optimize import differential_evolution, minimize

# Add paths for imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "ai_futures_simulator"))
sys.path.insert(0, str(repo_root / "app_backend"))

from ai_futures_simulator import AIFuturesSimulator
from parameters.model_parameters import ModelParameters
from endpoints.black_project.world_data import extract_world_data


# Parameter definitions with bounds
# Each parameter: (name, lower_bound, upper_bound, default_value)
OPTIMIZABLE_PARAMS = [
    # Labor allocation (fractions must sum to <= 1.0)
    ("total_labor", 1000, 50000, 11300),
    ("fraction_of_labor_devoted_to_datacenter_construction", 0.0, 1.0, 0.885),
    ("fraction_of_labor_devoted_to_black_fab_construction", 0.0, 1.0, 0.022),
    ("fraction_of_labor_devoted_to_black_fab_operation", 0.0, 1.0, 0.049),
    ("fraction_of_labor_devoted_to_ai_research", 0.0, 1.0, 0.044),
    # Diverted resources
    ("fraction_of_initial_compute_stock_to_divert_at_black_project_start", 0.0, 0.5, 0.05),
    ("fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start", 0.0, 0.2, 0.01),
    ("fraction_of_lithography_scanners_to_divert_at_black_project_start", 0.0, 0.5, 0.10),
    # Energy constraint
    ("max_fraction_of_total_national_energy_consumption", 0.01, 0.20, 0.05),
    # Timing
    ("years_before_black_project_start_to_begin_datacenter_construction", 0.0, 5.0, 1.0),
]

# Indices for labor fraction parameters (for constraint enforcement)
LABOR_FRACTION_INDICES = [1, 2, 3, 4]  # datacenter, fab_construction, fab_operation, ai_research


@dataclass
class OptimizationResult:
    """Result of optimization run."""
    best_params: Dict[str, float]
    best_median_compute: float
    optimization_history: List[Tuple[int, float]]
    total_evaluations: int
    elapsed_time: float


def params_to_dict(x: np.ndarray) -> Dict[str, float]:
    """Convert parameter array to dictionary."""
    return {name: float(x[i]) for i, (name, _, _, _) in enumerate(OPTIMIZABLE_PARAMS)}


def dict_to_params(d: Dict[str, float]) -> np.ndarray:
    """Convert dictionary to parameter array."""
    return np.array([d[name] for name, _, _, _ in OPTIMIZABLE_PARAMS])


def get_bounds() -> List[Tuple[float, float]]:
    """Get bounds for all parameters."""
    return [(lb, ub) for _, lb, ub, _ in OPTIMIZABLE_PARAMS]


def get_default_params() -> np.ndarray:
    """Get default parameter values."""
    return np.array([default for _, _, _, default in OPTIMIZABLE_PARAMS])


def normalize_labor_fractions(x: np.ndarray) -> np.ndarray:
    """Normalize labor fractions to sum to 1.0 if they exceed it."""
    x = x.copy()
    labor_fractions = x[LABOR_FRACTION_INDICES]
    total = labor_fractions.sum()
    if total > 1.0:
        # Normalize to sum to 1.0
        x[LABOR_FRACTION_INDICES] = labor_fractions / total
    return x


def run_simulations_with_params(
    params_dict: Dict[str, float],
    base_model_params: ModelParameters,
    num_simulations: int = 50,
    agreement_year: float = 2030.0,
    end_year: float = 2037.0,
) -> List[Dict[str, Any]]:
    """
    Run simulations with the given black project parameters.

    Returns list of extracted world data dictionaries, one per simulation.
    """
    # Deep copy the model parameters to avoid modifying the original
    model_params = copy.deepcopy(base_model_params)

    # Update black project properties
    if model_params.black_project is None:
        model_params.black_project = {}
    if 'properties' not in model_params.black_project:
        model_params.black_project['properties'] = {}

    for param_name, value in params_dict.items():
        model_params.black_project['properties'][param_name] = value

    # Disable software progress updates (only compute matters for black projects)
    model_params.software_r_and_d['update_software_progress'] = False

    # Set simulation time range
    start_year = model_params.settings.get('simulation_start_year', 2026)
    model_params.settings['simulation_end_year'] = end_year
    n_years = end_year - start_year
    model_params.settings['n_eval_points'] = int(n_years * 10) + 1

    # Create simulator and run
    simulator = AIFuturesSimulator(model_parameters=model_params)
    simulation_results = simulator.run_simulations(num_simulations=num_simulations)

    # Extract world data from each simulation
    world_data_list = []
    for result in simulation_results:
        world_data = extract_world_data(result)
        world_data_list.append(world_data)

    return world_data_list


def compute_objective(
    world_data_list: List[Dict[str, Any]],
    agreement_year: float = 2030.0,
) -> float:
    """
    Compute the optimization objective: median of max operational compute.

    For each simulation, we compute the maximum operational compute achieved
    after the agreement year. Then we return the median across all simulations.

    Higher is better (we maximize this).
    """
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

        # Find max operational compute after agreement year
        max_compute = 0.0
        for i, year in enumerate(years):
            if year >= agreement_year and i < len(operational_compute):
                compute = operational_compute[i]
                if compute > max_compute:
                    max_compute = compute

        max_operational_computes.append(max_compute)

    # Return median (negated because scipy minimizes)
    if not max_operational_computes:
        return 0.0

    return float(np.median(max_operational_computes))


class BlackProjectOptimizer:
    """
    Hybrid optimizer for black project properties.

    Combines Differential Evolution (global search) with local refinement.
    """

    def __init__(
        self,
        num_simulations: int = 50,
        agreement_year: float = 2030.0,
        end_year: float = 2037.0,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.num_simulations = num_simulations
        self.agreement_year = agreement_year
        self.end_year = end_year
        self.seed = seed
        self.verbose = verbose

        # Load base model parameters
        config_path = repo_root / "ai_futures_simulator" / "parameters" / "default_parameters.yaml"
        self.base_model_params = ModelParameters.from_yaml(config_path)

        # Tracking
        self.evaluation_count = 0
        self.best_objective = -np.inf
        self.best_params = None
        self.history = []

    def objective_function(self, x: np.ndarray) -> float:
        """
        Objective function for optimization.

        Returns negative median operational compute (since scipy minimizes).
        """
        self.evaluation_count += 1

        # Normalize labor fractions to satisfy constraint
        x = normalize_labor_fractions(x)

        # Convert to dictionary
        params_dict = params_to_dict(x)

        # Run simulations
        try:
            world_data_list = run_simulations_with_params(
                params_dict,
                self.base_model_params,
                num_simulations=self.num_simulations,
                agreement_year=self.agreement_year,
                end_year=self.end_year,
            )

            # Compute objective (median max operational compute)
            objective_value = compute_objective(world_data_list, self.agreement_year)

        except Exception as e:
            if self.verbose:
                print(f"  Error in evaluation {self.evaluation_count}: {e}")
            objective_value = 0.0

        # Track best
        if objective_value > self.best_objective:
            self.best_objective = objective_value
            self.best_params = params_dict.copy()
            self.history.append((self.evaluation_count, objective_value))
            if self.verbose:
                # Convert to K H100e for readability
                obj_k = objective_value / 1000
                print(f"  [Eval {self.evaluation_count}] New best: {obj_k:.0f} K H100e")

        # Return negative for minimization
        return -objective_value

    def de_callback(self, xk, convergence):
        """Callback for Differential Evolution to report progress."""
        if self.verbose and self.evaluation_count % 10 == 0:
            obj_k = self.best_objective / 1000
            print(f"  DE iteration: {self.evaluation_count} evaluations, best: {obj_k:.0f} K H100e")
        return False  # Don't stop early

    def optimize(
        self,
        max_iterations: int = 100,
        population_size: int = 15,
        local_refinement: bool = True,
    ) -> OptimizationResult:
        """
        Run the hybrid optimization.

        Args:
            max_iterations: Maximum iterations for Differential Evolution
            population_size: Population size for DE
            local_refinement: Whether to run local optimization after DE

        Returns:
            OptimizationResult with best parameters and history
        """
        start_time = time.time()
        bounds = get_bounds()

        print(f"Starting optimization with {self.num_simulations} simulations per evaluation")
        print(f"  DE max iterations: {max_iterations}")
        print(f"  DE population size: {population_size}")
        print(f"  Local refinement: {local_refinement}")
        print()

        # Phase 1: Differential Evolution (global search)
        print("Phase 1: Differential Evolution (global search)...")

        de_result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=max_iterations,
            popsize=population_size,
            mutation=(0.5, 1.0),  # Adaptive mutation
            recombination=0.7,
            seed=self.seed,
            callback=self.de_callback,
            polish=False,  # We'll do our own local refinement
            disp=self.verbose,
            workers=1,  # Single-threaded (simulations are already parallel internally)
        )

        de_best = normalize_labor_fractions(de_result.x)
        de_objective = -de_result.fun

        print(f"\nDE complete: {self.evaluation_count} evaluations")
        print(f"  Best median operational compute: {de_objective/1000:.0f} K H100e")

        # Phase 2: Local refinement (optional)
        if local_refinement:
            print("\nPhase 2: Local refinement (L-BFGS-B)...")

            # Use bounded optimization for local search
            local_result = minimize(
                self.objective_function,
                de_best,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 50,
                    'disp': self.verbose,
                }
            )

            local_best = normalize_labor_fractions(local_result.x)
            local_objective = -local_result.fun

            print(f"\nLocal refinement complete: {self.evaluation_count} total evaluations")
            print(f"  Best median operational compute: {local_objective/1000:.0f} K H100e")

            # Use local result if better
            if local_objective > de_objective:
                final_params = params_to_dict(local_best)
                final_objective = local_objective
            else:
                final_params = params_to_dict(de_best)
                final_objective = de_objective
        else:
            final_params = params_to_dict(de_best)
            final_objective = de_objective

        elapsed = time.time() - start_time

        return OptimizationResult(
            best_params=final_params,
            best_median_compute=final_objective,
            optimization_history=self.history,
            total_evaluations=self.evaluation_count,
            elapsed_time=elapsed,
        )


def print_results(result: OptimizationResult, defaults: Dict[str, float]):
    """Print optimization results with comparison to defaults."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\nBest median operational compute: {result.best_median_compute/1000:.0f} K H100e")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Elapsed time: {result.elapsed_time:.1f}s")

    print("\nOptimized parameters (vs defaults):")
    print("-" * 70)

    for name, _, _, default in OPTIMIZABLE_PARAMS:
        optimized = result.best_params[name]
        change = ((optimized - default) / default * 100) if default != 0 else 0
        change_str = f"{change:+.1f}%" if abs(change) > 0.1 else "no change"
        print(f"  {name}:")
        print(f"    Default: {default:.4f} -> Optimized: {optimized:.4f} ({change_str})")

    # Check labor fraction constraint
    labor_fractions = [
        result.best_params["fraction_of_labor_devoted_to_datacenter_construction"],
        result.best_params["fraction_of_labor_devoted_to_black_fab_construction"],
        result.best_params["fraction_of_labor_devoted_to_black_fab_operation"],
        result.best_params["fraction_of_labor_devoted_to_ai_research"],
    ]
    print(f"\nLabor fractions sum: {sum(labor_fractions):.4f} (should be <= 1.0)")


def save_results(result: OptimizationResult, output_path: Path):
    """Save optimization results to JSON file."""
    output = {
        "best_params": result.best_params,
        "best_median_compute": result.best_median_compute,
        "total_evaluations": result.total_evaluations,
        "elapsed_time_seconds": result.elapsed_time,
        "optimization_history": [
            {"evaluation": ev, "objective": obj}
            for ev, obj in result.optimization_history
        ],
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize black project properties to maximize median covert computation"
    )
    parser.add_argument(
        "--num-simulations", "-n",
        type=int,
        default=50,
        help="Number of simulations per evaluation (default: 50)"
    )
    parser.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=100,
        help="Maximum DE iterations (default: 100)"
    )
    parser.add_argument(
        "--population-size", "-p",
        type=int,
        default=15,
        help="DE population size (default: 15)"
    )
    parser.add_argument(
        "--no-local-refinement",
        action="store_true",
        help="Skip local refinement phase"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
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
        help="Output file path for results (default: scripts/optimization_results.json)"
    )
    args = parser.parse_args()

    # Create optimizer
    optimizer = BlackProjectOptimizer(
        num_simulations=args.num_simulations,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Run optimization
    result = optimizer.optimize(
        max_iterations=args.max_iterations,
        population_size=args.population_size,
        local_refinement=not args.no_local_refinement,
    )

    # Get defaults for comparison
    defaults = {name: default for name, _, _, default in OPTIMIZABLE_PARAMS}

    # Print results
    print_results(result, defaults)

    # Save results
    output_path = Path(args.output) if args.output else repo_root / "scripts" / "optimization_results.json"
    save_results(result, output_path)


if __name__ == "__main__":
    main()
