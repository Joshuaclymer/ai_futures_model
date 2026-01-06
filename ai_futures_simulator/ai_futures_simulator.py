"""
AI Futures Simulator

A differentiable simulation of AI futures using continuous-time ODE integration.
Supports gradient-based optimization of parameters.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torchdiffeq import odeint, odeint_event
from typing import Dict, List, Tuple

from classes.world.world import World
from classes.flat_world import FlatWorld
from classes.simulation_primitives import SimulationResult
from parameters.classes import SimulationParameters, ModelParameters
from initialize_world_history import initialize_world_for_year
from world_updaters.combined_updater import CombinedUpdater, FlatCombinedUpdater


# ODE solver settings
DEFAULT_ODE_RTOL = 0.002  # Relative tolerance
DEFAULT_ODE_ATOL = 5e-5   # Absolute tolerance


class AIFuturesSimulator(nn.Module):
    """
    Main differentiable simulator for AI futures.

    Accepts ModelParameters which specifies parameter distributions (or point estimates).
    Use run_simulation() for a single trajectory, run_simulations() for Monte Carlo.
    """

    def __init__(self, model_parameters: ModelParameters):
        """
        Initialize the simulator.

        Args:
            model_parameters: Parameter specifications (distributions or point estimates).
                              Load from YAML using ModelParameters.from_yaml().
        """
        super().__init__()
        self.model_parameters = model_parameters

        # Sample initial params for the default combined updater
        self._default_params = model_parameters.sample()
        # Let CombinedUpdater create updaters based on params (includes black project if enabled)
        self.combined = CombinedUpdater(self._default_params)

    def _discover_events(
        self,
        combined: CombinedUpdater,
        initial_state: Tensor,
        initial_template: World,
        start_time: float,
        end_time: float,
        rtol: float,
        atol: float,
    ) -> List[Tuple[float, Tensor, World]]:
        """
        Discover discrete events dynamically using odeint_event.

        Returns list of (time, state_tensor, world_template) tuples for each segment start.
        """
        segments = [(start_time, initial_state, initial_template)]

        current_state = initial_state
        current_time = start_time
        current_template = initial_template

        while current_time < end_time:
            event_fn = combined.make_event_fn(end_time)
            event_t, event_state = odeint_event(
                combined,
                current_state,
                torch.tensor(current_time),
                event_fn=event_fn,
                method='dopri5',
                rtol=rtol,
                atol=atol
            )

            event_time = event_t.item()
            current_time = event_time

            # odeint_event returns trajectory [N, state_dim], extract final state
            final_state = event_state[-1] if event_state.dim() > 1 else event_state

            if event_time < end_time:
                world = World.from_state_tensor(final_state, current_template)
                updated_world = combined.set_state_attributes(event_t, world)
                if updated_world is not None:
                    current_state = updated_world.to_state_tensor()
                    current_template = updated_world
                    combined.set_world_template(updated_world)
                    segments.append((event_time, current_state, current_template))
                else:
                    current_state = final_state
            else:
                current_state = final_state

        return segments

    def run_simulation(
        self,
        world_history: Dict[int, World] = None,
        params: SimulationParameters = None,
        use_flat_mode: bool = True,
    ) -> SimulationResult:
        """
        Run a single simulation trajectory.

        Uses two-phase approach:
        1. Discover discrete events dynamically using odeint_event
        2. Collect trajectory at eval points using segment-wise integration

        Args:
            world_history: Dictionary mapping years to World states. If None, creates default.
                          Simulation starts from the last year in history.
            params: SimulationParameters to use. If None, samples from model_parameters.
            use_flat_mode: If True (default), use flat tensor optimization for ~5x speedup.

        Returns:
            SimulationResult containing trajectory and metadata.
        """
        if params is None:
            params = self._default_params

        # Create a fresh combined updater if params differ from default
        if params is not self._default_params:
            # Let CombinedUpdater create updaters based on params (includes black project if enabled)
            combined = CombinedUpdater(params)
        else:
            combined = self.combined

        # Get initial world state for simulation start year
        start_year = params.settings.simulation_start_year
        if world_history is None:
            initial_world = initialize_world_for_year(params, start_year)
        else:
            if start_year not in world_history:
                raise ValueError(
                    f"simulation_start_year {start_year} not in world_history. "
                    f"Available years: {sorted(world_history.keys())}"
                )
            initial_world = world_history[start_year]

        combined.set_world_template(initial_world)

        start_time = initial_world.current_time.item()
        end_time = params.settings.simulation_end_year

        # Get ODE settings
        settings = params.settings
        rtol = getattr(settings, 'ode_rtol', DEFAULT_ODE_RTOL)
        atol = getattr(settings, 'ode_atol', DEFAULT_ODE_ATOL)

        # Integrate all eval points at once, then check for discrete events
        eval_times = torch.linspace(start_time, end_time, settings.n_eval_points)

        # Single odeint call for all eval times
        initial_state = initial_world.to_state_tensor()

        if use_flat_mode:
            # Use flat tensor optimization for fast ODE integration
            flat_world = FlatWorld.from_world(initial_world)
            flat_combined = FlatCombinedUpdater(combined, flat_world)
            states = odeint(
                flat_combined, initial_state, eval_times,
                method='dopri5', rtol=rtol, atol=atol
            )
        else:
            # Use original nested dataclass approach
            states = odeint(
                combined, initial_state, eval_times,
                method='dopri5', rtol=rtol, atol=atol
            )

        # Convert states to World objects and apply discrete events
        trajectory = []
        current_template = initial_world

        for i in range(len(eval_times)):
            t = eval_times[i]
            state_tensor = states[i]

            world = World.from_state_tensor(state_tensor, current_template)

            # Check for and apply discrete events
            updated = combined.set_state_attributes(t, world)
            if updated is not None:
                world = updated
                current_template = world
                combined.set_world_template(world)

            # Compute metrics
            world = combined.set_metric_attributes(t, world)
            trajectory.append(world)

        return SimulationResult(
            times=eval_times,
            trajectory=trajectory,
            params=params,
        )

    def run_simulations(
        self,
        num_simulations: int = 1,
        model_parameters: ModelParameters = None,
    ) -> List[SimulationResult]:
        """
        Run multiple simulation trajectories.

        Args:
            num_simulations: Number of trajectories to generate.
            model_parameters: Parameters to use. If None, uses instance defaults.

        Returns:
            List of SimulationResult objects.
        """
        import numpy as np

        if model_parameters is not None:
            self.model_parameters = model_parameters

        # Create a single rng to use across all samples for proper randomization
        rng = np.random.default_rng(self.model_parameters.seed)

        results = []
        for _ in range(num_simulations):
            params = self.model_parameters.sample(rng=rng)
            result = self.run_simulation(params=params)
            results.append(result)

        return results

    def forward(
        self,
        world_history: Dict[int, World] = None,
    ) -> SimulationResult:
        """
        Forward pass for gradient computation.

        Equivalent to run_simulation but follows nn.Module convention.
        """
        return self.run_simulation(world_history)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from utils.printing import print_simulation_summary

    parser = argparse.ArgumentParser(description="AI Futures Simulator")
    parser.add_argument(
        "--params",
        type=str,
        default="ai_futures_simulator/parameters/monte_carlo_parameters.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="simulation_output.json",
        help="Path to output JSON file"
    )
    args = parser.parse_args()

    print("AI Futures Simulator")
    print("=" * 60)

    # Load parameters from YAML
    params_path = Path(args.params)
    print(f"\nLoading parameters from: {params_path}")
    model_params = ModelParameters.from_yaml(params_path)
    simulator = AIFuturesSimulator(model_parameters=model_params)

    print("Running simulation...")

    # Run single simulation
    result = simulator.run_simulation()

    # Print summary and save to output file
    print_simulation_summary(result, output_file=args.output)
