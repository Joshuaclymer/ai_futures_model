"""
AI Futures Simulator

A differentiable simulation of AI futures using continuous-time ODE integration.
Supports gradient-based optimization of parameters.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torchdiffeq import odeint, odeint_event
from typing import Dict, List
from dataclasses import dataclass

from classes.world.world import World
from classes.simulation_primitives import SimulationResult
from parameters.simulation_parameters import SimulationParameters, ModelParameters
from initialize_world_history import initialize_world_for_year
from world_updaters.combined_updater import CombinedUpdater
from world_updaters.software_r_and_d import SoftwareRAndD
from world_updaters.ai_software_developers import AISoftwareDeveloperUpdater


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
        self.combined = CombinedUpdater(
            self._default_params,
            updaters=[SoftwareRAndD(self._default_params), AISoftwareDeveloperUpdater(self._default_params)]
        )

    def run_simulation(
        self,
        world_history: Dict[int, World] = None,
        params: SimulationParameters = None,
    ) -> SimulationResult:
        """
        Run a single simulation trajectory.

        Args:
            world_history: Dictionary mapping years to World states. If None, creates default.
                          Simulation starts from the last year in history.
            params: SimulationParameters to use. If None, samples from model_parameters.

        Returns:
            SimulationResult containing trajectory and metadata.
        """
        if params is None:
            params = self._default_params

        # Create a fresh combined updater if params differ from default
        # This ensures the TakeoffParameters are rebuilt with the new values
        if params is not self._default_params:
            combined = CombinedUpdater(
                params,
                updaters=[SoftwareRAndD(params), AISoftwareDeveloperUpdater(params)]
            )
        else:
            combined = self.combined

        # Get initial world state for simulation start year
        start_year = params.settings.simulation_start_year
        if world_history is None:
            # Only initialize the year we need, not all historical years
            initial_world = initialize_world_for_year(params, start_year)
        else:
            if start_year not in world_history:
                raise ValueError(
                    f"simulation_start_year {start_year} not in world_history. "
                    f"Available years: {sorted(world_history.keys())}"
                )
            initial_world = world_history[start_year]

        # Set template for tensor reconstruction
        combined.set_world_template(initial_world)

        # Time points to evaluate
        start_time = initial_world.current_time.item()
        end_time = params.settings.simulation_end_year

        # Integrate with event detection for discrete state changes
        current_state = initial_world.to_state_tensor()
        current_time = start_time
        current_template = initial_world

        while current_time < end_time:
            event_fn = combined.make_event_fn(end_time)
            event_t, event_state = odeint_event(
                combined,
                current_state,
                torch.tensor(current_time),
                event_fn=event_fn,
                method='dopri5',
                rtol=1e-5,
                atol=1e-6
            )

            event_time = event_t.item()
            current_time = event_time

            # Check if a discrete state change triggered (not just end_time)
            if event_time < end_time:
                world = World.from_state_tensor(event_state, current_template)
                updated_world = combined.set_state_attributes(event_t, world)
                if updated_world is not None:
                    current_state = updated_world.to_state_tensor()
                    current_template = updated_world
                    combined.set_world_template(updated_world)
                else:
                    current_state = event_state
            else:
                current_state = event_state

        # Final integration pass to get states at evenly-spaced evaluation points
        times = torch.linspace(start_time, end_time, params.settings.n_eval_points)
        combined.set_world_template(initial_world)
        state_trajectory = odeint(
            combined,
            initial_world.to_state_tensor(),
            times,
            method='dopri5',
            rtol=1e-5,
            atol=1e-6
        )

        # Reconstruct world states from tensors and compute metrics
        trajectory = []
        for i, t in enumerate(times):
            state_tensor = state_trajectory[i]
            world = World.from_state_tensor(state_tensor, initial_world)
            world = combined.set_metric_attributes(t, world)
            trajectory.append(world)

        return SimulationResult(
            times=times,
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
        "--config",
        type=str,
        default="parameters/modal_parameters.yaml",
        help="Path to YAML config file (default: parameters/modal_parameters.yaml)"
    )
    args = parser.parse_args()

    print("AI Futures Simulator")
    print("=" * 60)

    # Load parameters from YAML
    config_path = Path(args.config)
    print(f"\nLoading parameters from: {config_path}")
    model_params = ModelParameters.from_yaml(config_path)
    simulator = AIFuturesSimulator(model_parameters=model_params)

    print("Running simulation...")

    # Run single simulation
    result = simulator.run_simulation()

    # Print summary
    print_simulation_summary(result)
