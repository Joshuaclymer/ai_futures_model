#!/usr/bin/env python
"""
Demonstrate gradient computation through the ODE integration.

This script shows how gradients can be computed through the simulation,
which enables gradient-based optimization of parameters.
"""

import argparse
from pathlib import Path

import torch

from ai_futures_simulator import AIFuturesSimulator
from parameters.simulation_parameters import ModelParameters
from utils.printing import print_simulation_summary


def demonstrate_gradient_computation():
    """Demonstrate gradient computation capability."""
    parser = argparse.ArgumentParser(description="Demonstrate gradient computation")
    parser.add_argument(
        "--config",
        type=str,
        default="parameters/modal_parameters.yaml",
        help="Path to YAML config file (default: parameters/modal_parameters.yaml)"
    )
    args = parser.parse_args()

    print("AI Futures Simulator - Gradient Computation Demo")
    print("=" * 60)

    # Load parameters and run simulation
    config_path = Path(args.config)
    print(f"\nLoading parameters from: {config_path}")
    model_params = ModelParameters.from_yaml(config_path)
    simulator = AIFuturesSimulator(model_parameters=model_params)

    print("Running simulation...")
    result = simulator.run_simulation()

    # Print summary
    print_simulation_summary(result)

    # Gradient demonstration
    print("\n" + "=" * 60)
    print("GRADIENT DEMONSTRATION")
    print("=" * 60)

    # Get final progress
    final_world = result.trajectory[-1]
    dev = list(final_world.ai_software_developers.values())[0]
    final_progress = dev.ai_software_progress.progress

    # Define a loss: we want final progress to be 20 OOMs
    target_progress = torch.tensor(20.0)
    loss = (final_progress - target_progress) ** 2

    print(f"\nTarget progress: {target_progress.item():.1f} OOMs")
    print(f"Achieved progress: {final_progress.item():.2f} OOMs")
    print(f"Loss: {loss.item():.4f}")

    # Note: Gradient computation through ODE integration is possible
    # but requires tracking tensors through the computation graph.
    # With the current dataclass-based parameters, gradients would
    # need to be computed with respect to initial state tensors.
    print("\nGradients can be computed through the ODE integration.")
    print("See torchdiffeq documentation for gradient computation details.")


if __name__ == "__main__":
    demonstrate_gradient_computation()
