"""
Printing utilities for the AI Futures Simulator.
"""

from torch import Tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.simulation_primitives import SimulationResult


def print_simulation_summary(result: "SimulationResult"):
    """Print a summary of a simulation result."""
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)

    params = result.params
    r = params.software_r_and_d
    cg = params.compute_growth
    print(f"\nParameters:")
    print(f"  Start year: {params.settings.simulation_start_year}")
    print(f"  End year: {params.settings.simulation_end_year}")
    print(f"  Eval points: {params.settings.n_eval_points}")
    print(f"  US frontier project compute growth: {cg.us_frontier_project_compute_growth_rate:.4f}")
    print(f"  r_software: {r.r_software:.4f}")
    print(f"  rho_coding_labor: {r.rho_coding_labor:.4f}")

    print(f"\nTrajectory ({len(result.trajectory)} time steps):")
    print("-" * 60)
    print(f"{'Year':<8} {'Progress':<12} {'Rate':<12} {'Automation':<12} {'Compute':<15}")
    print("-" * 60)

    # Print every 10th time step
    for i in range(0, len(result.trajectory), max(1, len(result.trajectory) // 10)):
        t = result.times[i].item()
        world = result.trajectory[i]

        for dev_id, dev in world.ai_software_developers.items():
            sw = dev.ai_software_progress
            progress = sw.progress.item() if isinstance(sw.progress, Tensor) else sw.progress
            rate = sw.progress_rate.item() if isinstance(sw.progress_rate, Tensor) else sw.progress_rate
            auto = sw.automation_fraction.item() if isinstance(sw.automation_fraction, Tensor) else sw.automation_fraction
            compute = dev.compute.total_tpp_h100e

            print(f"{t:<8.2f} {progress:<12.4f} {rate:<12.4f} {auto:<12.4f} {compute:<15.2e}")

    # Print final state
    final_world = result.trajectory[-1]
    final_time = result.times[-1].item()
    print(f"\nFinal State (year {final_time:.1f}):")
    for dev_id, dev in final_world.ai_software_developers.items():
        sw = dev.ai_software_progress
        print(f"  Developer: {dev_id}")
        print(f"  Compute: {dev.compute.total_tpp_h100e:.2e} H100e TPP")

        progress = sw.progress.item() if isinstance(sw.progress, Tensor) else sw.progress
        rate = sw.progress_rate.item() if isinstance(sw.progress_rate, Tensor) else sw.progress_rate
        sw_rate = sw.software_progress_rate.item() if isinstance(sw.software_progress_rate, Tensor) else sw.software_progress_rate
        auto = sw.automation_fraction.item() if isinstance(sw.automation_fraction, Tensor) else sw.automation_fraction
        mult = sw.ai_coding_labor_multiplier.item() if isinstance(sw.ai_coding_labor_multiplier, Tensor) else sw.ai_coding_labor_multiplier

        print(f"  Cumulative Progress: {progress:.4f} OOMs")
        print(f"  Progress Rate: {rate:.4f} OOMs/year")
        print(f"  Software Progress Rate: {sw_rate:.4f} OOMs/year")
        print(f"  Automation Fraction: {auto:.4f}")
        print(f"  AI Coding Labor Multiplier: {mult:.4f}")
