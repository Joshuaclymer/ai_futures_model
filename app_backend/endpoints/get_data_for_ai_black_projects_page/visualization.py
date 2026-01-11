"""
Visualization and chart building functions for black project simulation.

Contains functions for building transistor density distributions, watts_per_tpp curves,
and dashboard summary values.
"""

from collections import Counter
import numpy as np
from typing import Dict, List, Any


def build_transistor_density_distribution(all_data: List[Dict]) -> List[Dict]:
    """
    Build transistor density data showing probability distribution across process nodes.

    Returns a list of dicts with {node, density, probability, wattsPerTpp} for each
    process node observed in simulations.
    """
    # Collect process nodes and their densities from all simulations
    node_data = []
    for d in all_data:
        if d['black_project']:
            node_nm = d['black_project'].get('fab_process_node_nm', 28.0)
            density = d['black_project'].get('fab_transistor_density_relative_to_h100', 0.0)
            if node_nm > 0:
                node_data.append((node_nm, density))

    if not node_data:
        # Fallback to static values if no data
        return [
            {"node": "28nm", "density": 0.06, "probability": 0.1, "wattsPerTpp": 9.1},
            {"node": "14nm", "density": 0.15, "probability": 0.2, "wattsPerTpp": 3.8},
            {"node": "7nm", "density": 0.43, "probability": 0.7, "wattsPerTpp": 1.9},
        ]

    # Count occurrences of each process node
    node_counts = Counter([nd[0] for nd in node_data])
    total = len(node_data)

    # Build result with probabilities
    result = []
    for node_nm, count in sorted(node_counts.items(), reverse=True):  # Sort by node size (28nm first)
        # Get density for this node (should be same for all simulations at same node)
        density = next((nd[1] for nd in node_data if nd[0] == node_nm), 0.0)
        probability = count / total

        # Estimate watts per TPP based on density (using power law approximation)
        watts_per_tpp = density ** (-0.5) if density > 0 else 10.0

        result.append({
            "node": f"{int(node_nm)}nm",
            "density": density,
            "probability": probability,
            "wattsPerTpp": watts_per_tpp,
        })

    return result


def build_watts_per_tpp_curve() -> Dict[str, List[float]]:
    """
    Build watts_per_tpp_curve using the Dennard scaling model.

    This curve shows how watts per TPP (relative to H100) varies with transistor density.
    Uses the same model as calculate_watts_per_tpp_from_transistor_density in black_compute.py.
    """
    # Reference values for H100
    h100_transistor_density_m_per_mm2 = 98.28
    h100_watts_per_tpp = 0.326493  # W/TPP for H100

    # Dennard scaling parameters - aligned with reference model
    transistor_density_at_end_of_dennard = 1.98  # M/mm²
    watts_per_tpp_exponent_before_dennard = -2.0
    watts_per_tpp_exponent_after_dennard = -0.91

    # Generate 100 log-spaced density values from 0.001 to 10.0 (relative to H100)
    density_relative_values = np.logspace(-3, 1, 100).tolist()
    watts_per_tpp_relative_values = []

    for density_relative in density_relative_values:
        # Convert relative density to absolute (M/mm²)
        transistor_density = density_relative * h100_transistor_density_m_per_mm2

        # Calculate watts_per_tpp at the Dennard transition point using post-Dennard relationship
        transition_density_ratio = transistor_density_at_end_of_dennard / h100_transistor_density_m_per_mm2
        transition_watts_per_tpp = h100_watts_per_tpp * (transition_density_ratio ** watts_per_tpp_exponent_after_dennard)

        if transistor_density < transistor_density_at_end_of_dennard:
            # Before Dennard scaling ended - anchor to transition point
            exponent = watts_per_tpp_exponent_before_dennard
            density_ratio = transistor_density / transistor_density_at_end_of_dennard
            watts_per_tpp = transition_watts_per_tpp * (density_ratio ** exponent)
        else:
            # After Dennard scaling ended - anchor to H100
            exponent = watts_per_tpp_exponent_after_dennard
            density_ratio = transistor_density / h100_transistor_density_m_per_mm2
            watts_per_tpp = h100_watts_per_tpp * (density_ratio ** exponent)

        # Convert to relative (relative to H100)
        watts_per_tpp_relative = watts_per_tpp / h100_watts_per_tpp
        watts_per_tpp_relative_values.append(watts_per_tpp_relative)

    return {
        "density_relative": density_relative_values,
        "watts_per_tpp_relative": watts_per_tpp_relative_values,
    }


def compute_fab_dashboard(
    fab_built_sims: List[Dict],
    all_data: List[Dict],
    detection_times: List[float],
    num_sims: int,
    dt: float,
    fab_individual_h100e: List[float] = None,
    fab_individual_time: List[float] = None,
    fab_individual_energy: List[float] = None
) -> Dict[str, Any]:
    """
    Compute pre-formatted dashboard values for the covert fab section.
    This avoids frontend computation.

    If fab_individual_* parameters are provided (from compute_fab_detection_data), use those
    for correct fab-specific detection calculations. Otherwise falls back to project detection.
    """
    num_fab_built = len(fab_built_sims)
    prob_fab_built = f"{(num_fab_built / num_sims * 100):.1f}%" if num_sims > 0 else "--"

    # Use fab-specific detection data if provided
    if fab_individual_h100e is not None:
        h100e_before = fab_individual_h100e
    else:
        # Fallback: compute using project detection times (old behavior)
        h100e_before = []
        for d in fab_built_sims:
            idx = all_data.index(d)
            if d['black_project'] and d['black_project'].get('fab_cumulative_production_h100e'):
                fab_prod = d['black_project']['fab_cumulative_production_h100e']
                det_idx = min(int(detection_times[idx] / dt) if dt > 0 else 0, len(fab_prod) - 1)
                h100e_before.append(fab_prod[det_idx])

    if h100e_before:
        sorted_h100e = sorted(h100e_before)
        median_h100e = sorted_h100e[len(sorted_h100e) // 2]
        if median_h100e >= 1_000_000:
            production_str = f"{median_h100e / 1_000_000:.1f}M H100e"
        elif median_h100e >= 1_000:
            production_str = f"{median_h100e / 1_000:.0f}K H100e"
        else:
            production_str = f"{median_h100e:.0f} H100e"
    else:
        median_h100e = 0
        production_str = "--"

    # Use fab-specific time if provided
    if fab_individual_time is not None:
        time_before = fab_individual_time
    else:
        # Fallback: compute using project detection times (old behavior)
        time_before = [detection_times[all_data.index(d)] for d in fab_built_sims]

    if time_before:
        sorted_time = sorted(time_before)
        median_time = sorted_time[len(sorted_time) // 2]
        years_operational_str = f"{median_time:.1f} yrs"
    else:
        years_operational_str = "--"

    # Get most common process node
    process_nodes = [
        f"{int(d['black_project'].get('fab_process_node_nm', 28))}nm"
        if d['black_project'] else "28nm"
        for d in fab_built_sims
    ]
    if process_nodes:
        node_counts = Counter(process_nodes)
        process_node_str = node_counts.most_common(1)[0][0]
    else:
        process_node_str = "--"

    # Compute median energy before detection
    energy_before = []
    for d in fab_built_sims:
        idx = all_data.index(d)
        if d['black_project'] and d['black_project'].get('total_compute_energy_gw'):
            energy = d['black_project']['total_compute_energy_gw']
            det_idx = min(int(detection_times[idx] / dt) if dt > 0 else 0, len(energy) - 1)
            energy_before.append(energy[det_idx])

    if energy_before:
        sorted_energy = sorted(energy_before)
        median_energy = sorted_energy[len(sorted_energy) // 2]
        if median_energy >= 1:
            energy_str = f"{median_energy:.1f} GW"
        else:
            energy_str = f"{median_energy * 1000:.0f} MW"
    else:
        energy_str = "--"

    return {
        "production": production_str,
        "energy": energy_str,
        "probFabBuilt": prob_fab_built,
        "yearsOperational": years_operational_str,
        "processNode": process_node_str,
    }
