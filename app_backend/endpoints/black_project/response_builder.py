"""
Main response builder for black project simulation API.

Orchestrates the extraction of simulation data and builds the complete
API response by assembling sections from modular builders.
"""

import logging
import numpy as np
from typing import Dict, List, Any

from .world_data import extract_world_data
from .detection import (
    compute_detection_times,
    compute_h100_years_before_detection,
    compute_h100e_before_detection,
    compute_fab_detection_data,
    get_detection_year,
)
from .response_sections import (
    build_black_project_model_section,
    build_black_datacenters_section,
    build_black_fab_section,
    build_initial_black_project_section,
    build_initial_stock_section,
)

logger = logging.getLogger(__name__)

# Constants for energy calculations
H100_TPP_PER_CHIP = 2144.0
H100_WATTS_PER_TPP = 0.326493
ENERGY_EFFICIENCY_RELATIVE_TO_SOTA = 0.20
SOTA_IMPROVEMENT_PER_YEAR = 1.26
H100_RELEASE_YEAR = 2022


def extract_black_project_plot_data(
    simulation_results: Dict[str, Any],
    frontend_params: dict,
) -> Dict[str, Any]:
    """
    Extract plot data from SimulationResult trajectories.
    Returns data formatted for the frontend (reference format).

    This function delegates to extract_reference_format() to produce
    the exact format expected by the frontend.
    """
    return extract_reference_format(simulation_results, frontend_params)


def extract_reference_format(
    simulation_results: Dict[str, Any],
    frontend_params: dict,
) -> Dict[str, Any]:
    """
    Extract plot data from SimulationResult trajectories.
    Returns data formatted to match the reference API.

    This format includes:
    - num_simulations, prob_fab_built, p_project_exists, researcher_headcount (top-level)
    - black_project_model (34 keys)
    - black_datacenters (16 keys)
    - black_fab (25 keys)
    - initial_black_project (4 keys)
    - initial_stock (12 keys)
    """
    results = simulation_results.get('simulation_results', [])
    agreement_year = simulation_results.get('agreement_year', 2029)

    if not results:
        return {"error": "No simulation results"}

    # Extract data from all simulations
    all_data = [extract_world_data(r) for r in results]

    # Use first simulation as reference for years
    raw_years = all_data[0]['years'] if all_data else []
    num_sims = len(all_data)

    # Filter data to only include time points >= agreement_year
    years, all_data = _filter_data_to_agreement_year(all_data, raw_years, agreement_year)

    dt = years[1] - years[0] if len(years) > 1 else 0.1

    # Identify fab-built simulations
    fab_built_sims = [
        d for d in all_data
        if d['black_project'] and any(d['black_project'].get('fab_is_operational', []))
    ]
    num_fab_built = len(fab_built_sims)
    prob_fab_built = num_fab_built / num_sims if num_sims > 0 else 0.0

    # Compute energy data
    energy_by_source, source_labels = _compute_energy_data(all_data, years)

    # Detection metrics
    # Use use_final_year_for_never_detected=True for dashboard individual values
    # Reference model uses final_year for individual values, but 1000 for CCDFs
    detection_times = compute_detection_times(all_data, years, agreement_year, use_final_year_for_never_detected=True)
    h100_years_before_detection = compute_h100_years_before_detection(all_data, years, agreement_year)
    h100e_before_detection = compute_h100e_before_detection(all_data, years, agreement_year)

    # Fab-specific detection data
    fab_individual_h100e, fab_individual_time, fab_individual_process_nodes, fab_individual_energy = \
        compute_fab_detection_data(all_data, years, agreement_year)

    # Energy before detection
    energy_before_detection = _compute_energy_before_detection(all_data, agreement_year, detection_times)

    # PRC capacity data
    prc_capacity_years, prc_capacity_gw, prc_capacity_at_agreement, prc_capacity_samples = \
        _compute_prc_capacity_data(all_data, agreement_year)

    # Build response by assembling sections
    return {
        # Top-level metadata
        "num_simulations": num_sims,
        "prob_fab_built": prob_fab_built,
        "p_project_exists": 0.2,  # Prior probability
        "researcher_headcount": 500,  # Default from reference

        # Sections
        "black_project_model": build_black_project_model_section(
            all_data, years, dt, agreement_year,
            energy_by_source, source_labels,
            detection_times, h100_years_before_detection,
            h100e_before_detection, energy_before_detection
        ),

        "black_datacenters": build_black_datacenters_section(
            all_data, years, dt, agreement_year,
            energy_by_source, source_labels,
            prc_capacity_years, prc_capacity_gw,
            prc_capacity_at_agreement, prc_capacity_samples
        ),

        "black_fab": build_black_fab_section(
            all_data, fab_built_sims, years, dt, agreement_year,
            detection_times, num_sims,
            fab_individual_h100e, fab_individual_time,
            fab_individual_process_nodes, fab_individual_energy
        ),

        "initial_black_project": build_initial_black_project_section(
            all_data, fab_built_sims, years
        ),

        "initial_stock": build_initial_stock_section(
            all_data, years, detection_times, num_sims,
            agreement_year, prc_capacity_years
        ),
    }


def _filter_data_to_agreement_year(
    all_data: List[Dict],
    raw_years: List[float],
    agreement_year: float
) -> tuple:
    """Filter simulation data to only include time points >= agreement_year."""
    if not raw_years:
        return raw_years, all_data

    # Find start index
    start_idx = 0
    for i, y in enumerate(raw_years):
        if y >= agreement_year:
            start_idx = i
            break
    years = raw_years[start_idx:]

    # Filter all time series data
    for d in all_data:
        d['years'] = d['years'][start_idx:]
        d['prc_compute_stock'] = d['prc_compute_stock'][start_idx:]
        d['prc_operating_compute'] = d['prc_operating_compute'][start_idx:]
        # Also filter counterfactual compute stocks (used in reduction ratio calculations)
        d['prc_counterfactual_compute_stock'] = d['prc_counterfactual_compute_stock'][start_idx:]
        d['usa_counterfactual_compute_stock'] = d['usa_counterfactual_compute_stock'][start_idx:]

        if d['black_project']:
            bp = d['black_project']
            time_series_keys = [
                'operational_compute', 'total_compute', 'datacenter_capacity_gw',
                'fab_is_operational', 'cumulative_lr', 'lr_other_intel', 'posterior_prob',
                'lr_reported_energy', 'fab_cumulative_production_h100e', 'fab_monthly_production_h100e',
                'survival_rate', 'initial_compute_surviving_h100e',
                'initial_stock_energy_gw', 'fab_compute_energy_gw', 'total_compute_energy_gw',
                'lr_fab_other', 'lr_fab_combined'
            ]
            for key in time_series_keys:
                if key in bp and isinstance(bp[key], list):
                    bp[key] = bp[key][start_idx:]

    return years, all_data


def _compute_energy_data(all_data: List[Dict], years: List[float]) -> tuple:
    """Compute energy by source and source labels."""
    energy_by_source = []
    for i in range(len(years)):
        initial_energy = float(np.median([
            d['black_project']['initial_stock_energy_gw'][i]
            if d['black_project'] and 'initial_stock_energy_gw' in d['black_project'] and i < len(d['black_project']['initial_stock_energy_gw'])
            else 0.0
            for d in all_data
        ])) if all_data else 0.0

        fab_energy = float(np.median([
            d['black_project']['fab_compute_energy_gw'][i]
            if d['black_project'] and 'fab_compute_energy_gw' in d['black_project'] and i < len(d['black_project']['fab_compute_energy_gw'])
            else 0.0
            for d in all_data
        ])) if all_data else 0.0

        energy_by_source.append([initial_energy, fab_energy])

    # Source labels with energy efficiency
    initial_eff = 700.0 / all_data[0]['black_project']['initial_stock_watts_per_h100e'] if all_data and all_data[0]['black_project'] and 'initial_stock_watts_per_h100e' in all_data[0]['black_project'] else 1.27
    fab_eff = 700.0 / all_data[0]['black_project']['fab_watts_per_h100e'] if all_data and all_data[0]['black_project'] and 'fab_watts_per_h100e' in all_data[0]['black_project'] else 0.07

    source_labels = [
        f"Initial Dark Compute ({initial_eff:.2f}x energy eff.)",
        f"Covert Fab Compute ({fab_eff:.2f}x energy eff.)",
    ]

    return energy_by_source, source_labels


def _compute_energy_before_detection(
    all_data: List[Dict],
    agreement_year: float,
    detection_times: List[float]
) -> List[float]:
    """Compute energy at detection time for each simulation."""
    ENERGY_EFFICIENCY_PRC = 0.2

    energy_before_detection = []
    for i, d in enumerate(all_data):
        bp = d.get('black_project')
        sim_years = d.get('years', [])

        if not bp or not sim_years:
            energy_before_detection.append(0.0)
            continue

        operational_compute = bp.get('operational_compute', [])
        if not operational_compute:
            energy_before_detection.append(0.0)
            continue

        # Get detection year
        det_year = agreement_year + detection_times[i]

        # Find index at detection time
        det_idx = 0
        for j, year in enumerate(sim_years):
            if year >= det_year:
                det_idx = j
                break
            det_idx = j

        if det_idx < len(operational_compute):
            op_h100e = operational_compute[det_idx]
        else:
            op_h100e = operational_compute[-1] if operational_compute else 0.0

        # Use reference formula
        energy_gw = op_h100e * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / ENERGY_EFFICIENCY_PRC / 1e9
        energy_before_detection.append(energy_gw)

    return energy_before_detection


def _compute_prc_capacity_data(
    all_data: List[Dict],
    agreement_year: float
) -> tuple:
    """Compute PRC capacity data from simulation parameters."""
    # Compute state-of-the-art efficiency at agreement year
    years_since_h100 = agreement_year - H100_RELEASE_YEAR
    sota_efficiency_at_agreement = SOTA_IMPROVEMENT_PER_YEAR ** years_since_h100
    combined_efficiency_at_agreement = ENERGY_EFFICIENCY_RELATIVE_TO_SOTA * sota_efficiency_at_agreement

    # Compute PRC energy capacity for each simulation
    prc_capacity_at_agreement_samples = []
    for d in all_data:
        if d.get('prc_params'):
            base_compute = d['prc_params']['total_prc_compute_tpp_h100e_in_2025']
            growth_rate = d['prc_params']['annual_growth_rate']
            years_since_2025 = agreement_year - 2025
            compute_stock = base_compute * (growth_rate ** years_since_2025)
            energy_watts = compute_stock * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / combined_efficiency_at_agreement
            energy_gw = energy_watts / 1e9
            prc_capacity_at_agreement_samples.append(energy_gw)
        else:
            prc_capacity_at_agreement_samples.append(0.175 * (1.74 ** (agreement_year - 2025)))

    # PRC capacity years and trajectories
    prc_capacity_years = list(range(2025, int(agreement_year) + 1))

    prc_capacity_by_sim = []
    for d in all_data:
        if d.get('prc_params'):
            base_compute = d['prc_params']['total_prc_compute_tpp_h100e_in_2025']
            growth_rate = d['prc_params']['annual_growth_rate']
            capacity_trajectory = []
            for year in prc_capacity_years:
                years_since_2025 = year - 2025
                compute_stock = base_compute * (growth_rate ** years_since_2025)
                years_since_h100 = year - H100_RELEASE_YEAR
                sota_eff = SOTA_IMPROVEMENT_PER_YEAR ** years_since_h100
                combined_eff = ENERGY_EFFICIENCY_RELATIVE_TO_SOTA * sota_eff
                energy_gw = (compute_stock * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / combined_eff) / 1e9
                capacity_trajectory.append(energy_gw)
            prc_capacity_by_sim.append(capacity_trajectory)
        else:
            prc_capacity_by_sim.append([0.175 * (1.74 ** (y - 2025)) for y in prc_capacity_years])

    # Compute percentiles
    prc_capacity_array = np.array(prc_capacity_by_sim)
    prc_capacity_gw = {
        "median": np.percentile(prc_capacity_array, 50, axis=0).tolist(),
        "p25": np.percentile(prc_capacity_array, 25, axis=0).tolist(),
        "p75": np.percentile(prc_capacity_array, 75, axis=0).tolist(),
    }
    prc_capacity_at_agreement = float(np.median(prc_capacity_at_agreement_samples))

    return prc_capacity_years, prc_capacity_gw, prc_capacity_at_agreement, prc_capacity_at_agreement_samples
