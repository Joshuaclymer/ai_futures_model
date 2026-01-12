"""
World data extraction functions for black project simulation.

Extracts time series data from SimulationTrajectory trajectories.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add ai_futures_simulator subdirectory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "ai_futures_simulator"))

from classes.simulation_primitives import SimulationTrajectory
from classes.world.entities import NamedNations
from .utils import to_float


def extract_world_data(result: SimulationTrajectory) -> Dict[str, Any]:
    """Extract time series data from a single SimulationTrajectory."""
    trajectory = result.trajectory
    times = result.times.tolist()

    # Extract sampled parameters for PRC compute calculations
    prc_params = None
    if hasattr(result, 'params') and result.params:
        compute = result.params.compute
        if compute and hasattr(compute, 'PRCComputeParameters'):
            prc_params = {
                'total_prc_compute_tpp_h100e_in_2025': compute.PRCComputeParameters.total_prc_compute_tpp_h100e_in_2025,
                'annual_growth_rate': compute.PRCComputeParameters.annual_growth_rate_of_prc_compute_stock,
                'proportion_domestic_2026': compute.PRCComputeParameters.proportion_of_prc_chip_stock_produced_domestically_2026,
                'proportion_domestic_2030': compute.PRCComputeParameters.proportion_of_prc_chip_stock_produced_domestically_2030,
            }

    data = {
        'years': times,
        'prc_compute_stock': [],
        'prc_operating_compute': [],
        # Counterfactual (no slowdown) nation data for reduction ratio calculations
        'prc_counterfactual_compute_stock': [],  # PRC compute trajectory without slowdown
        'usa_counterfactual_compute_stock': [],  # Largest AI company compute trajectory without slowdown
        'black_project': None,
        'prc_params': prc_params,  # Store sampled parameters for yearly calculations
    }

    for world in trajectory:
        # Extract PRC nation data
        prc = world.nations.get(NamedNations.PRC)
        if prc:
            data['prc_compute_stock'].append(
                to_float(prc.compute_stock.functional_tpp_h100e) if prc.compute_stock else 0.0
            )
            data['prc_operating_compute'].append(to_float(prc.operating_compute_tpp_h100e) if prc.operating_compute_tpp_h100e else 0.0)
        else:
            data['prc_compute_stock'].append(0.0)
            data['prc_operating_compute'].append(0.0)

        # Extract counterfactual nation data (for AI R&D reduction ratio calculations)
        prc_cf = world.nations.get(NamedNations.PRC_COUNTERFACTUAL_NO_SLOWDOWN)
        if prc_cf:
            data['prc_counterfactual_compute_stock'].append(
                to_float(prc_cf.compute_stock.functional_tpp_h100e) if prc_cf.compute_stock else 0.0
            )
        else:
            data['prc_counterfactual_compute_stock'].append(0.0)

        usa_cf = world.nations.get(NamedNations.USA_COUNTERFACTUAL_NO_SLOWDOWN)
        if usa_cf:
            data['usa_counterfactual_compute_stock'].append(
                to_float(usa_cf.compute_stock.functional_tpp_h100e) if usa_cf.compute_stock else 0.0
            )
        else:
            data['usa_counterfactual_compute_stock'].append(0.0)

    # Extract black project data if present
    if world.black_projects:
        bp_id = list(world.black_projects.keys())[0]

        # Get detection-related data from the first world state (these are set at init)
        first_bp = trajectory[0].black_projects.get(bp_id) if trajectory else None

        bp_data = {
            'id': bp_id,
            # Black project start year (for filtering output data)
            'black_project_start_year': to_float(first_bp.black_project_start_year) if first_bp and hasattr(first_bp, 'black_project_start_year') else None,
            # Time series (extracted from trajectory, one value per time step)
            'operational_compute': [],
            'total_compute': [],
            'datacenter_capacity_gw': [],
            'fab_is_operational': [],
            'cumulative_lr': [],  # Combined LR over time (full project)
            'lr_other_intel': [],  # Direct evidence LR over time (full project)
            'posterior_prob': [],  # Posterior probability over time
            'lr_reported_energy': [],  # Energy accounting LR over time
            'lr_fab_other': [],  # Fab-specific worker detection LR (uses fab labor)
            'lr_fab_combined': [],  # Fab's combined LR = lr_inventory × lr_procurement × lr_fab_other
            'fab_cumulative_production_h100e': [],  # Cumulative fab production
            'fab_monthly_production_h100e': [],  # Monthly production rate
            'survival_rate': [],  # Chip survival rate
            'initial_compute_surviving_h100e': [],  # Initial diverted compute surviving
            # Energy consumption by source (GW)
            'initial_stock_energy_gw': [],  # Energy from initial diverted compute
            'fab_compute_energy_gw': [],  # Energy from fab-produced compute
            'total_compute_energy_gw': [],  # Total energy consumption
            # Static values (from first/last world state)
            'sampled_detection_time': to_float(first_bp.sampled_detection_time) if first_bp and hasattr(first_bp, 'sampled_detection_time') else None,
            'lr_compute_accounting': to_float(first_bp.lr_compute_accounting) if first_bp and hasattr(first_bp, 'lr_compute_accounting') else 1.0,
            'lr_sme_inventory': to_float(first_bp.lr_sme_inventory) if first_bp and hasattr(first_bp, 'lr_sme_inventory') else 1.0,
            'lr_satellite_datacenter': to_float(first_bp.lr_satellite_datacenter) if first_bp and hasattr(first_bp, 'lr_satellite_datacenter') else 1.0,
            # Initial diverted compute (constant, set at black project start, never changes)
            'initial_diverted_compute_h100e': to_float(first_bp.initial_diverted_compute_h100e) if first_bp and hasattr(first_bp, 'initial_diverted_compute_h100e') else 0.0,
            # Fab static properties (from final world state for accurate values)
            'fab_wafer_starts_per_month': 0.0,  # Will be updated from final state
            'fab_architecture_efficiency': 1.0,  # Will be updated from final state
            'fab_transistor_density_relative_to_h100': 0.0,  # Will be updated from final state
            'fab_process_node_nm': 28.0,  # Will be updated from final state
            'fab_chips_per_wafer': 28,  # Will be updated from final state
            # Energy efficiency values (for labels)
            'initial_stock_watts_per_h100e': 700.0,  # Will be updated from first state
            'fab_watts_per_h100e': 700.0,  # Will be updated from final state
        }

        for world in trajectory:
            bp = world.black_projects.get(bp_id)
            if bp:
                # Core compute metrics
                bp_data['operational_compute'].append(to_float(bp.operating_compute_tpp_h100e))
                bp_data['total_compute'].append(
                    to_float(bp.compute_stock.functional_tpp_h100e) if bp.compute_stock else 0.0
                )
                bp_data['datacenter_capacity_gw'].append(
                    to_float(bp.datacenters.data_center_capacity_gw) if bp.datacenters else 0.0
                )
                bp_data['fab_is_operational'].append(bp.fab_is_operational if hasattr(bp, 'fab_is_operational') else False)

                # LR metrics (point-in-time values extracted from trajectory)
                bp_data['cumulative_lr'].append(to_float(bp.cumulative_lr) if hasattr(bp, 'cumulative_lr') else 1.0)
                bp_data['lr_other_intel'].append(to_float(bp.lr_other_intel) if hasattr(bp, 'lr_other_intel') else 1.0)
                bp_data['posterior_prob'].append(to_float(bp.posterior_prob) if hasattr(bp, 'posterior_prob') else 0.3)
                bp_data['lr_reported_energy'].append(to_float(bp.lr_reported_energy) if hasattr(bp, 'lr_reported_energy') else 1.0)

                # Fab-specific LR metrics
                bp_data['lr_fab_other'].append(to_float(bp.lr_fab_other) if hasattr(bp, 'lr_fab_other') else 1.0)
                bp_data['lr_fab_combined'].append(to_float(bp.lr_fab_combined) if hasattr(bp, 'lr_fab_combined') else 1.0)

                # Fab production metrics
                bp_data['fab_cumulative_production_h100e'].append(
                    to_float(bp.fab_cumulative_production_h100e) if hasattr(bp, 'fab_cumulative_production_h100e') else 0.0
                )
                bp_data['fab_monthly_production_h100e'].append(
                    to_float(bp.fab_monthly_production_h100e) if hasattr(bp, 'fab_monthly_production_h100e') else 0.0
                )

                # Survival metrics
                bp_data['survival_rate'].append(to_float(bp.survival_rate) if hasattr(bp, 'survival_rate') else 1.0)
                bp_data['initial_compute_surviving_h100e'].append(
                    to_float(bp.initial_compute_surviving_h100e) if hasattr(bp, 'initial_compute_surviving_h100e') else 0.0
                )

                # Energy consumption by source
                bp_data['initial_stock_energy_gw'].append(
                    to_float(bp.initial_stock_energy_gw) if hasattr(bp, 'initial_stock_energy_gw') else 0.0
                )
                bp_data['fab_compute_energy_gw'].append(
                    to_float(bp.fab_compute_energy_gw) if hasattr(bp, 'fab_compute_energy_gw') else 0.0
                )
                bp_data['total_compute_energy_gw'].append(
                    to_float(bp.total_compute_energy_gw) if hasattr(bp, 'total_compute_energy_gw') else 0.0
                )
            else:
                bp_data['operational_compute'].append(0.0)
                bp_data['total_compute'].append(0.0)
                bp_data['datacenter_capacity_gw'].append(0.0)
                bp_data['fab_is_operational'].append(False)
                bp_data['cumulative_lr'].append(1.0)
                bp_data['lr_other_intel'].append(1.0)
                bp_data['posterior_prob'].append(0.3)
                bp_data['lr_reported_energy'].append(1.0)
                bp_data['lr_fab_other'].append(1.0)
                bp_data['lr_fab_combined'].append(1.0)
                bp_data['fab_cumulative_production_h100e'].append(0.0)
                bp_data['fab_monthly_production_h100e'].append(0.0)
                bp_data['survival_rate'].append(1.0)
                bp_data['initial_compute_surviving_h100e'].append(0.0)
                bp_data['initial_stock_energy_gw'].append(0.0)
                bp_data['fab_compute_energy_gw'].append(0.0)
                bp_data['total_compute_energy_gw'].append(0.0)

        # Extract fab static properties from final world state
        final_bp = trajectory[-1].black_projects.get(bp_id) if trajectory else None
        if final_bp:
            bp_data['fab_wafer_starts_per_month'] = to_float(final_bp.fab_wafer_starts_per_month) if hasattr(final_bp, 'fab_wafer_starts_per_month') else 0.0
            bp_data['fab_architecture_efficiency'] = to_float(final_bp.fab_architecture_efficiency) if hasattr(final_bp, 'fab_architecture_efficiency') else 1.0
            bp_data['fab_transistor_density_relative_to_h100'] = to_float(final_bp.fab_transistor_density_relative_to_h100) if hasattr(final_bp, 'fab_transistor_density_relative_to_h100') else 0.0
            bp_data['fab_process_node_nm'] = to_float(final_bp.fab_process_node_nm) if hasattr(final_bp, 'fab_process_node_nm') else 28.0
            bp_data['fab_chips_per_wafer'] = to_float(final_bp.fab_chips_per_wafer) if hasattr(final_bp, 'fab_chips_per_wafer') else 28
            # Fab construction timing - fab construction starts at black_project_start_year (black_project_start_year)
            bp_data['fab_construction_start_year'] = to_float(final_bp.black_project_start_year) if hasattr(final_bp, 'black_project_start_year') else 2029.0
            bp_data['fab_construction_duration'] = to_float(final_bp.fab_construction_duration) if hasattr(final_bp, 'fab_construction_duration') else 1.5
            # Extract watts_per_h100e for fab
            fab_watts = to_float(final_bp.fab_watts_per_chip) if hasattr(final_bp, 'fab_watts_per_chip') else 700.0
            fab_h100e = to_float(final_bp.fab_h100e_per_chip) if hasattr(final_bp, 'fab_h100e_per_chip') else 1.0
            bp_data['fab_watts_per_h100e'] = fab_watts / fab_h100e if fab_h100e > 0 else 700.0

        # Extract initial stock watts_per_h100e from first state
        if first_bp and hasattr(first_bp, 'compute_stock') and first_bp.compute_stock:
            bp_data['initial_stock_watts_per_h100e'] = to_float(first_bp.compute_stock.watts_per_h100e) if hasattr(first_bp.compute_stock, 'watts_per_h100e') else 700.0

        data['black_project'] = bp_data

    return data
