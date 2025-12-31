"""
Black project simulation using the actual AIFuturesSimulator.

Runs Monte Carlo simulations and extracts plot data from World trajectories.
"""

import sys
import time
import logging
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add ai_futures_simulator subdirectory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator"))

from ai_futures_simulator import AIFuturesSimulator
from parameters.simulation_parameters import ModelParameters
from classes.simulation_primitives import SimulationResult
from classes.world.world import World
from classes.world.entities import NamedNations

logger = logging.getLogger(__name__)

# Constants
LIKELIHOOD_RATIO_THRESHOLDS = [1, 2, 4]

# =============================================================================
# GLOBAL COMPUTE PRODUCTION (loaded from global_compute_production.csv)
# =============================================================================

_cached_global_compute_production = None


def _parse_compute_value(s: str) -> Optional[float]:
    """Parse a compute value string with optional K/M/B/T suffix."""
    if not s:
        return None
    s = s.strip().replace(',', '')
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                return float(s[:-1]) * mult
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_global_compute_production() -> Dict[str, List]:
    """Load the global compute production data from global_compute_production.csv.

    Returns a dict with:
        - years: list of years
        - total_stock: list of total H100e in world (no decay)
    """
    global _cached_global_compute_production
    if _cached_global_compute_production is not None:
        return _cached_global_compute_production

    csv_path = Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator" / "data" / "global_compute_production.csv"

    years = []
    total_stock = []

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            # Skip header rows (first 3 rows: header + 2 metadata rows)
            for _ in range(3):
                next(reader)

            for row in reader:
                if len(row) > 0 and row[0]:
                    try:
                        year = int(row[0])
                        # Column U (index 20) is "H100e in world, no decay"
                        stock = _parse_compute_value(row[20]) if len(row) > 20 else None
                        years.append(year)
                        total_stock.append(stock)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        logger.warning(f"Global compute production CSV not found at {csv_path}")
        years = []
        total_stock = []

    _cached_global_compute_production = {
        'years': years,
        'total_stock': total_stock
    }
    return _cached_global_compute_production


def get_global_compute_stock(year: float) -> float:
    """Get the global compute stock (H100e) for a given year.

    Uses the "H100e in world, no decay" column from global_compute_production.csv.
    Linearly interpolates between years.
    """
    data = _load_global_compute_production()

    # Filter to valid entries
    valid_years = []
    valid_stocks = []
    for y, s in zip(data['years'], data['total_stock']):
        if s is not None:
            valid_years.append(y)
            valid_stocks.append(s)

    if not valid_years:
        # Fallback: use simple exponential model
        base_2025 = 500000  # ~500K H100e globally in 2025
        growth_rate = 2.5
        return base_2025 * (growth_rate ** (year - 2025))

    return float(np.interp(year, valid_years, valid_stocks))


def get_global_compute_production_between_years(start_year: float, end_year: float) -> float:
    """Calculate total global compute production between two years.

    Uses the change in global compute stock (H100e in world, no decay) between years.
    Production = Stock(end_year) - Stock(start_year)
    """
    start_stock = get_global_compute_stock(start_year)
    end_stock = get_global_compute_stock(end_year)
    return max(0.0, end_stock - start_stock)


def run_black_project_simulations(
    frontend_params: dict,
    num_simulations: int = 100,
    time_range: list = None,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulations using the actual AIFuturesSimulator.

    Returns results dict with:
    - simulation_results: List of SimulationResult objects
    - agreement_year: Start year
    - end_year: End year
    """
    start_time = time.perf_counter()

    # Load model parameters from YAML - use dedicated black project monte carlo config
    config_path = Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator" / "parameters" / "black_project_monte_carlo_parameters.yaml"
    logger.info(f"[black-project] Loading config from: {config_path}")

    try:
        model_params = ModelParameters.from_yaml(config_path)
        logger.info("[black-project] Model parameters loaded successfully")
    except Exception as e:
        logger.exception(f"[black-project] Failed to load model parameters: {e}")
        raise

    agreement_year = float(time_range[0]) if time_range else 2027.0
    end_year = float(time_range[1]) if len(time_range) > 1 else 2037.0
    logger.info(f"[black-project] Time range: {agreement_year} to {end_year}")

    # Create simulator
    simulator = AIFuturesSimulator(model_parameters=model_params)

    # Run Monte Carlo simulations
    logger.info(f"[black-project] Running {num_simulations} simulations...")

    simulation_results = simulator.run_simulations(num_simulations=num_simulations)

    elapsed = time.perf_counter() - start_time
    logger.info(f"[black-project] Completed {len(simulation_results)} simulations in {elapsed:.2f}s")

    return {
        'simulation_results': simulation_results,
        'agreement_year': agreement_year,
        'end_year': end_year,
    }


def to_float(value, default: float = 0.0) -> float:
    """Convert tensor or number to Python float, handling Infinity/NaN for JSON compatibility."""
    if value is None:
        return default
    if hasattr(value, 'item'):
        result = float(value.item())
    else:
        result = float(value)
    # Handle Infinity and NaN which are not valid JSON
    if np.isinf(result) or np.isnan(result):
        return default
    return result


def _is_fab_built(bp_props, black_project_start_year: float) -> bool:
    """
    Determine if a fab is built based on localization years and minimum process node requirement.

    A fab is built if the best available process node (that is localized by black_project_start_year)
    meets the minimum requirement specified in black_fab_min_process_node.
    """
    if bp_props is None:
        return True

    min_node = bp_props.black_fab_min_process_node

    # Check each node from most advanced to least advanced
    localization_years = {
        7: bp_props.prc_localization_year_7nm,
        14: bp_props.prc_localization_year_14nm,
        28: bp_props.prc_localization_year_28nm,
    }

    # Find best available node that is localized by start year
    for node_nm in [7, 14, 28]:
        if localization_years[node_nm] <= black_project_start_year:
            # Found a localized node - check if it meets minimum requirement
            return node_nm <= min_node

    # No node is localized by start year
    return False


def compute_detection_times(all_data: List[Dict], years: List[float], agreement_year: float, lr_threshold: float = 5.0) -> List[float]:
    """
    Compute detection times based on when cumulative LR exceeds threshold.

    Returns time (in years) from agreement year to detection for each simulation.
    This matches the reference implementation which uses LR threshold = 5 for dashboard.

    Detection is defined as the first year >= agreement_year where cumulative_lr >= lr_threshold.
    """
    detection_times = []
    for d in all_data:
        bp = d.get('black_project')
        sim_years = d.get('years', [])

        if not bp or not sim_years:
            detection_times.append(10.0)  # Default to end of 10-year simulation
            continue

        cumulative_lr = bp.get('cumulative_lr', [])

        # Find first year >= agreement_year where LR >= threshold
        detection_year = None
        for i, year in enumerate(sim_years):
            if year >= agreement_year and i < len(cumulative_lr):
                if cumulative_lr[i] >= lr_threshold:
                    detection_year = year
                    break

        if detection_year is not None:
            # Time from agreement year to detection
            time_before_detection = detection_year - agreement_year
        else:
            # No detection within simulation - use time to end of simulation
            end_year = max(sim_years) if sim_years else agreement_year + 10
            time_before_detection = end_year - agreement_year

        detection_times.append(max(0.0, time_before_detection))  # Ensure non-negative

    return detection_times


def get_detection_year(d: Dict, agreement_year: float, lr_threshold: float = 5.0) -> Optional[float]:
    """
    Get the detection year for a simulation based on cumulative LR threshold.

    Returns the first year >= agreement_year where cumulative_lr >= lr_threshold,
    or None if no detection occurs.
    """
    bp = d.get('black_project')
    sim_years = d.get('years', [])

    if not bp or not sim_years:
        return None

    cumulative_lr = bp.get('cumulative_lr', [])

    for i, year in enumerate(sim_years):
        if year >= agreement_year and i < len(cumulative_lr):
            if cumulative_lr[i] >= lr_threshold:
                return year

    return None


def compute_h100_years_before_detection(all_data: List[Dict], years: List[float], agreement_year: float, lr_threshold: float = 5.0) -> List[float]:
    """
    Compute cumulative H100-years of compute before detection for each simulation.
    Uses LR threshold to determine detection time.
    """
    if not years or len(years) < 2:
        return [0.0 for _ in all_data]

    dt = years[1] - years[0]
    h100_years = []

    for d in all_data:
        bp = d.get('black_project')
        sim_years = d.get('years', [])

        if not bp or not sim_years:
            h100_years.append(0.0)
            continue

        operational_compute = bp.get('operational_compute', [])
        if not operational_compute:
            h100_years.append(0.0)
            continue

        # Get detection year based on LR threshold
        detection_year = get_detection_year(d, agreement_year, lr_threshold)
        if detection_year is None:
            detection_year = max(sim_years) if sim_years else agreement_year + 10

        # Sum operational compute from agreement_year to detection_year
        cumulative = 0.0
        for i, year in enumerate(sim_years):
            if year < agreement_year:
                continue
            if year >= detection_year:
                break
            if i < len(operational_compute):
                cumulative += operational_compute[i] * dt  # H100e * years = H100-years

        h100_years.append(cumulative)

    return h100_years


def compute_h100e_before_detection(all_data: List[Dict], years: List[float], agreement_year: float, lr_threshold: float = 5.0) -> List[float]:
    """
    Compute chip stock (H100e) at detection time for each simulation.
    Uses LR threshold to determine detection time.
    """
    if not years:
        return [0.0 for _ in all_data]

    h100e = []

    for d in all_data:
        bp = d.get('black_project')
        sim_years = d.get('years', [])

        if not bp or not sim_years:
            h100e.append(0.0)
            continue

        total_compute = bp.get('total_compute', [])
        if not total_compute:
            h100e.append(0.0)
            continue

        # Get detection year based on LR threshold
        detection_year = get_detection_year(d, agreement_year, lr_threshold)
        if detection_year is None:
            detection_year = max(sim_years) if sim_years else agreement_year + 10

        # Find chip stock at detection time
        detection_idx = 0
        for i, year in enumerate(sim_years):
            if year >= detection_year:
                detection_idx = i
                break
            detection_idx = i

        if detection_idx < len(total_compute):
            h100e.append(total_compute[detection_idx])
        else:
            h100e.append(total_compute[-1] if total_compute else 0.0)

    return h100e


def extract_world_data(result: SimulationResult) -> Dict[str, Any]:
    """Extract time series data from a single SimulationResult."""
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

    # Extract black project data if present
    if world.black_projects:
        bp_id = list(world.black_projects.keys())[0]

        # Get detection-related data from the first world state (these are set at init)
        first_bp = trajectory[0].black_projects.get(bp_id) if trajectory else None

        bp_data = {
            'id': bp_id,
            # Black project start year (for filtering output data)
            'preparation_start_year': to_float(first_bp.preparation_start_year) if first_bp and hasattr(first_bp, 'preparation_start_year') else None,
            # Time series (extracted from trajectory, one value per time step)
            'operational_compute': [],
            'total_compute': [],
            'datacenter_capacity_gw': [],
            'fab_is_operational': [],
            'cumulative_lr': [],  # Combined LR over time
            'lr_other_intel': [],  # Direct evidence LR over time
            'posterior_prob': [],  # Posterior probability over time
            'lr_reported_energy': [],  # Energy accounting LR over time
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
            'lr_prc_accounting': to_float(first_bp.lr_prc_accounting) if first_bp and hasattr(first_bp, 'lr_prc_accounting') else 1.0,
            'lr_sme_inventory': to_float(first_bp.lr_sme_inventory) if first_bp and hasattr(first_bp, 'lr_sme_inventory') else 1.0,
            'lr_satellite_datacenter': to_float(first_bp.lr_satellite_datacenter) if first_bp and hasattr(first_bp, 'lr_satellite_datacenter') else 1.0,
            # Initial diverted compute (constant, set at black project start, never changes)
            # This is calculated from PRC compute at black project start year, not simulation start year
            'initial_diverted_compute_h100e': to_float(first_bp.initial_diverted_compute_h100e) if first_bp and hasattr(first_bp, 'initial_diverted_compute_h100e') else 0.0,
            # Fab static properties (from final world state for accurate values)
            'fab_wafer_starts_per_month': 0.0,  # Will be updated from final state
            'fab_architecture_efficiency': 1.0,  # Will be updated from final state
            'fab_transistor_density_relative_to_h100': 0.0,  # Will be updated from final state
            'fab_process_node_nm': 28.0,  # Will be updated from final state
            'fab_chips_per_wafer': 28,  # Will be updated from final state
            # Energy efficiency values (for labels)
            'initial_stock_watts_per_h100e': 700.0,  # Will be updated from first state
            'fab_watts_per_h100e': 700.0,  # Will be updated from final state (fab_watts_per_chip / fab_h100e_per_chip)
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

                # Energy consumption by source (using corrected fab watts calculation)
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
            # Extract watts_per_h100e for fab (fab_watts_per_chip / fab_h100e_per_chip)
            fab_watts = to_float(final_bp.fab_watts_per_chip) if hasattr(final_bp, 'fab_watts_per_chip') else 700.0
            fab_h100e = to_float(final_bp.fab_h100e_per_chip) if hasattr(final_bp, 'fab_h100e_per_chip') else 1.0
            bp_data['fab_watts_per_h100e'] = fab_watts / fab_h100e if fab_h100e > 0 else 700.0

        # Extract initial stock watts_per_h100e from first state
        if first_bp and hasattr(first_bp, 'compute_stock') and first_bp.compute_stock:
            bp_data['initial_stock_watts_per_h100e'] = to_float(first_bp.compute_stock.watts_per_h100e) if hasattr(first_bp.compute_stock, 'watts_per_h100e') else 700.0

        data['black_project'] = bp_data

    return data


def _build_transistor_density_distribution(all_data: List[Dict]) -> List[Dict]:
    """
    Build transistor density data showing probability distribution across process nodes.

    Returns a list of dicts with {node, density, probability, wattsPerTpp} for each
    process node observed in simulations.
    """
    from collections import Counter

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
        # W/TPP ~ density^(-0.5) for post-Dennard scaling
        watts_per_tpp = density ** (-0.5) if density > 0 else 10.0

        result.append({
            "node": f"{int(node_nm)}nm",
            "density": density,
            "probability": probability,
            "wattsPerTpp": watts_per_tpp,
        })

    return result


def _build_watts_per_tpp_curve() -> Dict[str, List[float]]:
    """
    Build watts_per_tpp_curve using the Dennard scaling model.

    This curve shows how watts per TPP (relative to H100) varies with transistor density.
    Uses the same model as calculate_watts_per_tpp_from_transistor_density in black_compute.py.

    NOTE: Uses Monte Carlo parameters from black_project_monte_carlo_parameters.yaml
    for consistency with the simulation.
    """
    # Reference values for H100
    h100_transistor_density_m_per_mm2 = 98.28
    h100_watts_per_tpp = 0.326493  # W/TPP for H100

    # Dennard scaling parameters - aligned with reference model
    # These MUST match black_project_monte_carlo_parameters.yaml for consistency
    transistor_density_at_end_of_dennard = 1.98  # M/mm² (reference default)
    watts_per_tpp_exponent_before_dennard = -2.0  # (reference default)
    watts_per_tpp_exponent_after_dennard = -0.91  # (reference default)

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


def _compute_fab_dashboard(fab_built_sims: List[Dict], all_data: List[Dict], detection_times: List[float], num_sims: int, dt: float) -> Dict[str, Any]:
    """
    Compute pre-formatted dashboard values for the covert fab section.
    This avoids frontend computation.
    """
    from collections import Counter

    num_fab_built = len(fab_built_sims)
    prob_fab_built = f"{(num_fab_built / num_sims * 100):.1f}%" if num_sims > 0 else "--"

    # Compute median H100e before detection
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

    # Compute median time before detection
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


def extract_reference_format(
    simulation_results: Dict[str, Any],
    frontend_params: dict,
) -> Dict[str, Any]:
    """
    Extract plot data from SimulationResult trajectories.
    Returns data formatted to match the reference API at dark-compute.onrender.com/run_simulation.

    This format includes:
    - num_simulations, prob_fab_built, p_project_exists, researcher_headcount (top-level)
    - black_project_model (34 keys)
    - black_datacenters (16 keys)
    - black_fab (25 keys)
    - initial_black_project (4 keys)
    - initial_stock (12 keys)
    """
    results: List[SimulationResult] = simulation_results.get('simulation_results', [])
    agreement_year = simulation_results.get('agreement_year', 2029)

    if not results:
        return {"error": "No simulation results"}

    # Extract data from all simulations
    all_data = [extract_world_data(r) for r in results]

    # Use first simulation as reference for years
    raw_years = all_data[0]['years'] if all_data else []
    num_sims = len(all_data)

    # Get black project start year to filter output
    # Use agreement_year (not bp_start_year) to match reference API which starts at agreement year
    bp_start_year = None
    if all_data and all_data[0]['black_project']:
        bp_start_year = all_data[0]['black_project'].get('preparation_start_year')

    # Filter data to only include time points >= agreement_year (to match reference API)
    filter_start_year = agreement_year if agreement_year else bp_start_year
    if filter_start_year is not None and raw_years:
        start_idx = 0
        for i, y in enumerate(raw_years):
            if y >= filter_start_year:
                start_idx = i
                break
        years = raw_years[start_idx:]

        # Filter all time series data
        for d in all_data:
            d['years'] = d['years'][start_idx:]
            d['prc_compute_stock'] = d['prc_compute_stock'][start_idx:]
            d['prc_operating_compute'] = d['prc_operating_compute'][start_idx:]

            if d['black_project']:
                bp = d['black_project']
                for key in ['operational_compute', 'total_compute', 'datacenter_capacity_gw',
                            'fab_is_operational', 'cumulative_lr', 'lr_other_intel', 'posterior_prob',
                            'lr_reported_energy', 'fab_cumulative_production_h100e', 'fab_monthly_production_h100e',
                            'survival_rate', 'initial_compute_surviving_h100e',
                            'initial_stock_energy_gw', 'fab_compute_energy_gw', 'total_compute_energy_gw']:
                    if key in bp and isinstance(bp[key], list):
                        bp[key] = bp[key][start_idx:]
    else:
        years = raw_years

    dt = years[1] - years[0] if len(years) > 1 else 0.1

    # Helper to compute percentiles with individual data
    def get_percentiles_with_individual(extractor) -> Dict[str, Any]:
        try:
            values = [extractor(d) for d in all_data]
            arr = np.array(values)
            if arr.size == 0:
                return {"individual": [], "median": [], "p25": [], "p75": []}
            return {
                "individual": [list(v) for v in values],
                "median": np.percentile(arr, 50, axis=0).tolist(),
                "p25": np.percentile(arr, 25, axis=0).tolist(),
                "p75": np.percentile(arr, 75, axis=0).tolist(),
            }
        except Exception as e:
            logger.warning(f"Error computing percentiles: {e}")
            return {"individual": [], "median": [], "p25": [], "p75": []}

    def get_percentiles(extractor) -> Dict[str, List[float]]:
        result = get_percentiles_with_individual(extractor)
        return {"median": result["median"], "p25": result["p25"], "p75": result["p75"]}

    # Helper to compute percentiles ONLY over fab-built simulations (for black_fab section)
    def get_fab_percentiles_with_individual(extractor) -> Dict[str, Any]:
        """Compute percentiles only over fab_built_sims, not all simulations."""
        try:
            if not fab_built_sims:
                return {"individual": [], "median": [], "p25": [], "p75": []}
            values = [extractor(d) for d in fab_built_sims]
            arr = np.array(values)
            if arr.size == 0:
                return {"individual": [], "median": [], "p25": [], "p75": []}
            return {
                "individual": [list(v) for v in values],
                "median": np.percentile(arr, 50, axis=0).tolist(),
                "p25": np.percentile(arr, 25, axis=0).tolist(),
                "p75": np.percentile(arr, 75, axis=0).tolist(),
            }
        except Exception as e:
            logger.warning(f"Error computing fab percentiles: {e}")
            return {"individual": [], "median": [], "p25": [], "p75": []}

    # Helper to compute CCDF
    def compute_ccdf(values: List[float]) -> List[Dict[str, float]]:
        values = [v for v in values if v > 0 and v < float('inf')]
        if not values:
            return []
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        ccdf = []
        seen = set()
        for i, val in enumerate(sorted_vals):
            if val not in seen:
                ccdf.append({"x": float(val), "y": float((n - i) / n)})
                seen.add(val)
        return ccdf

    # Identify fab-built simulations
    fab_built_sims = [
        d for d in all_data
        if d['black_project'] and any(d['black_project'].get('fab_is_operational', []))
    ]
    num_fab_built = len(fab_built_sims)
    prob_fab_built = num_fab_built / num_sims if num_sims > 0 else 0.0

    # Energy by source: [initial_stock_energy, fab_energy] per time step
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
        f"Initial Dark Compute ({initial_eff:.2f}x energy efficiency)",
        f"Covert Fab Compute ({fab_eff:.2f}x energy efficiency)",
    ]

    # Detection times
    detection_times = compute_detection_times(all_data, years, agreement_year)
    h100_years_before_detection = compute_h100_years_before_detection(all_data, years, agreement_year)
    h100e_before_detection = compute_h100e_before_detection(all_data, years, agreement_year)

    # Energy before detection (at detection time)
    # Use detection_times which is now based on LR threshold
    energy_before_detection = []
    for i, d in enumerate(all_data):
        bp = d.get('black_project')
        sim_years = d.get('years', [])

        if not bp or not sim_years:
            energy_before_detection.append(0.0)
            continue

        total_energy = bp.get('total_compute_energy_gw', [])
        if not total_energy:
            energy_before_detection.append(0.0)
            continue

        # Get detection year (agreement_year + detection_times[i])
        det_year = agreement_year + detection_times[i]

        # Find energy at detection time
        det_idx = 0
        for j, year in enumerate(sim_years):
            if year >= det_year:
                det_idx = j
                break
            det_idx = j

        if det_idx < len(total_energy):
            energy_before_detection.append(total_energy[det_idx])
        else:
            energy_before_detection.append(total_energy[-1] if total_energy else 0.0)

    # Compute reduction ratios for CCDFs
    def compute_reduction_ratios(lr_threshold: float) -> Dict[str, List[float]]:
        pre_slowdown_growth_rate = 2.2
        largest_company_fraction = 0.3
        LARGE_REDUCTION = 1e12

        chip_global = []
        chip_prc = []
        ai_largest = []
        ai_prc = []

        for d in all_data:
            bp = d.get('black_project')
            prc_stock = d.get('prc_compute_stock', [])
            if not bp or not prc_stock:
                continue

            # Find detection year for this threshold
            cumulative_lr = bp.get('cumulative_lr', [])
            detection_year = years[-1] + 1
            for j, lr in enumerate(cumulative_lr):
                if lr >= lr_threshold and j < len(years):
                    detection_year = years[j]
                    break
            if detection_year > years[-1]:
                detection_year = years[-1]

            # Compute metrics up to detection
            op_compute = bp.get('operational_compute', [])
            bp_h100_years = 0.0
            for j, year in enumerate(years):
                if year >= detection_year:
                    break
                if j < len(op_compute):
                    bp_h100_years += op_compute[j] * dt

            fab_prod = bp.get('fab_cumulative_production_h100e', [])
            prod_at_detection = 0.0
            prod_at_agreement = 0.0
            for j, year in enumerate(years):
                if j < len(fab_prod):
                    if year <= agreement_year:
                        prod_at_agreement = fab_prod[j]
                    if year <= detection_year:
                        prod_at_detection = fab_prod[j]
            bp_chip_production = max(0.0, prod_at_detection - prod_at_agreement)

            agreement_idx = 0
            for j, year in enumerate(years):
                if year >= agreement_year:
                    agreement_idx = j
                    break
            prc_at_agreement = prc_stock[agreement_idx] if agreement_idx < len(prc_stock) else prc_stock[0]
            duration = max(0.0, detection_year - agreement_year)

            global_production = get_global_compute_production_between_years(agreement_year, detection_year)
            prc_production = prc_at_agreement * (pre_slowdown_growth_rate ** duration - 1)
            largest_production = prc_at_agreement * largest_company_fraction * (pre_slowdown_growth_rate ** duration - 1)

            if pre_slowdown_growth_rate > 1.001:
                growth_factor = (pre_slowdown_growth_rate ** duration - 1) / np.log(pre_slowdown_growth_rate)
            else:
                growth_factor = duration
            prc_h100_years = prc_at_agreement * growth_factor
            largest_h100_years = prc_h100_years * largest_company_fraction

            if bp_chip_production <= 0:
                chip_global.append(LARGE_REDUCTION)
                chip_prc.append(LARGE_REDUCTION)
            else:
                chip_global.append(global_production / bp_chip_production if global_production > 0 else 0.0)
                chip_prc.append(prc_production / bp_chip_production if prc_production > 0 else 0.0)

            if bp_h100_years <= 0:
                ai_largest.append(LARGE_REDUCTION)
                ai_prc.append(LARGE_REDUCTION)
            else:
                ai_largest.append(largest_h100_years / bp_h100_years if largest_h100_years > 0 else 0.0)
                ai_prc.append(prc_h100_years / bp_h100_years if prc_h100_years > 0 else 0.0)

        return {
            'chip_global': chip_global,
            'chip_prc': chip_prc,
            'chip_largest': chip_prc,  # Same as PRC for chip production
            'ai_largest': ai_largest,
            'ai_prc': ai_prc,
        }

    # Detection probability at each time point
    datacenter_detection_prob = []
    for i, year in enumerate(years):
        detected_count = sum(
            1 for d in all_data
            if d['black_project'] and d['black_project'].get('sampled_detection_time', float('inf')) <= (year - years[0])
        )
        datacenter_detection_prob.append(detected_count / num_sims if num_sims > 0 else 0.0)

    # Fraction diverted (this is the datacenter capacity diversion fraction, NOT compute diversion)
    # Reference model uses fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start = 0.01
    # This is fixed from parameters, not computed dynamically
    fraction_diverted = 0.01  # Fixed value from parameters

    # PRC capacity data (simplified - use static growth model)
    prc_capacity_years = list(range(2025, int(agreement_year) + 1))
    prc_base_gw = 0.175  # Starting capacity in 2025
    prc_growth = 1.74  # Growth rate per year
    prc_capacity_gw = {
        "median": [prc_base_gw * (prc_growth ** (y - 2025)) for y in prc_capacity_years],
        "p25": [prc_base_gw * (prc_growth ** (y - 2025)) * 0.8 for y in prc_capacity_years],
        "p75": [prc_base_gw * (prc_growth ** (y - 2025)) * 1.2 for y in prc_capacity_years],
    }
    prc_capacity_at_agreement = prc_base_gw * (prc_growth ** (agreement_year - 2025))

    # Build response in reference format
    response = {
        # Top-level metadata
        "num_simulations": num_sims,
        "prob_fab_built": prob_fab_built,
        "p_project_exists": 0.2,  # Prior probability
        "researcher_headcount": 500,  # Default from reference

        # black_project_model (34 keys)
        "black_project_model": {
            "years": years,

            # Time series with individual data
            "h100_years": get_percentiles_with_individual(
                lambda d: [sum((d['black_project']['operational_compute'] if d['black_project'] else [0])[:i+1]) * dt for i in range(len(years))]
            ),
            "cumulative_lr": get_percentiles_with_individual(
                lambda d: d['black_project']['cumulative_lr'] if d['black_project'] else [1.0] * len(years)
            ),
            "initial_black_project": get_percentiles_with_individual(
                lambda d: d['black_project']['initial_compute_surviving_h100e'] if d['black_project'] else [0.0] * len(years)
            ),
            "black_fab_flow": get_percentiles_with_individual(
                lambda d: d['black_project']['fab_cumulative_production_h100e'] if d['black_project'] else [0.0] * len(years)
            ),
            "black_fab_flow_all_sims": get_percentiles_with_individual(
                lambda d: d['black_project']['fab_cumulative_production_h100e'] if d['black_project'] else [0.0] * len(years)
            ),
            "survival_rate": get_percentiles_with_individual(
                lambda d: d['black_project']['survival_rate'] if d['black_project'] else [1.0] * len(years)
            ),
            "covert_chip_stock": get_percentiles_with_individual(
                lambda d: d['black_project']['total_compute'] if d['black_project'] else [0.0] * len(years)
            ),
            "total_black_project": get_percentiles_with_individual(
                lambda d: d['black_project']['total_compute'] if d['black_project'] else [0.0] * len(years)
            ),
            "datacenter_capacity": get_percentiles_with_individual(
                lambda d: d['black_project']['datacenter_capacity_gw'] if d['black_project'] else [0.0] * len(years)
            ),
            # operational_compute: raw H100e (same as reference black_project_model section)
            "operational_compute": get_percentiles_with_individual(
                lambda d: d['black_project']['operational_compute'] if d['black_project'] else [0.0] * len(years)
            ),

            # Energy data
            "black_project_energy": energy_by_source,
            "energy_source_labels": source_labels,

            # LR components
            "lr_initial_stock": get_percentiles_with_individual(
                lambda d: [d['black_project']['lr_prc_accounting']] * len(years) if d['black_project'] else [1.0] * len(years)
            ),
            "lr_diverted_sme": get_percentiles_with_individual(
                lambda d: [d['black_project']['lr_sme_inventory']] * len(years) if d['black_project'] else [1.0] * len(years)
            ),
            "lr_prc_accounting": get_percentiles_with_individual(
                lambda d: [d['black_project']['lr_prc_accounting']] * len(years) if d['black_project'] else [1.0] * len(years)
            ),
            "lr_sme_inventory": get_percentiles_with_individual(
                lambda d: [d['black_project']['lr_sme_inventory']] * len(years) if d['black_project'] else [1.0] * len(years)
            ),
            "lr_satellite_datacenter": {
                "individual": [d['black_project']['lr_satellite_datacenter'] if d['black_project'] else 1.0 for d in all_data]
            },
            "lr_other_intel": get_percentiles_with_individual(
                lambda d: d['black_project']['lr_other_intel'] if d['black_project'] else [1.0] * len(years)
            ),
            "lr_reported_energy": get_percentiles_with_individual(
                lambda d: d['black_project']['lr_reported_energy'] if d['black_project'] else [1.0] * len(years)
            ),
            "lr_combined_reported_assets": get_percentiles_with_individual(
                lambda d: [
                    d['black_project']['lr_prc_accounting'] * d['black_project']['lr_sme_inventory'] *
                    d['black_project']['lr_satellite_datacenter'] * (d['black_project']['lr_reported_energy'][i] if i < len(d['black_project'].get('lr_reported_energy', [])) else 1.0)
                    if d['black_project'] else 1.0
                    for i in range(len(years))
                ]
            ),
            "posterior_prob_project": get_percentiles_with_individual(
                lambda d: d['black_project']['posterior_prob'] if d['black_project'] else [0.3] * len(years)
            ),

            # Individual simulation values
            "individual_project_time_before_detection": detection_times,
            "individual_project_h100_years_before_detection": h100_years_before_detection,
            "individual_project_h100e_before_detection": h100e_before_detection,
            "individual_project_energy_before_detection": energy_before_detection,
            "project_80th_h100_years": h100_years_before_detection,
            "project_80th_h100e": h100e_before_detection,
            "project_80th_time": detection_times,

            # CCDFs
            "time_to_detection_ccdf": {
                str(lr): compute_ccdf(detection_times)
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "h100_years_ccdf": {
                str(lr): compute_ccdf(h100_years_before_detection)
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "average_covert_compute_ccdf": {
                str(lr): compute_ccdf([
                    sum((d['black_project']['operational_compute'] if d['black_project'] else [])[:int(detection_times[i] / dt) + 1]) / max(1, detection_times[i]) if detection_times[i] > 0 else 0
                    for i, d in enumerate(all_data)
                ])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "chip_production_reduction_ccdf": {
                "global": compute_ccdf(compute_reduction_ratios(4)['chip_global']),
                "largest_company": compute_ccdf(compute_reduction_ratios(4)['chip_largest']),
                "prc": compute_ccdf(compute_reduction_ratios(4)['chip_prc']),
            },
            "ai_rd_reduction_ccdf": {
                "largest_company": compute_ccdf(compute_reduction_ratios(4)['ai_largest']),
                "prc": compute_ccdf(compute_reduction_ratios(4)['ai_prc']),
            },

            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,
        },

        # black_datacenters (16 keys)
        "black_datacenters": {
            "years": years,
            "datacenter_capacity": get_percentiles_with_individual(
                lambda d: d['black_project']['datacenter_capacity_gw'] if d['black_project'] else [0.0] * len(years)
            ),
            # operational_compute: divide by 1000 to convert H100e to K H100e (reference units)
            "operational_compute": get_percentiles_with_individual(
                lambda d: [v / 1000.0 for v in (d['black_project']['operational_compute'] if d['black_project'] else [0.0] * len(years))]
            ),
            "lr_datacenters": get_percentiles_with_individual(
                lambda d: d['black_project']['cumulative_lr'] if d['black_project'] else [1.0] * len(years)
            ),
            "datacenter_detection_prob": datacenter_detection_prob,
            "energy_by_source": energy_by_source,
            "source_labels": source_labels,
            "fraction_diverted": fraction_diverted,
            "capacity_ccdfs": {
                str(lr): compute_ccdf([
                    d['black_project']['datacenter_capacity_gw'][
                        min(int(detection_times[i] / dt) if dt > 0 else 0, len(d['black_project']['datacenter_capacity_gw']) - 1)
                    ] if d['black_project'] and d['black_project'].get('datacenter_capacity_gw') else 0.0
                    for i, d in enumerate(all_data)
                ])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "individual_capacity_before_detection": [
                d['black_project']['datacenter_capacity_gw'][
                    min(int(detection_times[i] / dt) if dt > 0 else 0, len(d['black_project']['datacenter_capacity_gw']) - 1)
                ] if d['black_project'] and d['black_project'].get('datacenter_capacity_gw') else 0.0
                for i, d in enumerate(all_data)
            ],
            "individual_time_before_detection": detection_times,
            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,
            "prc_capacity_years": prc_capacity_years,
            "prc_capacity_gw": prc_capacity_gw,
            "prc_capacity_at_agreement_year_gw": prc_capacity_at_agreement,
            "prc_capacity_at_agreement_year_samples": [prc_capacity_at_agreement * (0.8 + 0.4 * np.random.random()) for _ in range(num_sims)],
        },

        # black_fab (25 keys)
        "black_fab": {
            "years": years,
            "fab_built": [
                any(d['black_project'].get('fab_is_operational', [])) if d['black_project'] else False
                for d in all_data
            ],
            "is_operational": {
                "individual": [
                    [1.0 if op else 0.0 for op in (d['black_project']['fab_is_operational'] if d['black_project'] else [])]
                    for d in fab_built_sims
                ] if fab_built_sims else [],
                "proportion": [
                    sum(1 for d in fab_built_sims if d['black_project'] and i < len(d['black_project'].get('fab_is_operational', [])) and d['black_project']['fab_is_operational'][i]) / max(1, len(fab_built_sims))
                    for i in range(len(years))
                ] if fab_built_sims else [0.0] * len(years),
            },
            "wafer_starts": get_fab_percentiles_with_individual(
                lambda d: [d['black_project']['fab_wafer_starts_per_month']] * len(years) if d['black_project'] else [0.0] * len(years)
            ),
            "chips_per_wafer": get_fab_percentiles_with_individual(
                lambda d: [d['black_project']['fab_chips_per_wafer']] * len(years) if d['black_project'] else [28.0] * len(years)
            ),
            "architecture_efficiency": get_fab_percentiles_with_individual(
                lambda d: [d['black_project']['fab_architecture_efficiency']] * len(years) if d['black_project'] else [1.0] * len(years)
            ),
            "architecture_efficiency_at_agreement": fab_built_sims[0]['black_project']['fab_architecture_efficiency'] if fab_built_sims and fab_built_sims[0]['black_project'] else 1.0,
            "compute_per_wafer_2022_arch": get_fab_percentiles_with_individual(
                lambda d: [d['black_project']['fab_transistor_density_relative_to_h100'] * d['black_project']['fab_chips_per_wafer']] * len(years)
                if d['black_project'] else [1.0] * len(years)
            ),
            "transistor_density": get_fab_percentiles_with_individual(
                lambda d: [d['black_project']['fab_transistor_density_relative_to_h100']] * len(years) if d['black_project'] else [0.1] * len(years)
            ),
            # watts_per_tpp: Use fab-only percentiles to avoid dilution from non-fab simulations
            # Convert fab_watts_per_h100e (absolute watts, e.g. 700W) to watts_per_tpp_relative
            # watts_per_tpp_relative = fab_watts_per_h100e / H100_WATTS_PER_H100E
            # For H100: 700/700 = 1.0, for less efficient chips: higher values
            "watts_per_tpp": get_fab_percentiles_with_individual(
                lambda d: [d['black_project']['fab_watts_per_h100e'] / 700.0] * len(years) if d['black_project'] else [1.0] * len(years)
            ),
            "watts_per_tpp_curve": _build_watts_per_tpp_curve(),
            # LR values: Use fab-only percentiles (black_fab section only includes fab-built simulations)
            "lr_combined": get_fab_percentiles_with_individual(
                lambda d: d['black_project']['cumulative_lr'] if d['black_project'] else [1.0] * len(years)
            ),
            "lr_inventory": get_fab_percentiles_with_individual(
                lambda d: [d['black_project']['lr_sme_inventory']] * len(years) if d['black_project'] else [1.0] * len(years)
            ),
            "lr_procurement": get_fab_percentiles_with_individual(
                lambda d: [d['black_project']['lr_prc_accounting']] * len(years) if d['black_project'] else [1.0] * len(years)
            ),
            "lr_other": get_fab_percentiles_with_individual(
                lambda d: d['black_project']['lr_other_intel'] if d['black_project'] else [1.0] * len(years)
            ),
            "process_node_by_sim": [
                f"{int(d['black_project'].get('fab_process_node_nm', 28))}nm" if d['black_project'] else "28nm"
                for d in fab_built_sims
            ] if fab_built_sims else [],
            "individual_process_node": [
                f"{int(d['black_project'].get('fab_process_node_nm', 28))}nm" if d['black_project'] else "28nm"
                for d in fab_built_sims
            ] if fab_built_sims else [],
            "individual_h100e_before_detection": [
                d['black_project']['fab_cumulative_production_h100e'][
                    min(int(detection_times[all_data.index(d)] / dt) if dt > 0 else 0, len(d['black_project']['fab_cumulative_production_h100e']) - 1)
                ] if d['black_project'] and d['black_project'].get('fab_cumulative_production_h100e') else 0.0
                for d in fab_built_sims
            ] if fab_built_sims else [],
            "individual_time_before_detection": [
                detection_times[all_data.index(d)] for d in fab_built_sims
            ] if fab_built_sims else [],
            "individual_energy_before_detection": [
                d['black_project']['total_compute_energy_gw'][
                    min(int(detection_times[all_data.index(d)] / dt) if dt > 0 else 0, len(d['black_project']['total_compute_energy_gw']) - 1)
                ] if d['black_project'] and d['black_project'].get('total_compute_energy_gw') else 0.0
                for d in fab_built_sims
            ] if fab_built_sims else [],
            "compute_ccdf": [],
            "compute_ccdfs": {
                str(lr): compute_ccdf([
                    d['black_project']['fab_cumulative_production_h100e'][-1]
                    if d['black_project'] and d['black_project'].get('fab_cumulative_production_h100e')
                    else 0.0
                    for d in fab_built_sims
                ])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "op_time_ccdf": [],
            "op_time_ccdfs": {
                str(lr): compute_ccdf([
                    sum(1 for op in d['black_project'].get('fab_is_operational', []) if op) * dt
                    if d['black_project'] else 0.0
                    for d in fab_built_sims
                ])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,

            # Pre-computed dashboard values (avoid frontend computation)
            "dashboard": _compute_fab_dashboard(fab_built_sims, all_data, detection_times, num_sims, dt),
        },

        # initial_black_project (4 keys)
        "initial_black_project": {
            "years": years,
            "black_project": get_percentiles_with_individual(
                lambda d: d['black_project']['initial_compute_surviving_h100e'] if d['black_project'] else [0.0] * len(years)
            ),
            "h100e": get_percentiles_with_individual(
                lambda d: d['black_project']['total_compute'] if d['black_project'] else [0.0] * len(years)
            ),
            "survival_rate": get_percentiles_with_individual(
                lambda d: d['black_project']['survival_rate'] if d['black_project'] else [1.0] * len(years)
            ),
        },

        # initial_stock (13 keys - includes initial_energy_samples for frontend)
        "initial_stock": {
            "years": years,
            "diversion_proportion": 0.05,
            "initial_prc_stock_samples": [
                d['black_project']['initial_diverted_compute_h100e'] / 0.05
                if d['black_project'] and d['black_project'].get('initial_diverted_compute_h100e', 0) > 0
                else 0
                for d in all_data
            ],
            "initial_compute_stock_samples": [
                d['black_project']['initial_diverted_compute_h100e']
                if d['black_project'] and d['black_project'].get('initial_diverted_compute_h100e')
                else 0
                for d in all_data
            ],
            # Energy samples computed from compute stock and efficiency
            # energy (GW) = compute * H100_power_kw / (efficiency * 1e6)
            "initial_energy_samples": [
                (d['black_project']['initial_diverted_compute_h100e'] * 0.7) /
                ((1.26 ** (agreement_year - 2022)) * 0.2 * 1e6)
                if d['black_project'] and d['black_project'].get('initial_diverted_compute_h100e')
                else 0
                for d in all_data
            ],
            "lr_prc_accounting_samples": [
                d['black_project']['lr_prc_accounting'] if d['black_project'] else 1.0
                for d in all_data
            ],
            "lr_sme_inventory_samples": [
                d['black_project']['lr_sme_inventory'] if d['black_project'] else 1.0
                for d in all_data
            ],
            "lr_satellite_datacenter_samples": [
                d['black_project']['lr_satellite_datacenter'] if d['black_project'] else 1.0
                for d in all_data
            ],
            "initial_black_project_detection_probs": {
                "1x": sum(1 for t in detection_times if t < 1) / max(1, num_sims),
                "2x": sum(1 for t in detection_times if t < 2) / max(1, num_sims),
                "4x": sum(1 for t in detection_times if t < 4) / max(1, num_sims),
            },
            "prc_compute_years": prc_capacity_years,
            # Compute PRC stock for each year using sampled growth rate
            # Each simulation has its own sampled growth rate
            "prc_compute_over_time": get_percentiles_with_individual(
                lambda d: [
                    d['prc_params']['total_prc_compute_tpp_h100e_in_2025'] * (d['prc_params']['annual_growth_rate'] ** (year - 2025))
                    if d.get('prc_params') else 100000.0 * (2.2 ** (year - 2025))
                    for year in prc_capacity_years
                ]
            ),
            # Compute domestic production proportion for each year
            # Reference pattern: 0 for years < 2027, then linear interpolation from 0.175 to 0.7
            "prc_domestic_compute_over_time": get_percentiles_with_individual(
                lambda d: [
                    (d['prc_params']['total_prc_compute_tpp_h100e_in_2025'] * (d['prc_params']['annual_growth_rate'] ** (year - 2025))
                     if d.get('prc_params') else 100000.0 * (2.2 ** (year - 2025)))
                    * (0.0 if year < 2027 else 0.175 * (year - 2026) if year <= int(agreement_year) else 0.7)
                    for year in prc_capacity_years
                ]
            ),
            # Proportion domestic by year: 0 for years < 2027, then linear from 0.175 to 0.7
            "proportion_domestic_by_year": [
                0.0 if year < 2027 else 0.175 * (year - 2026) if year <= int(agreement_year) else 0.7
                for year in prc_capacity_years
            ],
            # Largest company compute: US frontier compute growing at ~2.91x per year
            # Reference uses ~120000 as starting value, 2.91x growth
            "largest_company_compute_over_time": [
                120000.0 * (2.91 ** (year - 2025))
                for year in prc_capacity_years
            ],
            "state_of_the_art_energy_efficiency_relative_to_h100": 1.26 ** (agreement_year - 2022),
        },
    }

    return response


def extract_black_project_plot_data(
    simulation_results: Dict[str, Any],
    frontend_params: dict,
) -> Dict[str, Any]:
    """
    Extract plot data from SimulationResult trajectories.
    Returns data formatted for the frontend (reference format).

    This function delegates to extract_reference_format() to produce
    the exact format expected by the frontend (matching dark-compute.onrender.com/run_simulation).
    """
    return extract_reference_format(simulation_results, frontend_params)


def get_default_parameters() -> Dict[str, Any]:
    """Get default parameters from YAML for frontend initialization."""
    config_path = Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator" / "parameters" / "black_project_monte_carlo_parameters.yaml"
    model_params = ModelParameters.from_yaml(config_path)

    # Extract relevant parameters for frontend
    sim_params = model_params.sample()

    bp = sim_params.black_project
    bp_props = bp.black_project_properties if bp else None
    compute = sim_params.compute
    prc_compute = compute.PRCComputeParameters if compute else None
    energy = sim_params.datacenter_and_energy
    prc_energy = energy.prc_energy_consumption if energy else None
    perceptions = sim_params.perceptions
    bp_perceptions = perceptions.black_project_perception_parameters if perceptions else None

    defaults = {
        # Simulation settings
        "numYearsToSimulate": 10,
        "numSimulations": 100,
        "agreementYear": int(sim_params.policy.ai_slowdown_start_year) if sim_params.policy else 2030,
        "blackProjectStartYear": int(bp.black_project_start_year) if bp else 2029,

        # Black project properties
        "workersInCovertProject": bp_props.total_labor if bp_props else 11300,
        "fractionOfLaborDevotedToDatacenterConstruction": bp_props.fraction_of_labor_devoted_to_datacenter_construction if bp_props else 0.885,
        "fractionOfLaborDevotedToBlackFabConstruction": bp_props.fraction_of_labor_devoted_to_black_fab_construction if bp_props else 0.022,
        "fractionOfLaborDevotedToBlackFabOperation": bp_props.fraction_of_labor_devoted_to_black_fab_operation if bp_props else 0.049,
        "fractionOfLaborDevotedToAiResearch": bp_props.fraction_of_labor_devoted_to_ai_research if bp_props else 0.044,
        "proportionOfInitialChipStockToDivert": bp_props.fraction_of_initial_compute_stock_to_divert_at_black_project_start if bp_props else 0.05,
        "fractionOfDatacenterCapacityToDivert": bp_props.fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start if bp_props else 0.5,
        "fractionOfLithographyScannersToDivert": bp_props.fraction_of_lithography_scanners_to_divert_at_black_project_start if bp_props else 0.10,
        "maxFractionOfTotalNationalEnergyConsumption": bp_props.max_fraction_of_total_national_energy_consumption if bp_props else 0.05,
        # build_a_black_fab is derived: True if best available node meets min requirement by black_project_start_year
        "buildCovertFab": _is_fab_built(bp_props, bp.black_project_start_year if bp else 2029) if bp_props else True,
        "blackFabMaxProcessNode": str(int(bp_props.black_fab_min_process_node)) if bp_props else "28",

        # Detection parameters
        "priorOddsOfCovertProject": bp_perceptions.prior_odds_of_covert_project if bp_perceptions else 0.111,
        "intelligenceMedianError": bp_perceptions.intelligence_median_error_in_estimate_of_compute_stock if bp_perceptions else 0.07,
        "meanDetectionTime100": bp_perceptions.mean_detection_time_for_100_workers if bp_perceptions else 6.95,
        "meanDetectionTime1000": bp_perceptions.mean_detection_time_for_1000_workers if bp_perceptions else 3.42,
        "varianceDetectionTime": bp_perceptions.variance_of_detection_time_given_num_workers if bp_perceptions else 3.88,
        "detectionThreshold": bp_perceptions.detection_threshold if bp_perceptions else 100.0,

        # PRC compute
        "totalPrcComputeTppH100eIn2025": prc_compute.total_prc_compute_tpp_h100e_in_2025 if prc_compute else 100000,
        "annualGrowthRateOfPrcComputeStock": prc_compute.annual_growth_rate_of_prc_compute_stock if prc_compute else 2.2,
        "h100SizedChipsPerWafer": prc_compute.h100_sized_chips_per_wafer if prc_compute else 28,
        "wafersPerMonthPerLithographyScanner": prc_compute.wafers_per_month_per_lithography_scanner if prc_compute else 1000,

        # PRC energy
        "energyEfficiencyOfComputeStockRelativeToStateOfTheArt": prc_energy.energy_efficiency_of_compute_stock_relative_to_state_of_the_art if prc_energy else 0.20,
        "totalPrcEnergyConsumptionGw": prc_energy.total_prc_energy_consumption_gw if prc_energy else 1100,
        "dataCenterMwPerYearPerConstructionWorker": prc_energy.data_center_mw_per_year_per_construction_worker if prc_energy else 1.0,

        # Survival parameters
        "initialAnnualHazardRate": compute.survival_rate_parameters.initial_annual_hazard_rate if compute else 0.05,
        "annualHazardRateIncreasePerYear": compute.survival_rate_parameters.annual_hazard_rate_increase_per_year if compute else 0.02,

        # Exogenous compute trends (for Dennard scaling curves)
        "transistorDensityScalingExponent": compute.exogenous_trends.transistor_density_scaling_exponent if compute else 1.49,
        "stateOfTheArtArchitectureEfficiencyImprovementPerYear": compute.exogenous_trends.state_of_the_art_architecture_efficiency_improvement_per_year if compute else 1.23,
        "transistorDensityAtEndOfDennardScaling": compute.exogenous_trends.transistor_density_at_end_of_dennard_scaling_m_per_mm2 if compute else 10.0,
        "wattsTppDensityExponentBeforeDennard": compute.exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended if compute else -1.0,
        "wattsTppDensityExponentAfterDennard": compute.exogenous_trends.watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended if compute else -0.33,
        "stateOfTheArtEnergyEfficiencyImprovementPerYear": compute.exogenous_trends.state_of_the_art_energy_efficiency_improvement_per_year if compute else 1.26,
    }

    return defaults
