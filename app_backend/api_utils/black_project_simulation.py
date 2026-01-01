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
from typing import List, Dict, Any, Optional, Tuple

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

    # Override simulation_end_year from time_range (YAML default is 2040)
    # Keep simulation_start_year from YAML (needs historical data to initialize)
    # But override end_year and calculate n_eval_points for full simulation range
    start_year = model_params.settings.get('simulation_start_year', 2026)
    model_params.settings['simulation_end_year'] = end_year
    # Calculate n_eval_points to maintain 0.1-year resolution for full simulation
    n_years = end_year - start_year
    model_params.settings['n_eval_points'] = int(n_years * 10) + 1
    logger.info(f"[black-project] Updated settings: start_year={start_year}, end_year={end_year}, n_eval_points={model_params.settings['n_eval_points']}")

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


def compute_fab_detection_data(all_data: List[Dict], years: List[float], agreement_year: float, lr_threshold: float = 5.0) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Compute fab-specific detection data for dashboard display.

    Matches reference model's extract_individual_fab_detection_data which:
    1. Uses fab-specific LR (lr_fab_combined) for detection threshold
    2. Reports time as operational time before detection (not time from agreement_year)
    3. Gets h100e at the fab detection time (not project detection time)

    Returns:
        Tuple of (individual_h100e, individual_time, individual_process_nodes, individual_energy)
        Only includes simulations where fab was built.
    """
    individual_h100e = []
    individual_time = []
    individual_process_nodes = []
    individual_energy = []

    for d in all_data:
        bp = d.get('black_project')
        sim_years = d.get('years', years)

        if not bp:
            continue

        # Check if fab was built
        fab_is_operational = bp.get('fab_is_operational', [])
        if not any(fab_is_operational):
            continue

        # Get fab-specific LR for detection calculation
        lr_fab_combined = bp.get('lr_fab_combined', [])

        # Find fab detection year based on fab LR threshold
        fab_detection_year = None
        for i, year in enumerate(sim_years):
            if year >= agreement_year and i < len(lr_fab_combined):
                if lr_fab_combined[i] >= lr_threshold:
                    fab_detection_year = year
                    break

        # If no detection, use end of simulation
        if fab_detection_year is None:
            fab_detection_year = sim_years[-1] if sim_years else agreement_year + 7

        # Calculate when fab became operational
        construction_start = bp.get('fab_construction_start_year', agreement_year)
        construction_duration = bp.get('fab_construction_duration', 1.5)
        operational_start = construction_start + construction_duration

        # Time operational before detection (matching reference model)
        # Reference model checks LR from agreement_year (not operational_start) and uses:
        # operational_time = max(0.0, detection_year - operational_start)
        # If detection happens BEFORE fab is operational, operational_time = 0.0
        operational_time = max(0.0, fab_detection_year - operational_start)

        # Get h100e at detection
        fab_prod = bp.get('fab_cumulative_production_h100e', [])
        if fab_prod and sim_years:
            # Find index closest to detection year
            dt = sim_years[1] - sim_years[0] if len(sim_years) > 1 else 0.1
            det_idx = min(int((fab_detection_year - sim_years[0]) / dt), len(fab_prod) - 1)
            det_idx = max(0, det_idx)
            h100e_at_detection = fab_prod[det_idx]
        else:
            h100e_at_detection = 0.0

        # Get process node
        process_node = f"{int(bp.get('fab_process_node_nm', 28))}nm"

        # Calculate energy using same formula as reference model:
        # energy_gw = h100e * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / energy_efficiency / 1e9
        # where H100_TPP_PER_CHIP * H100_WATTS_PER_TPP ≈ 700W and energy_efficiency = 0.2
        # Note: Reference uses energy_efficiency_of_prc_stock_relative_to_state_of_the_art (0.2)
        # not the fab-specific watts_per_h100e
        H100_WATTS_PER_H100E = 700.0
        ENERGY_EFFICIENCY_OF_PRC_STOCK = 0.2
        energy_gw = h100e_at_detection * H100_WATTS_PER_H100E / ENERGY_EFFICIENCY_OF_PRC_STOCK / 1e9

        individual_h100e.append(h100e_at_detection)
        individual_time.append(operational_time)
        individual_process_nodes.append(process_node)
        individual_energy.append(energy_gw)

    return individual_h100e, individual_time, individual_process_nodes, individual_energy


def extract_fab_ccdf_values_at_threshold(fab_built_sims: List[Dict], years: List[float], agreement_year: float, lr_threshold: float) -> Tuple[List[float], List[float]]:
    """
    Extract fab compute and operational time values at detection for CCDF calculation.

    This matches the reference model's extract_fab_compute_at_detection which:
    1. Uses fab-specific LR (lr_fab_combined) for detection threshold
    2. Calculates operational_time = max(0.0, detection_year - operational_start)
    3. Gets cumulative compute at detection year

    IMPORTANT: Different thresholds lead to different detection years, and thus
    different compute/op_time values. This is the key bug fix - previously the code
    ignored the threshold and always used final values.

    Args:
        fab_built_sims: List of simulation data dicts (filtered to sims with fab)
        years: Time points for the simulation
        agreement_year: Year when agreement starts
        lr_threshold: Detection threshold (1, 2, 4, etc.)

    Returns:
        Tuple of (compute_values, op_time_values) for CCDF calculation
    """
    compute_values = []
    op_time_values = []

    for d in fab_built_sims:
        bp = d.get('black_project')
        sim_years = d.get('years', years)

        if not bp or not sim_years:
            continue

        # Get fab-specific LR for detection calculation
        lr_fab_combined = bp.get('lr_fab_combined', [])

        # Find fab detection year based on fab LR threshold
        fab_detection_year = None
        for i, year in enumerate(sim_years):
            if year >= agreement_year and i < len(lr_fab_combined):
                if lr_fab_combined[i] >= lr_threshold:
                    fab_detection_year = year
                    break

        # If no detection, use end of simulation
        if fab_detection_year is None:
            fab_detection_year = sim_years[-1] if sim_years else agreement_year + 7

        # Calculate when fab became operational
        construction_start = bp.get('fab_construction_start_year', agreement_year)
        construction_duration = bp.get('fab_construction_duration', 1.5)
        operational_start = construction_start + construction_duration

        # Time operational before detection (matching reference model)
        operational_time = max(0.0, fab_detection_year - operational_start)

        # Get compute at detection
        fab_prod = bp.get('fab_cumulative_production_h100e', [])
        if fab_prod and sim_years:
            dt = sim_years[1] - sim_years[0] if len(sim_years) > 1 else 0.1
            det_idx = min(int((fab_detection_year - sim_years[0]) / dt), len(fab_prod) - 1)
            det_idx = max(0, det_idx)
            compute_at_detection = fab_prod[det_idx]
        else:
            compute_at_detection = 0.0

        compute_values.append(compute_at_detection)
        op_time_values.append(operational_time)

    return compute_values, op_time_values


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


def compute_average_covert_compute(all_data: List[Dict], years: List[float], agreement_year: float, lr_threshold: float) -> List[float]:
    """
    Compute average covert compute from agreement to detection for each simulation.

    This matches the reference implementation which calculates:
    average_compute = h100_years / time_duration
    where time_duration = detection_year - agreement_year

    For each simulation and detection threshold:
    1. Calculate detection year based on likelihood ratio threshold
    2. Compute cumulative H100-years from agreement year to detection
    3. Return average = h100_years / time_duration
    """
    if not years or len(years) < 2:
        return [0.0 for _ in all_data]

    dt = years[1] - years[0]
    average_compute_values = []

    for d in all_data:
        bp = d.get('black_project')
        sim_years = d.get('years', [])

        if not bp or not sim_years:
            average_compute_values.append(0.0)
            continue

        operational_compute = bp.get('operational_compute', [])
        if not operational_compute:
            average_compute_values.append(0.0)
            continue

        # Get detection year based on LR threshold
        detection_year = get_detection_year(d, agreement_year, lr_threshold)
        if detection_year is None:
            detection_year = max(sim_years) if sim_years else agreement_year + 10

        # Calculate time duration from agreement to detection
        time_duration = detection_year - agreement_year
        if time_duration <= 0:
            # If detection at or before agreement, average is 0
            average_compute_values.append(0.0)
            continue

        # Sum operational compute from agreement_year to detection_year (H100-years)
        cumulative = 0.0
        for i, year in enumerate(sim_years):
            if year < agreement_year:
                continue
            if year >= detection_year:
                break
            if i < len(operational_compute):
                cumulative += operational_compute[i] * dt  # H100e * years = H100-years

        # Average operational compute = total H100-years / time duration
        average_compute = cumulative / time_duration
        average_compute_values.append(average_compute)

    return average_compute_values


def compute_datacenter_capacity_at_detection(all_data: List[Dict], years: List[float], lr_threshold: float) -> List[float]:
    """
    Compute datacenter capacity (GW) at detection time for each simulation.

    This matches the reference implementation which:
    1. Uses datacenter-specific LR (combined: satellite * reported_energy * worker)
    2. Finds first year where LR >= threshold
    3. Returns capacity at that detection year (or end of simulation if not detected)
    """
    if not years or len(years) < 2:
        return [0.0 for _ in all_data]

    capacity_at_detection = []

    for d in all_data:
        bp = d.get('black_project')
        sim_years = d.get('years', years)

        if not bp or not sim_years:
            capacity_at_detection.append(0.0)
            continue

        datacenter_capacity_gw = bp.get('datacenter_capacity_gw', [])
        if not datacenter_capacity_gw:
            capacity_at_detection.append(0.0)
            continue

        # Get combined datacenter LR (satellite * reported_energy * worker)
        lr_satellite = bp.get('lr_satellite', [])
        lr_reported_energy = bp.get('lr_reported_energy', [])
        lr_worker = bp.get('lr_worker', [])

        # Find detection year based on combined datacenter LR
        detection_idx = None
        for i, year in enumerate(sim_years):
            # Compute combined LR at this time
            lr_sat = lr_satellite[i] if i < len(lr_satellite) else 1.0
            lr_energy = lr_reported_energy[i] if i < len(lr_reported_energy) else 1.0
            lr_work = lr_worker[i] if i < len(lr_worker) else 1.0
            combined_lr = lr_sat * lr_energy * lr_work

            if combined_lr >= lr_threshold:
                detection_idx = i
                break

        if detection_idx is not None:
            # Get capacity at detection
            capacity = datacenter_capacity_gw[detection_idx] if detection_idx < len(datacenter_capacity_gw) else datacenter_capacity_gw[-1]
        else:
            # Not detected - use final year capacity
            capacity = datacenter_capacity_gw[-1] if datacenter_capacity_gw else 0.0

        capacity_at_detection.append(capacity)

    return capacity_at_detection


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
            'lr_prc_accounting': to_float(first_bp.lr_prc_accounting) if first_bp and hasattr(first_bp, 'lr_prc_accounting') else 1.0,
            'lr_sme_inventory': to_float(first_bp.lr_sme_inventory) if first_bp and hasattr(first_bp, 'lr_sme_inventory') else 1.0,
            'lr_satellite_datacenter': to_float(first_bp.lr_satellite_datacenter) if first_bp and hasattr(first_bp, 'lr_satellite_datacenter') else 1.0,
            'lr_fab_procurement': to_float(first_bp.lr_fab_procurement) if first_bp and hasattr(first_bp, 'lr_fab_procurement') else 1.0,
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

                # Fab-specific LR metrics (use fab labor and time from construction start)
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
            # Fab construction timing (needed for compute_fab_detection_data)
            bp_data['fab_construction_start_year'] = to_float(final_bp.fab_construction_start_year) if hasattr(final_bp, 'fab_construction_start_year') else 2030.0
            bp_data['fab_construction_duration'] = to_float(final_bp.fab_construction_duration) if hasattr(final_bp, 'fab_construction_duration') else 1.5
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


def _compute_fab_dashboard(fab_built_sims: List[Dict], all_data: List[Dict], detection_times: List[float], num_sims: int, dt: float,
                           fab_individual_h100e: List[float] = None, fab_individual_time: List[float] = None,
                           fab_individual_energy: List[float] = None) -> Dict[str, Any]:
    """
    Compute pre-formatted dashboard values for the covert fab section.
    This avoids frontend computation.

    If fab_individual_* parameters are provided (from compute_fab_detection_data), use those
    for correct fab-specific detection calculations. Otherwise falls back to project detection.
    """
    from collections import Counter

    num_fab_built = len(fab_built_sims)
    prob_fab_built = f"{(num_fab_built / num_sims * 100):.1f}%" if num_sims > 0 else "--"

    # Use fab-specific detection data if provided (from compute_fab_detection_data)
    # These use fab LR and operational time, matching the reference model
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
                            'initial_stock_energy_gw', 'fab_compute_energy_gw', 'total_compute_energy_gw',
                            'lr_fab_other', 'lr_fab_combined']:  # Added missing fab LR keys
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
    # Matches reference model's calculate_ccdf which computes P(X > x) not P(X >= x)
    # NOTE: Reference does NOT filter out zeros - they are valid data points (e.g., op_time = 0
    # when detection happens before fab becomes operational)
    def compute_ccdf(values: List[float]) -> List[Dict[str, float]]:
        # Only filter out infinities and NaNs, but keep zeros and negative values
        values = [v for v in values if v < float('inf') and not np.isnan(v)]
        if not values:
            return []
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        ccdf = []
        prev_val = None
        for i, val in enumerate(sorted_vals):
            if val != prev_val:
                # P(X > x) = (count of values strictly greater than x) / total
                # For value at index i, there are (n - (i + 1)) values greater than it
                num_greater = n - (i + 1)
                ccdf.append({"x": float(val), "y": float(num_greater / n)})
                prev_val = val
            else:
                # For duplicate values, update the last y (will be lower)
                num_greater = n - (i + 1)
                ccdf[-1]["y"] = float(num_greater / n)
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

    # Fab-specific detection data (uses fab LR, not project LR; time is operational time not from agreement)
    fab_individual_h100e, fab_individual_time, fab_individual_process_nodes, fab_individual_energy = \
        compute_fab_detection_data(all_data, years, agreement_year)

    # Energy before detection (at detection time)
    # Use detection_times which is now based on LR threshold
    # Match reference formula: energy_gw = operational_h100e * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / energy_efficiency / 1e9
    # Where H100_TPP_PER_CHIP = 2144, H100_WATTS_PER_TPP = 0.326493, energy_efficiency = 0.2
    # This simplifies to: energy_gw = operational_h100e * 3500 / 1e9
    H100_TPP_PER_CHIP = 2144.0
    H100_WATTS_PER_TPP = 0.326493
    ENERGY_EFFICIENCY_PRC = 0.2  # energy_efficiency_of_prc_stock_relative_to_state_of_the_art
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

        # Get detection year (agreement_year + detection_times[i])
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
            # Use reference formula: energy_gw = operational_h100e * 3500 / 1e9
            energy_gw = op_h100e * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / ENERGY_EFFICIENCY_PRC / 1e9
            energy_before_detection.append(energy_gw)
        else:
            op_h100e = operational_compute[-1] if operational_compute else 0.0
            energy_gw = op_h100e * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / ENERGY_EFFICIENCY_PRC / 1e9
            energy_before_detection.append(energy_gw)

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

            # Compute ratios as covert / no-slowdown (values between 0 and 1)
            # Small values mean covert production is much smaller than no-slowdown
            if bp_chip_production <= 0:
                chip_global.append(0.0)  # No covert production = 0 ratio
                chip_prc.append(0.0)
            else:
                chip_global.append(bp_chip_production / global_production if global_production > 0 else 0.0)
                chip_prc.append(bp_chip_production / prc_production if prc_production > 0 else 0.0)

            if bp_h100_years <= 0:
                ai_largest.append(0.0)  # No covert computation = 0 ratio
                ai_prc.append(0.0)
            else:
                ai_largest.append(bp_h100_years / largest_h100_years if largest_h100_years > 0 else 0.0)
                ai_prc.append(bp_h100_years / prc_h100_years if prc_h100_years > 0 else 0.0)

        return {
            'chip_global': chip_global,
            'chip_prc': chip_prc,
            'chip_largest': chip_prc,  # Same as PRC for chip production
            'ai_largest': ai_largest,
            'ai_prc': ai_prc,
        }

    # Detection probability at each time point
    # Reference model: datacenter_detection_prob = np.mean(np.array(lr_datacenters_by_sim) >= 5.0, axis=0)
    # This is the proportion of simulations where datacenter LR >= 5.0 at each time step
    #
    # IMPORTANT: Reference model's cumulative_lr_from_concealed_datacenters includes 3 components:
    #   1. lr_from_identifying_datacenters_with_satellites() - fixed per simulation
    #   2. lr_from_reported_energy_consumption(year) - time series
    #   3. cumulative_lr_from_direct_observation(year) - time series (worker-based)
    # So: lr_datacenters = lr_satellite * lr_reported_energy * lr_worker (lr_other_intel)
    lr_datacenters_by_sim = []
    for d in all_data:
        bp = d.get('black_project')
        if not bp:
            lr_datacenters_by_sim.append([1.0] * len(years))
            continue

        # Get the three LR components
        lr_satellite = bp.get('lr_satellite_datacenter', 1.0)  # Fixed value
        lr_reported_energy = bp.get('lr_reported_energy', [1.0] * len(years))  # Time series
        lr_worker = bp.get('lr_other_intel', [1.0] * len(years))  # Time series (worker detection)

        # Compute combined datacenter LR (product of all three components)
        combined_lr = []
        for i in range(len(years)):
            lr_energy = lr_reported_energy[i] if i < len(lr_reported_energy) else 1.0
            lr_work = lr_worker[i] if i < len(lr_worker) else 1.0
            combined_lr.append(lr_satellite * lr_energy * lr_work)
        lr_datacenters_by_sim.append(combined_lr)

    lr_datacenters_array = np.array(lr_datacenters_by_sim)
    datacenter_detection_prob = (np.mean(lr_datacenters_array >= 5.0, axis=0)).tolist()

    # Compute datacenter-specific detection times using COMBINED datacenter LR (threshold = 5.0)
    # This is different from project detection which uses cumulative_lr
    LR_DETECTION_THRESHOLD = 5.0
    datacenter_detection_times = []
    for i, d in enumerate(all_data):
        bp = d.get('black_project')
        sim_years = d.get('years', years)

        if not bp or not sim_years:
            # Default to end of simulation
            datacenter_detection_times.append(sim_years[-1] - agreement_year if sim_years else 7.0)
            continue

        # Use the combined datacenter LR we computed above
        lr_datacenters = lr_datacenters_by_sim[i]

        # Find first year where datacenter LR >= threshold
        detection_year = None
        for j, year in enumerate(sim_years):
            if j < len(lr_datacenters) and lr_datacenters[j] >= LR_DETECTION_THRESHOLD:
                detection_year = year
                break

        if detection_year is not None:
            datacenter_detection_times.append(detection_year - agreement_year)
        else:
            # Not detected - use time to end of simulation
            datacenter_detection_times.append(sim_years[-1] - agreement_year if sim_years else 7.0)

    # Fraction diverted (this is the datacenter capacity diversion fraction, NOT compute diversion)
    # Reference model uses fraction_of_datacenter_capacity_not_built_for_concealment_to_divert_at_black_project_start = 0.01
    # This is fixed from parameters, not computed dynamically
    fraction_diverted = 0.01  # Fixed value from parameters

    # Constants for energy calculation (matching reference model)
    H100_TPP_PER_CHIP = 2144.0  # Tera-Parameter-Passes per H100 chip
    H100_WATTS_PER_TPP = 0.326493  # Watts per Tera-Parameter-Pass
    ENERGY_EFFICIENCY_RELATIVE_TO_SOTA = 0.20  # PRC stock efficiency relative to SOTA
    SOTA_IMPROVEMENT_PER_YEAR = 1.26  # State-of-the-art efficiency improvement per year
    H100_RELEASE_YEAR = 2022

    # Compute state-of-the-art efficiency at agreement year relative to H100 (2022)
    years_since_h100 = agreement_year - H100_RELEASE_YEAR
    sota_efficiency_at_agreement = SOTA_IMPROVEMENT_PER_YEAR ** years_since_h100
    combined_efficiency_at_agreement = ENERGY_EFFICIENCY_RELATIVE_TO_SOTA * sota_efficiency_at_agreement

    # Compute PRC energy capacity for each simulation from sampled growth rates
    # This matches reference model: compute_stock * H100_TPP * H100_WATTS_PER_TPP / combined_efficiency
    prc_capacity_at_agreement_samples_computed = []
    for d in all_data:
        if d.get('prc_params'):
            # Use sampled growth rate for this simulation
            base_compute = d['prc_params']['total_prc_compute_tpp_h100e_in_2025']
            growth_rate = d['prc_params']['annual_growth_rate']
            years_since_2025 = agreement_year - 2025
            compute_stock = base_compute * (growth_rate ** years_since_2025)
            # Convert compute stock to energy capacity in GW
            energy_watts = compute_stock * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / combined_efficiency_at_agreement
            energy_gw = energy_watts / 1e9
            prc_capacity_at_agreement_samples_computed.append(energy_gw)
        else:
            # Fallback: use static model
            prc_capacity_at_agreement_samples_computed.append(0.175 * (1.74 ** (agreement_year - 2025)))

    # PRC capacity data - compute from actual sampled values
    prc_capacity_years = list(range(2025, int(agreement_year) + 1))

    # Compute capacity trajectories for each simulation
    prc_capacity_by_sim = []
    for d in all_data:
        if d.get('prc_params'):
            base_compute = d['prc_params']['total_prc_compute_tpp_h100e_in_2025']
            growth_rate = d['prc_params']['annual_growth_rate']
            capacity_trajectory = []
            for year in prc_capacity_years:
                years_since_2025 = year - 2025
                compute_stock = base_compute * (growth_rate ** years_since_2025)
                # Combined efficiency changes each year
                years_since_h100 = year - H100_RELEASE_YEAR
                sota_eff = SOTA_IMPROVEMENT_PER_YEAR ** years_since_h100
                combined_eff = ENERGY_EFFICIENCY_RELATIVE_TO_SOTA * sota_eff
                energy_gw = (compute_stock * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / combined_eff) / 1e9
                capacity_trajectory.append(energy_gw)
            prc_capacity_by_sim.append(capacity_trajectory)
        else:
            # Fallback
            prc_capacity_by_sim.append([0.175 * (1.74 ** (y - 2025)) for y in prc_capacity_years])

    # Compute percentiles from simulation data
    prc_capacity_array = np.array(prc_capacity_by_sim)
    prc_capacity_gw = {
        "median": np.percentile(prc_capacity_array, 50, axis=0).tolist(),
        "p25": np.percentile(prc_capacity_array, 25, axis=0).tolist(),
        "p75": np.percentile(prc_capacity_array, 75, axis=0).tolist(),
    }
    prc_capacity_at_agreement = float(np.median(prc_capacity_at_agreement_samples_computed))

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
            # initial_black_project: Reference model uses total surviving compute (initial + fab)
            # NOT just initial stock. See reference: black_project_by_sim = initial_h100e + fab_h100e
            "initial_black_project": get_percentiles_with_individual(
                lambda d: d['black_project']['total_compute'] if d['black_project'] else [0.0] * len(years)
            ),
            # black_fab_flow: FILTERED to only fab-built simulations (for Covert fab section plots)
            "black_fab_flow": get_fab_percentiles_with_individual(
                lambda d: d['black_project']['fab_cumulative_production_h100e'] if d['black_project'] else [0.0] * len(years)
            ),
            # black_fab_flow_all_sims: ALL simulations (for Dark Compute Stock Breakdown)
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

            # CCDFs - compute for each threshold separately (not using pre-computed values)
            "time_to_detection_ccdf": {
                str(lr): compute_ccdf(compute_detection_times(all_data, years, agreement_year, lr))
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "h100_years_ccdf": {
                str(lr): compute_ccdf(compute_h100_years_before_detection(all_data, years, agreement_year, lr))
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            # Compute average covert compute CCDF for each lr threshold separately
            # Reference computes: average_compute = h100_years / (detection_year - agreement_year)
            "average_covert_compute_ccdf": {
                str(lr): compute_ccdf(compute_average_covert_compute(all_data, years, agreement_year, lr))
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
            "lr_datacenters": {
                # Use combined datacenter LR (satellite * reported_energy * worker)
                # lr_datacenters_by_sim was computed earlier as the product of all 3 components
                "p25": np.percentile(lr_datacenters_array, 25, axis=0).tolist(),
                "median": np.percentile(lr_datacenters_array, 50, axis=0).tolist(),
                "p75": np.percentile(lr_datacenters_array, 75, axis=0).tolist(),
                "individual": [sim.tolist() if hasattr(sim, 'tolist') else list(sim) for sim in lr_datacenters_by_sim]
            },
            "datacenter_detection_prob": datacenter_detection_prob,
            "energy_by_source": energy_by_source,
            "source_labels": source_labels,
            "fraction_diverted": fraction_diverted,
            # Reference uses only threshold [1] for datacenter capacity CCDFs since detection is binary
            # Must compute detection using threshold 1 (not pre-computed datacenter_detection_times which uses 5.0)
            "capacity_ccdfs": {
                str(lr): compute_ccdf(compute_datacenter_capacity_at_detection(all_data, years, lr))
                for lr in [1]  # Only threshold 1 - datacenter detection is binary
            },
            "individual_capacity_before_detection": [
                d['black_project']['datacenter_capacity_gw'][
                    min(int(datacenter_detection_times[i] / dt) if dt > 0 else 0, len(d['black_project']['datacenter_capacity_gw']) - 1)
                ] if d['black_project'] and d['black_project'].get('datacenter_capacity_gw') else 0.0
                for i, d in enumerate(all_data)
            ],
            "individual_time_before_detection": datacenter_detection_times,
            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,
            "prc_capacity_years": prc_capacity_years,
            "prc_capacity_gw": prc_capacity_gw,
            "prc_capacity_at_agreement_year_gw": prc_capacity_at_agreement,
            "prc_capacity_at_agreement_year_samples": prc_capacity_at_agreement_samples_computed,
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
            # LR values: Use FAB-SPECIFIC LR metrics (lr_fab_combined and lr_fab_other)
            # These are computed using fab labor (construction + operation) and time from fab construction start
            # lr_fab_combined = lr_sme_inventory × lr_prc_accounting × lr_fab_other
            "lr_combined": get_fab_percentiles_with_individual(
                lambda d: d['black_project']['lr_fab_combined'] if d['black_project'] else [1.0] * len(years)
            ),
            "lr_inventory": get_fab_percentiles_with_individual(
                lambda d: [d['black_project']['lr_sme_inventory']] * len(years) if d['black_project'] else [1.0] * len(years)
            ),
            "lr_procurement": get_fab_percentiles_with_individual(
                lambda d: [d['black_project']['lr_fab_procurement']] * len(years) if d['black_project'] else [1.0] * len(years)
            ),
            "lr_other": get_fab_percentiles_with_individual(
                lambda d: d['black_project']['lr_fab_other'] if d['black_project'] else [1.0] * len(years)
            ),
            "process_node_by_sim": [
                f"{int(d['black_project'].get('fab_process_node_nm', 28))}nm" if d['black_project'] else "28nm"
                for d in fab_built_sims
            ] if fab_built_sims else [],
            "individual_process_node": fab_individual_process_nodes,  # From compute_fab_detection_data
            # Fab detection data uses fab-specific LR and operational time (matching reference model)
            "individual_h100e_before_detection": fab_individual_h100e,
            "individual_time_before_detection": fab_individual_time,
            "individual_energy_before_detection": fab_individual_energy,
            "compute_ccdf": [],
            # Use extract_fab_ccdf_values_at_threshold to get values at detection for each threshold
            # This is the key fix - different thresholds lead to different detection years and thus
            # different compute/op_time values (previously ignored threshold, used final values)
            "compute_ccdfs": {
                str(lr): compute_ccdf(extract_fab_ccdf_values_at_threshold(fab_built_sims, years, agreement_year, lr)[0])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "op_time_ccdf": [],
            "op_time_ccdfs": {
                str(lr): compute_ccdf(extract_fab_ccdf_values_at_threshold(fab_built_sims, years, agreement_year, lr)[1])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,

            # Pre-computed dashboard values (avoid frontend computation)
            "dashboard": _compute_fab_dashboard(fab_built_sims, all_data, detection_times, num_sims, dt,
                                                fab_individual_h100e=fab_individual_h100e,
                                                fab_individual_time=fab_individual_time,
                                                fab_individual_energy=fab_individual_energy),
        },

        # initial_black_project (4 keys)
        # Reference format: h100e = fab production (filtered to fab-built sims), in thousands
        # black_project = total surviving compute (initial + fab), in thousands
        "initial_black_project": {
            "years": years,
            # Total dark compute (surviving initial + fab), in thousands to match reference
            "black_project": get_percentiles_with_individual(
                lambda d: [v / 1000 for v in d['black_project']['total_compute']] if d['black_project'] else [0.0] * len(years)
            ),
            # Fab production only, filtered to fab-built sims, in thousands
            "h100e": get_fab_percentiles_with_individual(
                lambda d: [v / 1000 for v in d['black_project']['fab_cumulative_production_h100e']] if d['black_project'] else [0.0] * len(years)
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
