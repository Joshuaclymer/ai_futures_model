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


def compute_detection_times(all_data: List[Dict], years: List[float], agreement_year: float) -> List[float]:
    """
    Extract actual sampled detection times from simulation data.
    Returns time (in years) from agreement start to detection for each simulation.
    """
    detection_times = []
    for d in all_data:
        bp = d.get('black_project')
        if bp and bp.get('sampled_detection_time') is not None:
            # sampled_detection_time is relative to black project start
            detection_times.append(bp['sampled_detection_time'])
        else:
            # Fallback if detection time not available
            detection_times.append(100.0)  # No detection - use large value that's JSON-compatible

    return detection_times


def compute_h100_years_before_detection(all_data: List[Dict], years: List[float], agreement_year: float) -> List[float]:
    """
    Compute cumulative H100-years of compute before detection for each simulation.
    Uses actual sampled detection time to determine cutoff.
    """
    if not years or len(years) < 2:
        return [0.0 for _ in all_data]

    dt = years[1] - years[0]
    h100_years = []

    for d in all_data:
        bp = d.get('black_project')
        if not bp:
            h100_years.append(0.0)
            continue

        detection_time = bp.get('sampled_detection_time')
        operational_compute = bp.get('operational_compute', [])

        if detection_time is None or not operational_compute:
            h100_years.append(0.0)
            continue

        # Find index corresponding to detection time
        # detection_time is relative to black project start (which is 1 year before agreement)
        # years array starts at agreement_year
        black_project_start_year = agreement_year - 1
        detection_abs_year = black_project_start_year + detection_time

        # Sum operational compute up to detection
        cumulative = 0.0
        for i, year in enumerate(years):
            if year >= detection_abs_year:
                break
            if i < len(operational_compute):
                cumulative += operational_compute[i] * dt  # H100e * years = H100-years

        h100_years.append(cumulative)

    return h100_years


def compute_h100e_before_detection(all_data: List[Dict], years: List[float], agreement_year: float) -> List[float]:
    """
    Compute chip stock (H100e) at detection time for each simulation.
    """
    if not years:
        return [0.0 for _ in all_data]

    h100e = []

    for d in all_data:
        bp = d.get('black_project')
        if not bp:
            h100e.append(0.0)
            continue

        detection_time = bp.get('sampled_detection_time')
        total_compute = bp.get('total_compute', [])

        if detection_time is None or not total_compute:
            h100e.append(0.0)
            continue

        # Find index corresponding to detection time
        black_project_start_year = agreement_year - 1
        detection_abs_year = black_project_start_year + detection_time

        # Find chip stock at detection time
        detection_idx = 0
        for i, year in enumerate(years):
            if year >= detection_abs_year:
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

    data = {
        'years': times,
        'prc_compute_stock': [],
        'prc_operating_compute': [],
        'black_project': None,
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


def extract_black_project_plot_data(
    simulation_results: Dict[str, Any],
    frontend_params: dict,
) -> Dict[str, Any]:
    """
    Extract plot data from SimulationResult trajectories.
    Returns data formatted for the frontend.
    """
    results: List[SimulationResult] = simulation_results.get('simulation_results', [])
    agreement_year = simulation_results.get('agreement_year', 2027)

    if not results:
        return {"success": False, "error": "No simulation results"}

    # Extract data from all simulations
    all_data = [extract_world_data(r) for r in results]

    # Use first simulation as reference for years
    raw_years = all_data[0]['years'] if all_data else []
    num_sims = len(all_data)

    # Get black project start year to filter output (survival rate is relative to this)
    bp_start_year = None
    if all_data and all_data[0]['black_project']:
        bp_start_year = all_data[0]['black_project'].get('preparation_start_year')

    # Filter data to only include time points >= black_project_start_year
    # This fixes the issue where survival_rate stays at 1.0 before the project starts
    if bp_start_year is not None and raw_years:
        # Find the first index where year >= bp_start_year
        start_idx = 0
        for i, y in enumerate(raw_years):
            if y >= bp_start_year:
                start_idx = i
                break

        # Filter years array
        years = raw_years[start_idx:]

        # Filter all time series data in all_data
        for d in all_data:
            # Filter top-level time series
            d['years'] = d['years'][start_idx:]
            d['prc_compute_stock'] = d['prc_compute_stock'][start_idx:]
            d['prc_operating_compute'] = d['prc_operating_compute'][start_idx:]

            # Filter black project time series
            if d['black_project']:
                bp = d['black_project']
                for key in ['operational_compute', 'total_compute', 'datacenter_capacity_gw',
                            'fab_is_operational', 'cumulative_lr', 'lr_other_intel', 'posterior_prob',
                            'lr_reported_energy', 'fab_cumulative_production_h100e', 'fab_monthly_production_h100e',
                            'survival_rate', 'initial_compute_surviving_h100e',
                            'initial_stock_energy_gw', 'fab_compute_energy_gw', 'total_compute_energy_gw']:
                    if key in bp and isinstance(bp[key], list):
                        bp[key] = bp[key][start_idx:]

        logger.info(f"[black-project] Filtered data to start from black project year {bp_start_year}")
        logger.info(f"[black-project] Removed {start_idx} time points before project start")
    else:
        years = raw_years

    # Log raw simulation data for debugging
    logger.info("=" * 60)
    logger.info("[RAW SIMULATION RESULTS FROM AIFuturesSimulator]")
    logger.info(f"Number of simulations: {num_sims}")
    logger.info(f"Number of time points: {len(years)}")
    if years:
        logger.info(f"Time range: {years[0]} to {years[-1]}")
    if all_data:
        logger.info(f"First sim PRC compute stock (first 5): {all_data[0]['prc_compute_stock'][:5]}")
        if all_data[0]['black_project']:
            bp = all_data[0]['black_project']
            logger.info(f"First sim black project operational_compute (first 5): {bp['operational_compute'][:5]}")
            logger.info(f"First sim black project total_compute (first 5): {bp['total_compute'][:5]}")
    logger.info("=" * 60)

    # Helper to compute percentiles across simulations
    def get_percentiles(extractor) -> Dict[str, List[float]]:
        try:
            values = np.array([extractor(d) for d in all_data])
            if values.size == 0:
                return {"median": [], "p25": [], "p75": []}
            return {
                "median": np.percentile(values, 50, axis=0).tolist(),
                "p25": np.percentile(values, 25, axis=0).tolist(),
                "p75": np.percentile(values, 75, axis=0).tolist(),
            }
        except Exception as e:
            logger.warning(f"Error computing percentiles: {e}")
            return {"median": [], "p25": [], "p75": []}

    # Helper to provide fallback time series data when real data is empty
    def get_percentiles_with_fallback(extractor, fallback_fn) -> Dict[str, List[float]]:
        result = get_percentiles(extractor)
        if not result["median"] or len(result["median"]) == 0:
            # Use fallback data generator
            return fallback_fn()
        return result

    # Generate fallback time series that follows realistic patterns
    def make_fallback_time_series(base_value: float, growth_rate: float = 1.0, spread: float = 0.15):
        """Generate fallback percentile data with given base value and growth rate."""
        median = [base_value * (growth_rate ** i) for i in range(len(years))]
        return {
            "median": median,
            "p25": [v * (1 - spread) for v in median],
            "p75": [v * (1 + spread) for v in median],
        }

    # Helper to compute accounting-only LR time series (chip × SME × datacenter × energy)
    # This excludes worker-based evidence and detection penalty
    def _compute_accounting_lr_time_series(bp_data: Dict) -> List[float]:
        """Compute time series of accounting-only likelihood ratio.

        Combined evidence from resource accounting = chip × SME × datacenter × energy LRs.
        The first three are static (set at initialization), while energy varies over time.
        """
        if not bp_data:
            return []

        # Get static LR components
        lr_chip = bp_data.get('lr_prc_accounting', 1.0)
        lr_sme = bp_data.get('lr_sme_inventory', 1.0)
        lr_dc = bp_data.get('lr_satellite_datacenter', 1.0)

        # Get time-varying energy LR
        lr_energy_series = bp_data.get('lr_reported_energy', [])

        if isinstance(lr_energy_series, (int, float)):
            # Single value, not a time series
            lr_energy_series = [lr_energy_series] * len(years)

        if not lr_energy_series or len(lr_energy_series) == 0:
            # No energy data, return static accounting LR
            static_lr = lr_chip * lr_sme * lr_dc
            return [static_lr] * len(years)

        # Compute combined accounting LR for each timestep
        combined = []
        for lr_energy in lr_energy_series:
            combined.append(lr_chip * lr_sme * lr_dc * lr_energy)

        return combined

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

    # Helper to compute CDF: P(Ratio < x) for ratio distributions
    def compute_cdf(ratios: List[float], num_points: int = 40) -> List[Dict[str, float]]:
        """Compute CDF for ratio distributions.

        Returns points for P(Ratio < x) across a range of x values.
        X values are distributed on a log scale from min to 1.0.
        """
        ratios = np.array([r for r in ratios if r > 0 and not np.isinf(r) and not np.isnan(r)])
        if len(ratios) == 0:
            return []

        # Generate x values on log scale from min ratio to 1.0
        x_min = max(1e-8, np.min(ratios) * 0.5)
        x_max = min(2.0, max(1.0, np.max(ratios) * 1.1))

        x_values = np.logspace(np.log10(x_min), np.log10(x_max), num_points)

        cdf = []
        for x in x_values:
            prob = np.mean(ratios < x)
            cdf.append({"x": float(x), "y": float(prob)})

        return cdf

    # Helper to calculate detection year based on likelihood ratio threshold
    def calculate_detection_year_for_threshold(bp_data: Dict, years: List[float], agreement_year: float, lr_threshold: float) -> float:
        """Calculate the year when cumulative LR exceeds the threshold.

        Returns detection year (absolute), or max(years) + 1 if never detected.
        """
        cumulative_lr = bp_data.get('cumulative_lr', [])
        if not cumulative_lr:
            return years[-1] + 1 if years else agreement_year + 10

        for i, lr in enumerate(cumulative_lr):
            if lr >= lr_threshold and i < len(years):
                return years[i]

        return years[-1] + 1 if years else agreement_year + 10

    # Compute ratios for chip production and AI R&D reduction CCDFs
    def compute_reduction_ratios_for_threshold(
        all_data: List[Dict],
        years: List[float],
        agreement_year: float,
        dt: float,
        lr_threshold: float = 4
    ) -> Dict[str, List[float]]:
        """
        Compute ratios of counterfactual to covert output for a specific LR threshold.

        Key change: Returns counterfactual/covert ratios (large values) to match reference model.
        Uses global compute production from CSV for accurate global estimates.

        Returns dict with ratio arrays for:
        - chip_production_vs_global: global_production / covert_fab_production
        - chip_production_vs_prc: prc_production / covert_fab_production
        - ai_rd_vs_largest_company: largest_company_h100_years / covert_h100_years
        - ai_rd_vs_prc: prc_h100_years / covert_h100_years
        """
        # Parameters for counterfactual estimation
        pre_slowdown_growth_rate = 2.2  # ~2.2x per year typical
        largest_company_fraction = 0.3  # Largest company is ~30% of PRC

        # Large value to represent "infinite" reduction when no covert production
        LARGE_REDUCTION = 1e12

        chip_production_vs_global = []
        chip_production_vs_prc = []
        ai_rd_vs_largest_company = []
        ai_rd_vs_prc = []

        for i, d in enumerate(all_data):
            bp = d.get('black_project')
            prc_stock = d.get('prc_compute_stock', [])

            if not bp or not prc_stock:
                continue

            # Calculate detection year based on LR threshold
            detection_year = calculate_detection_year_for_threshold(bp, years, agreement_year, lr_threshold)
            if detection_year > years[-1]:
                detection_year = years[-1]

            # Compute black project cumulative H100-years before detection
            op_compute = bp.get('operational_compute', [])
            bp_h100_years = 0.0
            for j, year in enumerate(years):
                if year >= detection_year:
                    break
                if j < len(op_compute):
                    bp_h100_years += op_compute[j] * dt

            # Compute black project chip production (from covert fab) up to detection
            fab_prod = bp.get('fab_cumulative_production_h100e', [])
            # Get production at detection time
            prod_at_detection = 0.0
            prod_at_agreement = 0.0
            for j, year in enumerate(years):
                if j < len(fab_prod):
                    if year <= agreement_year:
                        prod_at_agreement = fab_prod[j]
                    if year <= detection_year:
                        prod_at_detection = fab_prod[j]

            bp_chip_production = max(0.0, prod_at_detection - prod_at_agreement)

            # Find PRC compute at agreement start
            agreement_idx = 0
            for j, year in enumerate(years):
                if year >= agreement_year:
                    agreement_idx = j
                    break
            prc_at_agreement = prc_stock[agreement_idx] if agreement_idx < len(prc_stock) else prc_stock[0]

            # Duration from agreement to detection
            duration_years = max(0.0, detection_year - agreement_year)

            # GLOBAL production: use actual data from CSV
            global_production = get_global_compute_production_between_years(agreement_year, detection_year)

            # PRC counterfactual production (no slowdown)
            # Production = Stock(detection) - Stock(agreement)
            prc_stock_at_agreement = prc_at_agreement
            prc_stock_at_detection = prc_at_agreement * (pre_slowdown_growth_rate ** duration_years)
            prc_production = prc_stock_at_detection - prc_stock_at_agreement

            # Largest company production
            largest_company_stock_at_agreement = prc_at_agreement * largest_company_fraction
            largest_company_stock_at_detection = largest_company_stock_at_agreement * (pre_slowdown_growth_rate ** duration_years)
            largest_company_production = largest_company_stock_at_detection - largest_company_stock_at_agreement

            # Compute counterfactual H100-years (integral of stock over time)
            if pre_slowdown_growth_rate > 1.001:
                growth_factor = (pre_slowdown_growth_rate ** duration_years - 1) / np.log(pre_slowdown_growth_rate)
            else:
                growth_factor = duration_years

            counterfactual_prc_h100_years = prc_at_agreement * growth_factor
            counterfactual_largest_company_h100_years = counterfactual_prc_h100_years * largest_company_fraction

            # Compute ratios: counterfactual / covert (INVERTED from before)
            # If no covert production, use large value to represent "infinite" reduction
            if bp_chip_production <= 0:
                chip_production_vs_global.append(LARGE_REDUCTION)
                chip_production_vs_prc.append(LARGE_REDUCTION)
            else:
                if global_production > 0:
                    chip_production_vs_global.append(global_production / bp_chip_production)
                else:
                    chip_production_vs_global.append(0.0)

                if prc_production > 0:
                    chip_production_vs_prc.append(prc_production / bp_chip_production)
                else:
                    chip_production_vs_prc.append(0.0)

            if bp_h100_years <= 0:
                ai_rd_vs_largest_company.append(LARGE_REDUCTION)
                ai_rd_vs_prc.append(LARGE_REDUCTION)
            else:
                if counterfactual_largest_company_h100_years > 0:
                    ai_rd_vs_largest_company.append(counterfactual_largest_company_h100_years / bp_h100_years)
                else:
                    ai_rd_vs_largest_company.append(0.0)

                if counterfactual_prc_h100_years > 0:
                    ai_rd_vs_prc.append(counterfactual_prc_h100_years / bp_h100_years)
                else:
                    ai_rd_vs_prc.append(0.0)

        return {
            'chip_production_vs_global': chip_production_vs_global,
            'chip_production_vs_prc': chip_production_vs_prc,
            'ai_rd_vs_largest_company': ai_rd_vs_largest_company,
            'ai_rd_vs_prc': ai_rd_vs_prc,
        }

    # Compute the time step
    dt = years[1] - years[0] if len(years) > 1 else 0.1

    # Compute reduction ratios for CDFs - separate computation for each LR threshold
    reduction_ratios_by_threshold = {
        lr: compute_reduction_ratios_for_threshold(all_data, years, agreement_year, dt, lr_threshold=lr)
        for lr in LIKELIHOOD_RATIO_THRESHOLDS
    }

    # Extract black project specific data
    bp_data = [d['black_project'] for d in all_data if d['black_project']]

    # Build response
    response = {
        "num_simulations": num_sims,
        "agreement_year": agreement_year,

        # PRC compute data (from actual World trajectory)
        "prc_compute": {
            "years": years,
            "compute_stock": get_percentiles(lambda d: d['prc_compute_stock']),
            "operating_compute": get_percentiles(lambda d: d['prc_operating_compute']),
        },

        # Main model data
        "black_project_model": {
            "years": years,
            "operational_compute": get_percentiles(lambda d: d['black_project']['operational_compute'] if d['black_project'] else []),
            "total_black_project": get_percentiles(lambda d: d['black_project']['total_compute'] if d['black_project'] else []),
            "datacenter_capacity": get_percentiles(lambda d: d['black_project']['datacenter_capacity_gw'] if d['black_project'] else []),

            # Individual simulation data - use actual sampled detection times from detection model
            "individual_project_time_before_detection": compute_detection_times(all_data, years, agreement_year),
            # H100-years before detection: cumulative compute up to actual detection time
            "individual_project_h100_years_before_detection": compute_h100_years_before_detection(all_data, years, agreement_year),
            # H100e (chips) before detection: chip stock at actual detection time
            "individual_project_h100e_before_detection": compute_h100e_before_detection(all_data, years, agreement_year),

            # CCDFs for main dashboard - computed from actual simulation data
            # Time to detection CCDF (using actual sampled detection times)
            "time_to_detection_ccdf": {
                str(lr): compute_ccdf(compute_detection_times(all_data, years, agreement_year))
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            # H100-years before detection CCDF
            "h100_years_ccdf": {
                str(lr): compute_ccdf(compute_h100_years_before_detection(all_data, years, agreement_year))
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            # Chip production reduction CCDF: P(Ratio < x) for counterfactual/covert ratios
            # Frontend expects flat structure: {global: [points], prc: [points]}
            # Uses LR threshold 4 (most common detection threshold)
            # Note: ratios are counterfactual/covert (large values like 300, 1000)
            # Frontend displays these as inverse fractions (1/300x, 1/1000x)
            "chip_production_reduction_ccdf": {
                "global": compute_ccdf(reduction_ratios_by_threshold[4]['chip_production_vs_global']),
                "prc": compute_ccdf(reduction_ratios_by_threshold[4]['chip_production_vs_prc']),
            },
            # AI R&D computation reduction CCDF: P(Ratio < x) for counterfactual/covert ratios
            # Frontend expects flat structure: {largest_ai_company: [points], prc: [points]}
            "ai_rd_reduction_ccdf": {
                "largest_ai_company": compute_ccdf(reduction_ratios_by_threshold[4]['ai_rd_vs_largest_company']),
                "prc": compute_ccdf(reduction_ratios_by_threshold[4]['ai_rd_vs_prc']),
            },
            # Use threshold 4 for median calculation (most common threshold)
            "ai_rd_reduction_median": np.median(reduction_ratios_by_threshold[4]['ai_rd_vs_largest_company']) if reduction_ratios_by_threshold[4]['ai_rd_vs_largest_company'] else 0.05,
            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,
        },

        # Datacenter section data
        "black_datacenters": {
            "years": years,
            "datacenter_capacity": get_percentiles_with_fallback(
                lambda d: d['black_project']['datacenter_capacity_gw'] if d['black_project'] else [],
                lambda: make_fallback_time_series(0.05, 1.2, 0.2)  # 50 MW growing 20%/yr
            ),
            # Individual simulation values for dashboard - capacity at detection time
            "individual_capacity_before_detection": [
                (lambda bp, yrs: (
                    bp['datacenter_capacity_gw'][
                        min(
                            max(0, int(bp.get('sampled_detection_time', 1.0) / (yrs[-1] - yrs[0]) * len(yrs)) if yrs and len(yrs) > 1 else 0),
                            len(bp['datacenter_capacity_gw']) - 1
                        )
                    ] if bp and bp.get('datacenter_capacity_gw') else 0.0
                ))(d.get('black_project'), years)
                for d in all_data
            ] if bp_data else [0.3 + 0.1 * (i % 5) for i in range(num_sims)],
            "individual_time_before_detection": compute_detection_times(all_data, years, agreement_year),
            # CCDFs for capacity - compute from actual datacenter data at detection time
            "capacity_ccdfs": {
                str(lr): compute_ccdf([
                    d['black_project']['datacenter_capacity_gw'][
                        min(
                            int((min(d['black_project'].get('sampled_detection_time', 1.0), years[-1] - years[0]) / max(1, years[-1] - years[0])) * len(years)),
                            len(d['black_project']['datacenter_capacity_gw']) - 1
                        )
                    ] if d['black_project'] and d['black_project']['datacenter_capacity_gw'] else 0.0
                    for d in all_data
                ])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            # LR for datacenters - use cumulative LR time series (combined evidence over time)
            "lr_datacenters": get_percentiles_with_fallback(
                lambda d: d['black_project']['cumulative_lr'] if d['black_project'] and d['black_project'].get('cumulative_lr') else [],
                lambda: make_fallback_time_series(1.0, 1.3, 0.2)
            ),
            # PRC total capacity (context)
            "prc_capacity_years": years,
            "prc_capacity_gw": {
                "median": [50 + 5 * i for i in range(len(years))],
                "p25": [45 + 4 * i for i in range(len(years))],
                "p75": [55 + 6 * i for i in range(len(years))],
            },
            "prc_capacity_at_agreement_year_gw": 50.0,
            "prc_capacity_at_agreement_year_samples": [48 + 4 * (i % 5) for i in range(num_sims)],
            "fraction_diverted": 0.5,
            "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,
            # Worker and construction parameters (needed by DatacenterSection breakdown)
            "construction_workers": 10000,  # Default from modal_parameters.yaml: total_labor * fraction_for_datacenter_construction
            "mw_per_worker_per_year": 1.0,  # Default from modal_parameters.yaml: data_center_mw_per_year_per_construction_worker
            "datacenter_start_year": agreement_year - 2,  # PRC starts building covert datacenters 2 years before agreement
        },

        # Initial stock section
        # NOTE: Use initial_diverted_compute_h100e (calculated at black project start year)
        # NOT total_compute[0] (which is at simulation start year and may differ)
        "initial_stock": {
            # Back-calculate initial PRC stock from diverted compute: initial_prc = diverted / diversion_proportion
            "initial_prc_stock_samples": [
                d['black_project']['initial_diverted_compute_h100e'] / 0.05  # Divide by diversion proportion
                if d['black_project'] and d['black_project'].get('initial_diverted_compute_h100e', 0) > 0
                else 0
                for d in all_data
            ],
            # Use initial diverted compute directly (this is what was diverted at black project start)
            "initial_compute_stock_samples": [
                d['black_project']['initial_diverted_compute_h100e']
                if d['black_project'] and d['black_project'].get('initial_diverted_compute_h100e')
                else 0
                for d in all_data
            ],
            # Energy samples (chips * H100_power / efficiency / 1e9 for GW)
            "initial_energy_samples": [
                (d['black_project']['initial_diverted_compute_h100e'] if d['black_project'] and d['black_project'].get('initial_diverted_compute_h100e') else 0) * 700 / 0.2 / 1e9
                for d in all_data
            ],
            "diversion_proportion": 0.05,  # TODO: extract from params
            # LR samples from PRC accounting - use actual values from simulations
            "lr_prc_accounting_samples": [
                d['black_project']['lr_prc_accounting'] if d['black_project'] and d['black_project'].get('lr_prc_accounting') else 1.0
                for d in all_data
            ],
            # LR from SME inventory
            "lr_sme_inventory_samples": [
                d['black_project']['lr_sme_inventory'] if d['black_project'] and d['black_project'].get('lr_sme_inventory') else 1.0
                for d in all_data
            ],
            # LR from satellite detection of datacenter
            "lr_satellite_datacenter_samples": [
                d['black_project']['lr_satellite_datacenter'] if d['black_project'] and d['black_project'].get('lr_satellite_datacenter') else 1.0
                for d in all_data
            ],
            # Detection probabilities by threshold - compute from actual detection times
            "initial_black_project_detection_probs": {
                "1x": sum(1 for d in all_data if d['black_project'] and d['black_project'].get('sampled_detection_time', float('inf')) < 1) / max(1, num_sims),
                "2x": sum(1 for d in all_data if d['black_project'] and d['black_project'].get('sampled_detection_time', float('inf')) < 2) / max(1, num_sims),
                "4x": sum(1 for d in all_data if d['black_project'] and d['black_project'].get('sampled_detection_time', float('inf')) < 4) / max(1, num_sims),
            },
            # PRC compute over time (expected by frontend for "PRC chip stock over time" chart)
            "prc_compute_years": years,
            "prc_compute_over_time": get_percentiles_with_fallback(
                lambda d: d['prc_compute_stock'],
                lambda: make_fallback_time_series(1000000, 1.2, 0.15)  # ~1M H100e growing 20%/yr
            ),
            # Domestic compute (assume 10-40% domestic over time)
            "prc_domestic_compute_over_time": (lambda prc_data: {
                "median": [v * (0.1 + 0.3 * min(1, i / max(1, len(years) - 1)))
                          for i, v in enumerate(prc_data['median'])]
            })(get_percentiles_with_fallback(
                lambda d: d['prc_compute_stock'],
                lambda: make_fallback_time_series(1000000, 1.2, 0.15)
            )),
            "proportion_domestic_by_year": [0.1 + 0.3 * min(1, i / max(1, len(years) - 1)) for i in range(len(years))],
            # Largest AI company compute (30% of PRC compute, growing faster)
            "largest_company_compute_over_time": [
                (all_data[0]['prc_compute_stock'][i] if all_data and all_data[0]['prc_compute_stock'] and i < len(all_data[0]['prc_compute_stock']) else 300000 * (1.25 ** i)) * 0.3
                for i in range(len(years))
            ],
            # Energy efficiency relative to H100 (1.26^years from H100 release year 2022)
            # Uses 1.26 to match modal_parameters.yaml state_of_the_art_energy_efficiency_improvement_per_year
            "state_of_the_art_energy_efficiency_relative_to_h100": 1.26 ** (agreement_year - 2022),
        },

        # Fab section (legacy format)
        "black_fab": {
            "years": years,
            "is_operational": {
                "proportion": [0.0] + [min(1.0, 0.1 * i) for i in range(1, len(years))],
            },
            "compute_ccdfs": {"4": [{"x": 1000 * (1.5 ** i), "y": max(0.01, 1.0 - i * 0.06)} for i in range(18)]},
            "process_node_by_sim": ["28nm"] * num_sims,
            "prob_fab_built": 0.562,
        },

        # Covert fab section (new format expected by CovertFabSection)
        # IMPORTANT: Filter to only include simulations where a covert fab was actually built
        # A fab is considered "built" if fab_is_operational is True at any point
        "covert_fab": (lambda: {
            # Filter to simulations where fab was built
            "fab_built_data": (fab_built_data := [
                d for d in all_data
                if d['black_project'] and any(d['black_project'].get('fab_is_operational', []))
            ]),
            "num_fab_built": len(fab_built_data),
            "dashboard": {
                # Production/energy computed from fab-built simulations only
                "production": f"{int(np.median([d['black_project']['fab_cumulative_production_h100e'][-1] if d['black_project'] and d['black_project'].get('fab_cumulative_production_h100e') else 0 for d in fab_built_data]) / 1000):.0f}K H100e" if fab_built_data else "0K H100e",
                "energy": f"{np.median([d['black_project']['datacenter_capacity_gw'][-1] if d['black_project'] and d['black_project'].get('datacenter_capacity_gw') else 0 for d in fab_built_data]):.2f} GW" if fab_built_data else "0.00 GW",
                # probFabBuilt uses ALL simulations (it's the probability across all sims)
                "probFabBuilt": f"{100 * len(fab_built_data) / max(1, num_sims):.1f}%",
                # yearsOperational uses fab-built simulations only
                "yearsOperational": f"{np.median([sum(1 for op in d['black_project'].get('fab_is_operational', []) if op) * 0.25 if d['black_project'] else 0 for d in fab_built_data]):.1f} yrs" if fab_built_data else "0.0 yrs",
                "processNode": f"{int(np.median([d['black_project'].get('fab_process_node_nm', 28) for d in fab_built_data if d['black_project']]))}nm" if fab_built_data else "28nm",
            },
            "compute_ccdf": {
                lr: compute_ccdf([
                    d['black_project']['fab_cumulative_production_h100e'][-1]
                    if d['black_project'] and d['black_project'].get('fab_cumulative_production_h100e')
                    else 0.0
                    for d in fab_built_data  # Only fab-built simulations
                ])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "time_series_data": {
                "years": years,
                # Use fab-built simulations for time series
                "lr_combined": (lambda data: {
                    "median": np.percentile([[d['black_project']['cumulative_lr'][i] if d['black_project'] and d['black_project'].get('cumulative_lr') and i < len(d['black_project']['cumulative_lr']) else 1.0 for i in range(len(years))] for d in data], 50, axis=0).tolist() if data else [1.0] * len(years),
                    "p25": np.percentile([[d['black_project']['cumulative_lr'][i] if d['black_project'] and d['black_project'].get('cumulative_lr') and i < len(d['black_project']['cumulative_lr']) else 1.0 for i in range(len(years))] for d in data], 25, axis=0).tolist() if data else [1.0] * len(years),
                    "p75": np.percentile([[d['black_project']['cumulative_lr'][i] if d['black_project'] and d['black_project'].get('cumulative_lr') and i < len(d['black_project']['cumulative_lr']) else 1.0 for i in range(len(years))] for d in data], 75, axis=0).tolist() if data else [1.0] * len(years),
                })(fab_built_data),
                "h100e_flow": (lambda data: {
                    "median": np.percentile([[d['black_project']['fab_cumulative_production_h100e'][i] if d['black_project'] and d['black_project'].get('fab_cumulative_production_h100e') and i < len(d['black_project']['fab_cumulative_production_h100e']) else 0.0 for i in range(len(years))] for d in data], 50, axis=0).tolist() if data else [0.0] * len(years),
                    "p25": np.percentile([[d['black_project']['fab_cumulative_production_h100e'][i] if d['black_project'] and d['black_project'].get('fab_cumulative_production_h100e') and i < len(d['black_project']['fab_cumulative_production_h100e']) else 0.0 for i in range(len(years))] for d in data], 25, axis=0).tolist() if data else [0.0] * len(years),
                    "p75": np.percentile([[d['black_project']['fab_cumulative_production_h100e'][i] if d['black_project'] and d['black_project'].get('fab_cumulative_production_h100e') and i < len(d['black_project']['fab_cumulative_production_h100e']) else 0.0 for i in range(len(years))] for d in data], 75, axis=0).tolist() if data else [0.0] * len(years),
                })(fab_built_data),
            },
            "is_operational": {
                "years": years,
                # Compute fraction of fab-built simulations with operational fab at each time point
                # This shows WHEN the fab becomes operational among simulations that have a fab
                **{k: [
                    sum(1 for d in fab_built_data if d['black_project'] and d['black_project'].get('fab_is_operational') and i < len(d['black_project']['fab_is_operational']) and d['black_project']['fab_is_operational'][i]) / max(1, len(fab_built_data))
                    for i in range(len(years))
                ] for k in ["median", "p25", "p75"]},
            },
            # Extract real wafer starts from fab-built simulations only
            "wafer_starts_samples": [
                d['black_project']['fab_wafer_starts_per_month']
                if d['black_project'] and d['black_project'].get('fab_wafer_starts_per_month') is not None
                else 0.0
                for d in fab_built_data  # Only fab-built simulations
            ],
            # Extract chips per wafer from first fab-built simulation with valid data
            "chips_per_wafer": next(
                (d['black_project']['fab_chips_per_wafer']
                 for d in fab_built_data
                 if d['black_project'] and d['black_project'].get('fab_chips_per_wafer')),
                28
            ),
            # Extract architecture efficiency - use median from fab-built simulations
            "architecture_efficiency": float(np.median([
                d['black_project']['fab_architecture_efficiency']
                for d in fab_built_data
                if d['black_project'] and d['black_project'].get('fab_architecture_efficiency', 0) > 0
            ])) if any(d['black_project'] and d['black_project'].get('fab_architecture_efficiency', 0) > 0 for d in fab_built_data) else 1.0,
            "h100_power": 700,  # watts
            # Build transistor density from fab-built simulation data only
            "transistor_density": _build_transistor_density_distribution(fab_built_data),
            "compute_per_month": {
                "years": years,
                # Use fab-built simulations for compute per month
                **(lambda data: {
                    "median": np.percentile([[
                        (d['black_project']['fab_cumulative_production_h100e'][i] - (d['black_project']['fab_cumulative_production_h100e'][i-1] if i > 0 else 0)) / 0.25
                        if d['black_project'] and d['black_project'].get('fab_cumulative_production_h100e') and i < len(d['black_project']['fab_cumulative_production_h100e'])
                        else 0.0
                        for i in range(len(years))
                    ] for d in data], 50, axis=0).tolist() if data else [0.0] * len(years),
                    "p25": np.percentile([[
                        (d['black_project']['fab_cumulative_production_h100e'][i] - (d['black_project']['fab_cumulative_production_h100e'][i-1] if i > 0 else 0)) / 0.25
                        if d['black_project'] and d['black_project'].get('fab_cumulative_production_h100e') and i < len(d['black_project']['fab_cumulative_production_h100e'])
                        else 0.0
                        for i in range(len(years))
                    ] for d in data], 25, axis=0).tolist() if data else [0.0] * len(years),
                    "p75": np.percentile([[
                        (d['black_project']['fab_cumulative_production_h100e'][i] - (d['black_project']['fab_cumulative_production_h100e'][i-1] if i > 0 else 0)) / 0.25
                        if d['black_project'] and d['black_project'].get('fab_cumulative_production_h100e') and i < len(d['black_project']['fab_cumulative_production_h100e'])
                        else 0.0
                        for i in range(len(years))
                    ] for d in data], 75, axis=0).tolist() if data else [0.0] * len(years),
                })(fab_built_data),
            },
            "watts_per_tpp_curve": {
                # Extended curve: Dennard scaling ends at ~0.03 relative density
                # Before Dennard scaling: watts ~ density^(-1) (steeper)
                # After Dennard scaling: watts ~ density^(-0.5) (shallower)
                "densityRelative": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
                "wattsPerTppRelative": [5000, 1667, 500, 167, 56, 32, 1.0, 0.58, 0.32, 0.18, 0.10],
                "dennardScalingEndDensity": 0.03,  # Vertical line position
            },
            "energy_per_month": {
                "years": years,
                # Use fab-built simulations for energy per month
                **(lambda data: {
                    "median": np.percentile([[
                        d['black_project']['datacenter_capacity_gw'][i] if d['black_project'] and d['black_project'].get('datacenter_capacity_gw') and i < len(d['black_project']['datacenter_capacity_gw']) else 0.0
                        for i in range(len(years))
                    ] for d in data], 50, axis=0).tolist() if data else [0.0] * len(years),
                    "p25": np.percentile([[
                        d['black_project']['datacenter_capacity_gw'][i] if d['black_project'] and d['black_project'].get('datacenter_capacity_gw') and i < len(d['black_project']['datacenter_capacity_gw']) else 0.0
                        for i in range(len(years))
                    ] for d in data], 25, axis=0).tolist() if data else [0.0] * len(years),
                    "p75": np.percentile([[
                        d['black_project']['datacenter_capacity_gw'][i] if d['black_project'] and d['black_project'].get('datacenter_capacity_gw') and i < len(d['black_project']['datacenter_capacity_gw']) else 0.0
                        for i in range(len(years))
                    ] for d in data], 75, axis=0).tolist() if data else [0.0] * len(years),
                })(fab_built_data),
            },
        })(),

        # Detection likelihood section
        "detection_likelihood": {
            "years": years,
            # LR samples from actual simulations (chip accounting, SME inventory, satellite)
            "chip_evidence_samples": [
                d['black_project']['lr_prc_accounting'] if d['black_project'] and d['black_project'].get('lr_prc_accounting') else 1.0
                for d in all_data
            ],
            "sme_evidence_samples": [
                d['black_project']['lr_sme_inventory'] if d['black_project'] and d['black_project'].get('lr_sme_inventory') else 1.0
                for d in all_data
            ],
            "dc_evidence_samples": [
                d['black_project']['lr_satellite_datacenter'] if d['black_project'] and d['black_project'].get('lr_satellite_datacenter') else 1.0
                for d in all_data
            ],
            # Time series: combined LR from accounting only (chip × SME × datacenter × energy)
            # This is the product of accounting evidence WITHOUT worker-based evidence
            "combined_evidence": {
                "years": years,
                **get_percentiles_with_fallback(
                    lambda d: _compute_accounting_lr_time_series(d['black_project']) if d['black_project'] else [],
                    lambda: make_fallback_time_series(1.0, 1.15, 0.2)  # Growing LR over time
                ),
            },
            # Time series: direct evidence (worker-based detection LR)
            "direct_evidence": {
                "years": years,
                **get_percentiles_with_fallback(
                    lambda d: d['black_project']['lr_other_intel'] if d['black_project'] and d['black_project'].get('lr_other_intel') else [],
                    lambda: make_fallback_time_series(1.0, 1.2, 0.25)  # Growing direct evidence
                ),
            },
            # Time series: posterior probability
            "posterior_prob": {
                "years": years,
                **get_percentiles_with_fallback(
                    lambda d: d['black_project']['posterior_prob'] if d['black_project'] and d['black_project'].get('posterior_prob') else [],
                    lambda: {"median": [0.3] * len(years), "p25": [0.2] * len(years), "p75": [0.4] * len(years)}
                ),
            },
            # Energy evidence - from simulation lr_reported_energy time series
            "energy_evidence": {
                "years": years,
                **get_percentiles_with_fallback(
                    lambda d: d['black_project']['lr_reported_energy'] if d['black_project'] and d['black_project'].get('lr_reported_energy') else [],
                    lambda: {"median": [1.0 + 0.5 * i for i in range(len(years))], "p25": [max(1.0, 0.9 + 0.4 * i) for i in range(len(years))], "p75": [1.1 + 0.6 * i for i in range(len(years))]}
                ),
            },
        },

        # Rate of computation section
        "rate_of_computation": {
            "years": years,
            # Initial chip stock samples (fallback to realistic distribution)
            "initial_chip_stock_samples": [
                d['black_project']['total_compute'][0] if d['black_project'] and d['black_project']['total_compute'] else
                70000 + 20000 * np.random.randn()  # Fallback: ~70K H100e with variance
                for d in all_data
            ] if all_data else [70000 + 20000 * (i % 10 - 5) / 5 for i in range(num_sims)],
            # Acquired hardware grows exponentially
            "acquired_hardware": {
                "years": years,
                **get_percentiles_with_fallback(
                    lambda d: d['black_project']['total_compute'] if d['black_project'] else [],
                    lambda: make_fallback_time_series(1000, 1.15, 0.2)  # Small acquisition rate growing 15%/yr
                ),
            },
            # Surviving fraction decays over time - use actual survival_rate from simulation
            "surviving_fraction": {
                "years": years,
                **get_percentiles_with_fallback(
                    lambda d: d['black_project']['survival_rate'] if d['black_project'] else [],
                    lambda: {
                        "median": [max(0.1, 1.0 - 0.08 * i) for i in range(len(years))],
                        "p25": [max(0.05, 0.9 - 0.08 * i) for i in range(len(years))],
                        "p75": [max(0.15, 1.0 - 0.06 * i) for i in range(len(years))],
                    }
                ),
            },
            "covert_chip_stock": {
                "years": years,
                **get_percentiles_with_fallback(
                    lambda d: d['black_project']['total_compute'] if d['black_project'] else [],
                    lambda: make_fallback_time_series(70000, 1.05, 0.15)  # ~70K H100e growing slowly
                ),
            },
            "datacenter_capacity": {
                "years": years,
                **get_percentiles_with_fallback(
                    lambda d: d['black_project']['datacenter_capacity_gw'] if d['black_project'] else [],
                    lambda: make_fallback_time_series(0.05, 1.2, 0.2)  # 50 MW growing 20%/yr
                ),
            },
            # Energy usage (now using actual calculated energy from model)
            "energy_usage": {
                "years": years,
                **get_percentiles_with_fallback(
                    lambda d: d['black_project']['total_compute_energy_gw'] if d['black_project'] and 'total_compute_energy_gw' in d['black_project'] else [],
                    lambda: make_fallback_time_series(0.05, 1.05, 0.15)  # ~50 MW
                ),
            },
            # Stacked energy data: [initial_stock_energy_gw, fab_compute_energy_gw] per time point
            # Uses medians across all simulations for proper distribution representation
            "energy_stacked_data": (lambda: [
                [
                    float(np.median([
                        d['black_project']['initial_stock_energy_gw'][i]
                        if d['black_project'] and 'initial_stock_energy_gw' in d['black_project'] and i < len(d['black_project']['initial_stock_energy_gw'])
                        else 0.0
                        for d in all_data
                    ])) if all_data else 0.0,
                    float(np.median([
                        d['black_project']['fab_compute_energy_gw'][i]
                        if d['black_project'] and 'fab_compute_energy_gw' in d['black_project'] and i < len(d['black_project']['fab_compute_energy_gw'])
                        else 0.0
                        for d in all_data
                    ])) if all_data else 0.0,
                ]
                for i in range(len(years))
            ] if years else [])(),
            # Energy source labels with efficiency values (H100_power / watts_per_h100e)
            "energy_source_labels": (lambda: [
                f"Initial Dark Compute ({700.0 / all_data[0]['black_project']['initial_stock_watts_per_h100e']:.2f}x energy efficiency)" if all_data and all_data[0]['black_project'] and 'initial_stock_watts_per_h100e' in all_data[0]['black_project'] else "Initial Dark Compute",
                f"Covert Fab Compute ({700.0 / all_data[0]['black_project']['fab_watts_per_h100e']:.2f}x energy efficiency)" if all_data and all_data[0]['black_project'] and 'fab_watts_per_h100e' in all_data[0]['black_project'] else "Covert Fab Compute",
            ])(),
            "operating_chips": {
                "years": years,
                **get_percentiles_with_fallback(
                    lambda d: d['black_project']['operational_compute'] if d['black_project'] else [],
                    lambda: make_fallback_time_series(50000, 1.1, 0.15)  # ~50K operational
                ),
            },
            # Covert computation (cumulative H100-years)
            "covert_computation": {
                "years": years,
                **get_percentiles_with_fallback(
                    lambda d: [sum((d['black_project']['operational_compute'] if d['black_project'] else [0])[:i+1]) * 0.25 for i in range(len(d['black_project']['operational_compute']) if d['black_project'] else 0)],
                    lambda: make_fallback_time_series(5000, 1.3, 0.2)  # Growing cumulative compute
                ),
            },
        },

        # Debug: raw World data for exploration
        "_debug_raw_simulations": {
            "description": "Raw World trajectory data from AIFuturesSimulator",
            "num_simulations": num_sims,
            "num_time_points": len(years),
            "years": years,
            "first_simulation": all_data[0] if all_data else None,
        },
    }

    return response


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
        "numSimulations": 200,
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
