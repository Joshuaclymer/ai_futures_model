"""
Detection computation functions for black project simulation.

Contains functions for computing detection times, H100-years, and other
metrics based on likelihood ratio thresholds.
"""

from typing import Dict, List, Optional, Tuple


def get_detection_year(d: Dict, ai_slowdown_start_year: float, lr_threshold: float = 5.0) -> Optional[float]:
    """
    Get the detection year for a simulation based on cumulative LR threshold.

    Returns the first year >= ai_slowdown_start_year where cumulative_lr >= lr_threshold,
    or None if no detection occurs.
    """
    bp = d.get('black_project')
    sim_years = d.get('years', [])

    if not bp or not sim_years:
        return None

    cumulative_lr = bp.get('cumulative_lr', [])

    for i, year in enumerate(sim_years):
        if year >= ai_slowdown_start_year and i < len(cumulative_lr):
            if cumulative_lr[i] >= lr_threshold:
                return year

    return None


def compute_detection_times(
    all_data: List[Dict],
    years: List[float],
    ai_slowdown_start_year: float,
    lr_threshold: float = 5.0,
    use_final_year_for_never_detected: bool = False
) -> List[float]:
    """
    Compute detection times based on when cumulative LR exceeds threshold.

    Returns time (in years) from agreement year to detection for each simulation.
    This matches the reference implementation which uses LR threshold = 5 for dashboard.

    Detection is defined as the first year >= ai_slowdown_start_year where cumulative_lr >= lr_threshold.

    Args:
        all_data: List of simulation data dicts
        years: Time points
        ai_slowdown_start_year: Year when agreement starts
        lr_threshold: LR threshold for detection
        use_final_year_for_never_detected: If True, use (final_year - ai_slowdown_start_year) for
            never-detected cases (for dashboard individual values). If False, use 1000
            (for CCDFs). Reference model uses different values for these two cases.
    """
    # Large value to represent "never detected" - used for CCDFs
    NEVER_DETECTED_VALUE = 1000

    detection_times = []
    for d in all_data:
        bp = d.get('black_project')
        sim_years = d.get('years', [])

        if not bp or not sim_years:
            if use_final_year_for_never_detected and years:
                detection_times.append(max(years) - ai_slowdown_start_year)
            else:
                detection_times.append(NEVER_DETECTED_VALUE)
            continue

        cumulative_lr = bp.get('cumulative_lr', [])

        # Find first year >= ai_slowdown_start_year where LR >= threshold
        detection_year = None
        for i, year in enumerate(sim_years):
            if year >= ai_slowdown_start_year and i < len(cumulative_lr):
                if cumulative_lr[i] >= lr_threshold:
                    detection_year = year
                    break

        if detection_year is not None:
            # Time from agreement year to detection
            time_before_detection = detection_year - ai_slowdown_start_year
        else:
            # No detection within simulation
            # Reference model uses final_year for dashboard values, 1000 for CCDFs
            if use_final_year_for_never_detected:
                final_year = max(sim_years) if sim_years else (max(years) if years else ai_slowdown_start_year + 7)
                time_before_detection = final_year - ai_slowdown_start_year
            else:
                time_before_detection = NEVER_DETECTED_VALUE

        detection_times.append(max(0.0, time_before_detection))  # Ensure non-negative

    return detection_times


def compute_fab_detection_data(all_data: List[Dict], years: List[float], ai_slowdown_start_year: float, lr_threshold: float = 5.0) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Compute fab-specific detection data for dashboard display.

    Matches reference model's extract_individual_fab_detection_data which:
    1. Uses fab-specific LR (lr_fab_combined) for detection threshold
    2. Reports time as operational time before detection (not time from ai_slowdown_start_year)
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
            if year >= ai_slowdown_start_year and i < len(lr_fab_combined):
                if lr_fab_combined[i] >= lr_threshold:
                    fab_detection_year = year
                    break

        # If no detection, use end of simulation
        if fab_detection_year is None:
            fab_detection_year = sim_years[-1] if sim_years else ai_slowdown_start_year + 7

        # Calculate when fab became operational
        # Fab construction starts at black_project_start_year (black_project_start_year)
        construction_start = bp.get('fab_construction_start_year', bp.get('black_project_start_year', ai_slowdown_start_year))
        construction_duration = bp.get('fab_construction_duration', 1.5)
        operational_start = construction_start + construction_duration

        # Time operational before detection (matching reference model)
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

        # Calculate energy using same formula as reference model
        H100_WATTS_PER_H100E = 700.0
        ENERGY_EFFICIENCY_OF_PRC_STOCK = 0.2
        energy_gw = h100e_at_detection * H100_WATTS_PER_H100E / ENERGY_EFFICIENCY_OF_PRC_STOCK / 1e9

        individual_h100e.append(h100e_at_detection)
        individual_time.append(operational_time)
        individual_process_nodes.append(process_node)
        individual_energy.append(energy_gw)

    return individual_h100e, individual_time, individual_process_nodes, individual_energy


def extract_fab_ccdf_values_at_threshold(fab_built_sims: List[Dict], years: List[float], ai_slowdown_start_year: float, lr_threshold: float) -> Tuple[List[float], List[float]]:
    """
    Extract fab compute and operational time values at detection for CCDF calculation.

    This matches the reference model's extract_fab_compute_at_detection which:
    1. Uses fab-specific LR (lr_fab_combined) for detection threshold
    2. Calculates operational_time = max(0.0, detection_year - operational_start)
    3. Gets cumulative compute at detection year

    Args:
        fab_built_sims: List of simulation data dicts (filtered to sims with fab)
        years: Time points for the simulation
        ai_slowdown_start_year: Year when agreement starts
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
            if year >= ai_slowdown_start_year and i < len(lr_fab_combined):
                if lr_fab_combined[i] >= lr_threshold:
                    fab_detection_year = year
                    break

        # If no detection, use end of simulation
        if fab_detection_year is None:
            fab_detection_year = sim_years[-1] if sim_years else ai_slowdown_start_year + 7

        # Calculate when fab became operational
        # Fab construction starts at black_project_start_year (black_project_start_year)
        construction_start = bp.get('fab_construction_start_year', bp.get('black_project_start_year', ai_slowdown_start_year))
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


def compute_h100_years_before_detection(all_data: List[Dict], years: List[float], ai_slowdown_start_year: float, lr_threshold: float = 5.0) -> List[float]:
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
        detection_year = get_detection_year(d, ai_slowdown_start_year, lr_threshold)
        if detection_year is None:
            detection_year = max(sim_years) if sim_years else ai_slowdown_start_year + 10

        # Sum operational compute from ai_slowdown_start_year to detection_year
        cumulative = 0.0
        for i, year in enumerate(sim_years):
            if year < ai_slowdown_start_year:
                continue
            if year >= detection_year:
                break
            if i < len(operational_compute):
                cumulative += operational_compute[i] * dt  # H100e * years = H100-years

        h100_years.append(cumulative)

    return h100_years


def compute_average_covert_compute(all_data: List[Dict], years: List[float], ai_slowdown_start_year: float, lr_threshold: float) -> List[float]:
    """
    Compute average covert compute from agreement to detection for each simulation.

    This matches the reference implementation which calculates:
    average_compute = h100_years / time_duration
    where time_duration = detection_year - ai_slowdown_start_year
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
        detection_year = get_detection_year(d, ai_slowdown_start_year, lr_threshold)
        if detection_year is None:
            detection_year = max(sim_years) if sim_years else ai_slowdown_start_year + 10

        # Calculate time duration from agreement to detection
        time_duration = detection_year - ai_slowdown_start_year
        if time_duration <= 0:
            average_compute_values.append(0.0)
            continue

        # Sum operational compute from ai_slowdown_start_year to detection_year (H100-years)
        cumulative = 0.0
        for i, year in enumerate(sim_years):
            if year < ai_slowdown_start_year:
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

    Uses datacenter-specific LR (combined: satellite * reported_energy * worker)
    for detection.
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


def compute_h100e_before_detection(all_data: List[Dict], years: List[float], ai_slowdown_start_year: float, lr_threshold: float = 5.0) -> List[float]:
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
        detection_year = get_detection_year(d, ai_slowdown_start_year, lr_threshold)
        if detection_year is None:
            detection_year = max(sim_years) if sim_years else ai_slowdown_start_year + 10

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
