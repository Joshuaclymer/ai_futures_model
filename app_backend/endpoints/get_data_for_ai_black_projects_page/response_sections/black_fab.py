"""
Build the black_fab section of the API response.

Contains covert fab metrics, production data, and process node information.
"""

from typing import Dict, List, Any

from ..percentile_helpers import get_fab_percentiles_with_individual, compute_ccdf
from ..detection import extract_fab_ccdf_values_at_threshold
from ..visualization import build_watts_per_tpp_curve, compute_fab_dashboard

# Constants
LIKELIHOOD_RATIO_THRESHOLDS = [1, 2, 4]


def build_black_fab_section(
    all_data: List[Dict],
    fab_built_sims: List[Dict],
    years: List[float],
    dt: float,
    ai_slowdown_start_year: float,
    detection_times: List[float],
    num_sims: int,
    fab_individual_h100e: List[float],
    fab_individual_time: List[float],
    fab_individual_process_nodes: List[str],
    fab_individual_energy: List[float],
) -> Dict[str, Any]:
    """
    Build the black_fab section of the response.

    This section contains 25 keys including fab metrics, LR values, and CCDFs.
    """
    return {
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
            fab_built_sims,
            lambda d: [d['black_project']['fab_wafer_starts_per_month']] * len(years) if d['black_project'] else [0.0] * len(years)
        ),
        "chips_per_wafer": get_fab_percentiles_with_individual(
            fab_built_sims,
            lambda d: [d['black_project']['fab_chips_per_wafer']] * len(years) if d['black_project'] else [28.0] * len(years)
        ),
        "architecture_efficiency": get_fab_percentiles_with_individual(
            fab_built_sims,
            lambda d: [d['black_project']['fab_architecture_efficiency']] * len(years) if d['black_project'] else [1.0] * len(years)
        ),
        "architecture_efficiency_at_agreement": fab_built_sims[0]['black_project']['fab_architecture_efficiency'] if fab_built_sims and fab_built_sims[0]['black_project'] else 1.0,
        "compute_per_wafer_2022_arch": get_fab_percentiles_with_individual(
            fab_built_sims,
            lambda d: [d['black_project']['fab_transistor_density_relative_to_h100'] * d['black_project']['fab_chips_per_wafer']] * len(years)
            if d['black_project'] else [1.0] * len(years)
        ),
        "transistor_density": get_fab_percentiles_with_individual(
            fab_built_sims,
            lambda d: [d['black_project']['fab_transistor_density_relative_to_h100']] * len(years) if d['black_project'] else [0.1] * len(years)
        ),
        "watts_per_tpp": get_fab_percentiles_with_individual(
            fab_built_sims,
            lambda d: [d['black_project']['fab_watts_per_h100e'] / 700.0] * len(years) if d['black_project'] else [1.0] * len(years)
        ),
        "watts_per_tpp_curve": build_watts_per_tpp_curve(),
        # LR values: Use FAB-SPECIFIC LR metrics
        "lr_combined": get_fab_percentiles_with_individual(
            fab_built_sims,
            lambda d: d['black_project']['lr_fab_combined'] if d['black_project'] else [1.0] * len(years)
        ),
        "lr_inventory": get_fab_percentiles_with_individual(
            fab_built_sims,
            lambda d: [d['black_project']['lr_sme_inventory']] * len(years) if d['black_project'] else [1.0] * len(years)
        ),
        "lr_procurement": get_fab_percentiles_with_individual(
            fab_built_sims,
            lambda d: [d['black_project']['lr_fab_procurement']] * len(years) if d['black_project'] else [1.0] * len(years)
        ),
        "lr_other": get_fab_percentiles_with_individual(
            fab_built_sims,
            lambda d: d['black_project']['lr_fab_other'] if d['black_project'] else [1.0] * len(years)
        ),
        "process_node_by_sim": [
            f"{int(d['black_project'].get('fab_process_node_nm', 28))}nm" if d['black_project'] else "28nm"
            for d in fab_built_sims
        ] if fab_built_sims else [],
        "individual_process_node": fab_individual_process_nodes,
        # Fab detection data uses fab-specific LR and operational time
        "individual_h100e_before_detection": fab_individual_h100e,
        "individual_time_before_detection": fab_individual_time,
        "individual_energy_before_detection": fab_individual_energy,
        "compute_ccdf": [],
        # Use extract_fab_ccdf_values_at_threshold for values at detection
        "compute_ccdfs": {
            str(lr): compute_ccdf(extract_fab_ccdf_values_at_threshold(fab_built_sims, years, ai_slowdown_start_year, lr)[0])
            for lr in LIKELIHOOD_RATIO_THRESHOLDS
        },
        "op_time_ccdf": [],
        "op_time_ccdfs": {
            str(lr): compute_ccdf(extract_fab_ccdf_values_at_threshold(fab_built_sims, years, ai_slowdown_start_year, lr)[1])
            for lr in LIKELIHOOD_RATIO_THRESHOLDS
        },
        "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,

        # Pre-computed dashboard values
        "dashboard": compute_fab_dashboard(
            fab_built_sims, all_data, detection_times, num_sims, dt,
            fab_individual_h100e=fab_individual_h100e,
            fab_individual_time=fab_individual_time,
            fab_individual_energy=fab_individual_energy
        ),
    }
