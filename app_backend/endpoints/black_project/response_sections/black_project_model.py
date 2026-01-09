"""
Build the black_project_model section of the API response.

Contains main metrics, LR values, and CCDFs for the overall black project.
"""

from typing import Dict, List, Any

from ..percentile_helpers import (
    get_percentiles_with_individual,
    compute_ccdf,
)
from ..reduction_ratios import compute_reduction_ratios
from ..detection import (
    compute_detection_times,
    compute_h100_years_before_detection,
    compute_average_covert_compute,
)

# Constants
LIKELIHOOD_RATIO_THRESHOLDS = [1, 2, 4]


def build_black_project_model_section(
    all_data: List[Dict],
    years: List[float],
    dt: float,
    agreement_year: float,
    energy_by_source: List[List[float]],
    source_labels: List[str],
    detection_times: List[float],
    h100_years_before_detection: List[float],
    h100e_before_detection: List[float],
    energy_before_detection: List[float],
    model_params=None,
) -> Dict[str, Any]:
    """
    Build the black_project_model section of the response.

    This section contains 34 keys including time series, LR components, and CCDFs.

    Args:
        model_params: ModelParameters object containing compute allocation parameters.
            Used to calculate AI R&D fraction for reduction ratio comparisons.
    """
    return {
        "years": years,

        # Time series with individual data
        "h100_years": get_percentiles_with_individual(
            all_data,
            lambda d: [sum((d['black_project']['operational_compute'] if d['black_project'] else [0])[:i+1]) * dt for i in range(len(years))]
        ),
        "cumulative_lr": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['cumulative_lr'] if d['black_project'] else [1.0] * len(years)
        ),
        # initial_black_project: Reference model uses total surviving compute (initial + fab)
        "initial_black_project": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['total_compute'] if d['black_project'] else [0.0] * len(years)
        ),
        # black_fab_flow: FILTERED to only fab-built simulations (cumulative)
        "black_fab_flow": _get_fab_flow_percentiles(all_data, years),
        # black_fab_flow_all_sims: ALL simulations - CUMULATIVE production (for Dark Compute Stock Breakdown)
        "black_fab_flow_all_sims": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['fab_cumulative_production_h100e'] if d['black_project'] else [0.0] * len(years)
        ),
        # black_fab_monthly_flow_all_sims: ALL simulations - MONTHLY production rate
        "black_fab_monthly_flow_all_sims": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['fab_monthly_production_h100e'] if d['black_project'] else [0.0] * len(years)
        ),
        "survival_rate": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['survival_rate'] if d['black_project'] else [1.0] * len(years)
        ),
        "covert_chip_stock": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['total_compute'] if d['black_project'] else [0.0] * len(years)
        ),
        "total_black_project": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['total_compute'] if d['black_project'] else [0.0] * len(years)
        ),
        "datacenter_capacity": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['datacenter_capacity_gw'] if d['black_project'] else [0.0] * len(years)
        ),
        "operational_compute": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['operational_compute'] if d['black_project'] else [0.0] * len(years)
        ),

        # Energy data
        "black_project_energy": energy_by_source,
        "energy_source_labels": source_labels,

        # LR components
        "lr_initial_stock": get_percentiles_with_individual(
            all_data,
            lambda d: [d['black_project']['lr_prc_accounting']] * len(years) if d['black_project'] else [1.0] * len(years)
        ),
        "lr_diverted_sme": get_percentiles_with_individual(
            all_data,
            lambda d: [d['black_project']['lr_sme_inventory']] * len(years) if d['black_project'] else [1.0] * len(years)
        ),
        "lr_prc_accounting": get_percentiles_with_individual(
            all_data,
            lambda d: [d['black_project']['lr_prc_accounting']] * len(years) if d['black_project'] else [1.0] * len(years)
        ),
        "lr_sme_inventory": get_percentiles_with_individual(
            all_data,
            lambda d: [d['black_project']['lr_sme_inventory']] * len(years) if d['black_project'] else [1.0] * len(years)
        ),
        "lr_satellite_datacenter": {
            "individual": [d['black_project']['lr_satellite_datacenter'] if d['black_project'] else 1.0 for d in all_data]
        },
        "lr_other_intel": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['lr_other_intel'] if d['black_project'] else [1.0] * len(years)
        ),
        "lr_reported_energy": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['lr_reported_energy'] if d['black_project'] else [1.0] * len(years)
        ),
        "lr_combined_reported_assets": get_percentiles_with_individual(
            all_data,
            lambda d: [
                d['black_project']['lr_prc_accounting'] * d['black_project']['lr_sme_inventory'] *
                d['black_project']['lr_satellite_datacenter'] * (d['black_project']['lr_reported_energy'][i] if i < len(d['black_project'].get('lr_reported_energy', [])) else 1.0)
                if d['black_project'] else 1.0
                for i in range(len(years))
            ]
        ),
        "posterior_prob_project": get_percentiles_with_individual(
            all_data,
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

        # CCDFs - compute for each threshold separately
        "time_to_detection_ccdf": {
            str(lr): compute_ccdf(compute_detection_times(all_data, years, agreement_year, lr))
            for lr in LIKELIHOOD_RATIO_THRESHOLDS
        },
        "h100_years_ccdf": {
            str(lr): compute_ccdf(compute_h100_years_before_detection(all_data, years, agreement_year, lr))
            for lr in LIKELIHOOD_RATIO_THRESHOLDS
        },
        "average_covert_compute_ccdf": {
            str(lr): compute_ccdf(compute_average_covert_compute(all_data, years, agreement_year, lr))
            for lr in LIKELIHOOD_RATIO_THRESHOLDS
        },
        # Chip production reduction CCDFs - nested by threshold for reference model compatibility
        "chip_production_reduction_ccdf": {
            "global": {
                str(lr): compute_ccdf(compute_reduction_ratios(all_data, years, agreement_year, dt, lr, model_params)['chip_global'])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "prc": {
                str(lr): compute_ccdf(compute_reduction_ratios(all_data, years, agreement_year, dt, lr, model_params)['chip_prc'])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "largest_company": {
                str(lr): compute_ccdf(compute_reduction_ratios(all_data, years, agreement_year, dt, lr, model_params)['chip_prc'])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
        },
        # AI R&D reduction CCDFs - nested by threshold for reference model compatibility
        # Frontend chart shows global vs prc (two entities) at default threshold 4
        "ai_rd_reduction_ccdf": {
            "global": {
                str(lr): compute_ccdf(compute_reduction_ratios(all_data, years, agreement_year, dt, lr, model_params)['ai_global'])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
            "prc": {
                str(lr): compute_ccdf(compute_reduction_ratios(all_data, years, agreement_year, dt, lr, model_params)['ai_prc'])
                for lr in LIKELIHOOD_RATIO_THRESHOLDS
            },
        },
        # Flat versions for frontend charts (uses threshold 4)
        # Frontend expects covert/counterfactual fractions, so invert the ratios (1/x)
        "ai_rd_reduction_ccdf_flat": _compute_inverted_ccdf_flat(all_data, years, agreement_year, dt, ['ai_global', 'ai_prc'], ['global', 'prc'], model_params),
        "chip_production_reduction_ccdf_flat": _compute_inverted_ccdf_flat(all_data, years, agreement_year, dt, ['chip_global', 'chip_prc'], ['global', 'prc'], model_params),

        "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,
    }


def _compute_inverted_ccdf_flat(
    all_data: List[Dict],
    years: List[float],
    agreement_year: float,
    dt: float,
    ratio_keys: List[str],
    output_keys: List[str],
    model_params=None,
) -> Dict[str, Any]:
    """
    Compute CCDF with inverted ratios for frontend display.

    The reference model uses counterfactual/covert ratios (large numbers).
    Frontend expects covert/counterfactual fractions (small numbers like 0.001).
    This function inverts the ratios (1/x) before computing the CCDF.
    """
    ratios = compute_reduction_ratios(all_data, years, agreement_year, dt, 4, model_params)
    result = {}
    for ratio_key, output_key in zip(ratio_keys, output_keys):
        # Invert ratios: counterfactual/covert -> covert/counterfactual
        inverted = [1.0 / r if r > 0 else 0.0 for r in ratios[ratio_key]]
        result[output_key] = compute_ccdf(inverted)
    return result


def _get_fab_flow_percentiles(all_data: List[Dict], years: List[float]) -> Dict[str, Any]:
    """Get fab flow percentiles filtered to only fab-built simulations."""
    from ..percentile_helpers import get_fab_percentiles_with_individual
    fab_built_sims = [
        d for d in all_data
        if d['black_project'] and any(d['black_project'].get('fab_is_operational', []))
    ]
    return get_fab_percentiles_with_individual(
        fab_built_sims,
        lambda d: d['black_project']['fab_cumulative_production_h100e'] if d['black_project'] else [0.0] * len(years)
    )
