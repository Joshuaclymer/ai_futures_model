"""
Build the initial_black_project and initial_stock sections of the API response.

Contains initial compute stock, diversion data, and PRC compute trajectories.
"""

from typing import Dict, List, Any

from ..percentile_helpers import (
    get_percentiles_with_individual,
    get_fab_percentiles_with_individual,
)


def build_initial_black_project_section(
    all_data: List[Dict],
    fab_built_sims: List[Dict],
    years: List[float],
) -> Dict[str, Any]:
    """
    Build the initial_black_project section of the response.

    This section contains 4 keys.
    """
    return {
        "years": years,
        # Total dark compute (surviving initial + fab), in thousands
        "black_project": get_percentiles_with_individual(
            all_data,
            lambda d: [v / 1000 for v in d['black_project']['total_compute']] if d['black_project'] else [0.0] * len(years)
        ),
        # Fab production only, filtered to fab-built sims, in thousands
        "h100e": get_fab_percentiles_with_individual(
            fab_built_sims,
            lambda d: [v / 1000 for v in d['black_project']['fab_cumulative_production_h100e']] if d['black_project'] else [0.0] * len(years)
        ),
        "survival_rate": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['survival_rate'] if d['black_project'] else [1.0] * len(years)
        ),
    }


def build_initial_stock_section(
    all_data: List[Dict],
    years: List[float],
    detection_times: List[float],
    num_sims: int,
    agreement_year: float,
    prc_capacity_years: List[int],
) -> Dict[str, Any]:
    """
    Build the initial_stock section of the response.

    This section contains 13 keys including initial compute samples and PRC projections.
    """
    # Extract lr_prc_accounting samples for detection probability calculation
    lr_prc_accounting_samples = [
        d['black_project']['lr_prc_accounting'] if d['black_project'] else 1.0
        for d in all_data
    ]

    return {
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
        "initial_energy_samples": [
            (d['black_project']['initial_diverted_compute_h100e'] * 0.7) /
            ((1.26 ** (agreement_year - 2022)) * 0.2 * 1e6)
            if d['black_project'] and d['black_project'].get('initial_diverted_compute_h100e')
            else 0
            for d in all_data
        ],
        "lr_prc_accounting_samples": lr_prc_accounting_samples,
        "lr_sme_inventory_samples": [
            d['black_project']['lr_sme_inventory'] if d['black_project'] else 1.0
            for d in all_data
        ],
        "lr_satellite_datacenter_samples": [
            d['black_project']['lr_satellite_datacenter'] if d['black_project'] else 1.0
            for d in all_data
        ],
        # Detection probabilities based on lr_prc_accounting likelihood ratio thresholds
        "initial_black_project_detection_probs": {
            "1x": sum(1 for lr in lr_prc_accounting_samples if lr >= 1) / max(1, num_sims),
            "2x": sum(1 for lr in lr_prc_accounting_samples if lr >= 2) / max(1, num_sims),
            "4x": sum(1 for lr in lr_prc_accounting_samples if lr >= 4) / max(1, num_sims),
        },
        "prc_compute_years": prc_capacity_years,
        # Compute PRC stock for each year using sampled growth rate
        "prc_compute_over_time": get_percentiles_with_individual(
            all_data,
            lambda d: [
                d['prc_params']['total_prc_compute_tpp_h100e_in_2025'] * (d['prc_params']['annual_growth_rate'] ** (year - 2025))
                if d.get('prc_params') else 100000.0 * (2.2 ** (year - 2025))
                for year in prc_capacity_years
            ]
        ),
        # Compute domestic production proportion for each year
        "prc_domestic_compute_over_time": get_percentiles_with_individual(
            all_data,
            lambda d: [
                (d['prc_params']['total_prc_compute_tpp_h100e_in_2025'] * (d['prc_params']['annual_growth_rate'] ** (year - 2025))
                 if d.get('prc_params') else 100000.0 * (2.2 ** (year - 2025)))
                * (0.0 if year < 2027 else 0.175 * (year - 2026) if year <= int(agreement_year) else 0.7)
                for year in prc_capacity_years
            ]
        ),
        # Proportion domestic by year
        "proportion_domestic_by_year": [
            0.0 if year < 2027 else 0.175 * (year - 2026) if year <= int(agreement_year) else 0.7
            for year in prc_capacity_years
        ],
        # Largest company compute
        "largest_company_compute_over_time": [
            120000.0 * (2.91 ** (year - 2025))
            for year in prc_capacity_years
        ],
        "state_of_the_art_energy_efficiency_relative_to_h100": 1.26 ** (agreement_year - 2022),
    }
