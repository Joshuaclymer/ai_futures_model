"""
Reduction ratio computations for black project simulation.

Computes chip production and AI R&D reduction ratios comparing
covert project output vs. no-slowdown scenarios.
"""

import numpy as np
from typing import Dict, List

from .global_compute import get_global_compute_production_between_years


# Largest AI project parameters (from reference model ExogenousTrends)
# These represent the compute trajectory of the largest global AI project (e.g., OpenAI/Google)
# Used for comparing covert project compute to what the largest company would achieve without slowdown
LARGEST_AI_PROJECT_COMPUTE_STOCK_IN_2025 = 1.2e5  # H100e
ANNUAL_GROWTH_RATE_OF_LARGEST_AI_PROJECT_COMPUTE_STOCK = 2.91  # multiplier per year


def get_largest_ai_project_compute_stock(year: float) -> float:
    """
    Get the compute stock of the largest AI project at a given year.

    Compute grows exponentially from the 2025 baseline.

    Args:
        year: The year to calculate compute for

    Returns:
        Compute stock in H100 equivalents
    """
    years_since_2025 = year - 2025
    return LARGEST_AI_PROJECT_COMPUTE_STOCK_IN_2025 * (
        ANNUAL_GROWTH_RATE_OF_LARGEST_AI_PROJECT_COMPUTE_STOCK ** years_since_2025
    )


def compute_reduction_ratios(
    all_data: List[Dict],
    years: List[float],
    agreement_year: float,
    dt: float,
    lr_threshold: float
) -> Dict[str, List[float]]:
    """
    Compute reduction ratios for CCDFs.

    Computes ratios of covert project chip production and AI compute
    relative to what would have been produced without slowdown.

    Args:
        all_data: List of simulation data dicts
        years: Time points
        agreement_year: Year when agreement starts
        dt: Time step size
        lr_threshold: LR threshold for detection

    Returns:
        Dict with 'chip_global', 'chip_prc', 'chip_largest', 'ai_largest', 'ai_prc' keys
    """
    pre_slowdown_growth_rate = 2.2

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

        # Compute covert project H100-years from agreement_year to detection_year
        # (matching reference model which filters: agreement_year <= y <= current_year)
        op_compute = bp.get('operational_compute', [])
        bp_h100_years = 0.0
        for j, year in enumerate(years):
            if year < agreement_year:
                continue  # Skip years before agreement
            if year >= detection_year:
                break
            if j < len(op_compute):
                bp_h100_years += op_compute[j] * dt

        # Compute covert chip production (fab production during agreement period)
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

        # Get PRC compute at agreement year
        # Apply fraction of PRC compute spent on AI R&D (from reference model)
        # This represents the portion of PRC compute used for AI research, not total PRC compute
        FRACTION_OF_PRC_COMPUTE_SPENT_ON_AI_RD = 0.5
        agreement_idx = 0
        for j, year in enumerate(years):
            if year >= agreement_year:
                agreement_idx = j
                break
        total_prc_at_agreement = prc_stock[agreement_idx] if agreement_idx < len(prc_stock) else prc_stock[0]
        prc_at_agreement = total_prc_at_agreement * FRACTION_OF_PRC_COMPUTE_SPENT_ON_AI_RD
        duration = max(0.0, detection_year - agreement_year)

        # Compute counterfactual chip production (no slowdown scenario)
        global_production = get_global_compute_production_between_years(agreement_year, detection_year)
        prc_production = prc_at_agreement * (pre_slowdown_growth_rate ** duration - 1)

        # Largest AI project chip production = stock(detection) - stock(agreement)
        largest_stock_at_agreement = get_largest_ai_project_compute_stock(agreement_year)
        largest_stock_at_detection = get_largest_ai_project_compute_stock(detection_year)
        largest_production = largest_stock_at_detection - largest_stock_at_agreement

        # Compute counterfactual H100-years (integrated compute over time)
        # For PRC: integrate prc_at_agreement * growth_rate^t from 0 to duration
        pre_slowdown_growth_rate = 2.2
        if pre_slowdown_growth_rate > 1.001 and duration > 0:
            growth_factor = (pre_slowdown_growth_rate ** duration - 1) / np.log(pre_slowdown_growth_rate)
        else:
            growth_factor = duration
        prc_h100_years = prc_at_agreement * growth_factor

        # For largest AI project: integrate compute stock over time from agreement to detection
        largest_growth_rate = ANNUAL_GROWTH_RATE_OF_LARGEST_AI_PROJECT_COMPUTE_STOCK
        if largest_growth_rate > 1.001 and duration > 0:
            largest_growth_factor = (largest_growth_rate ** duration - 1) / np.log(largest_growth_rate)
        else:
            largest_growth_factor = duration
        largest_h100_years = largest_stock_at_agreement * largest_growth_factor

        # Compute ratios as counterfactual / covert (matching reference model format)
        # Large values mean covert project got much less than counterfactual would have
        # For frontend display, these are inverted to fractions in black_project_model.py
        LARGE_RATIO = 1e12  # Used when covert compute is zero

        if bp_chip_production <= 0:
            chip_global.append(LARGE_RATIO)
            chip_prc.append(LARGE_RATIO)
        else:
            chip_global.append(global_production / bp_chip_production if global_production > 0 else 0.0)
            chip_prc.append(prc_production / bp_chip_production if prc_production > 0 else 0.0)

        if bp_h100_years <= 0:
            ai_largest.append(LARGE_RATIO)
            ai_prc.append(LARGE_RATIO)
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
