"""
Reduction ratio computations for black project simulation.

Computes chip production and AI R&D reduction ratios comparing
covert project output vs. no-slowdown scenarios.
"""

import numpy as np
from typing import Dict, List

from .global_compute import get_global_compute_production_between_years


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
    largest_company_fraction = 0.3
    SMALL_RATIO = 1e-10

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
        if bp_chip_production <= 0:
            chip_global.append(SMALL_RATIO)
            chip_prc.append(SMALL_RATIO)
        else:
            chip_global.append(bp_chip_production / global_production if global_production > 0 else SMALL_RATIO)
            chip_prc.append(bp_chip_production / prc_production if prc_production > 0 else SMALL_RATIO)

        if bp_h100_years <= 0:
            ai_largest.append(SMALL_RATIO)
            ai_prc.append(SMALL_RATIO)
        else:
            ai_largest.append(bp_h100_years / largest_h100_years if largest_h100_years > 0 else SMALL_RATIO)
            ai_prc.append(bp_h100_years / prc_h100_years if prc_h100_years > 0 else SMALL_RATIO)

    return {
        'chip_global': chip_global,
        'chip_prc': chip_prc,
        'chip_largest': chip_prc,  # Same as PRC for chip production
        'ai_largest': ai_largest,
        'ai_prc': ai_prc,
    }
