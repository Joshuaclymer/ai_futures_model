"""
Reduction ratio computations for black project simulation.

Computes chip production and AI R&D reduction ratios comparing
covert project output vs. no-slowdown scenarios.

The "no slowdown" counterfactual is computed using counterfactual nation entities
(PRC_COUNTERFACTUAL_NO_SLOWDOWN, USA_COUNTERFACTUAL_NO_SLOWDOWN) that are simulated
alongside the main nations but with exponential growth at fixed rates. This matches
the reference model's approach of using fixed baseline parameters.
"""

import numpy as np
from typing import Dict, List

from .global_compute import get_global_compute_production_between_years


# Fraction of PRC compute spent on AI R&D (from reference model SlowdownCounterfactualParameters)
FRACTION_OF_PRC_COMPUTE_SPENT_ON_AI_RD = 0.5


def compute_reduction_ratios(
    all_data: List[Dict],
    years: List[float],
    agreement_year: float,
    dt: float,
    lr_threshold: float
) -> Dict[str, List[float]]:
    """
    Compute reduction ratios for CCDFs.

    Computes ratios of counterfactual AI compute (no slowdown scenario) to
    covert project AI compute, using discrete time-step integration matching
    the reference model.

    The counterfactual is computed using counterfactual nation entities that
    track the no-slowdown compute trajectory.

    Args:
        all_data: List of simulation data dicts (including counterfactual nation data)
        years: Time points
        agreement_year: Year when agreement starts
        dt: Time step size
        lr_threshold: LR threshold for detection

    Returns:
        Dict with 'chip_global', 'chip_prc', 'chip_largest', 'ai_largest', 'ai_prc' keys
    """
    chip_global = []
    chip_prc = []
    ai_largest = []
    ai_prc = []

    for d in all_data:
        bp = d.get('black_project')
        if not bp:
            continue

        # Get counterfactual nation data
        prc_cf_stock = d.get('prc_counterfactual_compute_stock', [])
        usa_cf_stock = d.get('usa_counterfactual_compute_stock', [])

        # Find detection year for this threshold
        cumulative_lr = bp.get('cumulative_lr', [])
        detection_year = years[-1]  # Default to end of simulation
        for j, lr in enumerate(cumulative_lr):
            if lr >= lr_threshold and j < len(years):
                detection_year = years[j]
                break

        # Build years_in_range: years where agreement_year <= year <= detection_year
        # This matches reference model's filtering
        years_in_range = [y for y in years if agreement_year <= y <= detection_year]

        # Compute covert project H100-years from agreement_year to detection_year
        # Using discrete time-step integration to match reference model
        op_compute = bp.get('operational_compute', [])
        bp_h100_years = 0.0
        for i in range(len(years_in_range) - 1):
            year = years_in_range[i]
            next_year = years_in_range[i + 1]
            time_step = next_year - year
            # Find the index of this year in the full years list
            year_idx = None
            for j, y in enumerate(years):
                if abs(y - year) < 1e-6:
                    year_idx = j
                    break
            if year_idx is not None and year_idx < len(op_compute):
                bp_h100_years += op_compute[year_idx] * time_step

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

        # Compute counterfactual PRC AI R&D H100-years (no slowdown scenario)
        # Uses data from PRC_COUNTERFACTUAL_NO_SLOWDOWN entity
        # Reference: prc_h100_years += prc_compute * prc_ai_rd_fraction * time_step
        prc_h100_years = 0.0
        for i in range(len(years_in_range) - 1):
            year = years_in_range[i]
            next_year = years_in_range[i + 1]
            time_step = next_year - year
            # Find the index of this year in the full years list
            year_idx = None
            for j, y in enumerate(years):
                if abs(y - year) < 1e-6:
                    year_idx = j
                    break
            if year_idx is not None and year_idx < len(prc_cf_stock):
                prc_compute = prc_cf_stock[year_idx]
                prc_h100_years += prc_compute * FRACTION_OF_PRC_COMPUTE_SPENT_ON_AI_RD * time_step

        # Compute counterfactual largest AI project H100-years (no slowdown scenario)
        # Uses data from USA_COUNTERFACTUAL_NO_SLOWDOWN entity
        largest_h100_years = 0.0
        for i in range(len(years_in_range) - 1):
            year = years_in_range[i]
            next_year = years_in_range[i + 1]
            time_step = next_year - year
            # Find the index of this year in the full years list
            year_idx = None
            for j, y in enumerate(years):
                if abs(y - year) < 1e-6:
                    year_idx = j
                    break
            if year_idx is not None and year_idx < len(usa_cf_stock):
                largest_h100_years += usa_cf_stock[year_idx] * time_step

        # Compute counterfactual chip production (no slowdown scenario)
        # Global: from historical data
        global_production = get_global_compute_production_between_years(agreement_year, detection_year)

        # PRC chip production: stock at detection - stock at agreement (from counterfactual entity)
        prc_stock_at_agreement = 0.0
        prc_stock_at_detection = 0.0
        for j, year in enumerate(years):
            if j < len(prc_cf_stock):
                if year <= agreement_year:
                    prc_stock_at_agreement = prc_cf_stock[j]
                if year <= detection_year:
                    prc_stock_at_detection = prc_cf_stock[j]
        prc_production = prc_stock_at_detection - prc_stock_at_agreement

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
