"""
Reduction ratio computations for black project simulation.

Computes chip production and AI R&D reduction ratios comparing
covert project output vs. no-slowdown scenarios.

The "no slowdown" counterfactual is computed using counterfactual nation entities
(PRC_COUNTERFACTUAL_NO_SLOWDOWN, USA_COUNTERFACTUAL_NO_SLOWDOWN) that are simulated
alongside the main nations. These entities represent the "largest AI company" compute
trajectory in each region, initialized using parameter values (total compute * proportion
in largest company) and growing at parameter-specified rates.

To get TOTAL regional compute from these entities, we divide by the proportion_of_compute_in_largest_ai_sw_developer.

For AI R&D comparisons:
- Counterfactual (no slowdown): AI R&D compute = total compute × AI_R_AND_D_FRACTION
  (because some compute goes to external deployment)
- Covert project (slowdown): 100% of compute is AI R&D
  (because external deployment is 0% during slowdown)

Global chip production is computed as the sum of PRC + US chip production from the
counterfactual simulation data (not from a hard-coded CSV).
"""

import numpy as np
from typing import Dict, List


def compute_reduction_ratios(
    all_data: List[Dict],
    years: List[float],
    agreement_year: float,
    dt: float,
    lr_threshold: float,
    model_params=None,
) -> Dict[str, List[float]]:
    """
    Compute reduction ratios for CCDFs.

    Computes ratios comparing counterfactual (no slowdown) compute to covert project compute.
    These ratios show how much less the covert project achieves compared to what companies
    would have computed without any slowdown agreement.

    The ratio is: counterfactual_compute / covert_project_compute
    - Large values mean the covert project got much less than the counterfactual
    - These are inverted (1/x) in black_project_model.py for frontend display

    Args:
        all_data: List of simulation data dicts (including counterfactual nation data)
        years: Time points
        agreement_year: Year when agreement starts
        dt: Time step size
        lr_threshold: LR threshold for detection
        model_params: ModelParameters object containing compute allocation parameters.
            Used to calculate AI R&D fraction (sum of fraction_for_ai_r_and_d_inference +
            fraction_for_ai_r_and_d_training + fraction_for_frontier_training).
            Without slowdown, only this fraction goes to AI R&D; the rest goes to external deployment.

    Returns:
        Dict with ratio lists:
        - 'chip_global': global_chip_production_no_slowdown / covert_chip_production
        - 'chip_prc': total_prc_chip_production_no_slowdown / covert_chip_production
        - 'ai_global': total_global_ai_rd_h100_years_no_slowdown / covert_project_h100_years
          (global = PRC + US, multiplied by ai_rd_fraction since only 76% goes to AI R&D without slowdown)
        - 'ai_prc': total_prc_ai_rd_h100_years_no_slowdown / covert_project_h100_years
          (multiplied by ai_rd_fraction since only 76% goes to AI R&D without slowdown)
    """
    # Calculate AI R&D fraction from model_params
    # This is the fraction of compute used for AI R&D without slowdown
    # (sum of inference, training, and frontier training fractions)
    if model_params is not None:
        compute_allocations = model_params.compute.get('compute_allocations', {})
        ai_rd_fraction_no_slowdown = (
            compute_allocations.get('fraction_for_ai_r_and_d_inference', 0.33) +
            compute_allocations.get('fraction_for_ai_r_and_d_training', 0.33) +
            compute_allocations.get('fraction_for_frontier_training', 0.1)
        )
        # Get proportions to convert from "largest company" to "total regional" compute
        prc_compute = model_params.compute.get('prc_compute', {})
        us_compute = model_params.compute.get('us_compute', {})
        prc_proportion_in_largest = prc_compute.get('proportion_of_compute_in_largest_ai_sw_developer', 0.3)
        us_proportion_in_largest = us_compute.get('proportion_of_compute_in_largest_ai_sw_developer', 0.3)
    else:
        # Fallback if model_params not provided (should not happen in normal usage)
        ai_rd_fraction_no_slowdown = 0.76
        prc_proportion_in_largest = 0.3
        us_proportion_in_largest = 0.3

    # Output ratio lists
    ratio_chip_global = []
    ratio_chip_prc = []
    ratio_ai_global = []
    ratio_ai_prc = []

    for d in all_data:
        bp = d.get('black_project')
        if not bp:
            continue

        # Get counterfactual nation compute stocks
        # These represent "largest company" compute (scaled by proportion_of_compute_in_largest)
        # To get total regional compute, divide by the proportion
        prc_largest_company_compute_stock = d.get('prc_counterfactual_compute_stock', [])
        us_largest_company_compute_stock = d.get('usa_counterfactual_compute_stock', [])

        # Find detection year for this threshold
        cumulative_lr = bp.get('cumulative_lr', [])
        detection_year = years[-1]  # Default to end of simulation
        for j, lr in enumerate(cumulative_lr):
            if lr >= lr_threshold and j < len(years):
                detection_year = years[j]
                break

        # Build years_in_range: years where agreement_year <= year <= detection_year
        years_in_range = [y for y in years if agreement_year <= y <= detection_year]

        # =============================================================================
        # COVERT PROJECT COMPUTE (numerator in final displayed ratio after inversion)
        # =============================================================================

        # Covert project H100-years: integrated operational compute from agreement to detection
        covert_project_operational_compute = bp.get('operational_compute', [])
        covert_project_h100_years = 0.0
        for i in range(len(years_in_range) - 1):
            year = years_in_range[i]
            next_year = years_in_range[i + 1]
            time_step = next_year - year
            year_idx = _find_year_index(year, years)
            if year_idx is not None and year_idx < len(covert_project_operational_compute):
                covert_project_h100_years += covert_project_operational_compute[year_idx] * time_step

        # Covert project chip production: fab production during agreement period
        covert_fab_cumulative_production = bp.get('fab_cumulative_production_h100e', [])
        covert_fab_production_at_detection = 0.0
        covert_fab_production_at_agreement = 0.0
        for j, year in enumerate(years):
            if j < len(covert_fab_cumulative_production):
                if year <= agreement_year:
                    covert_fab_production_at_agreement = covert_fab_cumulative_production[j]
                if year <= detection_year:
                    covert_fab_production_at_detection = covert_fab_cumulative_production[j]
        covert_project_chip_production = max(0.0, covert_fab_production_at_detection - covert_fab_production_at_agreement)

        # =============================================================================
        # COUNTERFACTUAL AI R&D COMPUTE - NO SLOWDOWN SCENARIOS (denominator in final displayed ratio)
        # =============================================================================
        # Note: Without slowdown, only ai_rd_fraction_no_slowdown (76%) of compute goes to AI R&D
        # The rest goes to external deployment
        # The covert project uses 100% for AI R&D since external deployment is 0% during slowdown

        # Total PRC AI R&D H100-years if no slowdown
        # Counterfactual entity is scaled by proportion_of_compute_in_largest, so divide to get total
        # Then multiply by ai_rd_fraction_no_slowdown to get only the AI R&D portion
        total_prc_ai_rd_h100_years_no_slowdown = 0.0
        for i in range(len(years_in_range) - 1):
            year = years_in_range[i]
            next_year = years_in_range[i + 1]
            time_step = next_year - year
            year_idx = _find_year_index(year, years)
            if year_idx is not None and year_idx < len(prc_largest_company_compute_stock):
                largest_company_compute = prc_largest_company_compute_stock[year_idx]
                total_regional_compute = largest_company_compute / prc_proportion_in_largest
                ai_rd_compute = total_regional_compute * ai_rd_fraction_no_slowdown
                total_prc_ai_rd_h100_years_no_slowdown += ai_rd_compute * time_step

        # Total US AI R&D H100-years if no slowdown
        # Counterfactual entity is scaled by proportion_of_compute_in_largest, so divide to get total
        # Then multiply by ai_rd_fraction_no_slowdown to get only the AI R&D portion
        total_us_ai_rd_h100_years_no_slowdown = 0.0
        for i in range(len(years_in_range) - 1):
            year = years_in_range[i]
            next_year = years_in_range[i + 1]
            time_step = next_year - year
            year_idx = _find_year_index(year, years)
            if year_idx is not None and year_idx < len(us_largest_company_compute_stock):
                largest_company_compute = us_largest_company_compute_stock[year_idx]
                total_regional_compute = largest_company_compute / us_proportion_in_largest
                ai_rd_compute = total_regional_compute * ai_rd_fraction_no_slowdown
                total_us_ai_rd_h100_years_no_slowdown += ai_rd_compute * time_step

        # Total global AI R&D H100-years = PRC + US
        total_global_ai_rd_h100_years_no_slowdown = total_prc_ai_rd_h100_years_no_slowdown + total_us_ai_rd_h100_years_no_slowdown

        # Total PRC chip production if no slowdown (stock increase during period)
        # Divide by proportion to get total PRC, not just largest company
        prc_largest_stock_at_agreement = 0.0
        prc_largest_stock_at_detection = 0.0
        for j, year in enumerate(years):
            if j < len(prc_largest_company_compute_stock):
                if year <= agreement_year:
                    prc_largest_stock_at_agreement = prc_largest_company_compute_stock[j]
                if year <= detection_year:
                    prc_largest_stock_at_detection = prc_largest_company_compute_stock[j]
        prc_largest_chip_production = prc_largest_stock_at_detection - prc_largest_stock_at_agreement
        total_prc_chip_production_no_slowdown = prc_largest_chip_production / prc_proportion_in_largest

        # Total US chip production if no slowdown (stock increase during period)
        # Divide by proportion to get total US, not just largest company
        us_largest_stock_at_agreement = 0.0
        us_largest_stock_at_detection = 0.0
        for j, year in enumerate(years):
            if j < len(us_largest_company_compute_stock):
                if year <= agreement_year:
                    us_largest_stock_at_agreement = us_largest_company_compute_stock[j]
                if year <= detection_year:
                    us_largest_stock_at_detection = us_largest_company_compute_stock[j]
        us_largest_chip_production = us_largest_stock_at_detection - us_largest_stock_at_agreement
        total_us_chip_production_no_slowdown = us_largest_chip_production / us_proportion_in_largest

        # Global chip production = PRC + US chip production from counterfactual simulation
        global_chip_production_no_slowdown = total_prc_chip_production_no_slowdown + total_us_chip_production_no_slowdown

        # =============================================================================
        # COMPUTE RATIOS: counterfactual_no_slowdown / covert_project
        # =============================================================================
        # Large values mean covert project got much less than counterfactual would have
        # These are inverted (1/x) in black_project_model.py for frontend display

        LARGE_RATIO = 1e12  # Used when covert compute is zero

        # Chip production ratios
        if covert_project_chip_production <= 0:
            ratio_chip_global.append(LARGE_RATIO)
            ratio_chip_prc.append(LARGE_RATIO)
        else:
            ratio_chip_global.append(
                global_chip_production_no_slowdown / covert_project_chip_production
                if global_chip_production_no_slowdown > 0 else 0.0
            )
            ratio_chip_prc.append(
                total_prc_chip_production_no_slowdown / covert_project_chip_production
                if total_prc_chip_production_no_slowdown > 0 else 0.0
            )

        # AI R&D compute ratios
        # Numerator: counterfactual AI R&D compute (no slowdown, only 76% goes to AI R&D)
        # Denominator: covert project compute (100% is AI R&D since external deployment = 0 during slowdown)
        if covert_project_h100_years <= 0:
            ratio_ai_global.append(LARGE_RATIO)
            ratio_ai_prc.append(LARGE_RATIO)
        else:
            ratio_ai_global.append(
                total_global_ai_rd_h100_years_no_slowdown / covert_project_h100_years
                if total_global_ai_rd_h100_years_no_slowdown > 0 else 0.0
            )
            ratio_ai_prc.append(
                total_prc_ai_rd_h100_years_no_slowdown / covert_project_h100_years
                if total_prc_ai_rd_h100_years_no_slowdown > 0 else 0.0
            )

    return {
        'chip_global': ratio_chip_global,
        'chip_prc': ratio_chip_prc,
        'ai_global': ratio_ai_global,
        'ai_prc': ratio_ai_prc,
    }


def _find_year_index(target_year: float, years: List[float]) -> int | None:
    """Find the index of a year in the years list (with floating point tolerance)."""
    for j, y in enumerate(years):
        if abs(y - target_year) < 1e-6:
            return j
    return None
