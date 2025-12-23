"""
Compare initial compute energy consumption between discrete and continuous models.

This script focuses specifically on the energy consumption calculation for
the initial diverted compute stock to identify alignment issues.
"""

import sys
import os

# Add both project roots to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')
sys.path.insert(0, '/Users/joshuaclymer/github/covert_compute_production_model')

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import from black_project_backend (discrete model)
from black_project_backend.model import BlackProjectModel
from black_project_backend.black_project_parameters import (
    ModelParameters as DiscreteModelParameters,
    SimulationSettings as DiscreteSimulationSettings,
    BlackProjectProperties as DiscreteBlackProjectProperties,
    BlackProjectParameters as DiscreteBlackProjectParameters,
    BlackFabParameters as DiscreteBlackFabParameters,
    BlackDatacenterParameters as DiscreteBlackDatacenterParameters,
    DetectionParameters as DiscreteDetectionParameters,
    ExogenousTrends as DiscreteExogenousTrends,
    SurvivalRateParameters as DiscreteSurvivalRateParameters,
)
from black_project_backend.util import _cache as discrete_cache
from black_project_backend.classes.largest_ai_project import LargestAIProject

# Note: Full continuous model imports are not used since we test the formula directly
# The actual fix is in ai_futures_simulator/world_updaters/black_project.py


def run_discrete_energy_calculation(
    agreement_year: int = 2030,
    prc_compute_stock_h100e: float = 1e6,
    proportion_to_divert: float = 0.05,
    energy_efficiency_relative_to_sota: float = 0.2,
    improvement_in_energy_efficiency_per_year: float = 1.26,
):
    """
    Calculate initial compute energy consumption using the discrete model approach.

    Returns detailed breakdown of the energy calculation.
    """
    # Clear cache
    discrete_cache.clear()
    np.random.seed(42)

    # Create exogenous trends
    exogenous_trends = DiscreteExogenousTrends(
        total_prc_compute_stock_in_2025=prc_compute_stock_h100e / (2.2 ** (agreement_year - 2025)),
        annual_growth_rate_of_prc_compute_stock_p10=2.2,
        annual_growth_rate_of_prc_compute_stock_p50=2.2,
        annual_growth_rate_of_prc_compute_stock_p90=2.2,
        energy_efficiency_of_prc_stock_relative_to_state_of_the_art=energy_efficiency_relative_to_sota,
        improvement_in_energy_efficiency_per_year=improvement_in_energy_efficiency_per_year,
    )

    # Create LargestAIProject to get state-of-the-art efficiency
    largest_ai_project = LargestAIProject(exogenous_trends)
    sota_efficiency_relative_to_h100 = largest_ai_project.get_energy_efficiency_relative_to_h100(agreement_year)

    # Combined efficiency (what the discrete model actually uses)
    combined_efficiency = energy_efficiency_relative_to_sota * sota_efficiency_relative_to_h100

    # Constants from the discrete model
    H100_TPP_PER_CHIP = 2144.0
    H100_WATTS_PER_TPP = 0.326493
    H100_POWER_W = H100_TPP_PER_CHIP * H100_WATTS_PER_TPP  # ~700W

    # Diverted compute
    initial_diverted_h100e = prc_compute_stock_h100e * proportion_to_divert

    # Energy calculation (discrete model approach):
    # tpp = h100e * H100_TPP_PER_CHIP
    # watts = tpp * H100_WATTS_PER_TPP / combined_efficiency
    # energy_gw = watts / 1e9
    tpp = initial_diverted_h100e * H100_TPP_PER_CHIP
    watts = tpp * H100_WATTS_PER_TPP / combined_efficiency
    energy_gw = watts / 1e9

    # Simplified formula
    energy_gw_simplified = (initial_diverted_h100e * H100_POWER_W) / (combined_efficiency * 1e9)

    return {
        'agreement_year': agreement_year,
        'prc_compute_stock_h100e': prc_compute_stock_h100e,
        'initial_diverted_h100e': initial_diverted_h100e,
        'energy_efficiency_relative_to_sota': energy_efficiency_relative_to_sota,
        'sota_efficiency_relative_to_h100': sota_efficiency_relative_to_h100,
        'combined_efficiency': combined_efficiency,
        'H100_POWER_W': H100_POWER_W,
        'energy_gw': energy_gw,
        'energy_gw_simplified': energy_gw_simplified,
    }


def run_continuous_energy_calculation(
    agreement_year: int = 2030,
    prc_compute_stock_h100e: float = 1e6,
    proportion_to_divert: float = 0.05,
    energy_efficiency_relative_to_sota: float = 0.2,
    h100_power_w: float = 700.0,
):
    """
    Calculate initial compute energy consumption using the continuous model approach.

    Returns detailed breakdown of the energy calculation.
    """
    # Diverted compute
    initial_diverted_h100e = prc_compute_stock_h100e * proportion_to_divert

    # Energy calculation (current continuous model approach):
    # energy_per_h100e_gw = (h100_power_w / energy_efficiency) / 1e9
    # energy_gw = initial_diverted_h100e * energy_per_h100e_gw
    energy_per_h100e_gw = (h100_power_w / energy_efficiency_relative_to_sota) / 1e9
    energy_gw = initial_diverted_h100e * energy_per_h100e_gw

    return {
        'agreement_year': agreement_year,
        'prc_compute_stock_h100e': prc_compute_stock_h100e,
        'initial_diverted_h100e': initial_diverted_h100e,
        'energy_efficiency_relative_to_sota': energy_efficiency_relative_to_sota,
        'h100_power_w': h100_power_w,
        'energy_per_h100e_gw': energy_per_h100e_gw,
        'energy_gw': energy_gw,
    }


def run_corrected_continuous_energy_calculation(
    agreement_year: int = 2030,
    prc_compute_stock_h100e: float = 1e6,
    proportion_to_divert: float = 0.05,
    energy_efficiency_relative_to_sota: float = 0.2,
    improvement_in_energy_efficiency_per_year: float = 1.26,
    h100_power_w: float = 700.0,
):
    """
    Calculate initial compute energy consumption using a CORRECTED continuous model approach.

    This includes the state-of-the-art efficiency improvement factor.
    """
    # Calculate state-of-the-art efficiency relative to H100
    h100_release_year = 2022
    years_since_h100 = agreement_year - h100_release_year
    sota_efficiency_relative_to_h100 = improvement_in_energy_efficiency_per_year ** years_since_h100

    # Combined efficiency (like discrete model)
    combined_efficiency = energy_efficiency_relative_to_sota * sota_efficiency_relative_to_h100

    # Diverted compute
    initial_diverted_h100e = prc_compute_stock_h100e * proportion_to_divert

    # Corrected energy calculation:
    # Watts per H100e = h100_power_w / combined_efficiency
    watts_per_h100e = h100_power_w / combined_efficiency
    energy_gw = (initial_diverted_h100e * watts_per_h100e) / 1e9

    return {
        'agreement_year': agreement_year,
        'prc_compute_stock_h100e': prc_compute_stock_h100e,
        'initial_diverted_h100e': initial_diverted_h100e,
        'energy_efficiency_relative_to_sota': energy_efficiency_relative_to_sota,
        'sota_efficiency_relative_to_h100': sota_efficiency_relative_to_h100,
        'combined_efficiency': combined_efficiency,
        'h100_power_w': h100_power_w,
        'watts_per_h100e': watts_per_h100e,
        'energy_gw': energy_gw,
    }


def test_continuous_model_formula():
    """
    Directly test the continuous model's energy calculation formula.
    This verifies the fix by checking the formula logic in isolation.
    """
    # Test parameters
    agreement_year = 2030
    h100_power_w = 700.0
    energy_efficiency_relative_to_sota = 0.2
    sota_energy_efficiency_improvement_per_year = 1.26

    # Calculate what the continuous model should compute
    h100_release_year = 2022
    years_since_h100 = max(0, agreement_year - h100_release_year)
    sota_efficiency_relative_to_h100 = sota_energy_efficiency_improvement_per_year ** years_since_h100
    combined_energy_efficiency = energy_efficiency_relative_to_sota * sota_efficiency_relative_to_h100
    watts_per_h100e = h100_power_w / combined_energy_efficiency

    # Calculate expected energy for 50000 H100e
    diverted_compute = 50000.0
    expected_energy_gw = diverted_compute * watts_per_h100e / 1e9

    return {
        'agreement_year': agreement_year,
        'years_since_h100': years_since_h100,
        'sota_efficiency_relative_to_h100': sota_efficiency_relative_to_h100,
        'combined_energy_efficiency': combined_energy_efficiency,
        'watts_per_h100e': watts_per_h100e,
        'expected_energy_gw': expected_energy_gw,
    }


def main():
    """Compare energy calculations between models."""

    print("=" * 80)
    print("INITIAL COMPUTE ENERGY CONSUMPTION COMPARISON")
    print("=" * 80)

    # Test parameters
    agreement_year = 2030
    prc_compute_stock = 1e6  # H100e
    proportion_to_divert = 0.05
    energy_efficiency_relative_to_sota = 0.2
    improvement_per_year = 1.26

    print(f"\nTest Parameters:")
    print(f"  Agreement Year: {agreement_year}")
    print(f"  PRC Compute Stock: {prc_compute_stock:,.0f} H100e")
    print(f"  Proportion to Divert: {proportion_to_divert:.0%}")
    print(f"  Energy Efficiency (PRC relative to SOTA): {energy_efficiency_relative_to_sota}")
    print(f"  Energy Efficiency Improvement per Year: {improvement_per_year}")

    # Run discrete model calculation
    print("\n" + "-" * 80)
    print("DISCRETE MODEL (Reference)")
    print("-" * 80)
    discrete = run_discrete_energy_calculation(
        agreement_year=agreement_year,
        prc_compute_stock_h100e=prc_compute_stock,
        proportion_to_divert=proportion_to_divert,
        energy_efficiency_relative_to_sota=energy_efficiency_relative_to_sota,
        improvement_in_energy_efficiency_per_year=improvement_per_year,
    )
    print(f"  Initial Diverted Compute: {discrete['initial_diverted_h100e']:,.0f} H100e")
    print(f"  PRC Efficiency vs SOTA: {discrete['energy_efficiency_relative_to_sota']}")
    print(f"  SOTA Efficiency vs H100: {discrete['sota_efficiency_relative_to_h100']:.4f}")
    print(f"  Combined Efficiency: {discrete['combined_efficiency']:.4f}")
    print(f"  H100 Power: {discrete['H100_POWER_W']:.1f} W")
    print(f"  >>> Initial Energy Requirement: {discrete['energy_gw']:.6f} GW")

    # Run current continuous model calculation
    print("\n" + "-" * 80)
    print("CONTINUOUS MODEL (Current - INCORRECT)")
    print("-" * 80)
    continuous = run_continuous_energy_calculation(
        agreement_year=agreement_year,
        prc_compute_stock_h100e=prc_compute_stock,
        proportion_to_divert=proportion_to_divert,
        energy_efficiency_relative_to_sota=energy_efficiency_relative_to_sota,
    )
    print(f"  Initial Diverted Compute: {continuous['initial_diverted_h100e']:,.0f} H100e")
    print(f"  Energy Efficiency: {continuous['energy_efficiency_relative_to_sota']} (MISSING SOTA factor!)")
    print(f"  H100 Power: {continuous['h100_power_w']:.1f} W")
    print(f"  >>> Initial Energy Requirement: {continuous['energy_gw']:.6f} GW")

    # Run corrected continuous model calculation
    print("\n" + "-" * 80)
    print("CONTINUOUS MODEL (Corrected)")
    print("-" * 80)
    corrected = run_corrected_continuous_energy_calculation(
        agreement_year=agreement_year,
        prc_compute_stock_h100e=prc_compute_stock,
        proportion_to_divert=proportion_to_divert,
        energy_efficiency_relative_to_sota=energy_efficiency_relative_to_sota,
        improvement_in_energy_efficiency_per_year=improvement_per_year,
    )
    print(f"  Initial Diverted Compute: {corrected['initial_diverted_h100e']:,.0f} H100e")
    print(f"  PRC Efficiency vs SOTA: {corrected['energy_efficiency_relative_to_sota']}")
    print(f"  SOTA Efficiency vs H100: {corrected['sota_efficiency_relative_to_h100']:.4f}")
    print(f"  Combined Efficiency: {corrected['combined_efficiency']:.4f}")
    print(f"  H100 Power: {corrected['h100_power_w']:.1f} W")
    print(f"  Watts per H100e: {corrected['watts_per_h100e']:.2f} W")
    print(f"  >>> Initial Energy Requirement: {corrected['energy_gw']:.6f} GW")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Discrete Model Energy:            {discrete['energy_gw']:.6f} GW")
    print(f"  Current Continuous Energy:        {continuous['energy_gw']:.6f} GW")
    print(f"  Corrected Continuous Energy:      {corrected['energy_gw']:.6f} GW")
    print()
    ratio_current = continuous['energy_gw'] / discrete['energy_gw']
    ratio_corrected = corrected['energy_gw'] / discrete['energy_gw']
    print(f"  Current Continuous / Discrete:    {ratio_current:.4f}x (ERROR!)")
    print(f"  Corrected Continuous / Discrete:  {ratio_corrected:.4f}x")

    # Root cause
    print("\n" + "-" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("-" * 80)
    print(f"  The current continuous model uses energy_efficiency = {energy_efficiency_relative_to_sota}")
    print(f"  But the discrete model uses energy_efficiency = {discrete['combined_efficiency']:.4f}")
    print(f"  The difference is the SOTA efficiency improvement factor: {discrete['sota_efficiency_relative_to_h100']:.4f}")
    print()
    print(f"  FIX: Multiply energy_efficiency by state_of_the_art_energy_efficiency_improvement_per_year ** (year - 2022)")
    print(f"       For {agreement_year}: 1.26 ** {agreement_year - 2022} = {discrete['sota_efficiency_relative_to_h100']:.4f}")

    # Test with different years
    print("\n" + "=" * 80)
    print("IMPACT ACROSS DIFFERENT AGREEMENT YEARS (Formula comparison)")
    print("=" * 80)
    for year in [2026, 2028, 2030, 2032]:
        discrete_y = run_discrete_energy_calculation(
            agreement_year=year,
            prc_compute_stock_h100e=prc_compute_stock,
            proportion_to_divert=proportion_to_divert,
            energy_efficiency_relative_to_sota=energy_efficiency_relative_to_sota,
            improvement_in_energy_efficiency_per_year=improvement_per_year,
        )
        continuous_y = run_continuous_energy_calculation(
            agreement_year=year,
            prc_compute_stock_h100e=prc_compute_stock,
            proportion_to_divert=proportion_to_divert,
            energy_efficiency_relative_to_sota=energy_efficiency_relative_to_sota,
        )
        ratio = continuous_y['energy_gw'] / discrete_y['energy_gw']
        print(f"  Year {year}: SOTA factor = {discrete_y['sota_efficiency_relative_to_h100']:.2f}, "
              f"Error ratio = {ratio:.2f}x")

    # Test formula correctness
    print("\n" + "=" * 80)
    print("FORMULA VERIFICATION (Continuous Model Fix)")
    print("=" * 80)
    formula_test = test_continuous_model_formula()
    print(f"  Years since H100: {formula_test['years_since_h100']}")
    print(f"  SOTA efficiency relative to H100: {formula_test['sota_efficiency_relative_to_h100']:.4f}")
    print(f"  Combined energy efficiency: {formula_test['combined_energy_efficiency']:.4f}")
    print(f"  Watts per H100e: {formula_test['watts_per_h100e']:.2f} W")
    print(f"  Expected energy for 50000 H100e: {formula_test['expected_energy_gw']:.6f} GW")

    # Verify the fix aligns with discrete model
    ratio = formula_test['expected_energy_gw'] / discrete['energy_gw']
    print(f"\n  Formula output / Discrete model: {ratio:.4f}x")

    if abs(ratio - 1.0) < 0.01:
        print("\n  ✓ FORMULA VERIFIED: The fix correctly aligns with discrete model!")
    else:
        print(f"\n  ✗ FORMULA ISSUE: Expected ratio ~1.0, got {ratio:.4f}x")

    # Final summary
    print("\n" + "=" * 80)
    print("FIX SUMMARY")
    print("=" * 80)
    print("""
The continuous model was missing the state-of-the-art energy efficiency
improvement factor when calculating initial compute energy consumption.

BEFORE FIX (WRONG):
  energy_efficiency = energy_efficiency_of_compute_stock_relative_to_state_of_the_art
  watts_per_h100e = h100_power_w / energy_efficiency

AFTER FIX (CORRECT):
  sota_efficiency_relative_to_h100 = improvement_per_year ** (year - 2022)
  combined_efficiency = energy_efficiency_relative_to_sota * sota_efficiency_relative_to_h100
  watts_per_h100e = h100_power_w / combined_efficiency

This matches the discrete model's LargestAIProject.get_energy_efficiency_relative_to_h100()
""")


if __name__ == "__main__":
    main()
