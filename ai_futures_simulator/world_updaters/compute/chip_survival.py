"""
Chip survival and attrition dynamics.

=============================================================================
MATHEMATICAL MODEL
=============================================================================

We model chip attrition using a linearly increasing hazard rate:

    h(a) = h₀ + h₁·a

where:
    h₀ = initial_annual_hazard_rate (per year)
    h₁ = annual_hazard_rate_increase_per_year (per year²)
    a  = chip age (years)

KEY INSIGHT: Since the hazard rate is LINEAR in age, the instantaneous
attrition rate for the entire stock can be computed using just the AVERAGE
age. This is because for any linear function f(x) = mx + b:

    E[f(X)] = f(E[X])

So if we have a distribution of chip ages with mean ā, the expected hazard
rate equals the hazard rate evaluated at the mean:

    E[h(a)] = h(ā) = h₀ + h₁·ā

This makes average_functional_chip_age_years sufficient to track attrition.

=============================================================================
STATE VARIABLES
=============================================================================

C  = functional compute stock (TPP H100e)
ā  = average age of functional chips (years)
F  = production flow rate (TPP H100e per year)

=============================================================================
DIFFERENTIAL EQUATIONS
=============================================================================

1. COMPUTE STOCK DYNAMICS
   ----------------------
   Change in functional compute = Production - Attrition

       dC/dt = F - h(ā)·C = F - (h₀ + h₁·ā)·C

   where:
   - F is the rate of new chip production
   - h(ā)·C is the rate of chip failure (attrition)

2. AVERAGE AGE DYNAMICS
   --------------------
   Let A = C·ā be total "age-weighted compute" (units: TPP·H100e·years)

   Change in total age comes from:
   - Aging: All C chips age, adding C·dt to total age
   - Attrition: Chips fail, removing h(ā)·C·ā·dt from total age
   - Production: F·dt new chips with age 0, adding 0 to total age

       dA/dt = C - h(ā)·C·ā

   Expanding A = C·ā using the product rule:

       d(C·ā)/dt = C·(dā/dt) + ā·(dC/dt)

   Substituting and simplifying:

       C - h(ā)·C·ā = C·(dā/dt) + ā·(F - h(ā)·C)
       C - h(ā)·C·ā = C·(dā/dt) + ā·F - h(ā)·C·ā
       C = C·(dā/dt) + ā·F

   Solving for dā/dt:

       dā/dt = 1 - (ā·F)/C

   Interpretation:
   - If F = 0 (no production): dā/dt = 1 (age increases at 1 year per year)
   - High production: average age decreases (young chips dilute the average)
   - Steady state (dā/dt = 0): ā_ss = C/F

=============================================================================
RELATIONSHIP TO GROWTH RATE
=============================================================================

If the "growth rate" parameter g represents GROSS production (what growth
would be without attrition), then:

    F = C·ln(g)

And the NET change in compute is:

    dC/dt = C·ln(g) - h(ā)·C = C·(ln(g) - h(ā))

The effective growth multiplier is:

    g_effective = exp(ln(g) - h(ā)) = g·exp(-h(ā))

So attrition reduces the effective growth rate.

=============================================================================
"""

import math
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from parameters.compute_parameters import SurvivalRateParameters


def calculate_hazard_rate(
    average_age: float,
    initial_hazard_rate: float,
    hazard_rate_increase_per_year: float,
) -> float:
    """
    Calculate the hazard rate at a given average age.

    Args:
        average_age: Average age of chips (years)
        initial_hazard_rate: Base annual hazard rate h₀ (per year)
        hazard_rate_increase_per_year: Rate at which hazard increases h₁ (per year²)

    Returns:
        Hazard rate h(ā) = h₀ + h₁·ā (per year)
    """
    return initial_hazard_rate + hazard_rate_increase_per_year * average_age


def calculate_attrition_rate(
    functional_compute: float,
    average_age: float,
    initial_hazard_rate: float,
    hazard_rate_increase_per_year: float,
) -> float:
    """
    Calculate the rate of compute loss due to chip failure.

    Args:
        functional_compute: Current functional compute C (TPP H100e)
        average_age: Average age of functional chips ā (years)
        initial_hazard_rate: Base annual hazard rate h₀ (per year)
        hazard_rate_increase_per_year: Rate at which hazard increases h₁ (per year²)

    Returns:
        Attrition rate = h(ā)·C (TPP H100e per year)
    """
    hazard = calculate_hazard_rate(average_age, initial_hazard_rate, hazard_rate_increase_per_year)
    return hazard * functional_compute


def calculate_compute_derivative(
    functional_compute: float,
    average_age: float,
    production_rate: float,
    initial_hazard_rate: float,
    hazard_rate_increase_per_year: float,
) -> float:
    """
    Calculate dC/dt: the rate of change of functional compute.

    dC/dt = F - h(ā)·C

    Args:
        functional_compute: Current functional compute C (TPP H100e)
        average_age: Average age of functional chips ā (years)
        production_rate: Rate of new chip production F (TPP H100e per year)
        initial_hazard_rate: Base annual hazard rate h₀ (per year)
        hazard_rate_increase_per_year: Rate at which hazard increases h₁ (per year²)

    Returns:
        dC/dt (TPP H100e per year)
    """
    attrition = calculate_attrition_rate(
        functional_compute, average_age,
        initial_hazard_rate, hazard_rate_increase_per_year
    )
    return production_rate - attrition


def calculate_average_age_derivative(
    functional_compute: float,
    average_age: float,
    production_rate: float,
) -> float:
    """
    Calculate dā/dt: the rate of change of average chip age.

    dā/dt = 1 - (ā·F)/C

    Args:
        functional_compute: Current functional compute C (TPP H100e)
        average_age: Average age of functional chips ā (years)
        production_rate: Rate of new chip production F (TPP H100e per year)

    Returns:
        dā/dt (years per year, i.e., dimensionless)

    Note: When functional_compute is very small but production is positive,
    newly produced chips have age ~0, so the dilution term dominates and
    average age stays near 0. When functional_compute is 0 with no production,
    average age is undefined so we return 0.
    """
    if functional_compute <= 0:
        if production_rate > 0:
            # Compute is 0 but production is starting - chips are brand new (age 0)
            # Return 0 to keep average age at 0 until compute grows
            return 0.0
        else:
            # No compute and no production - age is undefined
            return 0.0

    # Standard formula: dā/dt = 1 - (ā·F)/C
    # When ā=0 and C>0: dā/dt = 1 (chips age)
    # High production dilutes average age (new chips have age 0)
    return 1.0 - (average_age * production_rate) / functional_compute


def calculate_production_rate_from_growth(
    functional_compute: float,
    gross_growth_rate: float,
) -> float:
    """
    Calculate production rate from gross growth rate.

    If growth_rate g represents what the growth would be without attrition:
        F = C·ln(g)

    Args:
        functional_compute: Current functional compute C (TPP H100e)
        gross_growth_rate: Annual growth multiplier g (e.g., 1.5 for 50% growth)

    Returns:
        Production rate F (TPP H100e per year)
    """
    if gross_growth_rate <= 1.0:
        return 0.0
    return functional_compute * math.log(gross_growth_rate)


def calculate_derivatives_with_attrition(
    functional_compute: float,
    average_age: float,
    gross_growth_rate: float,
    initial_hazard_rate: float,
    hazard_rate_increase_per_year: float,
) -> Tuple[float, float]:
    """
    Calculate both dC/dt and dā/dt accounting for attrition.

    This is the main function to use for computing state derivatives.

    Args:
        functional_compute: Current functional compute C (TPP H100e)
        average_age: Average age of functional chips ā (years)
        gross_growth_rate: Annual growth multiplier g (what growth would be without attrition)
        initial_hazard_rate: Base annual hazard rate h₀ (per year)
        hazard_rate_increase_per_year: Rate at which hazard increases h₁ (per year²)

    Returns:
        Tuple of (dC/dt, dā/dt):
        - dC/dt: Rate of change of functional compute (TPP H100e per year)
        - dā/dt: Rate of change of average age (years per year)
    """
    # Production rate from growth rate
    production_rate = calculate_production_rate_from_growth(functional_compute, gross_growth_rate)

    # Compute derivative: dC/dt = F - h(ā)·C
    dC_dt = calculate_compute_derivative(
        functional_compute, average_age, production_rate,
        initial_hazard_rate, hazard_rate_increase_per_year
    )

    # Age derivative: dā/dt = 1 - (ā·F)/C
    da_dt = calculate_average_age_derivative(functional_compute, average_age, production_rate)

    return dC_dt, da_dt


# =============================================================================
# LEGACY FUNCTIONS (for backwards compatibility with black project code)
# =============================================================================

def calculate_survival_rate(
    years_since_acquisition: float,
    initial_hazard_rate: float,
    hazard_rate_increase_per_year: float,
) -> float:
    """
    Calculate chip survival rate for a cohort acquired at a single point in time.

    This is for computing the survival of an initial stock that was all
    acquired at once (e.g., diverted compute in a black project).

    For a cohort of age t, the survival probability is:
        S(t) = exp(-H(t))

    where H(t) is the cumulative hazard:
        H(t) = ∫₀ᵗ h(s) ds = h₀·t + h₁·t²/2

    Args:
        years_since_acquisition: Time since chips were acquired t (years)
        initial_hazard_rate: Base annual hazard rate h₀ (per year)
        hazard_rate_increase_per_year: Rate at which hazard increases h₁ (per year²)

    Returns:
        Survival rate S(t) as a fraction [0, 1]
    """
    if years_since_acquisition <= 0:
        return 1.0

    # Cumulative hazard: H(t) = h₀·t + h₁·t²/2
    cumulative_hazard = (
        initial_hazard_rate * years_since_acquisition +
        hazard_rate_increase_per_year * years_since_acquisition ** 2 / 2
    )

    return math.exp(-cumulative_hazard)


def calculate_functional_compute(
    all_compute_h100e: float,
    survival_rate: float,
) -> float:
    """
    Calculate functional compute after applying survival rate.

    Args:
        all_compute_h100e: Total compute in H100-equivalents
        survival_rate: Fraction of chips that are functional [0, 1]

    Returns:
        Functional compute in H100-equivalents
    """
    return all_compute_h100e * survival_rate


def calculate_survival_rate_from_params(
    years_since_acquisition: float,
    survival_params: "SurvivalRateParameters",
) -> float:
    """
    Calculate chip survival rate using parameters dataclass.

    Convenience function that extracts values from SurvivalRateParameters.

    Args:
        years_since_acquisition: Time since chips were acquired (years)
        survival_params: SurvivalRateParameters dataclass

    Returns:
        Survival rate as a fraction [0, 1]
    """
    return calculate_survival_rate(
        years_since_acquisition=years_since_acquisition,
        initial_hazard_rate=survival_params.initial_annual_hazard_rate,
        hazard_rate_increase_per_year=survival_params.annual_hazard_rate_increase_per_year,
    )
