"""
Nation initialization.

Initializes Nation entities for a given year.
"""

from classes.world.entities import Nation, NamedNations
from classes.world.assets import Compute, Fabs, Datacenters
from parameters.classes import SimulationParameters

# H100 reference power consumption in watts
H100_POWER_W = 700.0


def initialize_usa(
    params: SimulationParameters,
    year: int,
    total_compute: float = 0.0,
) -> Nation:
    """
    Initialize the USA nation for a given year.

    Args:
        params: SimulationParameters containing model configuration
        year: The integer year to initialize for (e.g., 2026)
        total_compute: Total operating compute in TPP H100e

    Returns:
        Initialized Nation for the USA
    """
    # Create compute stock (national level)
    compute_stock = Compute(
        tpp_h100e_including_attrition=total_compute,
        functional_tpp_h100e=total_compute,
        watts_per_h100e=700.0,
        average_functional_chip_age_years=0.0,
    )

    # Create empty fabs (not modeling fab production for US currently)
    fabs = Fabs(monthly_compute_production=Compute(
        tpp_h100e_including_attrition=0.0,
        functional_tpp_h100e=0.0,
        watts_per_h100e=700.0,
        average_functional_chip_age_years=0.0,
    ))

    # Create datacenters (energy capacity based on compute)
    # Assuming 700W per H100e, convert to GW
    datacenter_capacity_gw = (total_compute * 700.0) / 1e9 if total_compute > 0 else 0.0
    datacenters = Datacenters(data_center_capacity_gw=datacenter_capacity_gw)

    return Nation(
        id=NamedNations.USA,
        fabs=fabs,
        compute_stock=compute_stock,
        datacenters=datacenters,
        total_energy_consumption_gw=datacenter_capacity_gw,  # Simplified: just AI compute
        operating_compute_tpp_h100e=total_compute,
    )


def initialize_prc(
    params: SimulationParameters,
    year: int,
) -> Nation:
    """
    Initialize the PRC nation for a given year.

    Args:
        params: SimulationParameters containing model configuration
        year: The integer year to initialize for (e.g., 2026)

    Returns:
        Initialized Nation for the PRC
    """
    # Get PRC compute parameters
    prc_compute_params = params.compute.PRCComputeParameters
    prc_energy_params = params.datacenter_and_energy.prc_energy_consumption

    # Calculate PRC compute stock at the given year based on growth from 2025
    base_compute = prc_compute_params.total_prc_compute_tpp_h100e_in_2025
    growth_rate = prc_compute_params.annual_growth_rate_of_prc_compute_stock
    years_since_2025 = year - 2025
    prc_compute_stock = base_compute * (growth_rate ** years_since_2025)

    # Energy efficiency
    energy_efficiency = prc_energy_params.energy_efficiency_of_compute_stock_relative_to_state_of_the_art
    watts_per_h100e = H100_POWER_W / energy_efficiency

    # Create compute stock
    compute_stock = Compute(
        tpp_h100e_including_attrition=prc_compute_stock,
        functional_tpp_h100e=prc_compute_stock,
        watts_per_h100e=watts_per_h100e,
        average_functional_chip_age_years=2.0,
    )

    # Create fabs (PRC domestic production)
    fabs = Fabs(monthly_compute_production=Compute(
        tpp_h100e_including_attrition=0.0,
        functional_tpp_h100e=0.0,
        watts_per_h100e=H100_POWER_W,
        average_functional_chip_age_years=0.0,
    ))

    # Create datacenters (based on total energy consumption)
    total_energy_gw = prc_energy_params.total_prc_energy_consumption_gw
    # Assume 5% of total energy goes to datacenters
    datacenter_capacity_gw = total_energy_gw * 0.05
    datacenters = Datacenters(data_center_capacity_gw=datacenter_capacity_gw)

    # Calculate operating compute (limited by datacenter capacity)
    max_operating_compute = (datacenter_capacity_gw * 1e9) / watts_per_h100e
    operating_compute = min(prc_compute_stock, max_operating_compute)

    return Nation(
        id=NamedNations.PRC,
        fabs=fabs,
        compute_stock=compute_stock,
        datacenters=datacenters,
        total_energy_consumption_gw=total_energy_gw,
        operating_compute_tpp_h100e=operating_compute,
    )


def initialize_prc_counterfactual_no_slowdown(
    params: SimulationParameters,
    year: int,
) -> Nation:
    """
    Initialize the PRC counterfactual (no slowdown) nation for a given year.

    This represents PRC's compute trajectory in a hypothetical scenario where there
    is no slowdown agreement. Used for computing AI R&D reduction ratios.

    The counterfactual uses fixed parameters from the reference model:
    - Base compute: 1e5 H100e in 2025
    - Growth rate: 2.2x per year (p50 value)
    - AI R&D fraction: 0.5 (tracked separately in reduction_ratios)

    Args:
        params: SimulationParameters containing model configuration
        year: The integer year to initialize for (e.g., 2026)

    Returns:
        Initialized Nation representing PRC's no-slowdown counterfactual
    """
    # Fixed counterfactual parameters (from reference model ExogenousTrends)
    # These are intentionally NOT sampled - they represent the baseline counterfactual
    COUNTERFACTUAL_BASE_COMPUTE_2025 = 1e5  # H100e
    COUNTERFACTUAL_GROWTH_RATE = 2.2  # p50 annual growth rate

    # Calculate counterfactual compute stock at the given year
    years_since_2025 = year - 2025
    counterfactual_compute = COUNTERFACTUAL_BASE_COMPUTE_2025 * (COUNTERFACTUAL_GROWTH_RATE ** years_since_2025)

    # Create compute stock (using standard H100 watts since this is hypothetical)
    compute_stock = Compute(
        tpp_h100e_including_attrition=counterfactual_compute,
        functional_tpp_h100e=counterfactual_compute,
        watts_per_h100e=H100_POWER_W,
        average_functional_chip_age_years=0.0,  # Not tracking attrition for counterfactual
    )

    # Empty fabs and minimal datacenters (we only care about compute stock for this entity)
    fabs = Fabs(monthly_compute_production=Compute(
        tpp_h100e_including_attrition=0.0,
        functional_tpp_h100e=0.0,
        watts_per_h100e=H100_POWER_W,
        average_functional_chip_age_years=0.0,
    ))
    datacenters = Datacenters(data_center_capacity_gw=0.0)

    return Nation(
        id=NamedNations.PRC_COUNTERFACTUAL_NO_SLOWDOWN,
        fabs=fabs,
        compute_stock=compute_stock,
        datacenters=datacenters,
        total_energy_consumption_gw=0.0,
        operating_compute_tpp_h100e=counterfactual_compute,
    )


def initialize_usa_counterfactual_no_slowdown(
    params: SimulationParameters,
    year: int,
) -> Nation:
    """
    Initialize the USA counterfactual (no slowdown) nation for a given year.

    This represents the "largest AI company" (e.g., OpenAI/Google) compute trajectory
    in a hypothetical scenario where there is no slowdown agreement.
    Used for computing AI R&D reduction ratios.

    The counterfactual uses fixed parameters from the reference model (ExogenousTrends):
    - Base compute: 1.2e5 H100e in 2025 (largest_ai_project_compute_stock_in_2025)
    - Growth rate: 2.91x per year (annual_growth_rate_of_largest_ai_project_compute_stock)

    Args:
        params: SimulationParameters containing model configuration
        year: The integer year to initialize for (e.g., 2026)

    Returns:
        Initialized Nation representing largest AI company's no-slowdown counterfactual
    """
    # Fixed counterfactual parameters (from reference model ExogenousTrends)
    # These represent the largest global AI project (e.g., OpenAI/Google)
    COUNTERFACTUAL_BASE_COMPUTE_2025 = 1.2e5  # H100e
    COUNTERFACTUAL_GROWTH_RATE = 2.91  # annual growth rate

    # Calculate counterfactual compute stock at the given year
    years_since_2025 = year - 2025
    counterfactual_compute = COUNTERFACTUAL_BASE_COMPUTE_2025 * (COUNTERFACTUAL_GROWTH_RATE ** years_since_2025)

    # Create compute stock (using standard H100 watts since this is hypothetical)
    compute_stock = Compute(
        tpp_h100e_including_attrition=counterfactual_compute,
        functional_tpp_h100e=counterfactual_compute,
        watts_per_h100e=H100_POWER_W,
        average_functional_chip_age_years=0.0,  # Not tracking attrition for counterfactual
    )

    # Empty fabs and minimal datacenters (we only care about compute stock for this entity)
    fabs = Fabs(monthly_compute_production=Compute(
        tpp_h100e_including_attrition=0.0,
        functional_tpp_h100e=0.0,
        watts_per_h100e=H100_POWER_W,
        average_functional_chip_age_years=0.0,
    ))
    datacenters = Datacenters(data_center_capacity_gw=0.0)

    return Nation(
        id=NamedNations.USA_COUNTERFACTUAL_NO_SLOWDOWN,
        fabs=fabs,
        compute_stock=compute_stock,
        datacenters=datacenters,
        total_energy_consumption_gw=0.0,
        operating_compute_tpp_h100e=counterfactual_compute,
    )
