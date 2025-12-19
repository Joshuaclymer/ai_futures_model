"""
Compare black project trajectories between ai_futures_simulator and black_project_backend.

The ai_futures_simulator uses continuous ODE integration (derivatives).
The black_project_backend uses discrete timesteps.

Results should be similar - the continuous model should converge to the discrete
model as timesteps become smaller.
"""

import sys
import os

# Add both project roots to path
sys.path.insert(0, '/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator')
sys.path.insert(0, '/Users/joshuaclymer/github/covert_compute_production_model')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math

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


@dataclass
class ComparisonConfig:
    """Configuration for the comparison test."""
    # Time settings
    agreement_year: float = 2030.0
    num_years: float = 7.0
    time_step: float = 0.1

    # Initial conditions
    prc_compute_stock_at_agreement: float = 1e6  # H100e
    proportion_to_divert: float = 0.05

    # Hazard rates (fixed for comparison)
    initial_hazard_rate: float = 0.05
    hazard_rate_increase_per_year: float = 0.02

    # Datacenter settings
    datacenter_construction_labor: int = 10000
    years_before_agreement_building_dc: float = 1.0
    max_proportion_prc_energy: float = 0.05
    mw_per_worker_per_year: float = 1.0
    operating_labor_per_mw: float = 0.1
    total_prc_energy_gw: float = 1100.0
    energy_efficiency_relative_to_sota: float = 0.2

    # Fab settings (disabled for simplicity in first comparison)
    build_fab: bool = False

    # Detection parameters
    researcher_headcount: int = 500
    mean_detection_time_100_workers: float = 6.95
    mean_detection_time_1000_workers: float = 3.42
    variance_of_detection_time: float = 3.88


def run_discrete_model(config: ComparisonConfig) -> dict:
    """Run the black_project_backend (discrete) model."""

    # Set up discrete model parameters with fixed values (no sampling)
    sim_settings = DiscreteSimulationSettings(
        agreement_start_year=int(config.agreement_year),
        num_years_to_simulate=config.num_years,
        time_step_years=config.time_step,
        num_simulations=1,
    )

    bp_properties = DiscreteBlackProjectProperties(
        run_a_black_project=True,
        proportion_of_initial_compute_stock_to_divert=config.proportion_to_divert,
        datacenter_construction_labor=config.datacenter_construction_labor,
        years_before_agreement_year_prc_starts_building_black_datacenters=int(config.years_before_agreement_building_dc),
        max_proportion_of_PRC_energy_consumption=config.max_proportion_prc_energy,
        fraction_of_datacenter_capacity_not_built_for_concealment_diverted_to_black_project_at_agreement_start=0.0,
        build_a_black_fab=config.build_fab,
        researcher_headcount=config.researcher_headcount,
    )

    bp_params = DiscreteBlackProjectParameters(
        survival_rate_parameters=DiscreteSurvivalRateParameters(
            initial_hazard_rate_p50=config.initial_hazard_rate,
            increase_of_hazard_rate_per_year_p50=config.hazard_rate_increase_per_year,
            hazard_rate_p25_relative_to_p50=1.0,  # No variation
            hazard_rate_p75_relative_to_p50=1.0,
        ),
        datacenter_model_parameters=DiscreteBlackDatacenterParameters(
            MW_per_construction_worker_per_year=config.mw_per_worker_per_year,
            relative_sigma_mw_per_construction_worker_per_year=0.0,  # Disable sampling
            operating_labor_per_MW=config.operating_labor_per_mw,
            relative_sigma_operating_labor_per_MW=0.0,  # Disable sampling
        ),
        black_fab_parameters=DiscreteBlackFabParameters(),
        detection_parameters=DiscreteDetectionParameters(
            mean_detection_time_for_100_workers=config.mean_detection_time_100_workers,
            mean_detection_time_for_1000_workers=config.mean_detection_time_1000_workers,
            variance_of_detection_time_given_num_workers=config.variance_of_detection_time,
        ),
        exogenous_trends=DiscreteExogenousTrends(
            total_prc_compute_stock_in_2025=config.prc_compute_stock_at_agreement / (2.2 ** (config.agreement_year - 2025)),
            annual_growth_rate_of_prc_compute_stock_p10=2.2,  # Fixed growth rate
            annual_growth_rate_of_prc_compute_stock_p50=2.2,
            annual_growth_rate_of_prc_compute_stock_p90=2.2,
            energy_efficiency_of_prc_stock_relative_to_state_of_the_art=config.energy_efficiency_relative_to_sota,
            total_GW_of_PRC_energy_consumption=config.total_prc_energy_gw,
        ),
    )

    model_params = DiscreteModelParameters(
        simulation_settings=sim_settings,
        black_project_properties=bp_properties,
        black_project_parameters=bp_params,
    )

    # Run simulation
    model = BlackProjectModel(model_params)
    model.run_simulations(1)

    # Extract trajectory
    project = model.simulation_results[0]['prc_black_project']

    years = np.arange(
        config.agreement_year,
        config.agreement_year + config.num_years + config.time_step,
        config.time_step
    )

    compute_stock = []
    datacenter_capacity = []
    cumulative_lr = []

    for year in years:
        # Surviving compute at this year
        surviving = project.black_project_stock.surviving_compute(year)
        compute_stock.append(surviving.total_h100e_tpp())

        # Datacenter capacity
        dc_cap = project.black_datacenters.get_covert_GW_capacity_total(year)
        datacenter_capacity.append(dc_cap)

        # Cumulative likelihood ratio (from fab if it exists, otherwise from datacenter workers)
        if project.black_fab is not None:
            lr = project.black_fab.cumulative_detection_likelihood_ratio(year)
        else:
            # Use datacenter worker-based detection
            lr = project.black_datacenters.cumulative_lr_from_direct_observation(year)
        cumulative_lr.append(lr)

    return {
        'years': years,
        'compute_stock': np.array(compute_stock),
        'datacenter_capacity': np.array(datacenter_capacity),
        'cumulative_lr': np.array(cumulative_lr),
        'initial_compute': project.black_project_stock.initial_prc_black_project,
    }


def run_continuous_model_buggy(config: ComparisonConfig) -> dict:
    """
    Run the BUGGY continuous model that matches current ai_futures_simulator.

    BUG: Uses average_age = t/2, but for pure decay all chips are age t.
    """

    # Initial state
    initial_compute = config.prc_compute_stock_at_agreement * config.proportion_to_divert

    # Calculate datacenter construction rate
    construction_rate_gw_per_year = (config.mw_per_worker_per_year / 1000.0) * config.datacenter_construction_labor
    max_capacity_gw = config.max_proportion_prc_energy * config.total_prc_energy_gw

    # Initial concealed capacity (built before agreement)
    initial_concealed_gw = construction_rate_gw_per_year * config.years_before_agreement_building_dc

    years = np.arange(
        config.agreement_year,
        config.agreement_year + config.num_years + config.time_step,
        config.time_step
    )

    compute_stock = []
    datacenter_capacity = []

    # Euler integration
    current_log_stock = math.log(initial_compute)
    current_log_dc = math.log(max(initial_concealed_gw, 1e-10))

    for i, year in enumerate(years):
        t = year - config.agreement_year

        # Record current state
        compute_stock.append(math.exp(current_log_stock))
        datacenter_capacity.append(math.exp(current_log_dc))

        if i < len(years) - 1:
            dt = config.time_step

            # BUG: Using average_age = t/2 instead of t
            average_age = t / 2.0

            # Hazard rate (depends on average age)
            hazard_rate = config.initial_hazard_rate + config.hazard_rate_increase_per_year * average_age

            # d(log_stock)/dt = -H
            d_log_stock = -hazard_rate

            current_log_stock += d_log_stock * dt

            # --- Datacenter capacity dynamics ---
            current_dc = math.exp(current_log_dc)

            # Only grow if below max capacity
            if current_dc < max_capacity_gw and construction_rate_gw_per_year > 0:
                headroom = max_capacity_gw - current_dc
                effective_rate = min(construction_rate_gw_per_year, headroom)

                if current_dc > 1e-10:
                    d_log_dc = effective_rate / current_dc
                else:
                    d_log_dc = 10.0
            else:
                d_log_dc = 0.0

            current_log_dc += d_log_dc * dt

    return {
        'years': years,
        'compute_stock': np.array(compute_stock),
        'datacenter_capacity': np.array(datacenter_capacity),
        'initial_compute': initial_compute,
    }


def run_continuous_model_fixed(config: ComparisonConfig) -> dict:
    """
    Run the FIXED continuous model.

    For pure decay (single cohort, no production), all chips have age = t.
    Hazard rate H(t) = H0 + k*t
    d(log_S)/dt = -H(t)
    Solution: S(t) = S0 * exp(-H0*t - k*t^2/2)

    Datacenter capacity uses asymptotic growth:
    dC/dt = rate * (1 - exp(-(max - C) / scale))
    This is ~linear when C << max, and asymptotes to max.
    """

    # Initial state
    initial_compute = config.prc_compute_stock_at_agreement * config.proportion_to_divert

    # Calculate datacenter construction rate
    construction_rate_gw_per_year = (config.mw_per_worker_per_year / 1000.0) * config.datacenter_construction_labor
    max_capacity_gw = config.max_proportion_prc_energy * config.total_prc_energy_gw

    # Initial concealed capacity (built before agreement)
    initial_concealed_gw = construction_rate_gw_per_year * config.years_before_agreement_building_dc

    # Scale for asymptotic behavior - smaller = sharper transition near max
    # Use ~5% of max as the transition scale
    asymptote_scale = max_capacity_gw * 0.05

    years = np.arange(
        config.agreement_year,
        config.agreement_year + config.num_years + config.time_step,
        config.time_step
    )

    compute_stock = []
    datacenter_capacity = []

    # Euler integration
    current_log_stock = math.log(initial_compute)
    current_dc = initial_concealed_gw

    for i, year in enumerate(years):
        t = year - config.agreement_year

        # Record current state
        compute_stock.append(math.exp(current_log_stock))
        datacenter_capacity.append(current_dc)

        if i < len(years) - 1:
            dt = config.time_step

            # FIXED: For pure decay, chip age = t (not t/2)
            chip_age = t

            # Hazard rate
            hazard_rate = config.initial_hazard_rate + config.hazard_rate_increase_per_year * chip_age

            # d(log_stock)/dt = -H
            d_log_stock = -hazard_rate

            current_log_stock += d_log_stock * dt

            # --- Datacenter capacity dynamics ---
            # Asymptotic growth: dC/dt = rate * (1 - exp(-(max - C) / scale))
            # When C << max: exp term ≈ 0, so dC/dt ≈ rate (linear)
            # When C → max: exp term → 1, so dC/dt → 0 (asymptote)
            headroom = max_capacity_gw - current_dc
            if headroom > 0 and construction_rate_gw_per_year > 0:
                asymptote_factor = 1.0 - math.exp(-headroom / asymptote_scale)
                d_dc = construction_rate_gw_per_year * asymptote_factor
            else:
                d_dc = 0.0

            current_dc += d_dc * dt

    return {
        'years': years,
        'compute_stock': np.array(compute_stock),
        'datacenter_capacity': np.array(datacenter_capacity),
        'initial_compute': initial_compute,
    }


def run_analytical_model(config: ComparisonConfig) -> dict:
    """
    Compute analytical solution for pure decay.

    S(t) = S0 * exp(-H0*t - k*t^2/2)

    where H0 = initial_hazard and k = hazard_increase.

    This is only exact when hazard rate doesn't depend on average age,
    but on actual years since start. The discrete model uses years_of_life
    for each cohort.
    """

    initial_compute = config.prc_compute_stock_at_agreement * config.proportion_to_divert

    years = np.arange(
        config.agreement_year,
        config.agreement_year + config.num_years + config.time_step,
        config.time_step
    )

    t = years - config.agreement_year

    # Pure decay: S(t) = S0 * exp(-H0*t - k*t^2/2)
    cumulative_hazard = config.initial_hazard_rate * t + config.hazard_rate_increase_per_year * t**2 / 2
    compute_stock = initial_compute * np.exp(-cumulative_hazard)

    return {
        'years': years,
        'compute_stock': compute_stock,
        'initial_compute': initial_compute,
    }


def run_continuous_likelihood_ratio(config: ComparisonConfig, use_sampled_detection: bool = False) -> dict:
    """
    Compute likelihood ratio using the ai_futures_simulator's detection utilities.

    Args:
        config: Comparison configuration
        use_sampled_detection: If True, sample a detection time and use binary detection.
                              If False, use pure survival function (no detection event).
    """
    from scipy import stats

    # Compute detection constants: mu(workers) = A / log10(workers)^B
    x1, mu1 = 100, config.mean_detection_time_100_workers
    x2, mu2 = 1000, config.mean_detection_time_1000_workers
    B = np.log(mu1 / mu2) / np.log(np.log10(x2) / np.log10(x1))
    A = mu1 * (np.log10(x1) ** B)

    # Total labor (researchers + datacenter construction labor)
    total_labor = config.researcher_headcount + config.datacenter_construction_labor

    # Calculate mean detection time for this labor level
    mu = A / (np.log10(total_labor) ** B)
    k = mu / config.variance_of_detection_time  # shape parameter
    theta = config.variance_of_detection_time  # scale parameter

    # Sample detection time if requested
    if use_sampled_detection:
        sampled_detection_time = np.random.gamma(k, theta)
    else:
        sampled_detection_time = float('inf')  # Never detected

    years = np.arange(
        config.agreement_year,
        config.agreement_year + config.num_years + config.time_step,
        config.time_step
    )

    cumulative_lr = []
    is_detected_list = []

    for year in years:
        years_since_start = year - config.agreement_year

        # Check if detected
        is_detected = years_since_start >= sampled_detection_time

        if is_detected:
            lr = 100.0  # Strong evidence project exists
        elif years_since_start <= 0:
            lr = 1.0  # Neutral at start
        else:
            # Survival function: P(T > t | project exists)
            # LR = P(not detected | project) / P(not detected | no project)
            #    = gamma.sf(t) / 1.0
            lr = stats.gamma.sf(years_since_start, a=k, scale=theta)
            lr = max(lr, 0.001)  # Floor

        cumulative_lr.append(lr)
        is_detected_list.append(is_detected)

    return {
        'years': years,
        'cumulative_lr': np.array(cumulative_lr),
        'is_detected': np.array(is_detected_list),
        'sampled_detection_time': sampled_detection_time,
        'total_labor': total_labor,
        'mean_detection_time': mu,
    }


def compare_models():
    """Compare the two models and plot results."""

    config = ComparisonConfig(
        agreement_year=2030.0,
        num_years=7.0,
        time_step=0.1,
        prc_compute_stock_at_agreement=1e6,
        proportion_to_divert=0.05,
        initial_hazard_rate=0.05,
        hazard_rate_increase_per_year=0.02,
        build_fab=False,  # Start without fab for simpler comparison
    )

    print("Running discrete model (black_project_backend)...")
    discrete_results = run_discrete_model(config)

    print("Running BUGGY continuous model (current ai_futures_simulator)...")
    buggy_results = run_continuous_model_buggy(config)

    print("Running FIXED continuous model...")
    fixed_results = run_continuous_model_fixed(config)

    print("Computing analytical solution...")
    analytical_results = run_analytical_model(config)

    print("Computing continuous likelihood ratio (no sampled detection)...")
    continuous_lr_no_detection = run_continuous_likelihood_ratio(config, use_sampled_detection=False)

    print("Computing continuous likelihood ratio (with sampled detection)...")
    # Use same random seed as discrete model for fair comparison
    np.random.seed(42)
    continuous_lr_with_detection = run_continuous_likelihood_ratio(config, use_sampled_detection=True)

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print(f"\nInitial compute stock:")
    print(f"  Discrete:   {discrete_results['initial_compute']:.2f} H100e")
    print(f"  Buggy:      {buggy_results['initial_compute']:.2f} H100e")
    print(f"  Fixed:      {fixed_results['initial_compute']:.2f} H100e")
    print(f"  Analytical: {analytical_results['initial_compute']:.2f} H100e")

    print(f"\nFinal compute stock (year {config.agreement_year + config.num_years}):")
    print(f"  Discrete:   {discrete_results['compute_stock'][-1]:.2f} H100e")
    print(f"  Buggy:      {buggy_results['compute_stock'][-1]:.2f} H100e")
    print(f"  Fixed:      {fixed_results['compute_stock'][-1]:.2f} H100e")
    print(f"  Analytical: {analytical_results['compute_stock'][-1]:.2f} H100e")

    print(f"\nDifference (Discrete vs Buggy):")
    print(f"  Relative:   {abs(discrete_results['compute_stock'][-1] - buggy_results['compute_stock'][-1]) / discrete_results['compute_stock'][-1] * 100:.2f}%")

    print(f"\nDifference (Discrete vs Fixed):")
    print(f"  Relative:   {abs(discrete_results['compute_stock'][-1] - fixed_results['compute_stock'][-1]) / discrete_results['compute_stock'][-1] * 100:.2f}%")

    print(f"\nFinal datacenter capacity:")
    print(f"  Discrete:   {discrete_results['datacenter_capacity'][-1]:.4f} GW")
    print(f"  Buggy:      {buggy_results['datacenter_capacity'][-1]:.4f} GW")
    print(f"  Fixed:      {fixed_results['datacenter_capacity'][-1]:.4f} GW")

    # Survival rate at final time
    t_final = config.num_years
    analytical_survival = np.exp(
        -config.initial_hazard_rate * t_final
        - config.hazard_rate_increase_per_year * t_final**2 / 2
    )

    print(f"\nExpected survival rate (analytical):")
    print(f"  exp(-H0*t - k*t^2/2) = {analytical_survival * 100:.2f}%")

    # Likelihood ratio comparison
    print(f"\n" + "="*60)
    print("LIKELIHOOD RATIO COMPARISON")
    print("="*60)

    print(f"\nDetection parameters:")
    print(f"  Total labor: {continuous_lr_no_detection['total_labor']}")
    print(f"  Mean detection time: {continuous_lr_no_detection['mean_detection_time']:.2f} years")
    print(f"  Sampled detection time (continuous): {continuous_lr_with_detection['sampled_detection_time']:.2f} years")

    print(f"\nLikelihood ratio at selected years:")
    for year_offset in [0, 2, 5, 7]:
        year = config.agreement_year + year_offset
        idx = int(year_offset / config.time_step)
        if idx < len(discrete_results['cumulative_lr']):
            discrete_lr = discrete_results['cumulative_lr'][idx]
            continuous_lr_nd = continuous_lr_no_detection['cumulative_lr'][idx]
            continuous_lr_wd = continuous_lr_with_detection['cumulative_lr'][idx]
            is_detected = continuous_lr_with_detection['is_detected'][idx]

            print(f"  Year {year:.0f} (t+{year_offset}):")
            print(f"    Discrete:                   {discrete_lr:.4f}")
            print(f"    Continuous (no detection):  {continuous_lr_nd:.4f}")
            print(f"    Continuous (w/ detection):  {continuous_lr_wd:.4f} (detected={is_detected})")

    print(f"\n" + "="*60)
    print("BUG IDENTIFIED")
    print("="*60)
    print(f"""
The bug is in black_project.py set_metric_attributes():
  - It sets average_age_years = years_since_agreement / 2.0
  - But for pure decay (no production), all chips are age = t

FIX: Instead of using a simplified average_age, the hazard rate
should be computed based on the actual age distribution of chips.

For the simple case (initial stock only, no fab production):
  - All chips have age = t (years since agreement)
  - H(t) = H0 + k*t
  - d(log_S)/dt = -H(t)

For the general case (with fab production):
  - Need to track multiple cohorts or use weighted average age
  - average_age should be weighted by chip count in each cohort
""")

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Compute stock over time
    ax1 = axes[0, 0]
    ax1.plot(discrete_results['years'], discrete_results['compute_stock'],
             'b-', linewidth=2, label='Discrete (backend)')
    ax1.plot(buggy_results['years'], buggy_results['compute_stock'],
             'r--', linewidth=2, label='Buggy (age=t/2)')
    ax1.plot(fixed_results['years'], fixed_results['compute_stock'],
             'g-.', linewidth=2, label='Fixed (age=t)')
    ax1.plot(analytical_results['years'], analytical_results['compute_stock'],
             'm:', linewidth=2, label='Analytical')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Compute Stock (H100e)')
    ax1.set_title('Compute Stock Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Datacenter capacity over time
    ax2 = axes[0, 1]
    ax2.plot(discrete_results['years'], discrete_results['datacenter_capacity'],
             'b-', linewidth=2, label='Discrete (backend)')
    ax2.plot(buggy_results['years'], buggy_results['datacenter_capacity'],
             'r--', linewidth=2, label='Buggy')
    ax2.plot(fixed_results['years'], fixed_results['datacenter_capacity'],
             'g-.', linewidth=2, label='Fixed')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Datacenter Capacity (GW)')
    ax2.set_title('Datacenter Capacity Growth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Difference in compute stock
    ax3 = axes[1, 0]
    min_len = min(len(discrete_results['compute_stock']), len(buggy_results['compute_stock']))
    diff_buggy = (discrete_results['compute_stock'][:min_len] - buggy_results['compute_stock'][:min_len]) / discrete_results['compute_stock'][:min_len] * 100
    diff_fixed = (discrete_results['compute_stock'][:min_len] - fixed_results['compute_stock'][:min_len]) / discrete_results['compute_stock'][:min_len] * 100
    ax3.plot(discrete_results['years'][:min_len], diff_buggy, 'r-', linewidth=2, label='Buggy')
    ax3.plot(discrete_results['years'][:min_len], diff_fixed, 'g-', linewidth=2, label='Fixed')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Relative Difference (%)')
    ax3.set_title('Error: (Discrete - Continuous) / Discrete')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Log-scale compute stock
    ax4 = axes[1, 1]
    ax4.semilogy(discrete_results['years'], discrete_results['compute_stock'],
                 'b-', linewidth=2, label='Discrete')
    ax4.semilogy(buggy_results['years'], buggy_results['compute_stock'],
                 'r--', linewidth=2, label='Buggy')
    ax4.semilogy(fixed_results['years'], fixed_results['compute_stock'],
                 'g-.', linewidth=2, label='Fixed')
    ax4.semilogy(analytical_results['years'], analytical_results['compute_stock'],
                 'm:', linewidth=2, label='Analytical')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Compute Stock (H100e) - Log Scale')
    ax4.set_title('Compute Stock (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Likelihood ratio over time
    ax5 = axes[0, 2]
    ax5.semilogy(discrete_results['years'], discrete_results['cumulative_lr'],
                 'b-', linewidth=2, label='Discrete (backend)')
    ax5.semilogy(continuous_lr_no_detection['years'], continuous_lr_no_detection['cumulative_lr'],
                 'g--', linewidth=2, label='Continuous (survival fn)')
    ax5.semilogy(continuous_lr_with_detection['years'], continuous_lr_with_detection['cumulative_lr'],
                 'r-.', linewidth=2, label='Continuous (w/ detection)')
    ax5.set_xlabel('Year')
    ax5.set_ylabel('Cumulative Likelihood Ratio (log scale)')
    ax5.set_title('Detection Likelihood Ratio')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=1, color='k', linestyle=':', alpha=0.5, label='LR=1 (neutral)')

    # Likelihood ratio difference
    ax6 = axes[1, 2]
    min_len = min(len(discrete_results['cumulative_lr']), len(continuous_lr_no_detection['cumulative_lr']))
    # Use relative difference where both are > 0
    mask = (discrete_results['cumulative_lr'][:min_len] > 0.001) & (continuous_lr_no_detection['cumulative_lr'][:min_len] > 0.001)
    diff_lr = np.zeros(min_len)
    diff_lr[mask] = (discrete_results['cumulative_lr'][:min_len][mask] - continuous_lr_no_detection['cumulative_lr'][:min_len][mask]) / discrete_results['cumulative_lr'][:min_len][mask] * 100
    ax6.plot(discrete_results['years'][:min_len], diff_lr, 'g-', linewidth=2, label='Survival fn')
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Relative Difference (%)')
    ax6.set_title('LR Error: (Discrete - Continuous) / Discrete')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/Users/joshuaclymer/github/ai_futures_simulator/scripts/black_project_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    compare_models()
