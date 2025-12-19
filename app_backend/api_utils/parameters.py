"""Parameter conversion and loading utilities."""

import sys
from pathlib import Path

# Add ai_futures_simulator subdirectory to path for imports
# Structure: ai_futures_simulator/app_backend/api_utils/parameters.py
#            ai_futures_simulator/ai_futures_simulator/parameters/...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator"))

from parameters.simulation_parameters import (
    ModelParameters,
    SimulationParameters,
    SimulationSettings,
    SoftwareRAndDParameters,
)
from parameters.compute_parameters import ComputeParameters
from parameters.energy_consumption_parameters import EnergyConsumptionParameters

# Developer ID in the simulator
DEVELOPER_ID = "us_frontier_lab"

# Default YAML config path
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "ai_futures_simulator" / "parameters" / "modal_parameters.yaml"

# Cache the model parameters (loaded once at startup)
_cached_model_params = None


def get_model_params():
    """Get cached model parameters, loading from YAML if needed."""
    global _cached_model_params
    if _cached_model_params is None:
        _cached_model_params = ModelParameters.from_yaml(DEFAULT_CONFIG_PATH)
    return _cached_model_params


def frontend_params_to_simulation_params(frontend_params: dict, time_range: list) -> SimulationParameters:
    """
    Convert frontend parameter format to SimulationParameters.

    The frontend sends parameters with names like 'rho_coding_labor', 'present_day', etc.
    We need to map these to the appropriate dataclass fields.
    """
    # Load defaults from YAML
    default_model_params = ModelParameters.from_yaml(DEFAULT_CONFIG_PATH)
    default_sim_params = default_model_params.sample()

    # Extract settings
    start_year = int(time_range[0]) if time_range else 2024
    end_year = float(time_range[1]) if len(time_range) > 1 else 2040.0
    n_eval_points = frontend_params.get('n_eval_points', 100)

    settings = SimulationSettings(
        simulation_start_year=start_year,
        simulation_end_year=end_year,
        n_eval_points=n_eval_points,
    )

    # Build software R&D parameters from frontend params, using defaults for missing
    sw_defaults = default_sim_params.software_r_and_d

    # Handle ac_time_horizon_minutes - frontend sends log scale
    ac_time_horizon = frontend_params.get('ac_time_horizon_minutes')
    if ac_time_horizon is not None:
        # Frontend sends log10 value, convert to linear
        if isinstance(ac_time_horizon, (int, float)) and ac_time_horizon < 1000:
            ac_time_horizon = 10 ** ac_time_horizon
    else:
        ac_time_horizon = sw_defaults.ac_time_horizon_minutes

    software_r_and_d = SoftwareRAndDParameters(
        # Mode flags
        human_only=frontend_params.get('human_only', sw_defaults.human_only),
        # Production function parameters
        rho_coding_labor=frontend_params.get('rho_coding_labor', sw_defaults.rho_coding_labor),
        coding_labor_normalization=frontend_params.get('coding_labor_normalization', sw_defaults.coding_labor_normalization),
        direct_input_exp_cap_ces_params=frontend_params.get('direct_input_exp_cap_ces_params', sw_defaults.direct_input_exp_cap_ces_params),
        rho_experiment_capacity=frontend_params.get('rho_experiment_capacity', sw_defaults.rho_experiment_capacity),
        alpha_experiment_capacity=frontend_params.get('alpha_experiment_capacity', sw_defaults.alpha_experiment_capacity),
        experiment_compute_exponent=frontend_params.get('experiment_compute_exponent', sw_defaults.experiment_compute_exponent),
        inf_labor_asymptote=frontend_params.get('inf_labor_asymptote', sw_defaults.inf_labor_asymptote),
        inf_compute_asymptote=frontend_params.get('inf_compute_asymptote', sw_defaults.inf_compute_asymptote),
        labor_anchor_exp_cap=frontend_params.get('labor_anchor_exp_cap', sw_defaults.labor_anchor_exp_cap),
        compute_anchor_exp_cap=frontend_params.get('compute_anchor_exp_cap', sw_defaults.compute_anchor_exp_cap),
        inv_compute_anchor_exp_cap=frontend_params.get('inv_compute_anchor_exp_cap', sw_defaults.inv_compute_anchor_exp_cap),
        parallel_penalty=frontend_params.get('parallel_penalty', sw_defaults.parallel_penalty),
        # Software progress parameters
        r_software=frontend_params.get('r_software', sw_defaults.r_software),
        software_progress_rate_at_reference_year=frontend_params.get('software_progress_rate_at_reference_year', sw_defaults.software_progress_rate_at_reference_year),
        # Automation schedule parameters
        automation_fraction_at_coding_automation_anchor=frontend_params.get('automation_fraction_at_coding_automation_anchor', sw_defaults.automation_fraction_at_coding_automation_anchor),
        automation_anchors=frontend_params.get('automation_anchors', sw_defaults.automation_anchors),
        automation_interp_type=frontend_params.get('automation_interp_type', sw_defaults.automation_interp_type),
        automation_logistic_asymptote=frontend_params.get('automation_logistic_asymptote', sw_defaults.automation_logistic_asymptote),
        swe_multiplier_at_present_day=frontend_params.get('swe_multiplier_at_present_day', sw_defaults.swe_multiplier_at_present_day),
        # Coding labor mode
        coding_labor_mode=frontend_params.get('coding_labor_mode', sw_defaults.coding_labor_mode),
        coding_automation_efficiency_slope=frontend_params.get('coding_automation_efficiency_slope', sw_defaults.coding_automation_efficiency_slope),
        optimal_ces_eta_init=frontend_params.get('optimal_ces_eta_init', sw_defaults.optimal_ces_eta_init),
        optimal_ces_grid_size=frontend_params.get('optimal_ces_grid_size', sw_defaults.optimal_ces_grid_size),
        optimal_ces_frontier_tail_eps=frontend_params.get('optimal_ces_frontier_tail_eps', sw_defaults.optimal_ces_frontier_tail_eps),
        optimal_ces_frontier_cap=frontend_params.get('optimal_ces_frontier_cap', sw_defaults.optimal_ces_frontier_cap),
        max_serial_coding_labor_multiplier=frontend_params.get('max_serial_coding_labor_multiplier', sw_defaults.max_serial_coding_labor_multiplier),
        # AI research taste parameters
        ai_research_taste_at_coding_automation_anchor_sd=frontend_params.get('ai_research_taste_at_coding_automation_anchor_sd', sw_defaults.ai_research_taste_at_coding_automation_anchor_sd),
        ai_research_taste_slope=frontend_params.get('ai_research_taste_slope', sw_defaults.ai_research_taste_slope),
        taste_schedule_type=frontend_params.get('taste_schedule_type', sw_defaults.taste_schedule_type),
        median_to_top_taste_multiplier=frontend_params.get('median_to_top_taste_multiplier', sw_defaults.median_to_top_taste_multiplier),
        top_percentile=frontend_params.get('top_percentile', sw_defaults.top_percentile),
        taste_limit=frontend_params.get('taste_limit', sw_defaults.taste_limit),
        taste_limit_smoothing=frontend_params.get('taste_limit_smoothing', sw_defaults.taste_limit_smoothing),
        # Horizon/milestone parameters
        progress_at_aa=frontend_params.get('progress_at_aa', sw_defaults.progress_at_aa),
        ac_time_horizon_minutes=ac_time_horizon,
        pre_gap_ac_time_horizon=frontend_params.get('pre_gap_ac_time_horizon', sw_defaults.pre_gap_ac_time_horizon),
        horizon_extrapolation_type=frontend_params.get('horizon_extrapolation_type', sw_defaults.horizon_extrapolation_type),
        # Manual horizon fitting
        present_day=frontend_params.get('present_day', sw_defaults.present_day),
        present_horizon=frontend_params.get('present_horizon', sw_defaults.present_horizon),
        present_doubling_time=frontend_params.get('present_doubling_time', sw_defaults.present_doubling_time),
        doubling_difficulty_growth_factor=frontend_params.get('doubling_difficulty_growth_factor', sw_defaults.doubling_difficulty_growth_factor),
        # Milestone multipliers
        strat_ai_m2b=frontend_params.get('strat_ai_m2b', sw_defaults.strat_ai_m2b),
        ted_ai_m2b=frontend_params.get('ted_ai_m2b', sw_defaults.ted_ai_m2b),
        # Gap mode
        include_gap=frontend_params.get('include_gap', sw_defaults.include_gap),
        gap_years=frontend_params.get('gap_years', sw_defaults.gap_years),
    )

    # Build compute growth parameters
    cg_defaults = default_sim_params.compute_growth
    compute_growth = ComputeParameters(
        us_frontier_project_compute_growth_rate=frontend_params.get('us_frontier_project_compute_growth_rate', cg_defaults.us_frontier_project_compute_growth_rate),
        slowdown_year=frontend_params.get('slowdown_year', cg_defaults.slowdown_year),
        post_slowdown_training_compute_growth_rate=frontend_params.get('post_slowdown_training_compute_growth_rate', cg_defaults.post_slowdown_training_compute_growth_rate),
        initial_hazard_rate=frontend_params.get('initial_hazard_rate', cg_defaults.initial_hazard_rate),
        hazard_rate_increase_per_year=frontend_params.get('hazard_rate_increase_per_year', cg_defaults.hazard_rate_increase_per_year),
        # PRC compute stock
        total_prc_compute_stock_in_2025=cg_defaults.total_prc_compute_stock_in_2025,
        annual_growth_rate_of_prc_compute_stock_p10=cg_defaults.annual_growth_rate_of_prc_compute_stock_p10,
        annual_growth_rate_of_prc_compute_stock_p50=cg_defaults.annual_growth_rate_of_prc_compute_stock_p50,
        annual_growth_rate_of_prc_compute_stock_p90=cg_defaults.annual_growth_rate_of_prc_compute_stock_p90,
        # PRC domestic production
        proportion_of_prc_chip_stock_produced_domestically_2026=cg_defaults.proportion_of_prc_chip_stock_produced_domestically_2026,
        proportion_of_prc_chip_stock_produced_domestically_2030=cg_defaults.proportion_of_prc_chip_stock_produced_domestically_2030,
        # US frontier project
        us_frontier_project_h100e_in_2025=cg_defaults.us_frontier_project_h100e_in_2025,
    )

    # Build energy consumption parameters (use defaults)
    ec_defaults = default_sim_params.energy_consumption
    energy_consumption = EnergyConsumptionParameters(
        energy_efficiency_of_prc_stock_relative_to_state_of_the_art=ec_defaults.energy_efficiency_of_prc_stock_relative_to_state_of_the_art,
        architecture_efficiency_improvement_per_year=ec_defaults.architecture_efficiency_improvement_per_year,
        total_GW_of_PRC_energy_consumption=ec_defaults.total_GW_of_PRC_energy_consumption,
        largest_ai_project_energy_efficiency_improvement_per_year=ec_defaults.largest_ai_project_energy_efficiency_improvement_per_year,
    )

    # Build policy parameters (use defaults from sampled params)
    policy = default_sim_params.policy

    return SimulationParameters(
        settings=settings,
        software_r_and_d=software_r_and_d,
        compute_growth=compute_growth,
        energy_consumption=energy_consumption,
        policy=policy,
    )
