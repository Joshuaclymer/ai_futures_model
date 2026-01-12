"""
Calibration Trajectory Computation

This module computes software progress trajectories for parameter calibration.
It consolidates the essential logic from ProgressModel and integration functions
into a focused calibration-specific implementation.

The trajectory computation:
1. Calibrates experiment capacity CES parameters from anchors
2. Runs human-only trajectory to calibrate r_software
3. Solves for automation anchor from swe_multiplier
4. Estimates horizon trajectory from benchmark data
5. Runs main ODE with automation feedback
6. Extracts trajectory data for later interpolation
"""

from pathlib import Path
import copy
import logging
import numpy as np
from scipy import integrate
from scipy.integrate import cumulative_trapezoid
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime
import yaml
from dataclasses import dataclass, field

# Path to software_r_and_d for loading benchmark data
SOFTWARE_R_AND_D_DIR = Path(__file__).resolve().parent.parent.parent / "world_updaters" / "software_r_and_d"

# Import from software_r_and_d modules
from world_updaters.software_r_and_d.data_types import TimeSeriesData
from world_updaters.software_r_and_d.automation_model import AutomationModel, solve_lower_anchor_via_automation_model
from world_updaters.software_r_and_d.taste_distribution import TasteDistribution
from world_updaters.software_r_and_d.progress_rate import (
    compute_research_effort,
    compute_software_progress_rate,
    compute_automation_fraction,
    compute_ai_research_taste,
    compute_aggregate_research_taste,
    progress_rate_at_time,
)
from world_updaters.software_r_and_d.ces_functions import compute_exp_capacity_params_from_anchors
from world_updaters.software_r_and_d.utils import _log_interp

logger = logging.getLogger(__name__)


@dataclass
class CalibrationParams:
    """
    Container for parameters during calibration.

    This is a mutable dataclass that holds all parameters needed for
    trajectory computation during calibration.
    """
    # CES parameters
    rho_coding_labor: float = -2.0
    parallel_penalty: float = 0.5
    coding_labor_normalization: float = 1.0
    rho_experiment_capacity: float = 0.5
    alpha_experiment_capacity: float = 0.5
    experiment_compute_exponent: float = 0.5

    # Software progress
    r_software: float = 1.0
    software_progress_rate_at_reference_year: float = 1.0

    # Experiment capacity anchors (for calibration)
    # Default values match reference model (model_config.py DEFAULT_PARAMETERS)
    direct_input_exp_cap_ces_params: bool = False
    inf_labor_asymptote: float = 15.0
    inf_compute_asymptote: float = 1000.0
    labor_anchor_exp_cap: float = 1.6
    compute_anchor_exp_cap: float = 1.0
    inv_compute_anchor_exp_cap: float = 2.8

    # Automation
    automation_interp_type: str = "linear"
    automation_logistic_asymptote: float = 1.0
    automation_fraction_at_coding_automation_anchor: float = 1.0  # Match original model (full automation at AC)
    automation_anchors: Dict[float, float] = field(default_factory=dict)
    automation_model: Optional[AutomationModel] = None
    swe_multiplier_at_present_day: float = 1.8

    # Optimal CES frontier parameters
    optimal_ces_eta_init: float = 0.05
    optimal_ces_grid_size: int = 4096
    optimal_ces_frontier_tail_eps: float = 1e-6
    optimal_ces_frontier_cap: Optional[float] = 1e12
    max_serial_coding_labor_multiplier: Optional[float] = None

    # Taste parameters (match reference model model_config.py DEFAULT_PARAMETERS)
    median_to_top_taste_multiplier: float = 3.7  # MEDIAN_TO_TOP_TASTE_MULTIPLIER
    top_percentile: float = 0.999  # TOP_PERCENTILE
    taste_limit: float = 8.0  # taste_limit_m2b exponent factor
    taste_limit_smoothing: float = 0.5  # Match reference model
    taste_distribution: Optional[TasteDistribution] = None
    ai_research_taste_slope: float = 2.1  # Match reference model (model_config.py DEFAULT_PARAMETERS)
    ai_research_taste_at_coding_automation_anchor_sd: float = 0.5  # Match reference model
    coding_automation_efficiency_slope: float = 3.0

    # Horizon parameters
    progress_at_aa: float = 100.0
    present_day: float = 2025.9  # Match reference model (model_config.py DEFAULT_PARAMETERS)
    present_horizon: Optional[float] = None
    present_doubling_time: Optional[float] = None
    doubling_difficulty_growth_factor: Optional[float] = None
    ac_time_horizon_minutes: Optional[float] = None
    horizon_extrapolation_type: str = "exponential"

    # Mode flags
    human_only: bool = False

    def __post_init__(self):
        """Create runtime objects if not provided."""
        # Convert inv_compute_anchor_exp_cap to compute_anchor_exp_cap
        # (matching original model's Parameters.__post_init__)
        if self.inv_compute_anchor_exp_cap is not None and self.inv_compute_anchor_exp_cap > 0:
            self.compute_anchor_exp_cap = 1.0 / self.inv_compute_anchor_exp_cap

        if self.taste_distribution is None:
            self.taste_distribution = TasteDistribution(
                top_percentile=self.top_percentile,
                median_to_top_gap=self.median_to_top_taste_multiplier,
                taste_limit_m2b=self.taste_limit,
                taste_limit_smoothing=self.taste_limit_smoothing,
            )


# Type alias for backward compatibility
Parameters = CalibrationParams

# Calibration constants (previously from model_config.py)
DENSE_OUTPUT_STEP_SIZE = 0.1  # Output grid spacing for trajectories (years)
ODE_MAX_STEP = 1.0  # Max step size for ODE solver (years)
PARAM_CLIP_MIN = 1e-6  # Minimum value for numerical stability
REFERENCE_COMPUTE_CHANGE = 0.1  # Anchor for experiment capacity calibration
REFERENCE_LABOR_CHANGE = 30.0  # Anchor for experiment capacity calibration
SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR = 2024.0  # Year for r_software calibration

# Cache for benchmark data
_BENCHMARK_DATA_CACHE = None


def _load_benchmark_data() -> Dict:
    """Load benchmark_results.yaml with caching."""
    global _BENCHMARK_DATA_CACHE
    if _BENCHMARK_DATA_CACHE is None:
        try:
            benchmark_path = SOFTWARE_R_AND_D_DIR / "benchmark_results.yaml"
            with open(benchmark_path, 'r') as f:
                _BENCHMARK_DATA_CACHE = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("benchmark_results.yaml not found")
            _BENCHMARK_DATA_CACHE = {}
        except Exception as e:
            logger.warning(f"Error loading benchmark_results.yaml: {e}")
            _BENCHMARK_DATA_CACHE = {}
    return _BENCHMARK_DATA_CACHE


def _integrate_human_only(
    time_range: List[float],
    initial_progress: float,
    initial_research_stock: float,
    data: TimeSeriesData,
    params: Parameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast vectorized integration for human-only trajectory.

    In human-only mode there's no feedback loop, so we can use vectorized
    cumulative trapezoid instead of ODE solver.
    """
    t_start, t_end = time_range

    # Calculate grid size
    time_span = t_end - t_start
    num_output_points = max(2, int(np.ceil(time_span / DENSE_OUTPUT_STEP_SIZE)) + 1)

    # Fine internal grid for accuracy
    internal_multiplier = 300
    num_internal_points = (num_output_points - 1) * internal_multiplier + 1
    internal_times = np.linspace(t_start, t_end, num_internal_points)

    # Interpolate time series on fine grid
    if data.can_use_log_L_HUMAN:
        L_HUMAN = np.exp(np.interp(internal_times, data.time, data.log_L_HUMAN))
    else:
        L_HUMAN = np.interp(internal_times, data.time, data.L_HUMAN)

    if data.can_use_log_experiment_compute:
        experiment_compute = np.exp(np.interp(internal_times, data.time, data.log_experiment_compute))
    else:
        experiment_compute = np.interp(internal_times, data.time, data.experiment_compute)

    training_compute_growth_rate = data.get_training_compute_growth_rate(internal_times)

    # Compute coding labor (human only)
    serial_coding_labor = np.power(L_HUMAN, params.parallel_penalty) * params.coding_labor_normalization

    # Compute research effort
    research_effort = compute_research_effort(
        experiment_compute,
        serial_coding_labor,
        params.alpha_experiment_capacity,
        params.rho_experiment_capacity,
        params.experiment_compute_exponent,
        1.0  # aggregate_research_taste = 1 for human-only
    )
    research_effort = np.where(np.isfinite(research_effort), research_effort, 0.0)
    research_effort = np.maximum(research_effort, 0.0)

    # Integrate research stock
    cumulative_rs = cumulative_trapezoid(research_effort, internal_times, initial=0)
    research_stock = initial_research_stock + cumulative_rs
    research_stock = np.maximum(research_stock, 1e-6)

    # Compute software progress rate
    sw_progress_rate = compute_software_progress_rate(research_stock, research_effort, params.r_software)
    sw_progress_rate = np.where(np.isfinite(sw_progress_rate), sw_progress_rate, 0.0)
    sw_progress_rate = np.maximum(sw_progress_rate, 0.0)

    # Integrate progress
    progress_rate = sw_progress_rate + training_compute_growth_rate
    cumulative_progress = cumulative_trapezoid(progress_rate, internal_times, initial=0)
    progress_values = initial_progress + cumulative_progress

    # Downsample to output grid
    output_indices = np.arange(0, num_internal_points, internal_multiplier)

    return (
        internal_times[output_indices],
        progress_values[output_indices],
        research_stock[output_indices],
        research_effort[output_indices],
        sw_progress_rate[output_indices]
    )


def _integrate_with_automation(
    time_range: List[float],
    initial_progress: float,
    initial_research_stock: float,
    data: TimeSeriesData,
    params: Parameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ODE integration with automation feedback.

    Uses scipy.integrate.solve_ivp with robust method selection.
    """
    def ode_func(t, y):
        try:
            rates = progress_rate_at_time(
                t, y, data,
                rho_coding_labor=params.rho_coding_labor,
                parallel_penalty=params.parallel_penalty,
                coding_labor_normalization=params.coding_labor_normalization,
                alpha_experiment_capacity=params.alpha_experiment_capacity,
                rho_experiment_capacity=params.rho_experiment_capacity,
                experiment_compute_exponent=params.experiment_compute_exponent,
                r_software=params.r_software,
                automation_model=params.automation_model,
                taste_distribution=params.taste_distribution,
                ai_research_taste_slope=params.ai_research_taste_slope,
                progress_at_aa=params.progress_at_aa,
                ai_research_taste_at_coding_automation_anchor_sd=params.ai_research_taste_at_coding_automation_anchor_sd,
                human_only=False,
            )
            if len(rates) != 2 or not all(np.isfinite(rate) and rate >= 0 for rate in rates):
                return [0.0, 0.0]
            return rates
        except Exception:
            return [0.0, 0.0]

    t_start, t_end = time_range
    initial_state = [initial_progress, initial_research_stock]

    # Try multiple integration methods
    methods = [
        ('Radau', {'rtol': 1e-3, 'atol': 1e-5}),
        ('RK23', {'rtol': 1e-4, 'atol': 1e-6}),
        ('RK45', {'rtol': 1e-4, 'atol': 1e-6}),
        ('LSODA', {'rtol': 1e-3, 'atol': 1e-5}),
    ]

    sol = None
    for method, tolerances in methods:
        try:
            sol = integrate.solve_ivp(
                ode_func,
                [t_start, t_end],
                initial_state,
                method=method,
                dense_output=True,
                max_step=ODE_MAX_STEP,
                **tolerances
            )
            if sol.success:
                break
        except Exception as e:
            logger.debug(f"Integration method {method} failed: {e}")
            continue

    if sol is None or not sol.success:
        raise RuntimeError("All integration methods failed")

    # Create dense output
    time_span = t_end - t_start
    num_points = max(2, int(np.ceil(time_span / DENSE_OUTPUT_STEP_SIZE)) + 1)
    times = np.linspace(t_start, t_end, num_points)
    solution = sol.sol(times)

    return times, solution[0], solution[1]


def _compute_initial_research_stock(
    data: TimeSeriesData,
    params: Parameters,
    initial_progress: float = 0.0
) -> float:
    """
    Calculate initial research stock using RS(0) = (RS'(0))^2 / RS''(0).
    """
    start_time = data.time[0]
    dt = 1e-6

    L_HUMAN_0 = _log_interp(start_time, data.time, data.L_HUMAN)
    experiment_compute_0 = _log_interp(start_time, data.time, data.experiment_compute)
    coding_labor_0 = L_HUMAN_0 ** params.parallel_penalty

    rs_rate_0 = compute_research_effort(
        experiment_compute_0, coding_labor_0,
        params.alpha_experiment_capacity, params.rho_experiment_capacity,
        params.experiment_compute_exponent, 1.0
    )

    L_HUMAN_dt = _log_interp(start_time + dt, data.time, data.L_HUMAN)
    experiment_compute_dt = _log_interp(start_time + dt, data.time, data.experiment_compute)
    coding_labor_dt = L_HUMAN_dt ** params.parallel_penalty

    rs_rate_dt = compute_research_effort(
        experiment_compute_dt, coding_labor_dt,
        params.alpha_experiment_capacity, params.rho_experiment_capacity,
        params.experiment_compute_exponent, 1.0
    )

    rs_rate_second_derivative = (rs_rate_dt - rs_rate_0) / dt

    if abs(rs_rate_second_derivative) < PARAM_CLIP_MIN:
        return max(PARAM_CLIP_MIN, rs_rate_0)

    initial_research_stock = (rs_rate_0 ** 2) / rs_rate_second_derivative

    if not np.isfinite(initial_research_stock) or initial_research_stock <= 0:
        return max(PARAM_CLIP_MIN, rs_rate_0)

    return initial_research_stock


def _estimate_horizon_trajectory(
    params: Parameters,
    human_only_times: np.ndarray,
    human_only_progress: np.ndarray,
    anchor_progress_rate: float,
) -> Optional[Callable[[float], float]]:
    """
    Estimate horizon trajectory function from benchmark data and/or parameters.

    Returns a function that maps progress -> horizon_length (minutes).
    """
    # Check if we need benchmark data
    needs_benchmark = False
    if params.horizon_extrapolation_type == "exponential":
        if params.present_horizon is None or params.present_doubling_time is None:
            needs_benchmark = True
    elif params.horizon_extrapolation_type == "decaying doubling time":
        if (params.present_horizon is None or params.present_doubling_time is None
            or params.doubling_difficulty_growth_factor is None):
            needs_benchmark = True

    progress_values = np.array([])
    horizon_values = np.array([])

    if needs_benchmark:
        benchmark_data = _load_benchmark_data()
        if not benchmark_data or 'results' not in benchmark_data:
            logger.warning("Cannot estimate horizon trajectory: no benchmark data")
            return None

        pairs = []
        for model_name, model_info in benchmark_data['results'].items():
            release_date_obj = model_info.get('release_date')
            if not release_date_obj:
                continue
            try:
                if isinstance(release_date_obj, str):
                    release_date = datetime.strptime(release_date_obj, '%Y-%m-%d').date()
                else:
                    release_date = release_date_obj
                decimal_year = release_date.year + (release_date.timetuple().tm_yday - 1) / 365.25
            except (ValueError, AttributeError):
                continue

            # Interpolate progress at release date
            if human_only_times.min() <= decimal_year <= human_only_times.max():
                progress = np.interp(decimal_year, human_only_times, human_only_progress)
            elif decimal_year < human_only_times.min():
                progress = human_only_progress[0]
            else:
                progress = human_only_progress[-1]

            # Get horizon from metrics
            if 'metrics' in model_info and 'p80_horizon_length' in model_info['metrics']:
                horizon = model_info['metrics']['p80_horizon_length'].get('estimate')
                if horizon is not None and horizon > 0:
                    pairs.append((progress, horizon))

        if len(pairs) < 2:
            logger.warning("Not enough benchmark data points for horizon trajectory")
            return None

        progress_values = np.array([p[0] for p in pairs])
        horizon_values = np.array([p[1] for p in pairs])

    # Build the trajectory function based on extrapolation type
    if params.horizon_extrapolation_type == "exponential":
        return _build_exponential_horizon_trajectory(
            params, progress_values, horizon_values,
            human_only_times, human_only_progress, anchor_progress_rate
        )
    elif params.horizon_extrapolation_type == "decaying doubling time":
        return _build_decaying_horizon_trajectory(
            params, progress_values, horizon_values,
            human_only_times, human_only_progress, anchor_progress_rate
        )

    return None


def _build_exponential_horizon_trajectory(
    params: Parameters,
    progress_values: np.ndarray,
    horizon_values: np.ndarray,
    human_only_times: np.ndarray,
    human_only_progress: np.ndarray,
    anchor_progress_rate: float,
) -> Callable[[float], float]:
    """Build exponential horizon trajectory: log(H) = a*progress + b."""
    from scipy.optimize import curve_fit

    present_day = params.present_day
    present_day_progress = np.interp(present_day, human_only_times, human_only_progress)

    if params.present_horizon is not None and params.present_doubling_time is not None:
        # Use provided parameters
        H_0 = params.present_horizon
        doubling_time = params.present_doubling_time
        # Convert doubling time from years to progress units
        doubling_progress = doubling_time * anchor_progress_rate
        # slope = ln(2) / doubling_progress
        slope = np.log(2) / doubling_progress if doubling_progress > 0 else 0.1

        def horizon_func(progress):
            return H_0 * np.exp(slope * (progress - present_day_progress))
        return horizon_func

    # Fit from data
    if len(progress_values) < 2:
        return lambda p: 60.0  # Default 1 hour

    log_horizon = np.log(horizon_values)

    def linear_func(x, a, b):
        return a * x + b

    try:
        popt, _ = curve_fit(linear_func, progress_values, log_horizon)
        slope, intercept = popt

        def horizon_func(progress):
            return np.exp(slope * progress + intercept)
        return horizon_func
    except Exception as e:
        logger.warning(f"Failed to fit horizon trajectory: {e}")
        return lambda p: 60.0


def _build_decaying_horizon_trajectory(
    params: Parameters,
    progress_values: np.ndarray,
    horizon_values: np.ndarray,
    human_only_times: np.ndarray,
    human_only_progress: np.ndarray,
    anchor_progress_rate: float,
) -> Callable[[float], float]:
    """Build decaying doubling time horizon trajectory."""
    present_day = params.present_day
    present_day_progress = np.interp(present_day, human_only_times, human_only_progress)

    H_0 = params.present_horizon if params.present_horizon is not None else 60.0
    doubling_time = params.present_doubling_time if params.present_doubling_time is not None else 1.0

    # A_0 is the decay parameter = 1 - doubling_difficulty_growth_factor
    # (matching original model: A_0 = 1 - doubling_difficulty_growth_factor)
    if params.doubling_difficulty_growth_factor is not None:
        A_0 = 1.0 - params.doubling_difficulty_growth_factor
    else:
        A_0 = 0.08  # Default for growth factor = 0.92

    # Convert doubling time to progress units
    T_0 = doubling_time * anchor_progress_rate if anchor_progress_rate > 0 else doubling_time

    def horizon_func(progress):
        delta = progress - present_day_progress
        if abs(A_0) < 1e-10:
            # No decay case: exponential growth
            return H_0 * np.exp(np.log(2) * delta / T_0) if T_0 > 0 else H_0

        # Decaying doubling time formula (matching frontend's computeHorizonFromProgress):
        # H(p) = H_0 * (1 - A_0 * (p - p_0) / T_0)^(ln(2) / ln(1-A_0))
        factor = 1 - A_0 * delta / T_0
        if isinstance(factor, np.ndarray):
            factor = np.maximum(factor, 1e-12)
        else:
            factor = max(factor, 1e-12)

        # Exponent = ln(2) / ln(1 - A_0)
        log_denominator = np.log(1 - A_0) if abs(1 - A_0) > 1e-10 else -1e-10
        exponent = np.log(2) / log_denominator

        return H_0 * np.power(factor, exponent)

    return horizon_func


class CalibrationTrajectoryResult:
    """Container for calibration trajectory results."""

    def __init__(
        self,
        params: Parameters,
        times: np.ndarray,
        progress: np.ndarray,
        research_stock: np.ndarray,
        automation_fractions: np.ndarray,
        horizon_lengths: Optional[np.ndarray],
        ai_sw_progress_mult: np.ndarray,
        horizon_trajectory: Optional[Callable],
        progress_at_aa: float,
    ):
        self.params = params
        self.times = times
        self.progress = progress
        self.research_stock = research_stock
        self.automation_fractions = automation_fractions
        self.horizon_lengths = horizon_lengths
        self.ai_sw_progress_mult = ai_sw_progress_mult
        self.horizon_trajectory = horizon_trajectory
        self.progress_at_aa = progress_at_aa


def compute_calibration_trajectory(
    params: Parameters,
    data: TimeSeriesData,
    time_range: List[float],
    initial_progress: float = 0.0,
) -> CalibrationTrajectoryResult:
    """
    Compute a calibration trajectory with all necessary outputs.

    This replaces ProgressModel.compute_progress_trajectory() for calibration purposes.

    Args:
        params: Model parameters (will be modified with calibrated values)
        data: Historical time series data
        time_range: [start_time, end_time]
        initial_progress: Initial progress value (default 0.0)

    Returns:
        CalibrationTrajectoryResult with calibrated params and trajectory data
    """
    # Make a copy of params to modify
    params = copy.deepcopy(params)

    # Step 1: Compute experiment capacity CES params from anchors
    if not params.direct_input_exp_cap_ces_params:
        present_day = params.present_day
        ref_exp_compute = _log_interp(present_day, data.time, data.experiment_compute)
        ref_coding_labor = _log_interp(present_day, data.time, data.L_HUMAN)

        rho, alpha, exp_exponent = compute_exp_capacity_params_from_anchors(
            params.inf_labor_asymptote,
            params.inf_compute_asymptote,
            (REFERENCE_COMPUTE_CHANGE, params.compute_anchor_exp_cap),
            (REFERENCE_LABOR_CHANGE, params.labor_anchor_exp_cap),
            ref_exp_compute, ref_coding_labor,
            params.parallel_penalty
        )
        params.rho_experiment_capacity = rho
        params.alpha_experiment_capacity = alpha
        params.experiment_compute_exponent = exp_exponent

    # Step 2: Compute initial research stock
    initial_research_stock = _compute_initial_research_stock(data, params, initial_progress)

    # Step 3: Run short human-only trajectory to calibrate r_software
    human_params = copy.deepcopy(params)
    human_params.human_only = True

    short_end = min(params.present_day + 1.0, time_range[1])
    short_range = [time_range[0], short_end]

    times_h, progress_h, research_stock_h, research_effort_h, sw_rate_h = _integrate_human_only(
        short_range, initial_progress, initial_research_stock, data, human_params
    )

    # Get reference sw_progress_rate and calibrate r_software
    reference_sw_rate = np.interp(SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR, times_h, sw_rate_h)
    if reference_sw_rate > 0:
        params.r_software = params.software_progress_rate_at_reference_year * params.r_software / reference_sw_rate

    # Step 4: Recompute human-only trajectory with calibrated r_software for full range
    human_params.r_software = params.r_software
    times_h, progress_h, research_stock_h, research_effort_h, sw_rate_h = _integrate_human_only(
        time_range, initial_progress, initial_research_stock, data, human_params
    )

    # Get anchor stats
    present_day = params.present_day
    present_day_progress = np.interp(present_day, times_h, progress_h)
    present_day_human_labor = _log_interp(present_day, data.time, data.L_HUMAN)
    present_day_inference_compute = _log_interp(present_day, data.time, data.inference_compute)
    present_day_sw_rate = np.interp(present_day, times_h, sw_rate_h)
    present_day_research_stock = _log_interp(present_day, times_h, research_stock_h)
    anchor_progress_rate = np.interp(present_day, times_h, sw_rate_h + data.get_training_compute_growth_rate(times_h))

    # Step 5: Estimate horizon trajectory
    horizon_trajectory = _estimate_horizon_trajectory(
        params, times_h, progress_h, anchor_progress_rate
    )

    # Step 5b: Compute progress_at_aa from horizon trajectory (where horizon reaches AC threshold)
    # This must happen BEFORE setting automation anchors
    if horizon_trajectory is not None and params.ac_time_horizon_minutes is not None:
        target_horizon = params.ac_time_horizon_minutes
        # Search for progress level where horizon reaches target
        # Start from present_day_progress and search upward
        progress_search = np.linspace(present_day_progress, present_day_progress + 50, 1000)
        horizons = np.array([horizon_trajectory(p) for p in progress_search])
        logger.debug(f"Step 5b: present_day_progress={present_day_progress:.4f}, anchor_progress_rate={anchor_progress_rate:.4f}")
        logger.debug(f"Step 5b: horizon at present_day_progress={horizon_trajectory(present_day_progress):.2f}, at +7 progress={horizon_trajectory(present_day_progress+7):.2e}")
        # Find first progress where horizon >= target_horizon
        above_threshold = horizons >= target_horizon
        if np.any(above_threshold):
            idx = np.argmax(above_threshold)
            computed_progress_at_aa = float(progress_search[idx])
            params.progress_at_aa = computed_progress_at_aa
            logger.debug(f"Computed progress_at_aa from horizon: {computed_progress_at_aa:.4f} (target horizon: {target_horizon} min)")
        else:
            # Horizon doesn't reach threshold in search range; use last value
            params.progress_at_aa = float(progress_search[-1])
            logger.debug(f"Horizon never reaches {target_horizon} min in search range; using {params.progress_at_aa:.4f}")

    # Step 6: Convert slopes from per-progress-year to per-progress-unit
    if anchor_progress_rate > 0:
        params.ai_research_taste_slope = params.ai_research_taste_slope / anchor_progress_rate
        params.coding_automation_efficiency_slope = params.coding_automation_efficiency_slope / anchor_progress_rate

    # Step 7: Solve for automation anchor
    anchor_aut_frac = solve_lower_anchor_via_automation_model(
        params.swe_multiplier_at_present_day,
        float(present_day_progress),
        float(present_day_human_labor),
        float(present_day_inference_compute),
        params,
    )

    params.automation_anchors = {
        present_day_progress: anchor_aut_frac,
        params.progress_at_aa: params.automation_fraction_at_coding_automation_anchor
    }
    params.automation_model = AutomationModel(
        params.automation_interp_type,
        params.automation_anchors,
        params.automation_logistic_asymptote
    )

    # Step 8: Run main trajectory with automation
    times, progress, research_stock = _integrate_with_automation(
        time_range, initial_progress, initial_research_stock, data, params
    )

    # Step 9: Compute derived metrics for each time point
    n_points = len(times)
    automation_fractions = np.zeros(n_points)
    ai_sw_progress_mult = np.ones(n_points)
    horizon_lengths = np.zeros(n_points) if horizon_trajectory else None

    for i in range(n_points):
        p = progress[i]
        automation_fractions[i] = compute_automation_fraction(p, params.automation_model)

        if horizon_trajectory:
            try:
                horizon_lengths[i] = horizon_trajectory(p)
            except Exception:
                horizon_lengths[i] = np.nan

        # Compute ai_sw_progress_mult = current_sw_rate / present_day_sw_rate
        # This requires computing research effort and sw_rate at current progress
        t = times[i]
        rs = research_stock[i]

        if data.can_use_log_experiment_compute:
            exp_compute = np.exp(np.interp(t, data.time, data.log_experiment_compute))
        else:
            exp_compute = np.interp(t, data.time, data.experiment_compute)

        if data.can_use_log_L_HUMAN:
            L_HUMAN = np.exp(np.interp(t, data.time, data.log_L_HUMAN))
        else:
            L_HUMAN = np.interp(t, data.time, data.L_HUMAN)

        if data.can_use_log_inference_compute:
            inf_compute = np.exp(np.interp(t, data.time, data.log_inference_compute))
        else:
            inf_compute = np.interp(t, data.time, data.inference_compute)

        # Compute research effort with present-day resources but current automation
        ai_taste = compute_ai_research_taste(
            p,
            params.taste_distribution,
            params.ai_research_taste_slope,
            params.progress_at_aa,
            params.ai_research_taste_at_coding_automation_anchor_sd,
        )
        agg_taste = compute_aggregate_research_taste(ai_taste, params.taste_distribution)

        # For sw_mult calculation, use present-day resources
        coding_labor = (L_HUMAN ** params.parallel_penalty) * params.coding_labor_normalization
        # Adjust for automation
        auto_frac = automation_fractions[i]
        if auto_frac > 0:
            # Simple approximation for coding labor with automation
            coding_labor = coding_labor * (1 + auto_frac * (inf_compute / L_HUMAN - 1)) if L_HUMAN > 0 else coding_labor

        research_effort = compute_research_effort(
            exp_compute, coding_labor,
            params.alpha_experiment_capacity, params.rho_experiment_capacity,
            params.experiment_compute_exponent, agg_taste
        )

        # Use present-day research stock for the multiplier calculation
        sw_rate = compute_software_progress_rate(present_day_research_stock, research_effort, params.r_software)

        if present_day_sw_rate > 0:
            ai_sw_progress_mult[i] = sw_rate / present_day_sw_rate

    # Determine progress_at_aa (where horizon reaches AC threshold)
    progress_at_aa = params.progress_at_aa
    if horizon_trajectory and params.ac_time_horizon_minutes:
        # Find where horizon crosses AC threshold
        for i in range(len(progress)):
            if horizon_lengths is not None and horizon_lengths[i] >= params.ac_time_horizon_minutes:
                progress_at_aa = progress[i]
                break

    return CalibrationTrajectoryResult(
        params=params,
        times=times,
        progress=progress,
        research_stock=research_stock,
        automation_fractions=automation_fractions,
        horizon_lengths=horizon_lengths,
        ai_sw_progress_mult=ai_sw_progress_mult,
        horizon_trajectory=horizon_trajectory,
        progress_at_aa=progress_at_aa,
    )
