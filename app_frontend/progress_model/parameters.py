#!/usr/bin/env python3
"""
Parameters Module

Contains the Parameters dataclass for model configuration with validation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Any
import logging

import model_config as cfg
from .utils import _coerce_float_scalar, should_reraise
from .taste_distribution import TasteDistribution, get_or_create_taste_distribution
from .automation_model import AutomationModel

logger = logging.getLogger(__name__)


@dataclass
class Parameters:
    """Model parameters with validation"""

    human_only: bool = field(default_factory=lambda: False)

    # Production function parameters
    rho_coding_labor: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_coding_labor'])
    direct_input_exp_cap_ces_params: bool = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['direct_input_exp_cap_ces_params'])
    rho_experiment_capacity: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['rho_experiment_capacity'])
    alpha_experiment_capacity: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['alpha_experiment_capacity'])
    r_software: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['r_software'])
    software_progress_rate_at_reference_year: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['software_progress_rate_at_reference_year'])
    experiment_compute_exponent: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['experiment_compute_exponent'])

    # Automation parameters
    automation_fraction_at_coding_automation_anchor: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_fraction_at_coding_automation_anchor'])
    automation_anchors: Optional[Dict[float, float]] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_anchors'])
    automation_model: Optional[AutomationModel] = field(default_factory=lambda: None)
    automation_interp_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_interp_type'])
    automation_logistic_asymptote: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['automation_logistic_asymptote'])
    swe_multiplier_at_present_day: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['swe_multiplier_at_present_day'])
    # AI Research Taste sigmoid parameters
    ai_research_taste_at_coding_automation_anchor: float = field(default=None, init=False)  # Computed from _sd
    # Always specify the superhuman-coder taste as SD within the human range
    ai_research_taste_at_coding_automation_anchor_sd: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_coding_automation_anchor_sd', 0.5))
    ai_research_taste_slope: float = field(default_factory=lambda: cfg.TASTE_SLOPE_DEFAULTS.get(cfg.DEFAULT_TASTE_SCHEDULE_TYPE, cfg.DEFAULT_PARAMETERS['ai_research_taste_slope']))
    taste_schedule_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['taste_schedule_type'])
    progress_at_aa: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('progress_at_aa'))
    ac_time_horizon_minutes: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['ac_time_horizon_minutes'])
    # Pre-gap SC horizon minutes (formerly saturation_horizon_minutes)
    pre_gap_ac_time_horizon: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['pre_gap_ac_time_horizon'])
    horizon_extrapolation_type: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['horizon_extrapolation_type'])

    # Manual horizon fitting parameters
    present_day: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['present_day'])
    present_horizon: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['present_horizon'])
    present_doubling_time: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['present_doubling_time'])
    doubling_difficulty_growth_factor: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['doubling_difficulty_growth_factor'])

    # Normalization
    coding_labor_normalization: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['coding_labor_normalization'])

    # exp capacity pseudoparameters
    inf_labor_asymptote: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['inf_labor_asymptote'])
    inf_compute_asymptote: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['inf_compute_asymptote'])
    labor_anchor_exp_cap: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['labor_anchor_exp_cap'])
    compute_anchor_exp_cap: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['compute_anchor_exp_cap'])
    inv_compute_anchor_exp_cap: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['inv_compute_anchor_exp_cap'])
    # penalty on parallel coding labor in exp capacity CES
    parallel_penalty: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['parallel_penalty'])

    # Research taste distribution parameter
    median_to_top_taste_multiplier: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['median_to_top_taste_multiplier'])
    top_percentile: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['top_percentile'])
    taste_limit: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['taste_limit'])
    taste_limit_smoothing: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['taste_limit_smoothing'])
    taste_distribution: Optional[TasteDistribution] = field(default=None, init=False)  # Initialized in __post_init__
    # Median-to-best multipliers for general capability milestones
    strat_ai_m2b: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['strat_ai_m2b'])
    ted_ai_m2b: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['ted_ai_m2b'])

    # Benchmarks and gaps mode
    include_gap: Union[str, bool] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['include_gap'])
    gap_years: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['gap_years'])

    # Coding labor mode and optimal CES params (see AUTOMATION_SUGGESTION.md)
    coding_labor_mode: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('coding_labor_mode', 'simple_ces'))
    coding_automation_efficiency_slope: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('coding_automation_efficiency_slope', 1.0))
    optimal_ces_eta_init: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('optimal_ces_eta_init', 1.0))
    optimal_ces_grid_size: int = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('optimal_ces_grid_size', 4096))
    optimal_ces_frontier_tail_eps: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('optimal_ces_frontier_tail_eps', 1e-6))
    optimal_ces_frontier_cap: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('optimal_ces_frontier_cap'))
    max_serial_coding_labor_multiplier: Optional[float] = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS.get('max_serial_coding_labor_multiplier'))

    # SIE mode
    sos_mode: bool = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['sos_mode'])
    sos_start_milestone: str = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['sos_start_milestone'])

    # Plan A mode
    plan_a_mode: bool = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['plan_a_mode'])
    plan_a_start_time: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['plan_a_start_time'])
    show_blacksite: bool = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['show_blacksite'])
    sw_leaks_to_blacksite: bool = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['sw_leaks_to_blacksite'])
    is_blacksite: bool = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['is_blacksite'])
    main_site: Any = field(default_factory=lambda: None)
    main_project_training_compute_growth_rate: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['main_project_training_compute_growth_rate'])
    main_project_software_progress_rate: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['main_project_software_progress_rate'])
    blacksite_initial_years_behind: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['blacksite_initial_years_behind'])
    blacksite_initial_model_progress: float = field(default_factory=lambda: None)
    blacksite_training_compute_penalty_ooms: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['blacksite_training_compute_penalty_ooms'])
    blacksite_training_compute_growth_rate: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['blacksite_training_compute_growth_rate'])
    blacksite_human_labor_penalty_ooms: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['blacksite_human_labor_penalty_ooms'])
    blacksite_experiment_compute_penalty_ooms: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['blacksite_experiment_compute_penalty_ooms'])
    blacksite_inference_compute_penalty_ooms: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['blacksite_inference_compute_penalty_ooms'])
    blacksite_human_taste_penalty: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['blacksite_human_taste_penalty'])
    blacksite_can_stack_training_compute: bool = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['blacksite_can_stack_training_compute'])
    blacksite_can_stack_software_progress: bool = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['blacksite_can_stack_software_progress'])
    blacksite_start_time: float = field(default_factory=lambda: cfg.DEFAULT_PARAMETERS['blacksite_start_time'])

    def __post_init__(self):
        """Validate and sanitize parameters after initialization"""
        # Normalize taste parameters early so downstream numpy ops see scalars
        self.median_to_top_taste_multiplier = _coerce_float_scalar(
            self.median_to_top_taste_multiplier, "median_to_top_taste_multiplier")
        self.top_percentile = _coerce_float_scalar(self.top_percentile, "top_percentile")
        self.taste_limit = _coerce_float_scalar(self.taste_limit, "taste_limit")
        self.taste_limit_smoothing = _coerce_float_scalar(self.taste_limit_smoothing, "taste_limit_smoothing")

        # Sanitize elasticity parameters
        if not np.isfinite(self.rho_coding_labor):
            logger.warning(f"Non-finite rho_coding_labor: {self.rho_coding_labor}, setting to 0")
            self.rho_coding_labor = 0.0


        if not np.isfinite(self.r_software):
            logger.warning(f"Non-finite r_software: {self.r_software}, setting to 1.0")
            self.r_software = 1.0

        # Sanitize automation parameters
        if not np.isfinite(self.automation_fraction_at_coding_automation_anchor):
            logger.warning(f"Non-finite automation_fraction_at_coding_automation_anchor: {self.automation_fraction_at_coding_automation_anchor}, setting to {cfg.DEFAULT_PARAMETERS['automation_fraction_at_coding_automation_anchor']}")
            self.automation_fraction_at_coding_automation_anchor = cfg.DEFAULT_PARAMETERS['automation_fraction_at_coding_automation_anchor']

        # Sanitize AI research taste parameters
        # Always compute ai_research_taste_at_coding_automation_anchor from _sd
        if self.ai_research_taste_at_coding_automation_anchor_sd is None:
            self.ai_research_taste_at_coding_automation_anchor_sd = cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_coding_automation_anchor_sd', 0.5)

        if not np.isfinite(self.ai_research_taste_at_coding_automation_anchor_sd):
            logger.warning(f"Non-finite ai_research_taste_at_coding_automation_anchor_sd: {self.ai_research_taste_at_coding_automation_anchor_sd}, using default")
            self.ai_research_taste_at_coding_automation_anchor_sd = cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_coding_automation_anchor_sd', 0.5)

        # Initialize taste distribution (used throughout the model)
        # Use cached version to avoid expensive reinitialization
        try:
            self.taste_distribution = get_or_create_taste_distribution(
                top_percentile=self.top_percentile,
                median_to_top_gap=self.median_to_top_taste_multiplier,
                taste_limit=self.taste_limit,
                taste_limit_smoothing=self.taste_limit_smoothing
            )
        except Exception as e:
            logger.error(f"Failed to initialize TasteDistribution: {e}")
            raise

        # Convert SD to taste value using the taste distribution
        try:
            converted_taste = self.taste_distribution.get_taste_at_sd(float(self.ai_research_taste_at_coding_automation_anchor_sd))
            if np.isfinite(converted_taste):
                self.ai_research_taste_at_coding_automation_anchor = float(converted_taste)
            else:
                logger.warning(f"Converted taste is non-finite, using fallback")
                self.ai_research_taste_at_coding_automation_anchor = cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_coding_automation_anchor_fallback', 0.95)
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed converting ai_research_taste_at_coding_automation_anchor_sd to taste: {e}, using fallback")
            self.ai_research_taste_at_coding_automation_anchor = cfg.DEFAULT_PARAMETERS.get('ai_research_taste_at_coding_automation_anchor_fallback', 0.95)

        if not np.isfinite(self.ai_research_taste_slope):
            logger.warning(f"Non-finite ai_research_taste_slope: {self.ai_research_taste_slope}, setting to 1.0")
            self.ai_research_taste_slope = 1.0
        # Sanitize time horizon parameter
        if not np.isfinite(self.ac_time_horizon_minutes) or self.ac_time_horizon_minutes <= 0:
            logger.warning(f"Invalid ac_time_horizon_minutes: {self.ac_time_horizon_minutes}, setting to {cfg.DEFAULT_PARAMETERS['ac_time_horizon_minutes']}")
            self.ac_time_horizon_minutes = cfg.DEFAULT_PARAMETERS['ac_time_horizon_minutes']

        # Sanitize parallel_penalty
        if not np.isfinite(self.parallel_penalty):
            logger.warning(f"Non-finite parallel_penalty: {self.parallel_penalty}, setting to {cfg.DEFAULT_PARAMETERS['parallel_penalty']}")
            self.parallel_penalty = cfg.DEFAULT_PARAMETERS['parallel_penalty']
        else:
            self.parallel_penalty = float(np.clip(self.parallel_penalty, cfg.PARALLEL_PENALTY_MIN, cfg.PARALLEL_PENALTY_MAX))
        # Validate pre-gap SC horizon
        if not np.isfinite(self.pre_gap_ac_time_horizon) or self.pre_gap_ac_time_horizon <= 0:
            logger.warning(f"Invalid pre_gap_ac_time_horizon: {self.pre_gap_ac_time_horizon}, setting to {cfg.DEFAULT_PARAMETERS['pre_gap_ac_time_horizon']}")
            self.pre_gap_ac_time_horizon = float(cfg.DEFAULT_PARAMETERS['pre_gap_ac_time_horizon'])

        # Sanitize categorical parameters
        if self.horizon_extrapolation_type not in cfg.HORIZON_EXTRAPOLATION_TYPES:
            logger.warning(f"Invalid horizon_extrapolation_type: {self.horizon_extrapolation_type}, setting to default")
            self.horizon_extrapolation_type = cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE
        if self.automation_interp_type not in ["linear", "exponential", "logistic"]:
            logger.warning(f"Invalid automation_interp_type: {self.automation_interp_type}, setting to default")
            self.automation_interp_type = cfg.DEFAULT_PARAMETERS['automation_interp_type']

        # Sanitize manual horizon fitting parameters
        if not np.isfinite(self.present_day):
            logger.warning(f"Non-finite present_day: {self.present_day}, setting to default")
            self.present_day = cfg.DEFAULT_present_day

        # Validate optional parameters - if provided, ensure they're finite and positive
        if self.present_horizon is not None:
            if not np.isfinite(self.present_horizon) or self.present_horizon <= 0:
                logger.warning(f"Invalid present_horizon: {self.present_horizon}, setting to None for optimization")
                self.present_horizon = None

        if self.present_doubling_time is not None:
            if not np.isfinite(self.present_doubling_time) or self.present_doubling_time <= 0:
                logger.warning(f"Invalid present_doubling_time: {self.present_doubling_time}, setting to None for optimization")
                self.present_doubling_time = None

        if self.doubling_difficulty_growth_factor is not None:
            if not np.isfinite(self.doubling_difficulty_growth_factor):
                logger.warning(f"Invalid doubling_difficulty_growth_factor: {self.doubling_difficulty_growth_factor}, setting to None for optimization")
                self.doubling_difficulty_growth_factor = None

        if self.inv_compute_anchor_exp_cap is not None:
            self.compute_anchor_exp_cap = 1 / self.inv_compute_anchor_exp_cap

        # Sanitize include_gap parameter (new API)
        include_gap_bool = False
        try:
            if isinstance(self.include_gap, str):
                val = self.include_gap.strip().lower()
                if val in ("gap", "yes", "true", "1"):  # accept common truthy variants
                    include_gap_bool = True
                elif val in ("no gap", "no", "false", "0"):
                    include_gap_bool = False
                else:
                    include_gap_bool = bool(cfg.DEFAULT_PARAMETERS['include_gap'] == 'gap')
            else:
                include_gap_bool = bool(self.include_gap)
        except Exception as e:
            if should_reraise(e):
                raise
            include_gap_bool = bool(cfg.DEFAULT_PARAMETERS['include_gap'] == 'gap')
        # Normalize include_gap to canonical string for consistency
        self.include_gap = "gap" if include_gap_bool else "no gap"
        if not np.isfinite(self.gap_years) or self.gap_years < 0:
            logger.warning(f"Invalid gap_years: {self.gap_years}, setting to default")
            self.gap_years = cfg.DEFAULT_PARAMETERS['gap_years']

        # Sanitize normalization parameters
        if not np.isfinite(self.coding_labor_normalization) or self.coding_labor_normalization <= 0:
            logger.warning(f"Invalid coding_labor_normalization: {self.coding_labor_normalization}, setting to 1.0")
            self.coding_labor_normalization = 1.0

        # Sanitize coding_labor_mode
        if self.coding_labor_mode not in ("simple_ces", "optimal_ces"):
            logger.warning(f"Invalid coding_labor_mode: {self.coding_labor_mode}, defaulting to 'simple_ces'")
            self.coding_labor_mode = 'simple_ces'
        # Sanitize optimal CES parameters
        try:
            if not np.isfinite(self.coding_automation_efficiency_slope) or self.coding_automation_efficiency_slope <= 0:
                self.coding_automation_efficiency_slope = float(cfg.DEFAULT_PARAMETERS.get('coding_automation_efficiency_slope', 1.0))
            if not np.isfinite(self.optimal_ces_eta_init) or self.optimal_ces_eta_init <= 0:
                self.optimal_ces_eta_init = float(cfg.DEFAULT_PARAMETERS.get('optimal_ces_eta_init', 1.0))
            if not np.isfinite(self.optimal_ces_grid_size) or int(self.optimal_ces_grid_size) < 256:
                self.optimal_ces_grid_size = int(cfg.DEFAULT_PARAMETERS.get('optimal_ces_grid_size', 4096))
            else:
                self.optimal_ces_grid_size = int(self.optimal_ces_grid_size)

            # Apply parallel_penalty transformation to coding_automation_efficiency_slope
            # Transform: slope_transformed = slope_input ^ (0.5 / parallel_penalty)
            if np.isfinite(self.parallel_penalty) and self.parallel_penalty != 0:
                try:
                    exponent = 0.5 / self.parallel_penalty
                    original_slope = self.coding_automation_efficiency_slope
                    self.coding_automation_efficiency_slope = float(np.power(self.coding_automation_efficiency_slope, exponent))
                    logger.info(f"Transformed coding_automation_efficiency_slope: {original_slope:.6f} ^ ({exponent:.6f}) = {self.coding_automation_efficiency_slope:.6f}")

                    if not np.isfinite(self.coding_automation_efficiency_slope) or self.coding_automation_efficiency_slope <= 0:
                        logger.warning(f"Non-finite result after parallel_penalty transformation, reverting to original value: {original_slope}")
                        self.coding_automation_efficiency_slope = original_slope
                except (OverflowError, ValueError) as e:
                    logger.warning(f"Error applying parallel_penalty transformation to coding_automation_efficiency_slope: {e}, keeping original value")
            bounds_map = getattr(cfg, 'PARAMETER_BOUNDS', {})

            tail_eps_limits = bounds_map.get('optimal_ces_frontier_tail_eps', (1e-12, 1e-1))
            tail_eps = getattr(self, 'optimal_ces_frontier_tail_eps', cfg.DEFAULT_PARAMETERS.get('optimal_ces_frontier_tail_eps', 1e-6))
            if not np.isfinite(tail_eps) or tail_eps <= 0:
                tail_eps = cfg.DEFAULT_PARAMETERS.get('optimal_ces_frontier_tail_eps', 1e-6)
            if isinstance(tail_eps_limits, (list, tuple)) and len(tail_eps_limits) == 2:
                tail_min, tail_max = float(tail_eps_limits[0]), float(tail_eps_limits[1])
            else:
                tail_min, tail_max = 1e-12, 1e-1
            tail_min = max(tail_min, 1e-24)
            tail_max = min(max(tail_max, tail_min * 10.0), 0.5)
            self.optimal_ces_frontier_tail_eps = float(np.clip(tail_eps, tail_min, tail_max))

            cap_limits = bounds_map.get('optimal_ces_frontier_cap', (1.0, 1e30))
            cap_min, cap_max = cap_limits if isinstance(cap_limits, (list, tuple)) and len(cap_limits) == 2 else (1.0, 1e30)
            cap_val = getattr(self, 'optimal_ces_frontier_cap', None)
            if cap_val is not None:
                if not np.isfinite(cap_val) or cap_val <= 0:
                    self.optimal_ces_frontier_cap = None
                else:
                    self.optimal_ces_frontier_cap = float(np.clip(cap_val, cap_min, cap_max))

            max_serial = getattr(self, 'max_serial_coding_labor_multiplier', None)
            if max_serial is not None:
                if not np.isfinite(max_serial) or max_serial <= 0:
                    logger.warning(f"Invalid max_serial_coding_labor_multiplier: {max_serial}, ignoring")
                    self.max_serial_coding_labor_multiplier = None
                else:
                    serial_limits = bounds_map.get('max_serial_coding_labor_multiplier', (1.0, 1e30))
                    serial_min, serial_max = serial_limits if isinstance(serial_limits, (list, tuple)) and len(serial_limits) == 2 else (1.0, 1e30)
                    max_serial = float(np.clip(max_serial, serial_min, serial_max))
                    self.max_serial_coding_labor_multiplier = max_serial
                    penalty = float(self.parallel_penalty)
                    if penalty <= 0:
                        logger.warning("parallel_penalty must be positive to derive optimal_ces_frontier_cap from max_serial_coding_labor_multiplier")
                    else:
                        try:
                            exponent = 1.0 / penalty
                            derived_cap = float(np.power(max_serial, exponent))
                        except (OverflowError, ValueError):
                            derived_cap = None
                        if derived_cap is not None and np.isfinite(derived_cap) and derived_cap > 0:
                            self.optimal_ces_frontier_cap = float(np.clip(derived_cap, cap_min, cap_max))
                        else:
                            logger.warning("Could not derive optimal_ces_frontier_cap from max_serial_coding_labor_multiplier; leaving cap unset")
                            self.optimal_ces_frontier_cap = None

            if self.optimal_ces_frontier_cap is not None:
                rho = float(self.rho_coding_labor)
                denom = 1.0 - rho
                if abs(denom) > 1e-12:
                    exponent = rho / denom
                    try:
                        derived_eps = float(np.power(self.optimal_ces_frontier_cap, exponent))
                    except (OverflowError, ValueError):
                        derived_eps = None
                    if derived_eps is not None and np.isfinite(derived_eps) and derived_eps > 0:
                        self.optimal_ces_frontier_tail_eps = float(np.clip(derived_eps, tail_min, tail_max))
                else:
                    logger.warning("optimal_ces_frontier_cap ignored because rho is too close to 1.0")
        except Exception as e:
            if should_reraise(e):
                raise
            self.coding_labor_mode = 'simple_ces'
