#!/usr/bin/env python3
"""
Parameters Module

Contains the Parameters dataclass for model configuration with validation.
User-configurable defaults are in default_parameters.yaml.
Internal/mode flags have sensible defaults here.
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
    """Model parameters with validation.

    Main configurable parameters should be passed from default_parameters.yaml.
    Internal/mode flags have sensible defaults.
    """

    # Production function parameters (from YAML)
    rho_coding_labor: float
    software_progress_rate_at_reference_year: float
    swe_multiplier_at_present_day: float
    present_day: float
    coding_labor_normalization: float
    inf_labor_asymptote: float
    inf_compute_asymptote: float
    labor_anchor_exp_cap: float
    inv_compute_anchor_exp_cap: float
    parallel_penalty: float

    # Parameters with sensible defaults (internal/optional)
    human_only: bool = False
    direct_input_exp_cap_ces_params: bool = False
    rho_experiment_capacity: Optional[float] = None
    alpha_experiment_capacity: Optional[float] = None
    r_software: float = 1.0  # Will be calibrated
    experiment_compute_exponent: Optional[float] = None
    compute_anchor_exp_cap: float = 0.357  # Derived from inv_compute_anchor_exp_cap

    # Automation parameters
    automation_fraction_at_coding_automation_anchor: float = 1.0
    automation_anchors: Optional[Dict[float, float]] = None
    automation_interp_type: str = "linear"
    automation_logistic_asymptote: float = 1.05

    # AI Research Taste parameters
    ai_research_taste_at_coding_automation_anchor_sd: float = 0.5
    ai_research_taste_slope: float = 2.2
    taste_schedule_type: str = "SDs per progress-year"
    progress_at_aa: Optional[float] = None
    ac_time_horizon_minutes: float = 12000000.0
    pre_gap_ac_time_horizon: float = 124600.0  # Horizon before capability gap
    horizon_extrapolation_type: str = "decaying doubling time"

    # Manual horizon fitting parameters
    present_horizon: Optional[float] = 26.0
    present_doubling_time: Optional[float] = 0.458
    doubling_difficulty_growth_factor: Optional[float] = 0.92

    # Research taste distribution parameters
    median_to_top_taste_multiplier: float = 3.7
    top_percentile: float = 0.999
    taste_limit: float = 8.0
    taste_limit_smoothing: float = 0.51

    # Capability milestones
    strat_ai_m2b: float = 2.0
    ted_ai_m2b: float = 3.0

    # Benchmarks and gaps mode
    include_gap: Union[str, bool] = "no gap"
    gap_years: float = 1.5

    # Coding labor mode and optimal CES params
    coding_labor_mode: str = "optimal_ces"
    coding_automation_efficiency_slope: float = 3.0
    optimal_ces_eta_init: float = 0.05
    optimal_ces_grid_size: int = 4096
    optimal_ces_frontier_tail_eps: float = 1e-6
    optimal_ces_frontier_cap: Optional[float] = 1e12
    max_serial_coding_labor_multiplier: Optional[float] = None

    # SOS mode (disabled by default)
    sos_mode: bool = False
    sos_start_milestone: str = "AC"

    # Plan A / Blacksite mode (disabled by default)
    plan_a_mode: bool = False
    plan_a_start_time: float = 2030.0
    show_blacksite: bool = False
    sw_leaks_to_blacksite: bool = False
    is_blacksite: bool = False
    main_project_training_compute_growth_rate: float = 0.6
    main_project_software_progress_rate: float = 1.0
    blacksite_initial_years_behind: float = 1.0
    blacksite_initial_model_progress: Optional[float] = None
    blacksite_training_compute_penalty_ooms: float = 0.0
    blacksite_training_compute_growth_rate: float = 0.6
    blacksite_human_labor_penalty_ooms: float = 0.0
    blacksite_experiment_compute_penalty_ooms: float = 0.0
    blacksite_inference_compute_penalty_ooms: float = 0.0
    blacksite_human_taste_penalty: float = 0.0
    blacksite_can_stack_training_compute: bool = False
    blacksite_can_stack_software_progress: bool = False
    blacksite_start_time: float = 2030.0

    # Fields that are computed/initialized in __post_init__
    automation_model: Optional[AutomationModel] = field(default=None, init=False)
    ai_research_taste_at_coding_automation_anchor: float = field(default=None, init=False)
    taste_distribution: Optional[TasteDistribution] = field(default=None, init=False)
    main_site: Any = field(default=None, init=False)

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

        if self.r_software is not None and not np.isfinite(self.r_software):
            logger.warning(f"Non-finite r_software: {self.r_software}, setting to 1.0")
            self.r_software = 1.0

        # Sanitize automation parameters
        if not np.isfinite(self.automation_fraction_at_coding_automation_anchor):
            raise ValueError(f"Non-finite automation_fraction_at_coding_automation_anchor: {self.automation_fraction_at_coding_automation_anchor}")

        # Initialize taste distribution (used throughout the model)
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
                raise ValueError(f"Converted taste is non-finite from SD={self.ai_research_taste_at_coding_automation_anchor_sd}")
        except Exception as e:
            if should_reraise(e):
                raise
            raise ValueError(f"Failed converting ai_research_taste_at_coding_automation_anchor_sd to taste: {e}")

        if not np.isfinite(self.ai_research_taste_slope):
            logger.warning(f"Non-finite ai_research_taste_slope: {self.ai_research_taste_slope}, setting to 1.0")
            self.ai_research_taste_slope = 1.0

        # Sanitize time horizon parameter
        if not np.isfinite(self.ac_time_horizon_minutes) or self.ac_time_horizon_minutes <= 0:
            raise ValueError(f"Invalid ac_time_horizon_minutes: {self.ac_time_horizon_minutes}")

        # Sanitize parallel_penalty
        if not np.isfinite(self.parallel_penalty):
            raise ValueError(f"Non-finite parallel_penalty: {self.parallel_penalty}")
        else:
            self.parallel_penalty = float(np.clip(self.parallel_penalty, cfg.PARALLEL_PENALTY_MIN, cfg.PARALLEL_PENALTY_MAX))

        # Sanitize categorical parameters
        if self.horizon_extrapolation_type not in cfg.HORIZON_EXTRAPOLATION_TYPES:
            raise ValueError(f"Invalid horizon_extrapolation_type: {self.horizon_extrapolation_type}, must be one of {cfg.HORIZON_EXTRAPOLATION_TYPES}")
        if self.automation_interp_type not in ["linear", "exponential", "logistic"]:
            raise ValueError(f"Invalid automation_interp_type: {self.automation_interp_type}")

        # Sanitize manual horizon fitting parameters
        if not np.isfinite(self.present_day):
            raise ValueError(f"Non-finite present_day: {self.present_day}")

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

        # Sanitize include_gap parameter
        include_gap_bool = False
        try:
            if isinstance(self.include_gap, str):
                val = self.include_gap.strip().lower()
                if val in ("gap", "yes", "true", "1"):
                    include_gap_bool = True
                elif val in ("no gap", "no", "false", "0"):
                    include_gap_bool = False
                else:
                    include_gap_bool = False
            else:
                include_gap_bool = bool(self.include_gap)
        except Exception as e:
            if should_reraise(e):
                raise
            include_gap_bool = False
        self.include_gap = "gap" if include_gap_bool else "no gap"

        if not np.isfinite(self.gap_years) or self.gap_years < 0:
            raise ValueError(f"Invalid gap_years: {self.gap_years}")

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
                self.coding_automation_efficiency_slope = 1.0
            if not np.isfinite(self.optimal_ces_eta_init) or self.optimal_ces_eta_init <= 0:
                self.optimal_ces_eta_init = 1.0
            if not np.isfinite(self.optimal_ces_grid_size) or int(self.optimal_ces_grid_size) < 256:
                self.optimal_ces_grid_size = 4096
            else:
                self.optimal_ces_grid_size = int(self.optimal_ces_grid_size)

            # Apply parallel_penalty transformation to coding_automation_efficiency_slope
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
            tail_eps = self.optimal_ces_frontier_tail_eps
            if not np.isfinite(tail_eps) or tail_eps <= 0:
                tail_eps = 1e-6
            if isinstance(tail_eps_limits, (list, tuple)) and len(tail_eps_limits) == 2:
                tail_min, tail_max = float(tail_eps_limits[0]), float(tail_eps_limits[1])
            else:
                tail_min, tail_max = 1e-12, 1e-1
            tail_min = max(tail_min, 1e-24)
            tail_max = min(max(tail_max, tail_min * 10.0), 0.5)
            self.optimal_ces_frontier_tail_eps = float(np.clip(tail_eps, tail_min, tail_max))

            cap_limits = bounds_map.get('optimal_ces_frontier_cap', (1.0, 1e30))
            cap_min, cap_max = cap_limits if isinstance(cap_limits, (list, tuple)) and len(cap_limits) == 2 else (1.0, 1e30)
            cap_val = self.optimal_ces_frontier_cap
            if cap_val is not None:
                if not np.isfinite(cap_val) or cap_val <= 0:
                    self.optimal_ces_frontier_cap = None
                else:
                    self.optimal_ces_frontier_cap = float(np.clip(cap_val, cap_min, cap_max))

            max_serial = self.max_serial_coding_labor_multiplier
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
