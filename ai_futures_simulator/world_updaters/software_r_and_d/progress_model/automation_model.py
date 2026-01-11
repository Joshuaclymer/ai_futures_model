#!/usr/bin/env python3
"""
Automation Model Module

Contains the AutomationModel class for computing automation fractions
and optimal CES frontier calculations.
"""

import numpy as np
import copy
import time
from typing import Optional, Dict, Any, Tuple, NamedTuple, TYPE_CHECKING
from scipy.optimize import brentq, minimize_scalar
import logging

import model_config as cfg
from .ces_functions import compute_coding_labor_deprecated

if TYPE_CHECKING:
    from .parameters import Parameters

logger = logging.getLogger(__name__)


class AutomationModel:
    """Automation model"""
    def __init__(self, params):
        self.initial_FTE_per_GPU = 1
        self.FTE_per_GPU_slope = 1.0
        self.progress_base_unit = cfg.BASE_FOR_SOFTWARE_LOM
        self.schedule_type = getattr(params, 'automation_interp_type', cfg.DEFAULT_PARAMETERS['automation_interp_type'])
        anchors = list(params.automation_anchors.items())
        anchors.sort(key=lambda x: x[0])
        self.anchor_points = anchors
        (prog_1, aut_1), (prog_2, aut_2) = self.anchor_points
        self.linear_aut_slope = (aut_2 - aut_1) / (prog_2 - prog_1)
        self.linear_aut_intercept = aut_1 - self.linear_aut_slope * prog_1
        self.exponential_aut_slope = (np.log(aut_2) - np.log(aut_1)) / (prog_2 - prog_1)

        # Logistic schedule parameters: f(x) = L / (1 + exp(-k(x - x0)))
        # Only compute if using logistic schedule to avoid errors with incompatible anchors
        if self.schedule_type == "logistic":
            self.logistic_L = getattr(params, 'automation_logistic_asymptote', cfg.DEFAULT_PARAMETERS['automation_logistic_asymptote'])
            # Compute k and x0 from anchor points
            # Need L > aut_1 and L > aut_2 for valid logistic fit
            if self.logistic_L > aut_1 and self.logistic_L > aut_2:
                ratio_1 = self.logistic_L / aut_1 - 1
                ratio_2 = self.logistic_L / aut_2 - 1
                if ratio_1 > 0 and ratio_2 > 0 and prog_2 != prog_1:
                    self.logistic_k = np.log(ratio_1 / ratio_2) / (prog_2 - prog_1)
                    if abs(self.logistic_k) > 1e-12:
                        self.logistic_x0 = prog_1 + np.log(ratio_1) / self.logistic_k
                    else:
                        assert False, "Logistic aut frac error, anchors are not as expected."
                        # k ~ 0 means roughly constant, use midpoint
                        self.logistic_x0 = (prog_1 + prog_2) / 2
                else:
                    # Fallback to linear-like behavior
                    self.logistic_k = 1.0
                    self.logistic_x0 = (prog_1 + prog_2) / 2
            else:
                assert False, f"Value for logistic_L not large enough."
                # L not large enough, fallback
                self.logistic_k = 1.0
                self.logistic_x0 = (prog_1 + prog_2) / 2
        else:
            # Set defaults for non-logistic modes (not used but avoids missing attributes)
            self.logistic_L = 1.0
            self.logistic_k = 1.0
            self.logistic_x0 = (prog_1 + prog_2) / 2

        # Optimal CES frontier precompute cache (lazy init)
        self._frontier_pc: Optional[Dict[str, Any]] = None
        self._frontier_params_signature: Optional[Tuple] = None

    def get_automation_fraction(self, progress):
        """
        Compute the automation fraction.
        Handles both scalar and array inputs.

        Args:
            progress: Progress value(s) (scalar or array)

        Returns:
            Automation fraction (scalar if input is scalar, array if input is array)
        """
        # Track if input was scalar
        input_was_scalar = np.ndim(progress) == 0

        # Ensure progress is an array
        progress = np.atleast_1d(progress)

        if self.schedule_type == "linear":
            result = np.clip(
                self.linear_aut_intercept + self.linear_aut_slope * progress,
                0.0,
                1.0
            )
        elif self.schedule_type == "logistic":
            # Logistic function: f(x) = L / (1 + exp(-k(x - x0))), clipped to [0, 1]
            exponent = -self.logistic_k * (progress - self.logistic_x0)
            # Clamp exponent to avoid overflow
            exponent = np.clip(exponent, -cfg.SIGMOID_EXPONENT_CLAMP, cfg.SIGMOID_EXPONENT_CLAMP)
            raw_value = self.logistic_L / (1 + np.exp(exponent))
            result = np.clip(raw_value, 0.0, 1.0)
        elif self.schedule_type == "exponential":
            raise NotImplementedError("Exponential schedule type not implemented")
        else:
            raise ValueError("Invalid schedule type")

        # Return scalar if input was scalar, array otherwise
        return float(result[0]) if input_was_scalar else result

    def get_progress_from_index(self, index):
        """
        Get the progress from the index.
        Handles both scalar and array inputs.

        Args:
            index: Index value(s) (scalar or array)

        Returns:
            Progress value(s) (scalar if input is scalar, array if input is array)
        """
        # Track if input was scalar
        input_was_scalar = np.ndim(index) == 0

        # Ensure index is an array
        indices = np.atleast_1d(index)

        if self.schedule_type == "linear":
            (prog_1, aut_1), (prog_2, aut_2) = self.anchor_points
            if abs(self.linear_aut_slope) < 1e-12:
                # Fallback: interpolate progress directly between anchors
                result = (1.0 - indices) * prog_1 + indices * prog_2
            else:
                prog_slope = 1.0 / self.linear_aut_slope
                prog_for_zero = prog_1 - prog_slope * aut_1
                prog_for_one = prog_2 + prog_slope * (1.0 - aut_2)
                result = indices * prog_for_one + (1.0 - indices) * prog_for_zero

        elif self.schedule_type == "logistic":
            # Clamp indices to valid range for logistic inverse
            clamped = np.clip(indices, 1e-9, self.logistic_L - 1e-9)
            if abs(self.logistic_k) < 1e-12:
                # k ~ 0, use midpoint
                result = np.full_like(indices, self.logistic_x0)
            else:
                ratio = self.logistic_L / clamped - 1.0
                # Handle ratio <= 0 case (index >= L)
                result = np.where(
                    ratio > 0,
                    self.logistic_x0 - np.log(np.maximum(ratio, 1e-300)) / self.logistic_k,
                    self.logistic_x0 + cfg.SIGMOID_EXPONENT_CLAMP / abs(self.logistic_k)
                )

        elif self.schedule_type == "exponential":
            raise NotImplementedError("Exponential schedule type not implemented")
        else:
            raise ValueError(f"Invalid schedule type: {self.schedule_type}")

        # Return scalar if input was scalar, array otherwise
        return float(result[0]) if input_was_scalar else result

    def get_progress_from_index_vectorized(self, indices: np.ndarray) -> np.ndarray:
        """
        Vectorized version of get_progress_from_index.
        Always returns an array, even for scalar input.

        Args:
            indices: Index value(s) as array

        Returns:
            Progress values as array
        """
        # This is just a wrapper that ensures array output
        # The get_progress_from_index method already handles arrays
        indices = np.atleast_1d(indices)
        result = self.get_progress_from_index(indices)
        return np.atleast_1d(result)

    def get_FTE_per_GPU(self, index:float, progress:float) -> float:
        """Compute the FTE per GPU for a given task index at a given E.C. level"""
        index_progress = self.get_progress_from_index(index)
        if progress < index_progress:
            return 0.0
        progress_diff = progress - index_progress
        growth_factor = cfg.BASE_FOR_SOFTWARE_LOM ** (self.FTE_per_GPU_slope * progress_diff)
        return self.initial_FTE_per_GPU * growth_factor

    def get_crit_index(self, progress:float, aut_compute: float, L_HUMAN: float, rho: float) -> float:
        """
        Compute the critical index for a given progress
        TODO: Implement actual optimization to find the critical index
        """
        return self.get_automation_fraction(progress)

    def get_compute_allocation(self, index:float, progress:float, aut_compute: float, L_HUMAN: float, crit_index:float, rho: float) -> float:
        """Compute the optimal compute allocation for a given task index."""
        return 0


    def get_coding_labor(self, crit_index:float, progress:float, aut_compute: float, L_HUMAN: float, rho: float) -> float:
        """Compute the coding labor for a given critical index and progress"""


        return self.get_FTE_per_GPU(crit_index, progress)

    # ===================== Optimal CES fast path (embedded) =====================
    class _FrontierPrecomp(NamedTuple):
        grid_i: np.ndarray
        log_Eaut: np.ndarray
        log_B: np.ndarray
        log_F: np.ndarray
        log_Q: np.ndarray
        log_R: np.ndarray
        rho: float
        theta: float
        eta_init: float
        eps_i1: float

    @staticmethod
    def _interp(x_grid: np.ndarray, y_grid: np.ndarray, x):
        """Linear interpolation on grid (handles both scalar and array inputs)."""
        # Handle both scalar and array inputs
        x_input = np.atleast_1d(x)
        is_scalar = np.ndim(x) == 0

        # Find indices for each x value
        j = np.searchsorted(x_grid, x_input, side='right') - 1
        j = np.clip(j, 0, len(x_grid) - 2)

        # Get endpoints
        x0 = x_grid[j]
        x1 = x_grid[j + 1]

        # Compute interpolation weights
        t = np.where(x1 != x0, (x_input - x0) / (x1 - x0), 0.0)
        t = np.clip(t, 0.0, 1.0)

        # Linear interpolation
        result = (1.0 - t) * y_grid[j] + t * y_grid[j + 1]

        # Return scalar if input was scalar
        return float(result[0]) if is_scalar else result

    @staticmethod
    def _invert_monotone(x_grid: np.ndarray, y_grid: np.ndarray, y):
        """Inverse interpolation on monotone grid (handles both scalar and array inputs)."""
        # Handle both scalar and array inputs
        y_input = np.atleast_1d(y)
        is_scalar = np.ndim(y) == 0

        # Use searchsorted to find bracketing indices
        hi = np.searchsorted(y_grid, y_input, side='left')
        lo = hi - 1

        # Clamp to valid range
        lo = np.clip(lo, 0, len(x_grid) - 2)
        hi = np.clip(hi, 1, len(x_grid) - 1)

        # Handle edge cases
        below_min = y_input <= y_grid[0]
        above_max = y_input >= y_grid[-1]

        # Get bracket values
        y0 = y_grid[lo]
        y1 = y_grid[hi]

        # Handle non-finite brackets to avoid NaN (vectorized version of old logic)
        # If y0 is -inf and y1 is finite, any finite y should map to hi
        # If y0 is finite and y1 is +inf, any finite y should map to lo
        y0_is_ninf = np.isinf(y0) & (y0 < 0)
        y1_is_pinf = np.isinf(y1) & (y1 > 0)
        y_is_finite = np.isfinite(y_input)

        # Check for error conditions that old version would have raised AssertionError for
        # Case 1: Both endpoints non-finite (and not the expected -inf/+inf pattern with finite y)
        y0_nonfinite = ~np.isfinite(y0)
        y1_nonfinite = ~np.isfinite(y1)
        # The "both non-finite" error occurs when we can't handle it with the -inf/+inf special cases
        # Old logic: if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y)) and neither
        # of the two special cases (-inf with finite y1, or +inf with finite y0) applies
        unhandled_nonfinite = (
            ~(np.isfinite(y0) & np.isfinite(y1) & y_is_finite) &  # Not all finite
            ~below_min & ~above_max &  # Not at edges (which are handled separately)
            ~(y0_is_ninf & y_is_finite & np.isfinite(y1)) &  # Not the -inf case
            ~(y1_is_pinf & y_is_finite & np.isfinite(y0))    # Not the +inf case
        )
        if np.any(unhandled_nonfinite):
            raise AssertionError("Non-finite endpoints in _invert_monotone")

        # Case 2: Denominator is 0 (y1 == y0) for points that aren't at edges
        denom = y1 - y0
        zero_denom = (denom == 0) & ~below_min & ~above_max
        if np.any(zero_denom):
            raise AssertionError("Denominator is 0 in _invert_monotone")

        # Compute interpolation weights
        # Use np.where to handle the division safely (edges are handled separately anyway)
        with np.errstate(invalid='ignore', divide='ignore'):
            t = np.where((denom != 0) & np.isfinite(denom), (y_input - y0) / denom, 0.0)
        t = np.clip(t, 0.0, 1.0)

        # Linear interpolation
        result = (1.0 - t) * x_grid[lo] + t * x_grid[hi]

        # Apply edge case results
        result = np.where(below_min, x_grid[0], result)
        result = np.where(above_max, x_grid[-1], result)

        # Handle non-finite brackets: if y0=-inf and y is finite, use hi
        result = np.where(y0_is_ninf & y_is_finite & np.isfinite(y1), x_grid[hi], result)
        # Handle non-finite brackets: if y1=+inf and y is finite, use lo
        result = np.where(y1_is_pinf & y_is_finite & np.isfinite(y0), x_grid[lo], result)

        # Return scalar if input was scalar
        return float(result[0]) if is_scalar else result

    def _precompute_frontier(self, params) -> Optional[_FrontierPrecomp]:
        try:
            M = int(max(256, min(int(params.optimal_ces_grid_size), 16384)))
            tail_eps = None
            tail_param = getattr(params, 'optimal_ces_frontier_tail_eps', None)
            if tail_param is not None:
                try:
                    tail_candidate = float(tail_param)
                    if np.isfinite(tail_candidate) and tail_candidate > 0.0:
                        tail_eps = tail_candidate
                except (TypeError, ValueError):
                    tail_eps = None
            cap_value = getattr(params, 'optimal_ces_frontier_cap', None)
            if cap_value is not None:
                try:
                    cap_float = float(cap_value)
                except (TypeError, ValueError):
                    cap_float = None
                if cap_float is not None and np.isfinite(cap_float) and cap_float > 0.0:
                    rho = float(params.rho_coding_labor)
                    denom = 1.0 - rho
                    if abs(denom) > 1e-12:
                        exponent = rho / denom
                        try:
                            tail_from_cap = float(np.power(cap_float, exponent))
                        except (OverflowError, ValueError):
                            tail_from_cap = None
                        if tail_from_cap is not None and np.isfinite(tail_from_cap) and tail_from_cap > 0.0:
                            tail_eps = tail_from_cap
                    else:
                        logger.warning("Unable to derive frontier tail epsilon from cap: rho too close to 1.0")
            if tail_eps is None:
                tail_eps = 1e-6
            bounds_map = getattr(cfg, 'PARAMETER_BOUNDS', {})
            tail_bounds = bounds_map.get('optimal_ces_frontier_tail_eps', (1e-12, 0.1))
            tail_low, tail_high = (float(tail_bounds[0]), float(tail_bounds[1])) if isinstance(tail_bounds, (list, tuple)) and len(tail_bounds) == 2 else (1e-12, 0.1)
            tail_low = max(min(tail_low, 0.1), 1e-24)
            tail_high = min(max(tail_high, tail_low * 10.0), 0.5)
            tail_eps = float(np.clip(tail_eps, tail_low, tail_high))
            grid_max = float(max(1e-6, 1.0 - tail_eps))
            grid_i = np.linspace(0.0, grid_max, M)

            # Build E_aut(i) using existing inverse mapping: get_progress_from_index(i)
            # Then convert progress (OOMs of effective compute) to effective compute.
            log_base = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM))
            # progress threshold at each index (OOMs of effective compute)
            # Use vectorized version for performance (avoids 512+ Python function calls)
            idx_progress = self.get_progress_from_index(grid_i)
            idx_progress = np.maximum.accumulate(idx_progress)
            # Effective capability threshold in log-space to avoid overflow
            log_Eaut = idx_progress * log_base

            rho = float(params.rho_coding_labor)
            theta = float(params.coding_automation_efficiency_slope)
            # Use FTE-per-GPU as baseline eta; allow multiplicative tweak via optimal_ces_eta_init
            eta_init = float(self.initial_FTE_per_GPU) * float(params.optimal_ces_eta_init)
            if not (rho < 1 and abs(rho) > 1e-18 and theta > 0 and eta_init > 0):
                return None

            di = grid_i[1] - grid_i[0]
            alpha = rho / (1.0 - rho)
            beta = alpha * theta
            gamma = theta / (1.0 - rho)

            log_w = -beta * log_Eaut          # can be very large, but safe
            log_di = np.log(di)
            log_half = np.log(0.5)

            # Vectorized log_B computation (cumulative logaddexp)
            # Compute all increments at once: log_inc[k] = log_di + log_half + logaddexp(log_w[k], log_w[k+1])
            log_inc = log_di + log_half + np.logaddexp(log_w[:-1], log_w[1:])
            # Cumulative logaddexp: log_B[k+1] = logaddexp(log_B[k], log_inc[k])
            # This is equivalent to log(cumsum(exp(log_inc))) but numerically stable
            log_B = np.empty(M, dtype=np.float64)
            log_B[0] = -np.inf  # B[0] = 0 => log_B[0] = -inf by convention
            # Use a vectorized cumulative logaddexp via reduce
            # For numerical stability, shift by max value
            max_log_inc = np.max(log_inc) if len(log_inc) > 0 else 0.0
            if np.isfinite(max_log_inc):
                # exp(log_inc - max) is numerically stable
                scaled_inc = np.exp(log_inc - max_log_inc)
                cumsum_scaled = np.cumsum(scaled_inc)
                # log_B[1:] = log(cumsum) + max = log(cumsum_scaled) + max_log_inc
                # Add small epsilon to avoid log(0) warnings when cumsum_scaled contains zeros
                log_B[1:] = np.log(np.maximum(cumsum_scaled, 1e-300)) + max_log_inc
            else:
                # Fallback for edge cases
                log_B[1:] = log_inc.cumsum()  # Not quite right but handles degenerate cases

            one_minus_i = 1.0 - grid_i
            # log_Q = (1-rho) * log(1-i)
            log_one_minus_i = np.log(one_minus_i + 1e-18)
            log_Q = (1.0 - rho) * log_one_minus_i
            # log_R = log_Q - theta * log_Eaut
            log_R = log_Q - theta * log_Eaut
            # log_F = log_B + gamma * log_Eaut - log(1 - i)
            log_F = log_B + (gamma * log_Eaut) - log_one_minus_i
            # Enforce monotone in log-space
            log_F = np.maximum.accumulate(log_F)

            return AutomationModel._FrontierPrecomp(grid_i, log_Eaut, log_B, log_F, log_Q, log_R, rho, theta, eta_init, float(1.0 - grid_i[-1]))
        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"Frontier precompute failed: {e}")
            return None

    def _ensure_frontier(self, params) -> Optional[_FrontierPrecomp]:
        cap_value = getattr(params, 'optimal_ces_frontier_cap', None)
        cap_signature = None
        if cap_value is not None:
            try:
                cap_signature = float(cap_value)
            except (TypeError, ValueError):
                cap_signature = None
        max_serial_value = getattr(params, 'max_serial_coding_labor_multiplier', None)
        max_serial_signature = None
        if max_serial_value is not None:
            try:
                max_serial_signature = float(max_serial_value)
            except (TypeError, ValueError):
                max_serial_signature = None
        tail_signature = getattr(params, 'optimal_ces_frontier_tail_eps', 1e-6)
        try:
            tail_signature = float(tail_signature)
        except (TypeError, ValueError):
            tail_signature = 1e-6
        signature = (
            float(params.rho_coding_labor),
            float(params.coding_automation_efficiency_slope),
            float(params.optimal_ces_eta_init),
            int(params.optimal_ces_grid_size),
            tail_signature,
            cap_signature,
            max_serial_signature,
            tuple(self.anchor_points),
        )
        if self._frontier_pc is None or self._frontier_params_signature != signature:
            pc = self._precompute_frontier(params)
            if pc is None:
                self._frontier_pc = None
                self._frontier_params_signature = None
            else:
                self._frontier_pc = pc
                self._frontier_params_signature = signature
        return self._frontier_pc

    def coding_labor_optimal_ces(self, H, C, logE, params, return_details: bool = False):
        """Optimal CES coding labor computation (handles both scalar and array inputs).

        Args:
            H: Human labor (scalar or array)
            C: Compute (scalar or array)
            logE: Log of effective compute (scalar or array)
            params: Parameters object
            return_details: If True, return tuple (L, details_dict). Only works for scalar inputs.

        Returns:
            Coding labor value(s). Returns scalar if all inputs are scalar, otherwise array.
        """
        pc = self._ensure_frontier(params)
        if pc is None:
            return None if return_details else None

        try:
            # Determine if inputs are scalar
            H_input = np.atleast_1d(H)
            C_input = np.atleast_1d(C)
            logE_input = np.atleast_1d(logE)
            is_scalar = np.ndim(H) == 0 and np.ndim(C) == 0 and np.ndim(logE) == 0

            # For return_details mode, only support scalar inputs
            if return_details and not is_scalar:
                raise ValueError("return_details=True only supported for scalar inputs")

            # Broadcast to same shape
            N = max(len(H_input), len(C_input), len(logE_input))
            H_arr = np.broadcast_to(H_input, N) if len(H_input) == 1 else H_input
            C_arr = np.broadcast_to(C_input, N) if len(C_input) == 1 else C_input
            logE_arr = np.broadcast_to(logE_input, N) if len(logE_input) == 1 else logE_input

            # Find capability-limited indices for all logE values
            j = np.searchsorted(pc.log_Eaut, logE_arr, side='right') - 1
            j = np.clip(j, 0, len(pc.grid_i) - 2)

            # Interpolate i_E for all points
            i0, i1 = pc.grid_i[j], pc.grid_i[j + 1]
            le0, le1 = pc.log_Eaut[j], pc.log_Eaut[j + 1]
            tE = np.where(le1 != le0, np.clip((logE_arr - le0) / (le1 - le0), 0.0, 1.0), 0.0)
            i_E = (1.0 - tE) * i0 + tE * i1

            # Compute kappa and logS for all points
            kappa = C_arr / np.maximum(H_arr, 1e-18)
            logS = np.log(np.maximum(kappa * pc.eta_init, 1e-300)) + (pc.theta * logE_arr)

            # Interpolate log_F at i_E for all points
            log_F_iE = AutomationModel._interp(pc.grid_i, pc.log_F, i_E)

            # Determine cases with masks
            H_zero = H_arr <= 0.0
            interior_case = ~H_zero & (logS <= log_F_iE)
            boundary_case = ~H_zero & ~interior_case

            # Initialize results
            log_L = np.zeros(N)

            # Handle interior case
            if np.any(interior_case):
                i_c_int = AutomationModel._invert_monotone(pc.grid_i, pc.log_F, logS[interior_case])
                log_human_int = AutomationModel._interp(pc.grid_i, pc.log_Q, i_c_int)
                log_R_i_int = AutomationModel._interp(pc.grid_i, pc.log_R, i_c_int)
                log_comp_int = logS[interior_case] + log_R_i_int
                log_L_norm_rho_int = np.logaddexp(log_human_int, log_comp_int)
                log_L[interior_case] = np.log(H_arr[interior_case]) + (1.0 / pc.rho) * log_L_norm_rho_int

            # Handle boundary case
            if np.any(boundary_case):
                i_c_bnd = i_E[boundary_case]
                log_human_bnd = AutomationModel._interp(pc.grid_i, pc.log_Q, i_c_bnd)
                log_B_i_bnd = AutomationModel._interp(pc.grid_i, pc.log_B, i_c_bnd)
                log_comp_bnd = (pc.rho * logS[boundary_case]) + ((1.0 - pc.rho) * log_B_i_bnd)
                log_L_norm_rho_bnd = np.logaddexp(log_human_bnd, log_comp_bnd)
                log_L[boundary_case] = np.log(H_arr[boundary_case]) + (1.0 / pc.rho) * log_L_norm_rho_bnd

            # Convert to linear space
            L_arr = np.exp(log_L)

            # Handle H_zero case (must be done after exp to avoid -inf issues)
            L_arr[H_zero] = 0.0

            # Return details if requested (scalar only)
            if return_details:
                i_c = i_E[0] if not (interior_case[0] or boundary_case[0]) else (
                    i_c_int[0] if interior_case[0] else i_c_bnd[0]
                )
                case = "H_zero" if H_zero[0] else ("interior" if interior_case[0] else "boundary")
                log_human = log_human_int[0] if interior_case[0] else (log_human_bnd[0] if boundary_case[0] else 0.0)
                log_comp = log_comp_int[0] if interior_case[0] else (log_comp_bnd[0] if boundary_case[0] else 0.0)
                human = float(np.exp(log_human)) if not H_zero[0] else 0.0
                comp = float(np.exp(log_comp)) if not H_zero[0] else 0.0
                return float(L_arr[0]), {"i_cut": float(i_c), "i_E": float(i_E[0]), "case": case, "human_term": human, "comp_term": comp}

            # Return scalar if input was scalar
            return float(L_arr[0]) if is_scalar else L_arr

        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"coding_labor_optimal_ces failed at runtime: {e}")
            return None if not return_details else (None, {"error": str(e)})

    def compute_ai_only_coding_labor(self, C, logE, params, automation_fraction=None):
        """Compute AI-only coding labor (limit as H -> 0).

        In this branch (main), AI can only perform tasks up to the automation fraction.
        When automation_fraction < 1, a pure AI-only workforce cannot complete all tasks,
        so AI-only coding labor is 0.

        When automation_fraction = 1 (full automation), the formula is:
            L_ai_only = C * eta_init * E^theta * B(1)^((1-rho)/rho)

        In log-space:
            log_L_ai_only = log(C) + log(eta_init) + theta*logE + ((1-rho)/rho) * log_B[grid_max]

        Args:
            C: Inference compute (scalar or array)
            logE: Log of effective compute at the target progress level (scalar or array)
            params: Parameters object
            automation_fraction: Automation fraction(s) (scalar or array). If None, returns 0.

        Returns:
            AI-only coding labor value(s). Returns 0 when automation_fraction < 1.
            Returns scalar if all inputs are scalar, array otherwise.
        """
        # Handle scalar vs array inputs
        C_input = np.atleast_1d(C)
        logE_input = np.atleast_1d(logE)
        is_scalar = np.ndim(C) == 0 and np.ndim(logE) == 0

        if automation_fraction is None:
            # No automation fraction provided, return zeros
            result = np.zeros(max(len(C_input), len(logE_input)))
            return float(result[0]) if is_scalar else result

        aut_frac_input = np.atleast_1d(automation_fraction)
        is_scalar = is_scalar and np.ndim(automation_fraction) == 0

        # Broadcast to same shape
        N = max(len(C_input), len(logE_input), len(aut_frac_input))
        C_arr = np.broadcast_to(C_input, N) if len(C_input) == 1 else C_input
        logE_arr = np.broadcast_to(logE_input, N) if len(logE_input) == 1 else logE_input
        aut_frac_arr = np.broadcast_to(aut_frac_input, N) if len(aut_frac_input) == 1 else aut_frac_input

        # Initialize result array with zeros
        result = np.zeros(N)

        # Only compute for points where automation_fraction >= 1
        # Use a small tolerance for floating point comparison
        full_automation_mask = aut_frac_arr >= (1.0 - 1e-9)

        if not np.any(full_automation_mask):
            # No points have full automation, return zeros
            return float(result[0]) if is_scalar else result

        # Ensure frontier is computed
        pc = self._ensure_frontier(params)
        if pc is None:
            return float(result[0]) if is_scalar else result

        try:
            # Get log_B at grid_max (i = 1 - tail_eps, meaning AI does all tasks)
            log_B_at_max = pc.log_B[-1]

            # Compute exponent: (1 - rho) / rho
            if abs(pc.rho) < 1e-18:
                logger.warning("rho too close to zero in compute_ai_only_coding_labor")
                return float(result[0]) if is_scalar else result

            exponent = (1.0 - pc.rho) / pc.rho

            # Compute in log-space for numerical stability (only for full automation points)
            C_full = C_arr[full_automation_mask]
            logE_full = logE_arr[full_automation_mask]

            log_C = np.log(np.maximum(C_full, 1e-300))
            log_eta = np.log(max(pc.eta_init, 1e-300))

            log_L_ai_only = log_C + log_eta + pc.theta * logE_full + exponent * log_B_at_max
            L_ai_only = np.exp(log_L_ai_only)

            # Replace non-finite values with 0
            L_ai_only = np.where(np.isfinite(L_ai_only), L_ai_only, 0.0)

            result[full_automation_mask] = L_ai_only

            return float(result[0]) if is_scalar else result

        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"compute_ai_only_coding_labor failed: {e}")
            return float(result[0]) if is_scalar else result


def aut_frac_from_swe_multiplier(swe_multiplier: float, L_HUMAN: float, inference_compute: float, params: "Parameters") -> float:
    """
    Compute automation fraction from swe multiplier.

    Solve for A in:
      (swe_multiplier)**params.parallel_penalty * compute_coding_labor_deprecated(A, inference_compute, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True) = compute_coding_labor_deprecated(A, inference_compute, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization)
    where p = params.rho_coding_labor.
    Returns A in (0, 1). If there are multiple solutions, return the lower one.

    """
    # Input validation
    if not all(np.isfinite([swe_multiplier, L_HUMAN, inference_compute])):
        logger.warning("Non-finite inputs to aut_frac_from_swe_multiplier")
        return 0.0

    if swe_multiplier <= 0 or L_HUMAN <= 0 or inference_compute < 0:
        logger.warning("Invalid inputs to aut_frac_from_swe_multiplier")
        return 0.0

    # Target value we want to achieve
    target_output = swe_multiplier**params.parallel_penalty * compute_coding_labor_deprecated(0, inference_compute, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True)

    # Define the objective function to minimize
    def objective(A_candidate):
        """Return the difference between target and actual cognitive output"""
        try:
            actual_output = compute_coding_labor_deprecated(
                A_candidate, inference_compute, L_HUMAN,
                params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
            )
            return actual_output - target_output
        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"Error in objective function: {e}")
            return float('inf')

    # Use bounds slightly inside (0, 1) to avoid numerical issues
    bounds = (cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)

    try:
        # Check if the function changes sign over the interval
        f_low = objective(bounds[0])
        f_high = objective(bounds[1])

        if f_low * f_high <= 0:
            # Sign change exists - we can find a root
            result = brentq(objective, bounds[0], bounds[1], xtol=1e-12, maxiter=100)
            result = np.clip(result, bounds[0], bounds[1])
            return float(result)
        else:
            # No sign change - target may not be achievable
            # Find the automation fraction that minimizes the absolute error
            result = minimize_scalar(
                lambda A: abs(objective(A)),
                bounds=bounds,
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 100}
            )

            if result.success:
                return float(np.clip(result.x, bounds[0], bounds[1]))
            else:
                raise RuntimeError("Optimization failed")

    except Exception as e:
        # Don't swallow timeout/interrupt exceptions - let them propagate
        if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
            raise
        logger.warning(f"Root finding/optimization failed in aut_frac_from_swe_multiplier: {e}")

        # Fallback: use grid search to find the best approximation
        try:
            A_candidates = np.linspace(bounds[0], bounds[1], 1000)
            # Vectorize grid search: compute_coding_labor_deprecated supports array inputs
            try:
                actual_outputs = compute_coding_labor_deprecated(
                    A_candidates, inference_compute, L_HUMAN,
                    params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
                )
                errors = np.abs(actual_outputs - target_output)
            except Exception as vec_e:
                # Don't swallow timeout/interrupt exceptions - let them propagate
                if isinstance(vec_e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(vec_e).__name__:
                    raise
                # Fallback to element-wise if vectorization fails
                errors = np.array([abs(objective(A)) for A in A_candidates])

            best_idx = np.argmin(errors)
            result = A_candidates[best_idx]

            logger.info(f"Used grid search fallback, error: {errors[best_idx]}")
            return float(result)

        except Exception as e2:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e2, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e2).__name__:
                raise
            logger.warning(f"Grid search fallback also failed: {e2}")
            # Return a reasonable default
            return 0.5


def solve_lower_anchor_via_automation_model(
    swe_multiplier: float,
    anchor_progress: float,
    L_HUMAN: float,
    inference_compute: float,
    params: "Parameters",
) -> float:
    """
    Solve for the lower anchor automation fraction at the anchor progress such that,
    when initializing AutomationModel with anchors
        { anchor_progress: A_lower, params.progress_at_aa: params.automation_fraction_at_coding_automation_anchor },
    the implied coding-labor multiplier at anchor_progress matches swe_multiplier.

    The multiplier is defined on coding labor after applying parallel_penalty and normalization, i.e.
        coding_labor_with_AI = swe_multiplier**parallel_penalty * coding_labor_human_only.

    If the optimal-CES frontier is unavailable for given parameters, falls back to the simple CES formulation.
    Returns a clipped value in (cfg.AUTOMATION_FRACTION_CLIP_MIN, 1 - cfg.AUTOMATION_FRACTION_CLIP_MIN).
    """
    # Profiling: track overall function time
    _prof_start = time.perf_counter()
    _prof_setup_time = 0.0
    _prof_implied_ratio_time = 0.0
    _prof_implied_ratio_calls = 0
    _prof_deepcopy_time = 0.0
    _prof_automationmodel_init_time = 0.0
    _prof_coding_labor_optimal_ces_time = 0.0
    _prof_solver_time = 0.0
    _prof_solver_method = "none"

    try:
        _prof_setup_start = time.perf_counter()
        # Validate inputs
        if not all(np.isfinite([swe_multiplier, anchor_progress, L_HUMAN, inference_compute])):
            logger.warning("Non-finite inputs to solve_lower_anchor_via_automation_model")
            return 0.01
        if swe_multiplier <= 0 or L_HUMAN <= 0 or inference_compute < 0:
            logger.warning("Invalid inputs to solve_lower_anchor_via_automation_model")
            return 0.01

        # Ensure an upper anchor exists
        progress_at_aa = getattr(params, 'progress_at_aa', None)
        aut_at_sc = getattr(params, 'automation_fraction_at_coding_automation_anchor', None)
        if progress_at_aa is None or not np.isfinite(progress_at_aa) or aut_at_sc is None:
            logger.warning("Missing progress_at_aa or automation_fraction_at_coding_automation_anchor; falling back to direct solver")
            return aut_frac_from_swe_multiplier(swe_multiplier, L_HUMAN, inference_compute, params)

        # Target coding-labor ratio in parallel_penalty space
        target_ratio = float(np.power(swe_multiplier, params.parallel_penalty))

        # Baseline human-only coding labor (consistent with definition of multiplier)
        baseline = compute_coding_labor_deprecated(
            0, inference_compute, L_HUMAN,
            params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization,
            human_only=True
        )
        if baseline <= 0 or not np.isfinite(baseline):
            logger.warning("Invalid baseline coding labor in anchor solver; using fallback")
            return aut_frac_from_swe_multiplier(swe_multiplier, L_HUMAN, inference_compute, params)

        logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * anchor_progress)
        _prof_setup_time = time.perf_counter() - _prof_setup_start

        def implied_ratio_for_anchor(a_lower: float) -> float:
            nonlocal _prof_implied_ratio_calls, _prof_implied_ratio_time
            nonlocal _prof_deepcopy_time, _prof_automationmodel_init_time, _prof_coding_labor_optimal_ces_time
            _prof_call_start = time.perf_counter()
            _prof_implied_ratio_calls += 1

            # Build temporary params with candidate anchors
            _prof_dc_start = time.perf_counter()
            p = copy.deepcopy(params)
            _prof_deepcopy_time += time.perf_counter() - _prof_dc_start

            # Clip candidate to safe open interval
            a_clipped = float(np.clip(a_lower, cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN))
            p.automation_anchors = {
                float(anchor_progress): a_clipped,
                float(progress_at_aa): float(np.clip(aut_at_sc, cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)),
            }
            try:
                _prof_am_start = time.perf_counter()
                am = AutomationModel(p)
                _prof_automationmodel_init_time += time.perf_counter() - _prof_am_start

                H = float(L_HUMAN)
                C = float(inference_compute)

                _prof_ces_start = time.perf_counter()
                L_opt = am.coding_labor_optimal_ces(H, C, logE, p)
                _prof_coding_labor_optimal_ces_time += time.perf_counter() - _prof_ces_start

                if L_opt is None or not np.isfinite(L_opt):
                    # Fallback to simple CES using the schedule's automation at the anchor (which equals a_lower)
                    A = am.get_automation_fraction(anchor_progress)
                    L_ai = compute_coding_labor_deprecated(
                        A, inference_compute, L_HUMAN,
                        p.rho_coding_labor, p.parallel_penalty, p.coding_labor_normalization
                    )
                else:
                    # Match units with compute_coding_labor_deprecated
                    L_ai = float((L_opt ** p.parallel_penalty) * p.coding_labor_normalization)
                if not np.isfinite(L_ai) or L_ai <= 0:
                    _prof_implied_ratio_time += time.perf_counter() - _prof_call_start
                    return 0.0
                _prof_implied_ratio_time += time.perf_counter() - _prof_call_start
                return float(L_ai / baseline)
            except Exception as e:
                # Don't swallow timeout/interrupt exceptions - let them propagate
                if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                    raise
                logger.warning(f"Error computing implied ratio for anchor {a_lower}: {e}")
                _prof_implied_ratio_time += time.perf_counter() - _prof_call_start
                return 0.0

        def objective(a_lower: float) -> float:
            return implied_ratio_for_anchor(a_lower) - target_ratio

        # Bounds slightly inside (0, 1) to avoid numerical issues
        lo = float(cfg.AUTOMATION_FRACTION_CLIP_MIN)
        hi = float(1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)

        try:
            _prof_solver_start = time.perf_counter()
            f_lo = objective(lo)
            f_hi = objective(hi)
            if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0:
                _prof_solver_method = "brentq"
                root = brentq(objective, lo, hi, xtol=1e-8, maxiter=100)
                _prof_solver_time = time.perf_counter() - _prof_solver_start
                result = float(np.clip(root, lo, hi))
                _log_profiling_results(_prof_start, _prof_setup_time, _prof_implied_ratio_time,
                                       _prof_implied_ratio_calls, _prof_deepcopy_time,
                                       _prof_automationmodel_init_time, _prof_coding_labor_optimal_ces_time,
                                       _prof_solver_time, _prof_solver_method)
                return result

            # If no sign change, minimize absolute error
            _prof_solver_method = "minimize_scalar"
            res = minimize_scalar(lambda x: abs(objective(x)), bounds=(lo, hi), method='bounded', options={'xatol': 1e-8, 'maxiter': 200})
            _prof_solver_time = time.perf_counter() - _prof_solver_start
            if getattr(res, 'success', False):
                result = float(np.clip(res.x, lo, hi))
                _log_profiling_results(_prof_start, _prof_setup_time, _prof_implied_ratio_time,
                                       _prof_implied_ratio_calls, _prof_deepcopy_time,
                                       _prof_automationmodel_init_time, _prof_coding_labor_optimal_ces_time,
                                       _prof_solver_time, _prof_solver_method)
                return result
            else:
                raise RuntimeError("Anchor solver bounded minimization failed")
        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"Anchor solver root-finding failed; using grid search fallback: {e}")
            try:
                _prof_solver_start = time.perf_counter()
                _prof_solver_method = "grid_search"
                grid = np.linspace(lo, hi, 512)
                # Note: objective involves deepcopy and state modification, cannot be vectorized
                errs = np.array([abs(objective(a)) for a in grid])
                idx = int(np.argmin(errs))
                _prof_solver_time = time.perf_counter() - _prof_solver_start
                result = float(grid[idx])
                _log_profiling_results(_prof_start, _prof_setup_time, _prof_implied_ratio_time,
                                       _prof_implied_ratio_calls, _prof_deepcopy_time,
                                       _prof_automationmodel_init_time, _prof_coding_labor_optimal_ces_time,
                                       _prof_solver_time, _prof_solver_method)
                return result
            except Exception as e2:
                # Don't swallow timeout/interrupt exceptions - let them propagate
                if isinstance(e2, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e2).__name__:
                    raise
                logger.warning(f"Grid search fallback failed in anchor solver: {e2}")
                return 0.01
    except Exception as e:
        # Don't swallow timeout/interrupt exceptions - let them propagate
        if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
            raise
        logger.warning(f"solve_lower_anchor_via_automation_model failed unexpectedly: {e}")
        return 0.01


def _log_profiling_results(
    prof_start: float,
    setup_time: float,
    implied_ratio_time: float,
    implied_ratio_calls: int,
    deepcopy_time: float,
    automationmodel_init_time: float,
    coding_labor_optimal_ces_time: float,
    solver_time: float,
    solver_method: str,
) -> None:
    """Log profiling results for solve_lower_anchor_via_automation_model."""
    total_time = time.perf_counter() - prof_start
    avg_call_time = implied_ratio_time / implied_ratio_calls if implied_ratio_calls > 0 else 0.0
    avg_deepcopy = deepcopy_time / implied_ratio_calls if implied_ratio_calls > 0 else 0.0
    avg_am_init = automationmodel_init_time / implied_ratio_calls if implied_ratio_calls > 0 else 0.0
    avg_ces = coding_labor_optimal_ces_time / implied_ratio_calls if implied_ratio_calls > 0 else 0.0

    logger.info(
        f"PROFILING solve_lower_anchor_via_automation_model:\n"
        f"  Total time: {total_time*1000:.2f}ms\n"
        f"  Setup time: {setup_time*1000:.2f}ms ({100*setup_time/total_time:.1f}%)\n"
        f"  Solver method: {solver_method}\n"
        f"  Solver time (incl. objective calls): {solver_time*1000:.2f}ms ({100*solver_time/total_time:.1f}%)\n"
        f"  implied_ratio_for_anchor:\n"
        f"    Total calls: {implied_ratio_calls}\n"
        f"    Total time: {implied_ratio_time*1000:.2f}ms ({100*implied_ratio_time/total_time:.1f}%)\n"
        f"    Avg per call: {avg_call_time*1000:.3f}ms\n"
        f"    Breakdown per call:\n"
        f"      copy.deepcopy: {avg_deepcopy*1000:.3f}ms ({100*deepcopy_time/implied_ratio_time:.1f}% of implied_ratio)\n"
        f"      AutomationModel.__init__: {avg_am_init*1000:.3f}ms ({100*automationmodel_init_time/implied_ratio_time:.1f}% of implied_ratio)\n"
        f"      coding_labor_optimal_ces: {avg_ces*1000:.3f}ms ({100*coding_labor_optimal_ces_time/implied_ratio_time:.1f}% of implied_ratio)"
    )
