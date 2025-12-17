#!/usr/bin/env python3
"""Dump parameter configuration from the Python model for the frontend."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np

from python_repo_utils import resolve_python_repo_root


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    return str(obj)


def build_ui_defaults(raw_defaults: dict[str, Any], parameter_bounds: dict[str, Any]) -> dict[str, Any]:
    def get_bound(name: str, index: int, fallback: float) -> float:
        bounds = parameter_bounds.get(name)
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            return float(bounds[index])
        return fallback

    sc_minutes = float(raw_defaults.get("ac_time_horizon_minutes", 1.0))
    sc_minutes = sc_minutes if sc_minutes > 0 else 1.0
    difficulty_growth = float(raw_defaults.get("doubling_difficulty_growth_factor", 0.92))

    return {
        "present_doubling_time": float(raw_defaults.get("present_doubling_time", 0.408)),
        "ac_time_horizon_minutes": math.log10(sc_minutes),
        "doubling_difficulty_growth_factor": difficulty_growth,
        "rho_coding_labor": float(raw_defaults.get("rho_coding_labor", -2.0)),
        "rho_experiment_capacity": float(raw_defaults.get("rho_experiment_capacity", -0.137)),
        "alpha_experiment_capacity": float(raw_defaults.get("alpha_experiment_capacity", 0.701)),
        "direct_input_exp_cap_ces_params": bool(raw_defaults.get("direct_input_exp_cap_ces_params", False)),
        "r_software": float(raw_defaults.get("r_software", 2.4)),
        "software_progress_rate_at_reference_year": float(raw_defaults.get("software_progress_rate_at_reference_year", 1.25)),
        "coding_labor_normalization": float(raw_defaults.get("coding_labor_normalization", 1.0)),
        "experiment_compute_exponent": float(raw_defaults.get("experiment_compute_exponent", 0.562)),
        # coding_labor_exponent has been replaced by parallel_penalty in Python; expose the same value for UI continuity
        "coding_labor_exponent": float(raw_defaults.get("parallel_penalty", 0.5)),
        "automation_fraction_at_coding_automation_anchor": float(raw_defaults.get("automation_fraction_at_coding_automation_anchor", 1.0)),
        "automation_interp_type": str(raw_defaults.get("automation_interp_type", "linear")),
        "automation_logistic_asymptote": float(raw_defaults.get("automation_logistic_asymptote", 1.05)),
        "swe_multiplier_at_present_day": float(raw_defaults.get("swe_multiplier_at_present_day", 1.35)),
        "automation_anchors": raw_defaults.get("automation_anchors"),
        "ai_research_taste_at_coding_automation_anchor_sd": float(raw_defaults.get("ai_research_taste_at_coding_automation_anchor_sd", 0.0)),
        "ai_research_taste_slope": float(raw_defaults.get("ai_research_taste_slope", 1.8)),
        "taste_schedule_type": str(raw_defaults.get("taste_schedule_type", "SDs per progress-year")),
        "progress_at_aa": (
            float(raw_defaults["progress_at_aa"]) if raw_defaults.get("progress_at_aa") is not None
            else get_bound("progress_at_aa", 0, 1.0)
        ),
        "saturation_horizon_minutes": float(raw_defaults.get("pre_gap_ac_time_horizon", sc_minutes)),
        "present_day": float(raw_defaults.get("present_day", 2025.6)),
        "present_horizon": float(raw_defaults.get("present_horizon", 26.0)),
        "horizon_extrapolation_type": str(raw_defaults.get("horizon_extrapolation_type", "decaying doubling time")),
        "inf_labor_asymptote": float(raw_defaults.get("inf_labor_asymptote", 15.0)),
        "inf_compute_asymptote": float(raw_defaults.get("inf_compute_asymptote", 5000.0)),
        "labor_anchor_exp_cap": float(raw_defaults.get("labor_anchor_exp_cap", 1.6)),
        "compute_anchor_exp_cap": raw_defaults.get("compute_anchor_exp_cap"),
        "inv_compute_anchor_exp_cap": float(raw_defaults.get("inv_compute_anchor_exp_cap", 3.33)),
        "benchmarks_and_gaps_mode": str(raw_defaults.get("include_gap", "no gap")) != "no gap",
        "gap_years": float(raw_defaults.get("gap_years", 0.0)),
        "coding_automation_efficiency_slope": float(raw_defaults.get("coding_automation_efficiency_slope", 3.0)),
        "max_serial_coding_labor_multiplier": float(raw_defaults.get("max_serial_coding_labor_multiplier", 1e12)),
        "median_to_top_taste_multiplier": float(raw_defaults.get("median_to_top_taste_multiplier", 3.25)),
        "top_percentile": float(raw_defaults.get("top_percentile", 0.999)),
        "taste_limit": float(raw_defaults.get("taste_limit", 8.0)),
        "taste_limit_smoothing": float(raw_defaults.get("taste_limit_smoothing", 0.51)),
        "strat_ai_m2b": float(raw_defaults.get("strat_ai_m2b", 2.0)),
        "ted_ai_m2b": float(raw_defaults.get("ted_ai_m2b", 4.0)),
        "optimal_ces_eta_init": float(raw_defaults.get("optimal_ces_eta_init", 0.05)),
        "constant_training_compute_growth_rate": float(raw_defaults.get("constant_training_compute_growth_rate", 0.6)),
        "slowdown_year": float(raw_defaults.get("slowdown_year", 2028.0)),
        "post_slowdown_training_compute_growth_rate": float(raw_defaults.get("post_slowdown_training_compute_growth_rate", 0.25)),
    }


def main() -> None:
    current_file = Path(__file__)
    python_repo = resolve_python_repo_root(current_file)

    import sys
    sys.path.insert(0, str(python_repo))

    import model_config as cfg  # type: ignore

    raw_defaults = to_jsonable(cfg.DEFAULT_PARAMETERS)
    parameter_bounds = to_jsonable(cfg.PARAMETER_BOUNDS)    

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_source": str((python_repo / "model_config.py").resolve()),
        "raw_defaults": raw_defaults,
        "parameter_bounds": parameter_bounds,
        "taste_schedule_types": to_jsonable(getattr(cfg, "TASTE_SCHEDULE_TYPES", [])),
        "taste_slope_defaults": to_jsonable(getattr(cfg, "TASTE_SLOPE_DEFAULTS", {})),
        "horizon_extrapolation_types": to_jsonable(getattr(cfg, "HORIZON_EXTRAPOLATION_TYPES", [])),
        "ui_defaults": build_ui_defaults(raw_defaults, parameter_bounds),
        "model_constants": {
            "training_compute_reference_year": float(cfg.TRAINING_COMPUTE_REFERENCE_YEAR),
            "training_compute_reference_ooms": float(cfg.TRAINING_COMPUTE_REFERENCE_OOMS),
            "software_progress_scale_reference_year": float(cfg.SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR),
            "base_for_software_lom": float(cfg.BASE_FOR_SOFTWARE_LOM),
        },
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
