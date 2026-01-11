"""
Software R&D module.

This module contains:
- SoftwareRAndD: World updater for AI software progress
- Core model functions for AI progress computation
"""

from .update_software_r_and_d import SoftwareRAndD

# Re-export core types
from .data_types import TimeSeriesData, AnchorConstraint, InitialConditions

# Re-export main classes
from .automation_model import AutomationModel
from .taste_distribution import TasteDistribution, get_or_create_taste_distribution

# Re-export CES functions
from .ces_functions import (
    _ces_function,
    compute_coding_labor_deprecated,
    compute_rho_from_asymptotes,
    compute_alpha_experiment_capacity_from_asymptotes,
    compute_experiment_compute_exponent_from_anchor,
    compute_exp_capacity_params_from_anchors,
)

# Re-export progress rate functions
from .progress_rate import (
    compute_research_effort,
    compute_software_progress_rate,
    compute_overall_progress_rate,
    compute_automation_fraction,
    compute_ai_research_taste,
    compute_aggregate_research_taste,
    progress_rate_at_time,
)

# Re-export utilities
from .utils import (
    should_reraise,
    _coerce_float_scalar,
    _gauss_hermite_expectation,
    _log_interp,
)

__all__ = [
    # World updater
    'SoftwareRAndD',

    # Types
    'TimeSeriesData',
    'AnchorConstraint',
    'InitialConditions',

    # Main classes
    'TasteDistribution',
    'AutomationModel',

    # Factory functions
    'get_or_create_taste_distribution',

    # CES functions
    '_ces_function',
    'compute_coding_labor_deprecated',
    'compute_rho_from_asymptotes',
    'compute_alpha_experiment_capacity_from_asymptotes',
    'compute_experiment_compute_exponent_from_anchor',
    'compute_exp_capacity_params_from_anchors',

    # Research functions
    'compute_research_effort',
    'compute_software_progress_rate',
    'compute_overall_progress_rate',
    'compute_automation_fraction',
    'compute_ai_research_taste',
    'compute_aggregate_research_taste',
    'progress_rate_at_time',

    # Utilities
    'should_reraise',
    '_coerce_float_scalar',
    '_gauss_hermite_expectation',
    '_log_interp',
]
