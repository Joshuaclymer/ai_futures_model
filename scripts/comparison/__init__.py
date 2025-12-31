"""
Comparison module for validating the local API against the reference model.

This module provides tools for:
- Fetching data from the reference API (discrete model)
- Fetching data from the local API (continuous ODE model)
- Comparing metrics between models (each metric in its own module)
- Generating alignment reports
"""

from .config import (
    REFERENCE_API_URL,
    LOCAL_API_URL,
    CACHE_DIR,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_START_YEAR,
    DEFAULT_AGREEMENT_YEAR,
    DEFAULT_NUM_YEARS,
    DEFAULT_TOTAL_LABOR,
)
from .reference_api import fetch_reference_api
from .local_simulator import fetch_local_api, extract_local_metrics, clear_local_cache
from .metrics import (
    ComparisonResult,
    ALL_COMPARISONS,
    compare_survival_rate,
    compare_cumulative_lr,
    compare_posterior_prob,
    compare_lr_other_intel,
    compare_lr_reported_energy,
    compare_lr_prc_accounting,
    compare_operating_compute,
    compare_datacenter_capacity,
)
from .reporting import print_results, generate_markdown_report

__all__ = [
    # Config
    'REFERENCE_API_URL',
    'LOCAL_API_URL',
    'CACHE_DIR',
    'DEFAULT_NUM_SAMPLES',
    'DEFAULT_START_YEAR',
    'DEFAULT_AGREEMENT_YEAR',
    'DEFAULT_NUM_YEARS',
    'DEFAULT_TOTAL_LABOR',
    # APIs
    'fetch_reference_api',
    'fetch_local_api',
    'extract_local_metrics',
    'clear_local_cache',
    # Metrics
    'ComparisonResult',
    'ALL_COMPARISONS',
    'compare_survival_rate',
    'compare_cumulative_lr',
    'compare_posterior_prob',
    'compare_lr_other_intel',
    'compare_lr_reported_energy',
    'compare_lr_prc_accounting',
    'compare_operating_compute',
    'compare_datacenter_capacity',
    # Reporting
    'print_results',
    'generate_markdown_report',
]
