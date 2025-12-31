"""
Comparison module for validating the local API against the reference model.

This module provides tools for:
- Fetching data from the reference API (discrete model)
- Fetching data from the local API (continuous ODE model)
- Automatically comparing all shared keys between APIs
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
from .local_simulator import fetch_local_api, clear_local_cache
from .auto_compare import (
    KeyComparisonResult,
    ComparisonSummary,
    compare_apis,
    print_comparison_results,
)
from .main import run_comparison

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
    'clear_local_cache',
    # Auto compare
    'KeyComparisonResult',
    'ComparisonSummary',
    'compare_apis',
    'print_comparison_results',
    # Main
    'run_comparison',
]
