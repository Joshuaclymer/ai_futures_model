"""
Comparison module for validating ai_futures_simulator against the reference model.

This module provides tools for:
- Fetching data from the reference API (discrete model)
- Fetching data from ai_futures_simulator (continuous ODE model)
- Automatically comparing all shared keys between APIs
- Generating alignment reports
"""

from .config import (
    REFERENCE_API_URL,
    CACHE_DIR,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_START_YEAR,
    DEFAULT_END_YEAR,
)
from .reference_api import fetch_reference_api
from .ai_futures_simulator_api import fetch_ai_futures_simulator
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
    'CACHE_DIR',
    'DEFAULT_NUM_SIMULATIONS',
    'DEFAULT_START_YEAR',
    'DEFAULT_END_YEAR',
    # APIs
    'fetch_reference_api',
    'fetch_ai_futures_simulator',
    # Auto compare
    'KeyComparisonResult',
    'ComparisonSummary',
    'compare_apis',
    'print_comparison_results',
    # Main
    'run_comparison',
]
