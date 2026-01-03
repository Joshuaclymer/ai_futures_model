"""
Utility functions for black project simulation.

Contains small helper functions used across multiple modules.
"""

import numpy as np
from typing import Optional


def to_float(value, default: float = 0.0) -> float:
    """Convert tensor or number to Python float, handling Infinity/NaN for JSON compatibility."""
    if value is None:
        return default
    if hasattr(value, 'item'):
        result = float(value.item())
    else:
        result = float(value)
    # Handle Infinity and NaN which are not valid JSON
    if np.isinf(result) or np.isnan(result):
        return default
    return result


def is_fab_built(bp_props, black_project_start_year: float) -> bool:
    """
    Determine if a fab is built based on localization years and minimum process node requirement.

    A fab is built if the best available process node (that is localized by black_project_start_year)
    meets the minimum requirement specified in black_fab_min_process_node.
    """
    if bp_props is None:
        return True

    min_node = bp_props.black_fab_min_process_node

    # Check each node from most advanced to least advanced
    localization_years = {
        7: bp_props.prc_localization_year_7nm,
        14: bp_props.prc_localization_year_14nm,
        28: bp_props.prc_localization_year_28nm,
    }

    # Find best available node that is localized by start year
    for node_nm in [7, 14, 28]:
        if localization_years[node_nm] <= black_project_start_year:
            # Found a localized node - check if it meets minimum requirement
            return node_nm <= min_node

    # No node is localized by start year
    return False
