"""
Policy parameters for modeling AI governance and international agreements.

NOTE: Default values are NOT stored here. All defaults are in modal_parameters.yaml.
These dataclasses define the structure only.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PolicyParameters:
    """
    Parameters for AI policy scenarios.

    Currently only contains the AI slowdown start year, which determines
    when international agreements take effect and covert development begins.
    """

    # Year when AI slowdown agreement takes effect
    ai_slowdown_start_year: float
