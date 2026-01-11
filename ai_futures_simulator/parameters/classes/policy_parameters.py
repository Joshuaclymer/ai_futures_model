"""
Policy parameters for modeling AI governance and international agreements.
"""

from dataclasses import dataclass

from parameters.classes.base_spec import BaseSpec
from parameters.distribution_spec import ParamValue


@dataclass
class PolicyParameters(BaseSpec):
    """
    Parameters for AI policy scenarios.

    Currently only contains the AI slowdown start year, which determines
    when international agreements take effect and covert development begins.
    """
    ai_slowdown_start_year: ParamValue = None
