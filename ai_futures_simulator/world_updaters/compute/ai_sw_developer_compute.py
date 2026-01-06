"""
AI Software Developer compute updater.

Updates compute stock for AI software developers based on their parent nation's compute.
The fraction of nation compute available to the largest developer is determined by parameters.
"""

import math
import torch
from torch import Tensor
from typing import Optional

from classes.world.world import World
from classes.world.entities import AISoftwareDeveloper, NamedNations
from classes.world.assets import Compute
from classes.simulation_primitives import WorldUpdater
from parameters.classes import SimulationParameters


class AISoftwareDeveloperComputeUpdater(WorldUpdater):
    """
    Updates compute for AI software developers based on nation compute.

    Each AI software developer gets a fraction of their nation's compute stock.
    The compute is then allocated according to the developer's compute_allocation.

    This updater should run AFTER NationComputeUpdater so nation compute is up to date.
    """

    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

    def _get_nation_for_developer(self, dev: AISoftwareDeveloper, world: World) -> Optional[str]:
        """
        Get the nation ID for a developer.

        For now, assumes US-based developers. Can be extended to support
        developer-nation mappings.
        """
        # Default to USA for now - could be extended with developer metadata
        return NamedNations.USA

    def _get_fraction_of_nation_compute(self, dev: AISoftwareDeveloper, nation_id: str) -> float:
        """
        Get the fraction of nation compute available to this developer.

        Uses parameter: proportion_of_compute_in_largest_ai_sw_developer
        """
        compute_params = self.params.compute

        if nation_id == NamedNations.USA:
            if hasattr(compute_params, 'USComputeParameters'):
                us_params = compute_params.USComputeParameters
                if hasattr(us_params, 'proportion_of_compute_in_largest_ai_sw_developer'):
                    return us_params.proportion_of_compute_in_largest_ai_sw_developer
        elif nation_id == NamedNations.PRC:
            if hasattr(compute_params, 'PRCComputeParameters'):
                prc_params = compute_params.PRCComputeParameters
                if hasattr(prc_params, 'proportion_of_compute_in_largest_ai_sw_developer'):
                    return prc_params.proportion_of_compute_in_largest_ai_sw_developer

        # Default fraction if parameter not found
        return 0.1  # 10% default

    def _get_training_compute_growth_rate(self, nation_id: str, current_time: float) -> float:
        """
        Get the training compute growth rate (OOMs/year) for a nation at a given time.

        This is derived from the nation's compute annual growth rate:
        training_compute_growth_rate = log10(annual_growth_rate)

        For example, 4x/year growth = log10(4) ≈ 0.6 OOMs/year

        After the slowdown_year, the rate is reduced to post_slowdown_operating_compute_growth_rate.
        """
        compute_params = self.params.compute

        annual_growth_rate = 1.0  # Default: no growth
        slowdown_year = float('inf')  # Default: no slowdown
        post_slowdown_rate = None

        if nation_id == NamedNations.USA:
            # US uses us_frontier_developer_operating_compute_annual_growth_rate
            # Note: The attribute is USComputeParameters (capitalized), not us_compute
            if hasattr(compute_params, 'USComputeParameters') and compute_params.USComputeParameters:
                us_params = compute_params.USComputeParameters
                if hasattr(us_params, 'us_frontier_developer_operating_compute_annual_growth_rate'):
                    annual_growth_rate = us_params.us_frontier_developer_operating_compute_annual_growth_rate
                if hasattr(us_params, 'slowdown_year'):
                    slowdown_year = us_params.slowdown_year
                if hasattr(us_params, 'post_slowdown_operating_compute_growth_rate'):
                    post_slowdown_rate = us_params.post_slowdown_operating_compute_growth_rate
        elif nation_id == NamedNations.PRC:
            # PRC uses annual_growth_rate_of_prc_compute_stock
            # Note: The attribute is PRCComputeParameters (capitalized), not prc_compute
            if hasattr(compute_params, 'PRCComputeParameters') and compute_params.PRCComputeParameters:
                prc_params = compute_params.PRCComputeParameters
                if hasattr(prc_params, 'annual_growth_rate_of_prc_compute_stock'):
                    annual_growth_rate = prc_params.annual_growth_rate_of_prc_compute_stock

        # After slowdown year, use the post-slowdown rate directly
        if current_time >= slowdown_year and post_slowdown_rate is not None:
            return post_slowdown_rate

        # Convert to OOMs/year: log10(growth_rate)
        if annual_growth_rate > 0:
            return math.log10(annual_growth_rate)
        return 0.0

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """
        Update AI software developer compute.

        For each developer:
        1. Apply growth to operating_compute based on growth rate
        2. Calculate frontier_training_compute from operating_compute * allocation
        3. Update training_compute_growth_rate for the current time
        """
        current_time = t.item() if isinstance(t, Tensor) else float(t)
        dt = world.dt if hasattr(world, 'dt') else 0.0
        if isinstance(dt, Tensor):
            dt = dt.item()

        for dev_id, dev in world.ai_software_developers.items():
            # Get nation for this developer
            nation_id = self._get_nation_for_developer(dev, world)
            if nation_id is None or nation_id not in world.nations:
                continue

            # Get the operating compute growth rate (OOMs/year)
            growth_rate_ooms_per_year = self._get_training_compute_growth_rate(nation_id, current_time)
            dev._set_frozen_field('training_compute_growth_rate', growth_rate_ooms_per_year)

            # Apply growth to operating_compute if we have a timestep
            if dt > 0 and dev.operating_compute:
                # Growth factor = 10^(growth_rate * dt)
                growth_factor = 10 ** (growth_rate_ooms_per_year * dt)

                # Get current operating compute
                current_compute = dev.operating_compute[0]
                current_functional = current_compute.functional_tpp_h100e
                if isinstance(current_functional, Tensor):
                    current_functional = current_functional.item()

                # Apply growth
                new_functional = current_functional * growth_factor

                # Update operating compute
                new_compute = Compute(
                    tpp_h100e_including_attrition=new_functional,
                    functional_tpp_h100e=new_functional,
                    watts_per_h100e=current_compute.watts_per_h100e,
                    average_functional_chip_age_years=current_compute.average_functional_chip_age_years,
                )
                dev.operating_compute[0] = new_compute

            # Always calculate allocation metrics from current operating_compute
            total_compute = sum(
                c.functional_tpp_h100e.item() if isinstance(c.functional_tpp_h100e, Tensor) else c.functional_tpp_h100e
                for c in dev.operating_compute
            ) if dev.operating_compute else 0.0
            ca = dev.compute_allocation

            dev._set_frozen_field(
                'ai_r_and_d_inference_compute_tpp_h100e',
                total_compute * ca.fraction_for_ai_r_and_d_inference
            )
            dev._set_frozen_field(
                'ai_r_and_d_training_compute_tpp_h100e',
                total_compute * ca.fraction_for_ai_r_and_d_training
            )
            dev._set_frozen_field(
                'external_deployment_compute_tpp_h100e',
                total_compute * ca.fraction_for_external_deployment
            )
            dev._set_frozen_field(
                'alignment_research_compute_tpp_h100e',
                total_compute * ca.fraction_for_alignment_research
            )
            dev._set_frozen_field(
                'frontier_training_compute_tpp_h100e',
                total_compute * ca.fraction_for_frontier_training
            )

        return world
