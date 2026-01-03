"""
State perceptions of covert compute world updater.

Updates entity perceptions about the probability that other entities
have covert AI projects based on cumulative likelihood ratios.

Uses Bayesian updating:
- Convert prior probability to odds: O = P / (1 - P)
- Multiply by cumulative likelihood ratio: O' = O * LR
- Convert back to probability: P' = O' / (1 + O')
"""

from torch import Tensor

from classes.world.world import World
from classes.world.entities import NamedNations
from classes.world.perceptions import Perceptions
from classes.simulation_primitives import StateDerivative, WorldUpdater
from parameters.simulation_parameters import SimulationParameters
from parameters.perceptions_parameters import PerceptionsParameters


def probability_to_odds(probability: float) -> float:
    """Convert probability to odds."""
    if probability >= 1.0:
        return float('inf')
    if probability <= 0.0:
        return 0.0
    return probability / (1.0 - probability)


def odds_to_probability(odds: float) -> float:
    """Convert odds to probability."""
    if odds == float('inf'):
        return 1.0
    if odds <= 0.0:
        return 0.0
    return odds / (1.0 + odds)


def bayesian_update_probability(prior: float, likelihood_ratio: float) -> float:
    """
    Update a probability using Bayesian inference with a likelihood ratio.

    P(H|E) = P(H) * LR / (P(H) * LR + (1 - P(H)))

    Or equivalently using odds:
    - Convert to odds: O = P / (1 - P)
    - Update odds: O' = O * LR
    - Convert back: P' = O' / (1 + O')

    Args:
        prior: Prior probability P(H)
        likelihood_ratio: LR = P(E|H) / P(E|~H)

    Returns:
        Posterior probability P(H|E)
    """
    prior_odds = probability_to_odds(prior)
    posterior_odds = prior_odds * likelihood_ratio
    return odds_to_probability(posterior_odds)


class StatePerceptionsOfCovertComputeUpdater(WorldUpdater):
    """
    Updates state perceptions about covert AI projects.

    Specifically updates US perceptions of the probability that PRC
    has a covert AI project based on the cumulative likelihood ratio
    from the PRC black project simulation.

    This updater:
    - Does NOT contribute to continuous ODE dynamics (no state derivatives)
    - Updates perceptions as metrics after each ODE step
    """

    def __init__(
        self,
        params: SimulationParameters,
        perceptions_params: PerceptionsParameters,
    ):
        super().__init__()
        self.params = params
        self.perceptions_params = perceptions_params

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """No continuous dynamics - perceptions are updated discretely."""
        return StateDerivative.zeros(world)

    def set_state_attributes(self, t: Tensor, world: World) -> World | None:
        """No discrete state changes."""
        return None

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """
        Update US perceptions of PRC covert project probability.

        Uses Bayesian updating with the cumulative likelihood ratio
        from the PRC black project.
        """
        # Get the PRC black project (if it exists)
        prc_black_project = world.black_projects.get(NamedNations.PRC)

        if prc_black_project is None:
            # No PRC black project - nothing to update
            return world

        # Get or create US perceptions
        usa_perceptions = world.perceptions.get(NamedNations.USA)
        if usa_perceptions is None:
            # Initialize US perceptions with default values
            usa_perceptions = Perceptions(
                probability_entity_has_covert_AI_project={
                    NamedNations.PRC: self.perceptions_params.prior_probability_prc_has_covert_project
                }
            )
            world.perceptions[NamedNations.USA] = usa_perceptions

        # Always use the initial prior from parameters
        # (The cumulative_likelihood_ratio already represents all evidence accumulated
        # from the start, so we apply it to the original prior, not the previous posterior)
        prior_probability = self.perceptions_params.prior_probability_prc_has_covert_project

        # Get cumulative likelihood ratio from PRC black project
        cumulative_lr = prc_black_project.cumulative_lr

        # Perform Bayesian update: P(H|E) = prior * LR / (prior * LR + (1 - prior))
        posterior_probability = bayesian_update_probability(prior_probability, cumulative_lr)

        # Clamp to min/max bounds
        posterior_probability = max(
            self.perceptions_params.min_probability,
            min(self.perceptions_params.max_probability, posterior_probability)
        )

        # Update US perceptions
        usa_perceptions.probability_entity_has_covert_AI_project[NamedNations.PRC] = posterior_probability

        return world
