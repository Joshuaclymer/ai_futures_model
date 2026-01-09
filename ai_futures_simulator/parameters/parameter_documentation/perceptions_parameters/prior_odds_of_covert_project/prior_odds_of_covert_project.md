# prior_odds_of_covert_project

| Modal value | Description |
|-------------|-------------|
| 0.25 | Prior odds (not probability) that US intelligence assigns to the existence of a covert AI project before observing any evidence |

**How we chose this parameter**

This parameter represents the baseline belief that US intelligence assigns to the existence of a covert AI project before observing any evidence. The model uses Bayesian updating to track how intelligence estimates evolve over time: as evidence of discrepancies in chip stock, energy consumption, or other indicators is observed, the likelihood ratio updates the probability estimate.

This parameter is a rough estimate and is not load-bearing to model predictions. Detection is modeled as a relative change in posterior probability (specifically, when the likelihood ratio exceeds a threshold), not as an absolute probability crossing a threshold. This means the exact prior value doesn't significantly affect detection timing. What matters is the rate at which evidence accumulates. Detection occurs when cumulative evidence provides sufficient update to cross the threshold.

We keep the prior as a configurable parameter because reasonable people disagree about baseline probabilities and because geopolitical context changes the prior significantly. The model's key outputs—time to detection and compute accumulated before detection—are robust to reasonable prior assumptions.
