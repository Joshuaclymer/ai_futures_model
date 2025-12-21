# Prior Odds of Covert Project

## Purpose

Represents the baseline probability that US intelligence assigns to the existence of a covert AI project before observing any evidence.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Prior probability | — | 0–1 | Initial belief probability that a covert project exists |

## How It Works

The model uses Bayesian updating to track how intelligence estimates evolve over time:

```
Posterior = Prior × Likelihood Ratio
```

When evidence of discrepancies (chip stock, energy, etc.) is observed, the likelihood ratio updates the probability estimate.

## Model Note

This parameter is a rough estimate and is **not load-bearing** to model predictions. Detection is treated as a relative change in posterior probability (i.e., when the likelihood ratio exceeds a threshold), not as an absolute probability crossing a threshold.

This means:
- The exact prior value doesn't significantly affect detection timing
- What matters is the *rate* at which evidence accumulates
- Detection occurs when cumulative evidence provides sufficient update

## Selection Reasoning

The prior is kept as a free parameter because:
- Reasonable people disagree about baseline probabilities
- Geopolitical context changes the prior significantly
- The model's key outputs (time to detection, compute accumulated) are robust to prior assumptions
