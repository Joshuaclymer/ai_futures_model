# Detection Time

## Purpose

Models how long a covert project can operate before being discovered through leaks, defections, or surveillance. Larger workforces increase the probability of detection.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Mean detection time (100 workers) | 6.95 years | 1–20 years | Expected time to detection for a 100-person project |
| Mean detection time (1000 workers) | 3.42 years | 1–20 years | Expected time to detection for a 1000-person project |
| Detection time variance | 3.88 | 0.1–10 | Controls the spread of the detection time distribution |

## Model Specification

Detection time follows a Gamma distribution with mean inversely proportional to the logarithm of workforce size:

```
μ(N) = A / log₁₀(N)^B
```

Where:
- `μ(N)` is the mean detection time for N workers
- `A` and `B` are fitted constants derived from historical data
- Variance is proportional to the mean: `variance = σ² × μ`

This produces a heavy-tailed distribution, capturing the reality that some projects remain hidden much longer than expected.

## Calibration

We estimated this relationship using data from 12 covert nuclear weapons programs over the past 50 years, tracking workforce size and time to external discovery.

![Detection latency vs workers](/parameter_docs/perceptions_parameters/mean_detection_time_for_100_workers/detection_latency.png)

Key observations from historical data:
- Small teams (< 100 people) can maintain secrecy for 5–10+ years
- Large programs (> 10,000 people) typically detected within 2–4 years
- Variance is substantial—some large programs evaded detection for extended periods

## Selection Reasoning

The logarithmic relationship captures two key dynamics:

1. **Diminishing marginal risk**: Adding workers to a small team significantly increases detection risk; adding workers to an already-large program has less marginal effect
2. **Heavy tails**: Real-world detection events show substantial variance, with some programs evading detection far longer than predicted by simple models
