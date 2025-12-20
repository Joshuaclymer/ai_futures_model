# Detection Time

Detection latency depends on workforce size—larger workforces increase the probability of leaks, defections, or surveillance detection.

We estimated this relationship using data from 12 covert nuclear weapons programs over the past 50 years, tracking workforce size and time to external discovery.

![Detection latency vs workers](/parameter_docs/detection_latency.png)

## Model specification

Detection time follows a Gamma distribution with mean inversely proportional to the logarithm of workforce size:

```
μ(N) = A / log₁₀(N)^B
```

Variance is proportional to the mean (variance = σ² × μ), producing a heavy-tailed distribution.
