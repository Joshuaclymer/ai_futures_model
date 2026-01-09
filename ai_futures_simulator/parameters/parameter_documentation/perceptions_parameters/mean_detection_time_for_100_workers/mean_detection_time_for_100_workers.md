# mean_detection_time_for_100_workers

| Modal value | Description |
|-------------|-------------|
| 6.95 years | Expected time until detection for a covert project with 100 workers |

**How we chose this parameter**

This parameter models how long a covert project can operate before being discovered through leaks, defections, or surveillance. The key insight is that larger workforces increase the probability of detection, but the relationship is logarithmic rather than linear: adding workers to a small team significantly increases detection risk, while adding workers to an already-large program has less marginal effect.

We estimated this relationship using data from 12 covert nuclear weapons programs over the past 50 years, tracking workforce size and time to external discovery. Detection time follows a Gamma distribution with mean inversely proportional to the logarithm of workforce size.

![Detection latency vs workers](/parameter_docs/perceptions_parameters/mean_detection_time_for_100_workers/detection_latency.png)

[Download data](/api/parameter_docs/perceptions_parameters/mean_detection_time_for_100_workers/nuclear_case_studies.csv)

The historical data shows that small teams (fewer than 100 people) can maintain secrecy for 5 to 10 or more years, while large programs (more than 10,000 people) are typically detected within 2 to 4 years. There is substantial variance in the data, with some large programs evading detection for extended periods. This variance is captured by modeling detection time with a heavy-tailed distribution.

![Bayesian fit](/parameter_docs/perceptions_parameters/mean_detection_time_for_100_workers/detection_latency_vs_workers_bayesian.png)
