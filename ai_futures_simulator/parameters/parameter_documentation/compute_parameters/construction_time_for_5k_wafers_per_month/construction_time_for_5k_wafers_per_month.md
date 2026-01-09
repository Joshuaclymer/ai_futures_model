# construction_time_for_5k_wafers_per_month

| Modal value | Description |
|-------------|-------------|
| 1.4 years | Time required to construct a fab with 5,000 wafers per month capacity, from groundbreaking to first production |

**How we chose this parameter**

Chip fabrication plants typically require 1 to 3 years from groundbreaking to first production, with larger fabs generally taking longer. We collected data on 16 fab construction projects spanning a range of capacities from 4,000 to 200,000 wafers per month. These include both completed projects and planned facilities with announced timelines.

The plot below shows construction time as a function of fab capacity. We fit a logarithmic regression to this data, which allows us to interpolate construction times for fabs of different sizes. At 5,000 wafers per month, the regression predicts a construction time of approximately 1 year.

![Construction time vs capacity](/parameter_docs/compute_parameters/construction_time_for_5k_wafers_per_month/construction_time_plot.png)

[Download data](/api/parameter_docs/compute_parameters/construction_time_for_5k_wafers_per_month/fab_construction_time.csv)

We also considered how the requirement that the construction must be performed in *secret* might lengthen the construction timeline. Comparing declared versus undeclared nuclear enrichment facilities provides a useful analogy: undeclared facilities took approximately 1.5 times longer to build (5.5 years vs 3.7 years on average). This suggests that the need for secrecy imposes meaningful overhead on construction projects. Applying this 1.5x factor to chip fabs yields estimated covert construction times of roughly 1.5 to 4.5 years depending on capacity.

![Adjusted construction time](/parameter_docs/compute_parameters/construction_time_for_5k_wafers_per_month/construction_time_plot_adjusted.png)