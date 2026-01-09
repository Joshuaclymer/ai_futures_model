# intelligence_median_error_in_estimate_of_compute_stock

| Modal value | Description |
|-------------|-------------|
| 7% | Median percentage error in US estimates of PRC AI chip inventory |

**How we chose this parameter**

This parameter captures how precisely US intelligence can estimate China's inventory of AI chips. If the US has accurate estimates of how many AI chips China possesses, then diverting a significant fraction to a covert project becomes risky because the discrepancy between expected and observed inventory would raise red flags. Conversely, if US estimates have wide error bars, small diversions can hide within the noise.

To calibrate this parameter, we looked at historical intelligence estimates of the sizes of adversary weapons stockpiles. Studies of these estimates from 1960–2025 show that US intelligence typically achieves median errors of around 15%. This provides a baseline, but AI chips differ from weapons systems in ways that may make them easier to track.

AI chips flow through a highly specialized supply chain that creates visibility. Manufacturing is concentrated at a small number of foundries (primarily TSMC and Samsung for advanced chips). US export controls have created tracking infrastructure for chip shipments to China. Commercial applications require some transparency, as chips are used by companies that report to investors and have some public disclosure.

We estimate US intelligence error at approximately 7%—roughly half the historical weapons estimate error. However, China has developed domestic chip production and may use smuggling routes to acquire chips, which adds uncertainty to these estimates.

[Download data](/api/parameter_docs/perceptions_parameters/intelligence_median_error_in_estimate_of_fab_stock/us_intelligence_estimates.csv)
