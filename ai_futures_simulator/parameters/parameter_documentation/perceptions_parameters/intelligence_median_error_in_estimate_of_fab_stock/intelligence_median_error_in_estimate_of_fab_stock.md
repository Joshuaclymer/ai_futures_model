# intelligence_median_error_in_estimate_of_fab_stock

| Modal value | Description |
|-------------|-------------|
| 7% | Median percentage error in US estimates of PRC semiconductor manufacturing equipment inventory |

**How we chose this parameter.**

This parameter captures how precisely US intelligence can estimate China's inventory of semiconductor manufacturing equipment. The key intuition is that if the US has a good sense of how many lithography scanners China possesses, then diverting a significant fraction of those scanners to a covert fab becomes risky—the discrepancy between expected and observed inventory would raise red flags. Conversely, if US estimates have wide error bars, small diversions can hide within the noise.

To calibrate this parameter, we looked at historical intelligence estimates of the sizes of adversary weapons stockpiles. Studies of these estimates from 1960–2025 show that US intelligence typically achieves median errors of around 15%. This provides a baseline, but semiconductor equipment differs from weapons systems in ways that make it easier to track.

China's semiconductor industry is more visible than its weapons programs for several reasons. The major Chinese chip manufacturers are quasi-private companies that report to investors and have some public disclosure requirements. The supply chain for semiconductor equipment is highly concentrated, with only a handful of suppliers worldwide—ASML dominates lithography, while Applied Materials, Lam Research, and others supply other critical tools. US export controls on advanced equipment have created extensive tracking and monitoring infrastructure. Additionally, this equipment requires ongoing maintenance and spare parts from the original manufacturers, which means those manufacturers have records of where their tools are installed and serviced.

Given these factors, we estimate that US intelligence can achieve roughly twice the precision for semiconductor equipment as for weapons systems, yielding a median error of about 7%. That said, some factors push toward higher error. China has been stockpiling equipment in anticipation of tighter export controls and may have unreported inventory. China's growing domestic equipment production also adds uncertainty, since domestically-produced tools are harder for US intelligence to track.
