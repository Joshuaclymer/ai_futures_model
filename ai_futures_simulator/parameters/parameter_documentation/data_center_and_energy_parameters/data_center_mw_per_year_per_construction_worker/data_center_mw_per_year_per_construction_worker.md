# data_center_mw_per_year_per_construction_worker

| Modal value | Description |
|-------------|-------------|
| 0.13 MW/worker/year | Datacenter capacity that can be constructed per worker per year under covert conditions |

**How we chose this parameter**

This parameter specifies datacenter construction rate per worker. We estimated this rate from publicly reported datacenter construction projects using data from Epoch's Frontier Datacenters dataset.

![Construction workers vs GW per year](/parameter_docs/data_center_and_energy_parameters/data_center_mw_per_year_per_construction_worker/workers_vs_gw_per_year.png)

[Download data](/api/parameter_docs/data_center_and_energy_parameters/data_center_mw_per_year_per_construction_worker/data_center_workers_vs_build_rate.csv)

Under normal (non-covert) conditions, datacenter construction achieves approximately 0.2 MW per worker per year. However, covert construction requires additional time due to security measures, compartmentalization, and the need to avoid attracting attention. Comparing declared versus undeclared nuclear enrichment facilities shows that undeclared facilities took approximately 1.5 times longer to build (5.5 years vs 3.7 years on average). Applying this secrecy overhead factor reduces the effective construction rate to approximately 0.13 MW per worker per year.
