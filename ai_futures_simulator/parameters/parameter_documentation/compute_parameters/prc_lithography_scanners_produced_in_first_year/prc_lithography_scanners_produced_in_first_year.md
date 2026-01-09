# prc_lithography_scanners_produced_in_first_year

| Modal value | Description |
|-------------|-------------|
| 20 scanners | Number of lithography scanners China produces in the first year after achieving high-volume manufacturing capability |

**How we chose this parameter**

This parameter models the rate at which China could scale domestic lithography scanner production once it achieves the capability to manufacture them. Lithography scanners are the critical bottleneck for chip production, and China's ability to produce them domestically would significantly affect the viability of a covert fab.

We estimated this from ASML's historical production ramp-up for each scanner generation after achieving high-volume manufacturing:

![ASML lithography sales history](/parameter_docs/compute_parameters/prc_lithography_scanners_produced_in_first_year/lithography_sales_plot.png)

The trendline shows typical ramp-up trajectory. ASML typically produced around 20 scanners in the first year of volume production for each new generation, then scaled up from there. We assume China's domestic scanner production would follow a similar pattern, though possibly slower given the challenges of building a supply chain from scratch.
