# intelligence_median_error_in_satellite_estimate_of_datacenter_capacity

| Modal value | Description |
|-------------|-------------|
| 5% | Median percentage error in satellite-based estimates of PRC datacenter capacity |

**How we chose this parameter**

This parameter captures how precisely US intelligence can estimate China's datacenter capacity using satellite imagery. If satellite-based estimates are highly accurate, then diverting existing datacenter capacity to a covert project becomes risky because the discrepancy would be observable from space.

Physical datacenter infrastructure is relatively easy to observe from satellites. Datacenters have large building footprints, distinctive cooling systems (cooling towers, chillers), and thermal signatures from continuous operation. Power infrastructure such as substations and transmission lines provides additional indicators. Datacenters also follow consistent location patterns, clustering near power sources, fiber optic lines, and water.

Modern satellite imagery has very high resolution, and thermal imaging can detect operating facilities even when visual inspection is inconclusive. For facilities that are not specifically built for concealment, we estimate satellite-based datacenter capacity estimates have approximately 5% median error.

This parameter specifically applies to existing, unconcealed facilities. Purpose-built covert facilities that use camouflage, distributed designs, or underground construction would require different detection methods such as human intelligence or energy accounting.
