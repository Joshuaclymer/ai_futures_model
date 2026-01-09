# wafers_per_month_per_lithography_scanner

| Modal value | Description |
|-------------|-------------|
| 1,000 wafers/month | Number of completed wafers a single lithography scanner can produce per month |

**How we chose this parameter**

Photolithography scanners are typically the production bottleneck in chip fabrication. These machines pattern circuits onto silicon wafers and cost over $100 million each.

ASML DUV machines can pattern approximately 250 wafers per hour, which translates to roughly 180,000 wafers per month at full utilization. However, each chip requires approximately 80 patterning steps (each layer of the circuit must be patterned separately), and machines typically run at around 50% utilization due to maintenance, changeovers, and other downtime. This yields:

```
0.5 × 180,000 / 80 ≈ 1,000 completed wafers per month per scanner
```

Combined with the labor constraint (24.6 wafers per worker per month), fab output is determined by whichever factor is more limiting:

```
Production = min(24.6 × workers, 1000 × scanners)
```
