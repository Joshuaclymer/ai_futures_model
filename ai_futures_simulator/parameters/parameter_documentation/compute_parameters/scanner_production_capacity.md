# Scanner Production Capacity

Photolithography scanners are typically the production bottleneck in chip fabrication. These machines pattern circuits onto silicon wafers and cost over $100 million each.

## Capacity calculation

ASML DUV machines pattern approximately 250 wafers per hour (~180,000 wafers/month). With ~80 patterning steps per chip and ~50% utilization:

```
0.5 × 180,000 / 80 = ~1,000 completed wafers per month per scanner
```

## Production formula

Fab output is constrained by the limiting factor—workers or scanners:

```
Production = min(24.6 × workers, 1000 × scanners)
```
