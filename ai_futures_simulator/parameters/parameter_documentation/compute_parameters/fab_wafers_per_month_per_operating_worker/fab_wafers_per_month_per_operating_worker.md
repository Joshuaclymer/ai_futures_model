# fab_wafers_per_month_per_operating_worker

| Modal value | Description |
|-------------|-------------|
| 24.64 wafers/month | Number of wafers per month that each operating worker can support in a fab |

**How we chose this parameter**

Fab production capacity is constrained by either workforce or lithography scanners, whichever is more limiting. This parameter captures the labor productivity side of that constraint.

![Operating labor vs production](/parameter_docs/compute_parameters/fab_wafers_per_month_per_operating_worker/labor_vs_production.png)

The relationship between workers and output was estimated from publicly available data on fab workforce sizes and production capacities. Each worker supports approximately 24.6 wafers per month. Combined with the scanner constraint (each scanner processes approximately 1,000 wafers per month), the production formula is:

```
Production = min(24.6 × workers, 1000 × scanners)
```

This means a fab needs roughly 40 workers per scanner to be fully labor-staffed. Understaffing reduces output proportionally; overstaffing beyond this ratio provides no benefit.
