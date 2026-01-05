# Operating Labor and Production

Fab production capacity is constrained by either workforce or lithography scanners, whichever is more limiting:

![Operating labor vs production](/parameter_docs/compute_parameters/labor_vs_production.png)

## Production formula

Output equals the minimum of labor-constrained and scanner-constrained capacity:

```
Production = min(24.6 × workers, 1000 × scanners)
```

Each worker supports approximately 24.6 wafers per month; each scanner processes approximately 1,000 wafers per month.
