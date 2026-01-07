# Known Problems

(No known problems at this time)

## Recently Fixed

### Covert computation time range (Fixed 2026-01-06)
**Problem:** The reference model computes covert computation (bp_h100_years) from agreement_year to detection_year, but our implementation was computing from the start of the simulation.

**Fix:** Updated `app_backend/endpoints/black_project/reduction_ratios.py` to filter bp_h100_years calculation to only include years >= agreement_year.

**Verification:** Comparison script shows 0.1% difference for ai_rd_reduction_ccdf metrics.
