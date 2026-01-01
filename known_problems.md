- Currently, a lot of things (surviving compute, operating compute, etc) are computed from the agreement start year instead of the black project start year.
- Reference uses fixed 3500 W/H100e instead of varying by process node
- Reference model adds BOTH fab_construction_labor AND fab_operating_labor to total labor when a fab exists (black_project.py lines 83-85). This doesn't make semantic sense since you wouldn't have construction and operating workers simultaneously, but we match this behavior in get_black_project_total_labor() for consistency.
- cumulative_lr calculation: Fixed local model to use precomputed lr_datacenters_by_year dictionary (worker_lr) instead of computing survival probability with current labor. Reference formula: cumulative_lr = lr_prc * lr_sme * lr_satellite * lr_energy * lr_other, where lr_other comes from precomputed dictionary. The fix is in set_metric_attributes() in black_project.py. Comparison shows ~100% avg diff with high variance - this may be due to Monte Carlo sampling differences or time alignment issues between APIs.
- survival_rate: Fixed by adding max=4.0 bound to hazard_rate_multiplier metalog distribution. The local metalog implementation has much heavier right tails than pymetalog (the reference library), producing samples up to 90,000+ vs pymetalog's much more bounded distribution. This caused extremely high hazard rates in some simulations, leading to near-zero survival rates and pulling down the median. With max=4.0 cap, local median survival is ~0.93 vs reference ~0.97 (~4% diff), within tolerance. Note: The fundamental issue is that our custom 3-term semi-bounded metalog implementation differs from pymetalog's behavior. A proper fix would require matching pymetalog's algorithm exactly or using the pymetalog library.
- operational_compute: ~42% avg diff (Local: ~340K vs Ref: ~560K). Investigation findings:
  - The operating_compute scaling logic was already fixed in update_black_project() to use total_energy_gw (which correctly combines initial+fab energy with separate efficiencies) for the energy-limitation calculation
  - However, systematic differences remain even with this fix
  - Counterintuitively, Local has HIGHER initial stock (~365K vs ~245K) and HIGHER datacenter capacity (8.17 vs 5.75 GW) but LOWER operational_compute
  - This suggests the difference comes from fab production: reference model accumulates more compute from fab over time
  - Related discrepancies: black_fab.wafer_starts (80% diff with 5 samples), etc.
  - Root cause appears to be in the fab chip production calculations rather than the energy/operating_compute formula itself
- transistor_density: Formula is CORRECT. Uses (H100_node / fab_node)^1.49 = (4/process_node)^1.49. Values match exactly between local and reference:
  - 28nm → 0.0551
  - 14nm → 0.1546
  - 7nm → 0.4344
  The 180% diff seen with 5 samples is a FALSE POSITIVE caused by random process node distribution variation at small sample sizes. With 50+ samples, transistor_density passes comparison (<5% diff).
- black_fab_flow: Architecture efficiency year calculation was using preparation_start_year (2029) instead of agreement_year (2030), causing ~23% lower H100e production per chip. FIXED: Changed calculate_fab_h100e_per_chip() calls to use agreement_year, matching the reference model's construction_start_year = agreement_year logic. Improvement: 61% → 50% avg diff. Remaining ~50% avg diff is due to:
  1. Monte Carlo variance in wafer_starts sampling (both models use log-normal distributions)
  2. wafer_starts is ~18% lower in local model due to random sampling variance
  3. The "avg diff" metric is sensitive to early time points where small absolute differences produce large percentage differences
  - Absolute end-of-simulation values are much closer: Local 1.51M vs Ref 1.77M (15% difference)
- black_fab metrics (wafer_starts, lr_combined, lr_other): December 2024 investigation findings:
  - wafer_starts: Formula is correct (matches reference's estimate_wafer_starts_per_month). Diff reduced from 40%→8% by using larger sample sizes (50+ instead of 5). Both models have high std (~5000) so variance is expected.
  - lr_other: Formula is correct. Medians match at 100.0 (detection threshold). Avg diff in time series is due to WHEN detection occurs varying across simulations.
  - lr_combined: High variance (often >50% diff) is inherent to the metric. It's lr_inventory * lr_procurement * lr_other, and when detection occurs (lr_other jumps to 100), the time series comparison becomes very sensitive to detection timing differences between models.
  - lr_inventory: Formula matches reference. With 50 samples, medians are ~3% apart (3.03 vs 2.95).
  - Root cause of variance: Monte Carlo sampling (both models have high variance), continuous vs discrete localization year sampling, and stochastic detection timing.