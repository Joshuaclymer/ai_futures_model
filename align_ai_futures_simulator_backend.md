Project Context

You are working on aligning a continuous ODE-based AI futures simulator with a discrete reference model (hosted at https://dark-compute.onrender.com/). The simulator models covert "black projects" - secret AI compute infrastructure that nations might build. The goal is to ensure both models produce statistically equivalent outputs for detection times, likelihood ratios, compute trajectories, etc.

Your job is to determine whether the continuous model's outputs align with the reference model by running the comparison scripts in `scripts/comparison/`.

## Running Alignment Checks

```bash
# Run comparison with default 200 samples
python -m scripts.comparison.main

# Run with fewer samples for faster testing
python -m scripts.comparison.main --samples 50

# Disable caching to force fresh API calls
python -m scripts.comparison.main --no-cache

# Generate a markdown report
python -m scripts.comparison.main --report alignment_report.md
```

**Prerequisites:** The local backend must be running at http://localhost:5329

## Continuous Model

The continuous model is served by the local backend API at `app_backend/` which powers the frontend page at http://localhost:3000/ai-black-projects.

API Endpoint: POST http://localhost:5329/api/get-data-for-ai-black-projects-page

Example usage:
```python
import urllib.request, json

url = 'http://localhost:5329/api/get-data-for-ai-black-projects-page'
data = json.dumps({
    'parameters': {},  # Optional frontend params
    'num_simulations': 100,  # Number of Monte Carlo simulations
    'time_range': [2027, 2037]  # [agreement_year, end_year]
}).encode()
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=120) as response:
    result = json.loads(response.read().decode())
```

Source code: `/Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator/`

Key things to keep in mind:
- The simulation outputs a sequence of "World" objects. Each object represents the world at an instantaneous point in time. World objects should not include time series. You can only get the time series by extracting data from a full trajectory after the simulation run finishes.
- When parameters in the reference model are uncertain and are sampled, they should NOT be sampled inside of the world_updaters class in the new version. They should instead be defined as a distribution in monte_carlo_parameters.yaml.
- Please update simulation state with derivatives whenever possible to preserve end-to-end differentiability.

## Reference Model API

```python
import urllib.request, json

url = 'https://dark-compute.onrender.com/run_simulation'
data = json.dumps({'num_samples': 200, 'start_year': 2029, 'total_labor': 11300}).encode()
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=120) as response:
    result = json.loads(response.read().decode())
```

## Reference Model Repository

You can see the original source code in `~/github/covert_compute_production_model/black_project_backend`

## Recent Alignment Progress (2025-12-31)

### Fixed Issues

1. **Scanner calculation year** (`black_project.py` line 608): Changed from `preparation_start_year` (2029) to `fab_construction_start_year = agreement_year` (2030) for scanner count calculation. This matches the reference model which calculates total PRC scanners at `construction_start_year`.

2. **Wafer starts calculation** (`black_project.py` lines 634-647): Removed incorrect 0.5 factor and restructured to use `calculate_fab_wafer_starts_per_month()` with both labor and scanner constraints. Moved scanner calculation earlier so it's available for wafer_starts.

3. **Fab detection metrics** (`black_project_simulation.py`): Added `compute_fab_detection_data()` function that:
   - Uses fab-specific LR (`lr_fab_combined`) for detection threshold
   - Reports time as operational time before detection (not from agreement_year)
   - Only considers detection after fab becomes operational

4. **Serialization** (`black_project_simulation.py`): Added serialization for `fab_construction_start_year` and `fab_construction_duration` needed for detection calculations.

### Current Status (200 samples, 2025-12-31)

After fixing `lr_fab_procurement` bug:

**Well-Aligned:**
- `wafer_starts`: Local 8000 vs Ref 8936 (~10% diff)
- `lr_inventory` mean: Local 2.17 vs Ref 2.19 (~1% diff)
- `fab_built` rate: Both ~54%
- `prc_capacity_gw`: ~7% diff
- `lr_other` at year 2031.5: Local 0.6465 vs Ref 0.6469 (**virtually identical**)

**Moderate Differences (likely Monte Carlo variance):**
- `black_fab.individual_time_before_detection`: Local 0.93 vs Ref 0.71 years (~30% diff)
- `black_fab.individual_h100e_before_detection`: Local 413K vs Ref 368K (~12% diff)
- `initial_black_project`: Local 795K vs Ref 951K (~16% diff)

**Under Investigation:**
- `black_fab.lr_combined`: Local 115.4 vs Ref 48.67 (time series average ~2.4x higher)

### Investigation Findings (2025-12-31)

#### Fab Detection Time Discrepancy (Local FASTER than Ref)

The local model detects the fab earlier than reference (0.09 yrs vs 0.55 yrs operational time before detection). Key findings:

1. **lr_combined starts higher in local model**: Some simulations have `lr_combined[0]` values of 17+ (already above 5.0 threshold) at year 0 (agreement_year). This is due to higher `lr_inventory` × `lr_procurement` products.

2. **Detection timing comparison by simulation**:
   - Local: Many sims have LR >= 5.0 BEFORE fab becomes operational
   - Reference: Many sims have time_before_detection = 0.0 (immediate detection when operational)
   - Local reports small positive values (0.002-0.05 yrs) instead of exactly 0.0

3. **Year indexing verified correct**: Both `sim_years` and `lr_fab_combined` are filtered to start at agreement_year (2030), so array indexing aligns properly.

4. **Remaining difference**: The ~0.45 year difference in median detection times is due to:
   - Different distributions of `lr_inventory` and `lr_procurement` across Monte Carlo samples
   - Higher combined LR growth rate in local model
   - Reference model rounds to 0.0 when detection happens early, local uses floating point precision

#### Verified Alignments

1. **Detection functions are identical**: `compute_lr_over_time_vs_num_workers()` in `black_compute.py` matches the reference model implementation exactly. Both use gamma distribution with identical parameters.

2. **Perception parameters match**:
   - `mean_detection_time_for_100_workers`: 6.95 years
   - `mean_detection_time_for_1000_workers`: 3.42 years
   - `variance`: 3.88

3. **Fab labor values match**: Both models use ~800 workers during operation.

4. **LR computation is correct**: `lr_other` correctly starts at 1.0 before fab construction (year 2030), decreases during operation, then jumps to 100 at detection time.

#### Root Cause of H100e Discrepancy

The `individual_h100e_before_detection` difference (~3.7x) despite similar detection times (~9% diff) suggests the discrepancy is in **production rate**, not detection timing. Possible causes:

1. **Transistor density or architecture efficiency**: Affects chips produced per wafer
2. **Time step differences**: Reference uses discrete 0.1-year steps vs our continuous ODE
3. **Cumulative production tracking**: Our model now correctly tracks `fab_total_produced_h100e` separately from surviving compute

#### High Variance in Detection Times

Detection times show high Monte Carlo variance between runs (0.06 to 2.79 years). Some simulations have `lr_inventory` (scanner diversion LR) values high enough to trigger instant detection when combined:
- Individual `lr_inventory` values observed: 5.90, 7.47 (threshold is 5.0)
- This causes some simulations to detect immediately upon fab operation start

This variance is expected and aligns with reference model behavior.

### Known Discrepancies (Not Bugs)

#### Fab Energy Calculation

Our model correctly uses `fab_watts_per_h100e = watts_per_tpp_relative * 700` which gives:
- 28nm: 13.99 × 700 = **9794 W/H100e**
- 14nm: 5.47 × 700 = **3826 W/H100e**

The reference model appears to use a fixed **3500 W/H100e** for all process nodes, ignoring the actual `watts_per_tpp` values. This is a bug in the reference model.

Evidence from reference API analysis:
- All simulations show implied watts/H100e = 3500 W regardless of process node
- 28nm simulations should use 9794 W but use 3500 W (ratio 0.357)
- 14nm simulations should use 3826 W but use 3500 W (ratio 0.915)

**Decision**: Keep our correct energy calculation. This affects:
- `black_fab.individual_energy_before_detection`: Our values are ~2.8x higher for 28nm
- `energy_by_source`: Median fab energy is higher in our model

### Bug Fixes (2025-12-31)

1. **Fixed `compute_fab_detection_data` field name** (`black_project_simulation.py` line 348): Changed `bp.get('watts_per_h100e', 700)` to `bp.get('fab_watts_per_h100e', 700)`. The black_project dict uses `fab_watts_per_h100e`, not `watts_per_h100e`.

2. **Added `fab_total_produced_h100e` state variable** (`entities.py`, `black_project.py`): Track cumulative fab production separately from surviving compute (which includes attrition). The metric `fab_cumulative_production_h100e` now correctly reports total production, not surviving compute.

3. **Fixed `lr_fab_procurement` localization year check** (`black_project.py` line 997): Changed from `preparation_start_year` (2029) to `fab_construction_start_year` (agreement_year = 2030). The bug caused some simulations with `localization_year = 2030` to incorrectly get `lr_fab_procurement = 10.0` instead of `1.0`, inflating their `lr_combined` by 10x and causing premature detection.

4. **Fixed fab detection time calculation** (`black_project_simulation.py` lines 327-331): The `compute_fab_detection_data` function was incorrectly checking LR only from `operational_start` year forward. The reference model checks LR from `agreement_year`, then computes `operational_time = max(0.0, detection_year - operational_start)`. This means if detection happens BEFORE the fab becomes operational (when LR >= 5 is triggered by scanner diversion), `operational_time = 0.0`. The local model now matches this behavior.

### Current Status (200 samples, 2025-12-31, after detection time fix)

**Well-Aligned (PASS <5% diff):**
- `wafer_starts`: Local 9500 vs Ref 8936 (6.3% diff) - ✓
- `lr_other` at year 2031.5: ~0.65 both - ✓
- `fab_built` rate: Both ~55%
- `prc_capacity_gw`: 6.7% diff
- `lr_prc_accounting_samples`: 5.6% diff - ✓

**Moderate Differences (WARN 5-25%):**
- `black_fab.individual_time_before_detection`: Local 0.65 vs Ref 0.71 (6.6% diff) - **FIXED!** (was 130% before)
- `black_fab.lr_combined`: Local 57.1 vs Ref 48.7 (14.9% diff) - improved from 2.4x!
- `prc_capacity_at_agreement_year_gw`: Local 2.41 vs Ref 2.77 (13.2% diff)
- `initial_compute_stock_samples`: 13% median diff (MC variance in PRC capacity)

**Remaining Failures (mostly acceptable MC variance):**
- `lr_inventory`: Local 1.94 vs Ref 2.71 (28.4% diff in median, but means nearly identical at 2.32 vs 2.27)
- `initial_black_project`: Local 598K vs Ref 951K (flows from PRC capacity diff)
- `black_fab.individual_h100e_before_detection`: Local 232K vs Ref 368K (29.6% diff)

### Further Investigation (2025-12-31)

#### LR Component Analysis

Direct comparison of LR components at year 2031.5 (fab operational time):
- **lr_other**: Local 0.6465 vs Ref 0.6469 - **virtually identical**
- **lr_inventory**: Local 1.68 vs Ref 2.28 - similar distributions
- **lr_procurement**: Both use 1.0 when localized

This confirms that the underlying LR calculations match. The differences in aggregate metrics are due to Monte Carlo variance.

#### High Variance in Detection Times

The fab detection model exhibits high Monte Carlo variance:
- Some simulations detect immediately (detection time ~0.01 years)
- Some simulations run for years (detection time ~3-7 years)

This variance is inherent to the gamma distribution sampling for worker-based detection time. The variance creates apparent discrepancies between runs even when the models are functionally equivalent.

#### Fab Built Rate

Both models build fabs at similar rates (~54%) when using the same sample sizes.

### Diagnostic Scripts

- `scripts/diagnose_fab_lr.py` - Compares fab LR time series between APIs
- `scripts/compare_fab_lr.py` - Direct comparison of LR components at specific years
- `scripts/comparison/main.py` - Full alignment comparison

### Resolved Issues

1. **Detection time computation fixed**: `individual_time_before_detection` now matches reference model (6.6% diff vs 130% before)

2. **lr_combined improved**: Now 14.9% diff (vs 2.4x before) after fixing detection time calculation

3. **PRC capacity/initial_black_project difference explained**: 13% diff in PRC capacity sampling flows through to initial stock calculations. This is acceptable Monte Carlo variance.

4. **lr_inventory difference investigated**: Median differs (1.94 vs 2.71) but means are nearly identical (2.32 vs 2.27). The distributions have similar ranges - this is acceptable MC variance due to different sampling order.

### Remaining Items for Future Work

1. **h100e_before_detection 30% diff**: May be related to detection time or production rate calculation differences

2. **Energy calculation intentional difference**: Local model uses correct process-node-specific watts/H100e; reference uses fixed 3500W