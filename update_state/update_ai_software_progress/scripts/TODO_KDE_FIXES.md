# TODO: Fix KDE Functions to Use Gamma Kernel with present_day

## Overview
Several scripts currently use `make_gaussian_kde` which does not enforce a lower bound at `present_day`. This allows probability mass to appear in the past (before the run's `present_day` value), which is incorrect.

All scripts need to be updated to:
1. Import `make_gamma_kernel_kde` instead of `make_gaussian_kde`
2. Import `load_present_day` from `plotting_utils.helpers`
3. Add `present_day` parameter to functions
4. Auto-load `present_day` if not provided
5. Filter data to remove times before `present_day`
6. Use gamma kernel KDE with `lower_bound=present_day`

## Completed Files ✅
- [x] `scripts/milestone_pdfs.py` - Fixed by user
- [x] `scripts/plot_post_sar_milestone_pdfs.py` - Fixed by user

## Files Requiring Fixes

### 1. `scripts/plot_transition_duration_pdf.py`
**Status:** ❌ Not fixed

**Current Issues:**
- Uses `make_gaussian_kde(data)` at line 70
- No `present_day` parameter in function signature
- No filtering of past data
- KDE range computed from raw data (line 73-74)

**Required Changes:**
1. Import `make_gamma_kernel_kde` instead of `make_gaussian_kde`
2. Import `load_present_day` from `plotting_utils.helpers`
3. Add `present_day: Optional[float] = None` parameter to function
4. Add auto-loading: `if present_day is None: present_day = load_present_day(out_path.parent)`
5. Filter data: Add `boundary = float(present_day)`, `valid_mask = raw_data >= boundary`, `data = raw_data[valid_mask]`
6. Replace `kde = make_gaussian_kde(data)` with `kde = make_gamma_kernel_kde(data, lower_bound=present_day)`
7. Update range calculation to respect `present_day`: `min_eval = max(present_day, float(np.min(data) - 5.0 * bw))`

**Reference:** See `plot_post_sar_milestone_pdfs.py` lines 32-34, 41, 51-53, 82-102, 111-121

---

### 2. `scripts/short_timelines_analysis.py`
**Status:** ❌ Not fixed

**Current Issues:**
- Has its own **local override** of `plot_milestone_pdfs_overlay()` function (line 240-247)
- Uses `make_gaussian_kde` instead of gamma kernel
- No `present_day` parameter accepted
- No filtering of past data

**Required Changes:**
**Option A (Recommended):** Remove the local override and use the fixed version from `milestone_pdfs.py`:
1. Delete lines 240-331 (the entire local `plot_milestone_pdfs_overlay` function)
2. Import from milestone_pdfs: `from milestone_pdfs import plot_milestone_pdfs_overlay`
3. Update the call at line 472 to pass `present_day` if needed

**Option B:** Fix the local function to match the pattern:
1. Import `make_gamma_kernel_kde` and `load_present_day`
2. Add `present_day` parameter
3. Auto-load `present_day` if not provided
4. Filter data before creating KDE
5. Use `make_gamma_kernel_kde(data, lower_bound=present_day)`
6. Update range calculation

**Note:** This file is likely a candidate for deletion/consolidation since it duplicates functionality.

---

### 3. `scripts/create_special_parameter_splits.py`
**Status:** ❌ Not fixed (but calls fixed function)

**Current Issues:**
- Calls `plot_milestone_pdfs_overlay()` without passing `present_day` at line 149

**Required Changes:**
1. **No changes needed** - This file already imports the fixed `plot_milestone_pdfs_overlay` from `plot_rollouts.py`
2. The function will auto-load `present_day` internally
3. **Test to verify:** Run the script and confirm it prints messages about dropped samples before present_day

**Priority:** Low (should work automatically after milestone_pdfs.py fix)

---

### 4. `scripts/run_extended_analyses.py`
**Status:** ❌ Not fixed (but calls fixed function)

**Current Issues:**
- Calls `plot_milestone_pdfs_overlay()` without passing `present_day` at lines 112 and 117

**Required Changes:**
1. **No changes needed** - This file imports the fixed `plot_milestone_pdfs_overlay` from `plot_rollouts.py`
2. The function will auto-load `present_day` internally
3. **Test to verify:** Run the script and confirm it prints messages about dropped samples before present_day

**Priority:** Low (should work automatically after milestone_pdfs.py fix)

---

### 5. `scripts/run_additional_analyses.py`
**Status:** ❌ Not fixed (but calls fixed function)

**Current Issues:**
- Calls `plot_milestone_pdfs_overlay()` without passing `present_day` at lines 223, 228, 328, 333

**Required Changes:**
1. **No changes needed** - This file imports the fixed `plot_milestone_pdfs_overlay` from `plot_rollouts.py`
2. The function will auto-load `present_day` internally
3. **Test to verify:** Run the script and confirm it prints messages about dropped samples before present_day

**Priority:** Low (should work automatically after milestone_pdfs.py fix)

---

### 6. `scripts/split_by_growth_factor.py`
**Status:** ❌ Not fixed (but calls fixed function)

**Current Issues:**
- Calls `plot_milestone_pdfs_overlay()` without passing `present_day` at line 118

**Required Changes:**
1. **No changes needed** - This file imports the fixed `plot_milestone_pdfs_overlay` from `plot_rollouts.py`
2. The function will auto-load `present_day` internally
3. **Test to verify:** Run the script and confirm it prints messages about dropped samples before present_day

**Priority:** Low (should work automatically after milestone_pdfs.py fix)

---

## Testing Strategy

### For scripts that need actual fixes (1-2):
1. Run the script on `outputs/20251113_014921/`
2. Verify it prints: `"Using present_day = 2025.600"`
3. Verify it prints dropped sample messages: `"SAR-level-experiment-selection-skill: dropped N samples before present_day=2025.600"`
4. Check the output PNG - should have NO probability mass before 2025.6
5. Check the output CSV - `time_decimal_year` column should start at or after 2025.6

### For scripts that just import (3-6):
1. Run the script on `outputs/20251113_014921/`
2. Verify it prints dropped sample messages from the imported function
3. Check output plots have no mass in the past

---

## Priority Order

**High Priority (need actual code changes):**
1. `plot_transition_duration_pdf.py` - Standalone script with its own KDE logic
2. `short_timelines_analysis.py` - Has problematic local override

**Low Priority (should auto-fix via imports):**
3. `create_special_parameter_splits.py`
4. `run_extended_analyses.py`
5. `run_additional_analyses.py`
6. `split_by_growth_factor.py`

---

## Implementation Template

For scripts that need fixing, use this pattern (from `plot_post_sar_milestone_pdfs.py`):

```python
# 1. Imports
from plotting_utils.kde import make_gamma_kernel_kde
from plotting_utils.helpers import load_present_day

# 2. Function signature
def plot_function(
    ...,
    present_day: Optional[float] = None,  # Add this parameter
):
    # 3. Auto-load present_day
    if present_day is None:
        present_day = load_present_day(Path(rollouts_file).parent)

    # 4. Filter data
    raw_data = np.asarray(times, dtype=float)
    boundary = float(present_day)
    valid_mask = raw_data >= boundary
    data = raw_data[valid_mask]
    dropped = raw_data.size - data.size
    if dropped:
        print(f"  {name}: dropped {dropped} samples before present_day={boundary:.3f}.")
    if data.size < 2:
        print(f"Warning: Not enough data after enforcing present_day boundary.")
        return

    # 5. Create gamma kernel KDE
    kde = make_gamma_kernel_kde(data, lower_bound=present_day)
    bw = float(kde.bandwidth)

    # 6. Compute range respecting present_day
    min_eval = max(present_day, float(np.min(data) - 5.0 * bw))
    max_eval = max(float(np.max(data) + 5.0 * bw), min_eval + bw)
    xs = np.linspace(min_eval, max_eval, 512)
```

---

## Questions/Decisions

1. **Should we delete `short_timelines_analysis.py`'s local override?**
   - Recommendation: Yes, use the fixed version from `milestone_pdfs.py` instead

2. **Should we pass `present_day` explicitly in calling scripts?**
   - Recommendation: No, let the functions auto-load it for consistency

3. **What about scripts using other KDE methods?**
   - Check for any other uses of `scipy.stats.gaussian_kde` directly
   - Search: `grep -r "gaussian_kde" scripts/` to find all instances
