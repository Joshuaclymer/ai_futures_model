# Changes and TODOs for ProgressModelIncremental

## Summary

Created `ProgressModelIncremental`, a version of `ProgressModel` that supports incremental time-stepping instead of running `compute_progress_trajectory` all at once. This is useful for scenarios where inputs (human labor, inference compute, experiment compute) are determined dynamically rather than from a fixed time series.

## Files Created

### 1. `progress_model_incremental.py`
The main incremental model implementation with:

- **`IncrementalState`** dataclass - holds current time, progress, research_stock, training_compute_ooms
- **`IncrementalMetrics`** dataclass - comprehensive metrics including:
  - progress_rate, human_only_progress_rate
  - coding_labor, ai_coding_labor, human_coding_labor
  - research_effort, automation_fraction
  - software_progress_rate, training_compute_growth_rate
  - horizon, taste (AI research taste)

- **`ProgressModelIncremental`** class with key methods:
  - `increment(next_time, human_labor, inference_compute, experiment_compute, training_compute_growth_rate)` - steps forward in time using RK4 integration
  - `get_metrics(human_labor, inference_compute, experiment_compute, training_compute_growth_rate)` - returns comprehensive metrics at current state
  - `from_progress_model(progress_model, time_range, initial_progress)` - **IMPORTANT**: factory method that creates an incremental model from a calibrated ProgressModel, ensuring identical calibration

### 2. `test_incremental_model.py`
Numerical validation test that:
- Runs both full ProgressModel and incremental model
- Compares trajectories point-by-point
- Verifies relative differences are within 1% tolerance

### 3. `test_incremental_plot.py`
Visual comparison test that:
- Creates 2x2 plot comparing both models
- Shows: AI R&D Speedups, Progress, Research Stock, Model Differences
- Saves to `incremental_model_comparison.png`

## Key Implementation Details

### Using the Incremental Model

```python
from progress_model import ProgressModel, Parameters, TimeSeriesData
from progress_model_incremental import ProgressModelIncremental
import model_config as cfg
import copy

# 1. Create and calibrate a full ProgressModel first
params_dict = copy.deepcopy(cfg.DEFAULT_PARAMETERS)
params_dict['automation_interp_type'] = 'linear'  # Avoid logistic bug
params = Parameters(**{k: v for k, v in params_dict.items() if k in Parameters.__dataclass_fields__})

time_series = load_time_series("input_data.csv")
full_model = ProgressModel(params, time_series)

# 2. Create incremental model FROM the calibrated full model
time_range = [2018.0, 2035.0]
incr_model = ProgressModelIncremental.from_progress_model(full_model, time_range, initial_progress=0.0)

# 3. Step through time incrementally
for t in time_points[1:]:
    incr_model.increment(t, human_labor, inference_compute, experiment_compute, training_rate)
    metrics = incr_model.get_metrics(human_labor, inference_compute, experiment_compute, training_rate)
    # Use metrics.progress_rate, metrics.ai_coding_labor, etc.
```

### Important Notes

1. **Always use `from_progress_model()`** to create the incremental model - this ensures identical calibration (r_software, automation anchors, etc.)

2. **Use `automation_interp_type = 'linear'`** - there's a bug in the original ProgressModel where logistic interpolation fails when the upper automation anchor = 1.0 (causes `logistic_x0 = None`)

3. **Pass `initial_progress=0.0` explicitly** - the default `None` can cause issues

4. **Log-interpolate inputs** for exponential trends:
   ```python
   human_labor = np.exp(np.interp(t, time_series.time, np.log(time_series.L_HUMAN)))
   ```

## Tests Run Successfully

The user confirmed `test_incremental_plot.py` runs successfully and produces matching trajectories between the full model and incremental model.

## TODOs / Future Work

1. **Black site integration** - Currently ignored per user request. Could be added later.

2. **Fix logistic automation bug** - In `progress_model.py` around line 780, when `aut_2 = 1.0`, the logistic calculation fails. This is a bug in the original model.

3. **Consider caching** - The incremental model recomputes some values that could be cached for performance.

## File Locations

All files are in: `/Users/joshuaclymer/github/covert_compute_production_model/ai-futures-calculator/`

- `progress_model_incremental.py` - Main implementation
- `test_incremental_model.py` - Numerical validation
- `test_incremental_plot.py` - Visual comparison
- `input_data.csv` - Default input data file
