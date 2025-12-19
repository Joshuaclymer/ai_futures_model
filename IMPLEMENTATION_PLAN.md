# Implementation Plan: Aligning ai_futures_simulator with black_project_backend

## Phase 1: Complete Model Alignment Verification

### 1.1 Additional Distribution Comparisons Needed
Compare these additional metrics between discrete and continuous models:

| Metric | Location (Discrete) | Status |
|--------|---------------------|--------|
| Detection time distribution | `black_project.sampled_detection_time` | Need to compare |
| Labor over time | `labor_by_year` dictionary | Need to compare |
| Datacenter capacity growth | `black_datacenters.get_covert_GW_capacity_total()` | Need to compare |
| Operational compute | `get_operational_compute()` | Need to compare |
| H100-years cumulative | `h100_years_to_date()` | Need to compare |
| Survival rate | `surviving_compute_energy_by_source()` | Need to compare |
| Fab production (if enabled) | `get_cumulative_compute_production_over_time()` | Need to compare |
| Posterior probability | Bayesian update from cumulative LR | Need to compare |

### 1.2 Tasks
1. Update comparison script to include all metrics above
2. Run 50+ simulations for each model
3. Generate comparison tables and plots
4. Document any remaining discrepancies

---

## Phase 2: Add Metrics to ai_futures_simulator Classes

### 2.1 New Metrics Classes Structure

```
ai_futures_simulator/classes/
├── world/
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── black_project_metrics.py    # Per-project metrics
│   │   ├── detection_metrics.py        # Detection/LR metrics
│   │   ├── compute_metrics.py          # Compute production metrics
│   │   └── ccdf_metrics.py             # CCDF data structures
│   └── ...existing files
```

### 2.2 Metrics to Add (from format_data_for_black_project_plots.py)

#### BlackProjectMetrics (attributes only, in classes/world/metrics/)
```python
@dataclass
class BlackProjectMetrics:
    # Time series data (one value per timestep)
    operational_compute_h100e: List[float]
    surviving_compute_h100e: List[float]
    cumulative_h100_years: List[float]
    datacenter_capacity_gw: List[float]
    survival_rate: List[float]

    # Cumulative LR components
    cumulative_lr: List[float]
    lr_initial: List[float]
    lr_sme: List[float]
    lr_other: List[float]
    lr_energy: List[float]
    lr_satellite: List[float]
    lr_compute_accounting: List[float]

    # Posterior probability
    posterior_probability: List[float]

    # Detection outcomes (at simulation end or detection)
    detection_year: Optional[float]
    h100_years_at_detection: float
    operational_compute_at_detection: float
    time_to_detection: float
```

#### FabMetrics (if fab enabled)
```python
@dataclass
class FabMetrics:
    cumulative_production_h100e: List[float]
    is_operational: List[bool]
    proportion_operational: List[float]
    wafer_starts_per_month: float
    chips_per_wafer: float
    transistor_density_relative_to_h100: float
    architecture_efficiency: List[float]
    watts_per_tpp_relative_to_h100: float
    process_node: str
    construction_start_year: float
    construction_duration: float

    # Detection-specific
    lr_inventory: float
    lr_procurement: float
    lr_other: List[float]
    lr_combined: List[float]
```

#### DatacenterMetrics
```python
@dataclass
class DatacenterMetrics:
    concealed_capacity_gw: List[float]
    unconcealed_capacity_gw: float
    total_capacity_gw: List[float]
    operating_labor: List[int]
    lr_datacenters: List[float]
    lr_energy: List[float]
    lr_satellite: float
```

#### InitialStockMetrics
```python
@dataclass
class InitialStockMetrics:
    initial_prc_stock: float
    initial_diverted_compute: float
    diversion_proportion: float
    lr_prc_accounting: float
```

### 2.3 World Updater Changes

Add metric computation to `world_updaters/`:
- `black_project.py`: Add `compute_metrics()` method
- `compute/black_compute.py`: Add helper functions for metrics

---

## Phase 3: Update Comparison Script

### 3.1 Extended Comparison Script
Update `scripts/compare_lr_distributions.py` to:
1. Compare all time series metrics
2. Compare detection outcomes
3. Compare CCDF distributions
4. Generate comprehensive comparison report

### 3.2 Comparison Output Format
```
METRIC COMPARISON TABLE
=======================
| Metric                  | Discrete (median) | Continuous (median) | Diff (%) |
|-------------------------|-------------------|---------------------|----------|
| Detection rate (t+5)    | 94%               | 98%                 | 4%       |
| Cumulative LR (t+5)     | 112.1             | 131.1               | 17%      |
| ...                     | ...               | ...                 | ...      |
```

---

## Phase 4: Convert Frontend to React

### 4.1 Component Structure for app_frontend

```
app_frontend/
├── app/
│   ├── page.tsx                          # Home page
│   ├── black-project/
│   │   └── page.tsx                      # Black project simulator page
│   └── api/
│       └── black-project/route.ts        # API endpoint
├── components/
│   ├── black-project/
│   │   ├── BlackProjectDashboard.tsx     # Main dashboard
│   │   ├── TopSection.tsx                # Overview metrics + CCDFs
│   │   ├── RateSection.tsx               # Methodology explanation
│   │   ├── DetectionSection.tsx          # Detection probability breakdown
│   │   ├── CovertFabSection.tsx          # Fab production breakdown
│   │   ├── CovertDatacenterSection.tsx   # Datacenter capacity breakdown
│   │   ├── InitialStockSection.tsx       # Initial stock breakdown
│   │   └── ParameterSidebar.tsx          # Parameter input controls
│   ├── charts/
│   │   ├── CCDFChart.tsx                 # CCDF visualization
│   │   ├── TimeSeriesChart.tsx           # Time series with percentiles
│   │   ├── BreakdownEquation.tsx         # Visual equation component
│   │   ├── MetricCard.tsx                # Dashboard metric display
│   │   └── DistributionHistogram.tsx     # PDF/histogram plot
│   └── ui/
│       ├── Tooltip.tsx                   # Tooltip system
│       └── Slider.tsx                    # Parameter slider
├── constants/
│   ├── blackProjectParameters.ts         # Parameter definitions
│   └── chartConfig.ts                    # Chart styling constants
└── types/
    └── blackProject.ts                   # TypeScript types
```

### 4.2 Migration Strategy

1. **Create new page**: `/black-project` route
2. **Convert sections one at a time**:
   - Start with TopSection (dashboard + CCDFs)
   - Then RateSection
   - Then DetectionSection
   - Then CovertFabSection
   - Then CovertDatacenterSection
   - Then InitialStockSection
   - Finally ParameterSidebar
3. **Preserve existing styles**: Use existing Tailwind conventions from app_frontend
4. **Create reusable chart components** that match black_project_frontend visualizations

### 4.3 Styling Guidelines
- Use Tailwind CSS (existing convention in app_frontend)
- Use existing color palette from app_frontend
- Use et-book font for headings
- Preserve responsive behavior from black_project_frontend
- Use CSS variables for theming

---

## Phase 5: Integration Testing

### 5.1 End-to-End Tests
1. Run simulations through both backends
2. Compare API response structures
3. Verify frontend renders correctly
4. Test parameter updates propagate correctly

### 5.2 Visual Regression Testing
- Screenshot comparison between old and new frontends
- Verify all plots render correctly
- Check responsive behavior

---

## Implementation Order

1. **Week 1**: Phase 1 - Complete model alignment verification
2. **Week 2**: Phase 2 - Add metrics to ai_futures_simulator classes
3. **Week 3**: Phase 3 - Update comparison script, verify all metrics match
4. **Week 4-5**: Phase 4 - Convert frontend components to React
5. **Week 6**: Phase 5 - Integration testing and polish

---

## Files to Create/Modify

### New Files
- `ai_futures_simulator/classes/world/metrics/__init__.py`
- `ai_futures_simulator/classes/world/metrics/black_project_metrics.py`
- `ai_futures_simulator/classes/world/metrics/detection_metrics.py`
- `ai_futures_simulator/classes/world/metrics/compute_metrics.py`
- `app_frontend/app/black-project/page.tsx`
- `app_frontend/components/black-project/*.tsx` (multiple files)
- `app_frontend/components/charts/*.tsx` (multiple files)
- `app_frontend/types/blackProject.ts`

### Modified Files
- `ai_futures_simulator/world_updaters/black_project.py`
- `ai_futures_simulator/world_updaters/compute/black_compute.py`
- `ai_futures_simulator/scripts/compare_lr_distributions.py`
- `app_frontend/components/HeaderContent.tsx` (add navigation)
