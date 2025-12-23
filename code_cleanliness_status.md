# Code Cleanliness Status

This document tracks code quality improvements identified and their implementation status.

## Frontend Improvements

| Issue | Status | Notes |
|-------|--------|-------|
| **Dashboard duplication** - Same dashboard layout repeated across TopChartsSection, DatacenterSection, InitialStockSection, CovertFabSection | ACCEPTED - DONE | Created shared `Dashboard` and `DashboardItem` components |
| **Monolithic section files** - 700+ line section components | REJECTED | User prefers keeping as-is, 1000 lines is acceptable limit |
| **Massive CSS file** - 1284 lines in ai-black-projects.css | ACCEPTED - DONE | Extracted to component CSS files, reduced to 792 lines |
| **Duplicate formatters** - formatH100e, formatEnergy scattered | REJECTED | User doesn't want to consolidate |
| **Type assertions** - `as unknown as` casts without validation | ACCEPTED - DONE | Created `utils/typeGuards.ts` with validation functions |
| **ParameterSidebar size** - 702 lines | REJECTED | User says it's fine under 1000 line limit |
| **Inline styles mixed with CSS** - Inconsistent styling approach | ACCEPTED - TODO | Consolidate to CSS classes. Found 116 occurrences across 17 files. Highest counts: DatacenterSection.tsx (34), CovertFabSection (27), InitialStockSection (17), Tooltip (10) |
| **Missing error boundaries** - No error handling around charts | ACCEPTED - DONE | Created `ErrorBoundary.tsx` and `ChartErrorBoundary` |
| **Duplicate CSS selectors** - .dashboard vs .bp-dashboard | ACCEPTED - DONE | Consolidated to `bp-dashboard*` classes |
| **Magic numbers in charts** - Hardcoded values like heights, thresholds | ACCEPTED - DONE | Centralized in `chartConfig.ts` |

## Backend Improvements

| Issue | Status | Notes |
|-------|--------|-------|
| **Long functions** - extract_sw_progress_from_raw (250+ lines) | ACCEPTED - DONE | Refactored into 5 helper functions |
| **Incomplete TODOs** - TODOs in active code | NOT DISCUSSED | Found 2 TODOs in `black_project_simulation.py`: line 256 ("extract diversion_proportion from params") and line 266 ("compute from fab_is_operational") |
| **Repeated sys.path.insert** - Path manipulation in 12+ files | NOT DISCUSSED | Found in: `software_r_and_d.py`, `calibrate.py`, `api.py`, `simulation.py`, `parameters.py`, `black_project_simulation.py`, and several scripts. Consider using a proper package structure or centralized path setup. |
| **Missing type hints** - Various functions lack type annotations | NOT DISCUSSED | |
| **Inconsistent logging styles** - Different formats across files | ACCEPTED - DONE | Minor cleanup - was already mostly consistent |
| **camelCase/snake_case mixing** - Inconsistent dict keys | DEFERRED | Different APIs use different conventions; changing requires coordinated frontend/backend updates |

## Implementation Progress

### Completed
- **Magic numbers in charts** - Created centralized constants in `chartConfig.ts` for chart heights, margins, and dimensions. Updated PDFChart, TimeSeriesChart, EnergyStackedAreaChart, PlotlyChart, HistoricalCharts, and DatacenterSection to use these constants.
- **Error boundaries** - Created `ErrorBoundary.tsx` with generic `ErrorBoundary` class and `ChartErrorBoundary` wrapper component.
- **Type guards** - Created `utils/typeGuards.ts` with runtime type validation for API data. Updated `InitialStockSection` to use `parseInitialStockData()` instead of unsafe `as unknown as` casts.
- **CSS selector consolidation** - Removed duplicate `.dashboard*` classes, standardized on `bp-dashboard*` prefix. Updated InitialStockSection to use consolidated classes.
- **Dashboard component extraction** - Created shared `Dashboard` and `DashboardItem` components in `ui/Dashboard.tsx`. Updated InitialStockSection, DatacenterSection, and CovertFabSection to use the shared components.

### In Progress
- (none)

### Not Started
- Inline style consolidation (116 occurrences found)
- TODO comments in black_project_simulation.py

### CSS Organization
- **Component CSS files** - Extracted styles from `ai-black-projects.css` (1284 -> 792 lines) into:
  - `TopChartsSection.css` - Top charts section styles
  - `ParameterSidebar/ParameterSidebar.css` - Sidebar styles
  - `ui/Dashboard.css` - Dashboard component styles

### Backend
- **Long function refactoring** - Refactored `extract_sw_progress_from_raw()` (250+ lines) into 5 focused helper functions: `_safe_get()`, `_extract_time_series_point()`, `_compute_training_and_efficiency()`, `_build_milestones()`, `_compute_milestone_times()`

## New Issues Identified (2025-12-21)

### Priority: Low
| Issue | Location | Description |
|-------|----------|-------------|
| **Hardcoded paths in scripts** | `compare_lr_distributions.py`, `compare_black_project_models.py` | Absolute paths like `/Users/joshuaclymer/github/...` should use relative paths |
| **Untracked pycache files** | Multiple locations | Git status shows many `.pyc` files being tracked. Consider adding `**/__pycache__/` to `.gitignore` |
| **Section file sizes approaching limit** | CovertFabSection (707 lines), InitialStockSection (680 lines), DatacenterSection (568 lines) | While under 1000 line limit, trending upward |

### Priority: Medium
| Issue | Location | Description |
|-------|----------|-------------|
| **sys.path manipulation** | 12+ files | Fragile path setup. Consider proper Python packaging or a central path configuration module |
| **Inline styles in DatacenterSection** | DatacenterSection.tsx | 34 inline styles - highest of any component. Should be moved to CSS |

---

*Last updated: 2025-12-21*
