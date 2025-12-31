# Comparing Local API with Reference Model

## Overview

This module automatically compares all shared keys between the local API and reference API.

**Local API:** `http://localhost:5329/api/get-data-for-ai-black-projects-page`
- Serves data for the frontend page at http://localhost:3000/ai-black-projects
- Powered by the continuous ODE-based AI futures simulator

**Reference API:** `https://dark-compute.onrender.com/run_simulation`
- The original discrete model implementation
- Source code at `~/github/covert_compute_production_model/black_project_backend`

## How It Works

The comparison module automatically:
1. Fetches data from both APIs
2. Finds all **shared keys** between the two responses
3. Classifies each key's value type (scalar, array, time_series, etc.)
4. Computes appropriate comparison metrics:
   - **Scalars**: Direct percent difference
   - **Arrays**: Median-based percent difference
   - **Time Series**: **Average percent difference** across all time points
5. Reports results grouped by status (PASS ≤5%, WARN 5-25%, FAIL >25%)

## Quick Start

```bash
# Run comparison with default 200 samples
python -m scripts.comparison.main

# Run with fewer samples for faster testing
python -m scripts.comparison.main --samples 50

# Disable caching to force fresh API calls
python -m scripts.comparison.main --no-cache

# Show all comparisons including passes
python -m scripts.comparison.main --show-all

# Clear cached responses
python -m scripts.comparison.main --clear-cache
```

**Prerequisites:** The local backend must be running at http://localhost:5329

## Status Thresholds

| Status | Avg % Diff | Meaning |
|--------|-----------|---------|
| ✓ PASS | ≤ 5% | Values closely aligned |
| ⚠ WARN | 5-25% | Minor differences |
| ✗ FAIL | > 25% | Significant differences |

## Module Structure

```
scripts/comparison/
├── __init__.py
├── config.py           # API URLs, default parameters
├── main.py             # CLI entry point
├── auto_compare.py     # Automatic key comparison logic
├── local_simulator.py  # Fetches from local API
├── reference_api.py    # Fetches from reference API
├── docs.md             # This file
└── metrics/            # (Legacy) Individual metric comparisons
```

## Configuration

Key parameters in `config.py`:

```python
# API URLs
REFERENCE_API_URL = 'https://dark-compute.onrender.com/run_simulation'
LOCAL_API_URL = 'http://localhost:5329/api/get-data-for-ai-black-projects-page'

# Default parameters
DEFAULT_NUM_SAMPLES = 200
DEFAULT_AGREEMENT_YEAR = 2030
DEFAULT_NUM_YEARS = 10
DEFAULT_START_YEAR = 2029  # Reference model's prep year
DEFAULT_TOTAL_LABOR = 11300
```

## Caching

Both API responses are cached in `scripts/cache/`:
- Local: `local_{samples}_{year}_{duration}.json`
- Reference: `reference_{samples}_{year}_{labor}.json`

Use `--no-cache` to bypass or `--clear-cache` to delete cached files.

## Value Types

The comparison handles these value types:

| Type | Description | Comparison Method |
|------|-------------|-------------------|
| `scalar` | Single number | Direct % diff |
| `array` | List of numbers | Median % diff |
| `time_series` | Dict with `median`, `p25`, `p75` | Avg % diff across time |
| `array_2d` | 2D array (e.g., individual sims) | Median across sims, then avg % diff |
| `ccdf` | CCDF data points | Skipped (structural) |
| `object` | Nested objects | Recursively compared |
| `string`, `boolean` | Non-numeric | Skipped |

## Example Output

```
================================================================================
AUTOMATIC KEY COMPARISON RESULTS
================================================================================

Summary:
  Total shared keys: 100
  Compared: 82
  Skipped: 18
  ✓ PASS: 23
  ⚠ WARN: 2
  ✗ FAIL: 57

────────────────────────────────────────────────────────────────────────────────
FAILURES (avg diff > 25%):
────────────────────────────────────────────────────────────────────────────────
  ✗ black_datacenters.operational_compute
    Type: time_series, Points: 71
    Local: 106583.5, Ref: 560.1
    Avg diff: 1000.0%, Max diff: 1000.0%
...
```

---

## Changelog

### 2024-12-31
- **Major refactor**: Automatic comparison of all shared keys
- Replaced individual metric comparisons with automatic key discovery
- For time series, now computes **average percent difference** across all time points
- Added `--show-all` flag to display passing comparisons

### 2024-12-30
- Initial version comparing local API with reference API
- Individual metric comparison functions in `metrics/` directory
