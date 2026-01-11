# AI Futures Simulator API

This document describes the Flask API endpoints for the AI Futures Simulator.

**Base URL:** `http://localhost:5329`

## Endpoint Directory Structure

Each endpoint has its own directory under `endpoints/`, named using snake_case matching the URL path:

```
endpoints/
├── run_simulation/                      # /api/run-simulation
├── run_sw_progress_simulation/          # /api/run-sw-progress-simulation
├── get_data_for_ai_black_projects_page/ # /api/get-data-for-ai-black-projects-page
├── black_project_defaults/              # /api/black-project-defaults
├── sampling_config/                     # /api/sampling-config
└── parameter_config/                    # /api/parameter-config
```

## Endpoints

### `POST /api/run-simulation`

Runs a single simulation and returns the full raw World trajectory.

**Request Body:**
```json
{
  "parameters": {
    "software_r_and_d.rho_coding_labor": 0.5,
    "software_r_and_d.present_doubling_time": 0.458
  },
  "time_range": [2024, 2040]
}
```

**Response:**
```json
{
  "success": true,
  "times": [2024.0, 2024.16, ...],
  "trajectory": [...],
  "params": {...},
  "generation_time_seconds": 1.234
}
```

---

### `POST /api/run-sw-progress-simulation`

Runs a simulation and returns software progress metrics formatted for visualization.

**Request Body:**
```json
{
  "parameters": {
    "software_r_and_d.rho_coding_labor": 0.5
  },
  "time_range": [2024, 2040]
}
```

**Response:**
```json
{
  "success": true,
  "time_series": [
    {
      "year": 2024.0,
      "progress": 0.0,
      "horizonLength": 26.0,
      "automationFraction": 0.0,
      "softwareProgressRate": 0.5,
      "trainingCompute": 26.54,
      "softwareEfficiency": 0.0,
      "effectiveCompute": 26.54
    }
  ],
  "milestones": {
    "AC": { "metric": "progress", "target": 4.5, "time": 2028.5 },
    "TED-AI": { "metric": "ai_research_taste_sd", "target": 12.36 }
  },
  "exp_capacity_params": {
    "rho": 0.5,
    "alpha": 0.5,
    "experiment_compute_exponent": 0.5
  }
}
```

---

### `POST /api/get-data-for-ai-black-projects-page`

Runs N Monte Carlo simulations for black project scenarios and returns aggregated plot data.

**Request Body:**
```json
{
  "parameters": {},
  "num_simulations": 100,
  "ai_slowdown_start_year": 2030,
  "end_year": 2037
}
```

**Response:**
```json
{
  "num_simulations": 100,
  "prob_fab_built": 0.55,
  "p_project_exists": 0.2,
  "researcher_headcount": 500,
  "black_project_model": {...},
  "black_datacenters": {...},
  "black_fab": {...},
  "initial_black_project": {...},
  "initial_stock": {...}
}
```

**Caching:** Set `USE_CACHE=true` environment variable to enable caching for default parameter requests.

---

### `GET /api/black-project-defaults`

Returns default parameter values for the Black Projects page.

**Response:**
```json
{
  "success": true,
  "defaults": {
    "ai_slowdown_start_year": 2030,
    "end_year": 2037
  }
}
```

---

### `GET /api/sampling-config`

Returns parameter distribution configuration for Monte Carlo sampling.

**Response:**
```json
{
  "success": true,
  "config": {...}
}
```

---

### `GET /api/parameter-config`

Returns parameter bounds and default values.

**Response:**
```json
{
  "success": true,
  "config": {...}
}
```

---

## Parameter Format

All simulation endpoints accept parameters using **dot-notation paths** that match the backend dataclass structure:

```
software_r_and_d.rho_coding_labor
software_r_and_d.present_doubling_time
compute.USComputeParameters.total_us_compute_annual_growth_rate
```

## Running the Server

```bash
cd app_backend
python api.py
```

Server runs on `http://localhost:5329` by default.
