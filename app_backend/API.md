# AI Futures Simulator API

This document describes the Flask API endpoints for the AI Futures Simulator.

**Base URL:** `http://localhost:5329`

## Architecture

The frontend communicates directly with this Flask backend. There are no intermediate proxy layers.

## Endpoints

### Core Simulation Endpoints

#### `POST /api/run-ai-futures-simulation`

Runs a **single** AI futures simulation rollout with the provided parameters.

**Request Body:**
```json
{
  "parameters": {
    // ModelParameters fields - see parameters/simulation_parameters.py
    "agreement_year": 2027,
    // ... other parameters
  },
  "time_range": [2027, 2037]  // [start_year, end_year]
}
```

**Response:**
```json
{
  "success": true,
  "simulation_result": {
    // Single SimulationResult object with world trajectory
  }
}
```

---

#### `POST /api/run-ai-futures-simulations`

Runs **multiple** AI futures simulations, sampling parameters for each run according to configured distributions (Monte Carlo).

**Request Body:**
```json
{
  "parameters": {
    // Base ModelParameters - distributions will be sampled around these
  },
  "num_simulations": 100,
  "time_range": [2027, 2037]
}
```

**Response:**
```json
{
  "success": true,
  "num_simulations": 100,
  "simulation_results": [
    // Array of SimulationResult objects
  ]
}
```

---

### Page Data Endpoints (Streaming)

These endpoints return data formatted for specific frontend pages. They use **Server-Sent Events (SSE)** to stream results progressively:

1. First, return data from a single "central" simulation (fast initial render)
2. Then, stream aggregated data from multiple Monte Carlo simulations

#### `GET /api/stream-ai-timelines-data`

Returns data for the AI Timelines and Takeoff page.

**Query Parameters:**
- `agreement_year` (int): Start year for simulation
- `num_simulations` (int): Total simulations to run (default: 100)
- Additional parameters as needed

**SSE Stream Format:**
```
event: initial
data: {"success": true, "type": "initial", "data": {...}}

event: progress
data: {"success": true, "type": "progress", "completed": 10, "total": 100}

event: aggregated
data: {"success": true, "type": "aggregated", "data": {...}}

event: done
data: {"success": true, "type": "done"}
```

**Initial Data Payload:**
- Single simulation trajectory data
- Formatted for immediate chart rendering

**Aggregated Data Payload:**
- Percentile bands (p25, median, p75)
- CCDFs for key metrics
- Distribution samples

---

#### `GET /api/stream-black-project-data`

Returns data for the Black Projects page.

**Query Parameters:**
- `agreement_year` (int): Year the agreement starts
- `num_simulations` (int): Total simulations to run (default: 100)
- `proportion_of_initial_chip_stock_to_divert` (float)
- `workers_in_covert_project` (int)
- Additional parameters as needed

**SSE Stream Format:**
```
event: initial
data: {"success": true, "type": "initial", "data": {...}}

event: progress
data: {"success": true, "type": "progress", "completed": 10, "total": 100}

event: aggregated
data: {"success": true, "type": "aggregated", "data": {...}}

event: done
data: {"success": true, "type": "done"}
```

**Initial Data Payload:**
Contains single-simulation data for:
- `initial_stock`: Initial compute stock data
- `rate_of_computation`: Chip stock and energy over time
- `covert_fab`: Fab production data
- `detection_likelihood`: Detection probability data
- `black_datacenters`: Datacenter capacity data
- `black_project_model`: Overall project metrics

**Aggregated Data Payload:**
Contains Monte Carlo aggregated data:
- Percentile bands for all time series
- CCDFs for detection thresholds (1x, 2x, 4x)
- Distribution samples for dashboard statistics

---

### Utility Endpoints

#### `GET /api/black-project-defaults`

Returns default parameter values for the Black Projects page.

**Response:**
```json
{
  "success": true,
  "defaults": {
    "agreementYear": 2027,
    "proportionOfInitialChipStockToDivert": 0.05,
    // ... other defaults
  }
}
```

---

#### `GET /api/config`

Returns simulation configuration and available parameter ranges.

---

## Streaming Implementation Notes

### Why Streaming?

Simulations can take 10-60+ seconds for 100 Monte Carlo runs. Streaming provides:

1. **Fast initial render**: Users see data within 1-2 seconds (single simulation)
2. **Progressive enhancement**: Charts update as more data arrives
3. **Progress feedback**: Users see simulation progress

### Frontend Consumption

```typescript
const eventSource = new EventSource('/api/stream-black-project-data?agreement_year=2027');

eventSource.addEventListener('initial', (e) => {
  const data = JSON.parse(e.data);
  // Render initial charts immediately
});

eventSource.addEventListener('progress', (e) => {
  const { completed, total } = JSON.parse(e.data);
  // Update progress indicator
});

eventSource.addEventListener('aggregated', (e) => {
  const data = JSON.parse(e.data);
  // Update charts with full Monte Carlo data
});

eventSource.addEventListener('done', () => {
  eventSource.close();
});
```

### Error Handling

```
event: error
data: {"success": false, "error": "Error message here"}
```

## Running the Server

```bash
cd app_backend
python api.py
```

Server runs on `http://localhost:5329` by default.
