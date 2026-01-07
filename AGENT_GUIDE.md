# Agent Guide: Aligning the AI Futures Simulator

## High-Level Objective

You are aligning a **continuous ODE-based AI futures simulator** with a **discrete reference model**. Both models simulate covert "black projects" - secret AI compute infrastructure that nations might build.

**Goal:** Make ai_futures_simulator produce statistically equivalent outputs to the reference model for metrics like:
- Detection times and likelihood ratios
- Compute trajectories (operating compute, chip stocks)
- Energy usage
- Fab production rates

The models will never match exactly due to Monte Carlo variance, but metrics should be within ~5-10% on average.

## Key Paths

| What | Path |
|------|------|
| **ai_futures_simulator source** | `ai_futures_simulator/` |
| **Main simulation logic** | `ai_futures_simulator/world_updaters/black_project.py` |
| **API/serialization** | `app_backend/api_utils/black_project_simulation.py` |
| **Parameters** | `ai_futures_simulator/parameters/` |
| **Comparison scripts** | `scripts/comparison/` |
| **Reference model source** | `~/github/covert_compute_production_model/black_project_backend/` |
| **Reference API (local)** | `http://127.0.0.1:5001/run_simulation` |

## Using the Comparison Script

The comparison script runs both models and compares all shared metrics automatically.

```bash
# Standard test (200 samples, ~20 seconds)
python -m scripts.comparison.main --samples 200 --no-cache

# Quick test (50 samples, ~5 seconds)
python -m scripts.comparison.main --samples 50 --no-cache

# Show all metrics including passes
python -m scripts.comparison.main --samples 200 --no-cache --show-all
```

**Output interpretation:**
- `✓ PASS` - Avg diff < 5% (good alignment)
- `⚠ WARN` - Avg diff 5-25% (acceptable, likely MC variance)
- `✗ FAIL` - Avg diff > 25% (needs investigation)

**Note:**
- ai_futures_simulator runs in-process (no HTTP server needed)
- The reference backend must be running locally at `http://127.0.0.1:5001`
- Reference results are cached by default; ai_futures_simulator results are never cached (always fresh)
- Use `--no-cache` to force fresh reference API calls

## Git Workflow (IMPORTANT)

Each agent must work in its own **git worktree** to avoid conflicts with other agents.

### Setup your worktree

```bash
# Create a new branch and worktree for your work
cd /Users/joshuaclymer/github/ai_futures_simulator
git worktree add ../ai_futures_simulator_agent1 -b agent1-fix-metric-name main

# Work in your worktree
cd ../ai_futures_simulator_agent1
```

### Rules

1. **Create your own worktree** - Branch off `main` with a descriptive name like `agent1-fix-metric-name`
2. **Commit your changes** - Make commits as you progress
3. **DO NOT merge** - Never merge your branch back to main. Merges will be handled by the human operator.
4. **DO NOT push** - Unless explicitly asked

## Example Workflow

### 1. Run comparison to identify your target metric
```bash
python -m scripts.comparison.main --samples 200 --no-cache 2>&1 | grep -E "(✗|your_metric_name)"
```

### 2. Understand the metric
- Find where it's computed in `black_project.py`
- Find the equivalent in reference model at `~/github/covert_compute_production_model/black_project_backend/`
- Compare the logic line by line

### 3. Make a targeted fix
- Change only what's necessary
- Add comments explaining the change

### 4. Test your fix
```bash
python -m scripts.comparison.main --samples 200 --no-cache 2>&1 | grep -E "(your_metric_name|RESULT)"
```

### 5. Commit when improved
```bash
git add -A
git commit -m "Fix [metric_name]: [brief description of change]"
```

## Reference Model API

The reference backend must be running locally. Start it with:

```bash
cd ~/github/covert_compute_production_model/black_project_backend
python app.py  # Runs on http://127.0.0.1:5001
```

Example API call:

```python
import urllib.request, json

url = 'http://127.0.0.1:5001/run_simulation'
data = json.dumps({
    'num_simulations': 200,
    'start_year': 2030,
    'num_years': 7,
}).encode()
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=120) as response:
    result = json.loads(response.read().decode())
```

## Important Notes

1. **Match the reference EXACTLY** - Even if you think the reference has a bug, match it first. Document suspected bugs but don't "fix" them.

2. **Monte Carlo variance is expected** - Metrics can vary 10-20% between runs. Run with 200 samples for stable comparisons.

3. **Focus on one metric at a time** - Don't try to fix multiple things at once.

4. **The reference model is the source of truth** - When in doubt, match what the reference does.

5. **Check both computation AND serialization** - Sometimes the metric is computed correctly but serialized wrong in `black_project_simulation.py`.

6. **World objects are instantaneous snapshots** - The simulation outputs a sequence of "World" objects. Each object represents the world at an instantaneous point in time. World objects should not include time series. You can only get the time series by extracting data from a full trajectory after the simulation run finishes.

7. **Sample parameters in YAML, not in code** - When parameters in the reference model are uncertain and are sampled, they should NOT be sampled inside of the world_updaters class in the new version. They should instead be defined as a distribution in `default_parameters.yaml`.

8. **Use derivatives for differentiability** - Update simulation state with derivatives whenever possible to preserve end-to-end differentiability.
