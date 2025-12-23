Project Context

You are working on aligning a continuous ODE-based AI futures simulator with a discrete reference model (hosted at https://dark-compute.onrender.com/). The simulator models covert "black projects" - secret AI compute infrastructure that nations might build. The goal is to ensure both models produce statistically equivalent outputs for detection times, likelihood ratios, compute trajectories, etc.

Please determine whether the model's outputs align with comparison scripts, such as the one defined at `scripts/compare_lr_distributions.py`

Repository for continuous model:

Path: /Users/joshuaclymer/github/ai_futures_simulator/ai_futures_simulator/

Key things to keep in mind:
- The simulation outputs a sequence of "World" objects. Each object represents the world at an instantaneous point in time. World objects should not include time series. You can only get the time series by extracting data from a full trajectory after the simulation run finishes.
- When parameters in the reference model are uncertain and are sampled, they should NOT be sampled inside of the world_updaters class in the new version. They should instead be defined as a distribution in monte_carlo_parameters.yaml.
- Please update simulation state with derivatives whenever possible to preserve end-to-end differentiability.

Reference Model API

import urllib.request, json

url = 'https://dark-compute.onrender.com/run_simulation'
data = json.dumps({'num_samples': 200, 'start_year': 2029, 'total_labor': 11300}).encode()
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=120) as response:
    result = json.loads(response.read().decode())

Reference Model Repository
You can see the original source code in `~/github/covert_compute_production_model/black_project_backend`