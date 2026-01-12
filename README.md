# AI Futures Model

This is a model of AI futures. The model is designed to incorporate many components together into a unified framework. Many elements of the model are differentiable, allowing gradient-based optimization of parameters.

## To use

The core model is in `./ai_futures_simulator`. Run the following command to simulate a single trajectory with default sampled parameters:

`python ai_futures_simulator/ai_futures_simulator.py --params ai_futures_simulator/parameters/default_parameters.yaml --output simulation_output.json`

This saves a file to `simulation_output.json` which contains a simulated world trajectory (which is a list of "World" objects) in json format.

You can also visualize the outputs of the model with the app in `app_frontend` and `app_backend`. To start the app run:

`scripts/restart_app.sh`

## Status of Components

✅ AI software R&D
- `world_updaters/software_r_and_d/update_software_r_and_d.py`

✅ Black project (covert compute)
- `world_updaters/black_project/update_black_project.py`
- `world_updaters/compute/black_compute.py`

✅ Compute infrastructure
- `world_updaters/compute/update_compute.py`

✅ AI researchers
- `world_updaters/ai_researchers/update_ai_researchers.py`

✅ Perceptions
- `world_updaters/perceptions/update_perceptions.py`

✅ Datacenters and energy
- `world_updaters/datacenters_and_energy/update_datacenters_and_energy.py`

⏳ AI infrastructure, AI hardware R&D, construction R&D, and economic output: I sketched a model here: https://docs.google.com/document/d/1RFDA30q1N21aKPH4aKZcDm2H44joTMWWppryQV053Kk/edit?usp=sharing. But I haven't written any code yet.

❌ Phases of AI slowdown agreements and export controls: not yet implemented

❌ Human power grab and AI takeover risk: not yet implemented (data class exists at `classes/world/takeover_risks.py`)

❌ Attacks (cyber / kinetic / bio): I haven't worked on this yet.

❌ Government awareness of AI natsec importance and takeover risks: I haven't worked on this yet.

## Design Principles

1. **Separate *modeling decisions* from *parameter values***. Modeling decisions are represented as code in the `/world_updaters` directory. Parameter values are stored in `/parameters` as yaml files.

2. **Represent uncertainty in the default_parameters.yaml configuration**. If you have uncertainty over parameters, represent this uncertainty with distributions over the parameters specified according to the format provided in `ai_futures_simulator/parameters/default_parameters.yaml`

3. **Separate *model state* from *metrics***. *State* includes the minimal set of variables needed to determine how the world changes at a given time. *Metrics* are derived quantities computed from state. For example, since all AI R&D metrics (time horizon, speedup, etc) can be calculated from OOMs of effective compute and research stock, OOMs of effective compute and research stock are the only state variables. State and metrics live in the same dataclass, but state fields are marked with `metadata={'is_state': True}` so they can be identified programmatically.

4. **Separate *data structures* from *object instances* and *logic***. Define all data structures in `classes/world`. This makes it easier to keep track of what kinds of data the model represents.

5. **Organize components by *what part of the state they affect***. Each module in `/world_updaters` can contribute to `d(state)/dt` or set state/metric values directly. We want it to be easy to determine *what code* affects a given part of the world state.

6. **Differentiable by default**. Avoid discrete logic to preserve differentiability when possible.

7. **Document components**. Document components with a README.md file that includes:
   - State the component reads
   - State the component writes (which derivatives it contributes to)
   - An explanation of the code, which ideally links to an interactive visualization in /app.

The rest of this document explains the high level structure of the model.

## Model Primitives

The model accepts as input `ModelParameters` and outputs a `SimulationTrajectory.`
- A `SimulationTrajectory` is a list of `World` objects that represents a prediction about how the future might unfold.
- `ModelParameters` specifies a *distribution* over configurations that affect the model's behavior. You can see an example of `ModelParameters` in `ai_futures_simulator/parameters/default_parameters.yaml`

Here's an excerpt of default model parameters:
```
  # US compute parameters
  us_compute:
    total_us_compute_tpp_h100e_in_2025: 400000
    total_us_compute_annual_growth_rate: 3.4 
    proportion_of_compute_in_largest_ai_sw_developer: 0.3

  # PRC compute parameters (with uncertainty for Monte Carlo)
  prc_compute:
    total_prc_compute_tpp_h100e_in_2025: 100000
    annual_growth_rate_of_prc_compute_stock:
      dist: metalog
      p10: 1.4
      p50: 3.4
      p90: 3.4
      modal: 3.4
      min: 1.0  
    proportion_of_compute_in_largest_ai_sw_developer: 0.3
```

The YAML format accepts point estimates and distributions.

To generate a `SimulationTrajectory`, three things happen:
1. First, `ModelParameters` are used to sample a single instance of `SimulationParameters`.
2. Then, a single `World` object is initialized to represent the world at the start of 2026 (`ai_futures_simulator/initialize_world_history`)
2. Finally, world updaters specify how the next state evolves from the previous state (`ai_futures_simulator/world_updaters`)

The `WorldUpdater` class has three methods:
- `contribute_state_derivatives`. This method outputs a derivative with respect to an attribute of the `World` object. These derivatives are added together and then integrated to generate a `SimulationTrajectory`. The reason we use derivatives is that it keeps updates *differentiable,* so we can optimize parameter values with gradient descent.
- `set_state_attributes`. This method allows for discrete state setting. It should generally not be used since it's NOT differentiable, but sometimes it's necessary to represent when, for example, fabs become operational, or a black project comes into existence. Under the hood, the simulation integrates the derivatives until it detects that `set_state_attributes` have been called, then terminates the integration, and begins a new integration over the next discrete segment.
- `set_metric_attributes`. Metrics are values that can be computed from state. Metrics are calculated after a trajectory of World states is already generated. So metrics do NOT affect core simulation dynamics. They are just meant to help you *interpret* simulation results.

## Directory Structure

```
ai_futures_simulator/
├── classes/
│   ├── simulation_primitives.py  # WorldUpdater base class, StateDerivative
│   ├── flat_world.py             # FlatWorld for fast ODE integration
│   ├── tensor_dataclass.py       # TensorDataclass base class
│   ├── state_schema.py           # State schema definitions
│   └── world/
│       ├── world.py              # World dataclass (state + metrics)
│       ├── entities.py           # Entity classes (Nation, AISoftwareDeveloper, etc.)
│       └── ...                   # Other data structures (assets, policies, etc.)
│
├── parameters/
│   ├── classes/
│   │   ├── simulation_parameters.py  # SimulationParameters and ModelParameters
│   │   ├── compute_parameters.py     # Compute growth parameters
│   │   ├── black_project_parameters.py  # Covert compute parameters
│   │   └── ...                       # Other parameter files
│   └── default_parameters.yaml       # Default parameter values
│
├── initialize_world_history/
│   ├── initialize_world_history.py  # Main initialization function
│   ├── initialize_nations.py        # Nation initialization
│   └── ...                          # Other initializers
│
├── world_updaters/
│   ├── combined_updater.py          # CombinedUpdater orchestrates all updaters
│   ├── software_r_and_d/            # AI software R&D
│   │   └── update_software_r_and_d.py
│   ├── black_project/               # Covert compute infrastructure
│   │   └── update_black_project.py
│   ├── compute/                     # Compute-related updaters
│   │   └── update_compute.py
│   ├── ai_researchers/              # AI researcher updaters
│   │   └── update_ai_researchers.py
│   ├── perceptions/                 # Perceptions updaters
│   │   └── update_perceptions.py
│   └── datacenters_and_energy/      # Datacenter and energy updaters
│       └── update_datacenters_and_energy.py
│
├── ai_futures_simulator.py          # AIFuturesSimulator main class
```

## Core Classes

### World

`World` represents the entire state of the world.

```python
@dataclass
class World(TensorDataclass):
    current_time: Tensor = field(metadata={'is_state': True})

    # Nested entities
    coalitions: Dict[str, Coalition]
    nations: Dict[str, Nation]
    ai_software_developers: Dict[str, AISoftwareDeveloper]
    ai_policies: Dict[str, AIPolicy]
    black_projects: Dict[str, AIBlackProject] = field(default_factory=dict)
    perceptions: Dict[str, Perceptions] = field(default_factory=dict)
```

### Entities

Entities are actors in the simulation (nations, organizations, etc.) that can have assets and be subject to policies. They are defined in `classes/world/entities.py`.

```python
@dataclass
class Nation(Entity):
    """A nation with compute stock tracking."""
    fabs: Fabs = field(metadata={'is_state': True})
    compute_stock: Compute = field(metadata={'is_state': True})
    datacenters: Datacenters = field(metadata={'is_state': True})
    total_energy_consumption_gw: float = field(metadata={'is_state': True})

    # Metrics
    operating_compute_tpp_h100e: float

@dataclass
class AISoftwareDeveloper(Entity):
    """An AI software development organization."""
    operating_compute: List[Compute] = field(metadata={'is_state': True})
    compute_allocation: ComputeAllocation = field(metadata={'is_state': True})
    human_ai_capability_researchers: float = field(metadata={'is_state': True})
    ai_software_progress: AISoftwareProgress = field(metadata={'is_state': True})

@dataclass
class AIBlackProject(AISoftwareDeveloper):
    """A covert AI compute project (extends AISoftwareDeveloper)."""
    parent_nation: Nation = field(metadata={'is_state': True})
    black_project_start_year: float = field(metadata={'is_state': True})
    # ... fab, datacenter, and detection-related fields
```

### Assets

Assets represent physical and compute resources owned by entities. Defined in `classes/world/assets.py`.

```python
@dataclass
class Compute(Assets, TensorDataclass):
    """Represents the stock of a single type of chip."""
    functional_tpp_h100e: float = field(metadata={'is_state': True})
    tpp_h100e_including_attrition: float = field(metadata={'is_state': True})
    watts_per_h100e: float = field(metadata={'is_state': True})
    average_functional_chip_age_years: float = field(metadata={'is_state': True})

@dataclass
class Datacenters(TensorDataclass):
    data_center_capacity_gw: float = field(metadata={'is_state': True})

@dataclass
class Fabs(TensorDataclass):
    monthly_compute_production: Compute = field(metadata={'is_state': True})
```