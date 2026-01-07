# AI Futures Model

This is a model of AI futures. The model is designed to incorporate many components together into a unified framework. Many elements of the model are differentiable, allowing gradient-based optimization of parameters.

## To use

The core model is in `./ai_futures_simulator`. Run the following command to simulate a single trajectory with parameters sampled from the default monte carlo distribution:

`python ai_futures_simulator/ai_futures_simulator.py --params ai_futures_simulator/parameters/default_parameters.yaml --output simulation_output.json`

This saves a file to `simulation_output.json` which contains a simulated world trajectory (which is a list of "World" objects) in json format.

You can also visualize the outputs of the model with the app in `app_frontend` and `app_backend`. To start the app run:

`scripts/restart_app.sh`


## Status of Components

✅ AI software R&D
- `world_updaters/software_r_and_d.py`

✅ Dark compute: I need to copy this over from another repo
- `world_updaters/covert_compute.py`

⏳ Phases of AI slowdown agreements and export controls: need to test code
- `world_updaters/policy_effects.py`


⏳ Human power grab and AI takeover risk: I need to copy this over from another repo
- `world_updaters/takeover_risks.py`

⏳ AI infrastructure, AI hardware R&D, construction R&D, and economic output: I sketched a model here: https://docs.google.com/document/d/1RFDA30q1N21aKPH4aKZcDm2H44joTMWWppryQV053Kk/edit?usp=sharing. But I haven't written any code yet.

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

The model accepts `SimulationParameters` (an `nn.Module` containing differentiable parameters) and an initial `World`. It produces a trajectory by integrating the ODE system:

```
d(state)/dt = f(world, t, params)
```

Each `WorldUpdater` module can contribute to this derivative. The ODE solver integrates state variables over time. Metrics are recomputed from state after integration.

For discrete changes to state (e.g., policy enacted), the simulation breaks into segments with event handling between them.

For Monte Carlo runs, `ModelParameters` defines distributions over `SimulationParameters`. Each sampled `SimulationParameters` produces one trajectory.

## Directory Structure

```
ai_futures_simulator/
├── classes/
│   ├── simulation_primitives.py  # WorldUpdater base class, StateDerivative
│   └── world/
│       ├── world.py              # World dataclass (state + metrics)
│       ├── entities.py           # Entity classes (Nation, AISoftwareDeveloper, etc.)
│       ├── flat_world.py         # FlatWorld for fast ODE integration
│       ├── tensor_dataclass.py   # TensorDataclass base class
│       └── ...                   # Other data structures (assets, policies, etc.)
│
├── parameters/
│   ├── simulation_parameters.py  # SimulationParameters and ModelParameters
│   ├── compute_parameters.py     # Compute growth parameters
│   ├── black_project_parameters.py  # Covert compute parameters
│   └── ...                       # Other parameter files
│
├── initialize_world_history/
│   ├── initialize_world_history.py  # Main initialization function
│   ├── initialize_nations.py        # Nation initialization
│   └── ...                          # Other initializers
│
├── world_updaters/
│   ├── combined_updater.py       # CombinedUpdater orchestrates all updaters
│   ├── software_r_and_d.py       # AI software R&D
│   ├── black_project.py          # Covert compute infrastructure
│   ├── compute/                  # Compute-related updaters
│   └── ai_researchers/           # AI researcher updaters
│
├── ai_futures_simulator.py       # AIFuturesSimulator main class
```

## Core Classes

### World

`World` is a dataclass containing nested entities (coalitions, nations, AI software developers, etc.) along with global state and metrics. State fields are marked with `metadata={'is_state': True}`:

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

@dataclass
class AISoftwareDeveloper(Entity):
    # State variables (integrated by ODE solver)
    operating_compute: List[Compute] = field(metadata={'is_state': True})
    compute_allocation: ComputeAllocation = field(metadata={'is_state': True})
    human_ai_capability_researchers: float = field(metadata={'is_state': True})
    ai_software_progress: AISoftwareProgress = field(metadata={'is_state': True})
    training_compute_growth_rate: float = field(metadata={'is_state': True})

    # Metrics (derived from state, recomputed after integration)
    ai_r_and_d_inference_compute_tpp_h100e: float = field(init=False, default=0.0)
    ai_r_and_d_training_compute_tpp_h100e: float = field(init=False, default=0.0)
```

The `TensorDataclass` base class provides:
- `to_state_tensor()` — recursively pack all `is_state=True` fields (including nested) into a tensor for the ODE solver
- `from_state_tensor(tensor)` — unpack tensor back into state fields
- `zeros()` — create zero-initialized instance
- `__add__` — element-wise addition (for combining derivative contributions)

### StateDerivative

A thin wrapper around `World` that marks it as containing derivative values rather than state values:

```python
@dataclass
class StateDerivative:
    """Wrapper indicating the World object contains d(state)/dt values."""
    world: World

    def to_state_tensor(self) -> Tensor:
        return self.world.to_state_tensor()

    def __add__(self, other: 'StateDerivative') -> 'StateDerivative':
        return StateDerivative(self.world + other.world)

    @classmethod
    def zeros(cls) -> 'StateDerivative':
        return cls(World.zeros())
```

This avoids duplicating the `World` class for derivatives.

### Adding New State Variables

Add fields to `World` or nested entity classes with the appropriate metadata:

```python
@dataclass
class AISoftwareDeveloper(Entity):
    # Existing fields...

    # New state variable
    alignment_tax: Tensor = field(metadata={'is_state': True})

    # New metric
    effective_compute: Tensor = field(init=False, default=0.0)
```

The `TensorDataclass` base class recursively traverses nested entities to collect all `is_state=True` fields. No separate derivative class to update.

## WorldUpdater

Each module extends `WorldUpdater` (in `classes/simulation_primitives.py`) and can implement three methods:

```python
class WorldUpdater(nn.Module):
    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """Continuous contribution to d(state)/dt. Differentiable."""
        return StateDerivative.zeros(world)

    def set_state_attributes(self, t: Tensor, world: World) -> World | None:
        """Discrete changes to state. Return None if no changes, or updated World.
        NOT differentiable through the change."""
        return None  # Default: no changes

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """Compute derived metrics from state. Can be called anytime."""
        return world  # Default: no changes
```

### Example: SoftwareRAndD

```python
class SoftwareRAndD(WorldUpdater):
    def __init__(self, params: SimulationParameters):
        super().__init__()
        self.params = params

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        automation = torch.sigmoid(self.params.slope * world.progress)
        sw_rate = self.params.base_rate * (1 + automation * self.params.mult)

        d_world = World.zeros()
        d_world.progress = sw_rate
        d_world.research_stock = sw_rate
        d_world.log_compute = 0.0

        return StateDerivative(d_world)

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        world.automation_fraction = torch.sigmoid(self.params.slope * world.progress)
        world.time_horizon = compute_horizon(world.progress, self.params)
        return world
```

### CombinedUpdater

Orchestrates all world updaters (in `world_updaters/combined_updater.py`):

```python
class CombinedUpdater(WorldUpdater):
    def __init__(self, params: SimulationParameters, updaters: List[WorldUpdater] = None):
        super().__init__()
        self.params = params

        # If no updaters provided, create default set based on params
        if updaters is None:
            updaters = [ComputeUpdater(params)]
            if params.software_r_and_d.update_software_progress:
                updaters.append(SoftwareRAndD(params))
            if params.black_project and params.black_project.run_a_black_project:
                updaters.append(BlackProjectUpdater(params, ...))

        self.updaters = nn.ModuleList(updaters)

    def forward(self, t: Tensor, state_tensor: Tensor) -> Tensor:
        """Called by ODE solver."""
        world = World.from_state_tensor(state_tensor, self._world_template)

        total_derivative = self.contribute_state_derivatives(t, world)
        return total_derivative.to_state_tensor()

    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """Combine state derivative contributions from all updaters."""
        total_derivative = StateDerivative.zeros(world)
        for updater in self.updaters:
            total_derivative = total_derivative + updater.contribute_state_derivatives(t, world)
        return total_derivative

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """Compute metrics from all updaters."""
        for updater in self.updaters:
            world = updater.set_metric_attributes(t, world)
        return world
```

## Event Handling for Discrete Changes

When state variables need to change discretely (e.g., policy enacted), the simulation breaks into segments:

```python
def simulate(y0, times, event_times):
    trajectory = []

    for segment_start, segment_end in get_segments(times, event_times):
        segment_times = times[(times >= segment_start) & (times <= segment_end)]

        # Integrate continuous dynamics
        segment_traj = odeint(combined.forward, y0, segment_times)
        trajectory.append(segment_traj)

        # Apply discrete changes at segment boundary
        y0 = combined.apply_discrete_changes(segment_end, segment_traj[-1])

    # Compute metrics for full trajectory
    full_traj = torch.cat(trajectory)
    return combined.compute_metrics(times, full_traj)
```

Gradients flow through the continuous segments but not through discrete changes.

## Parameters

Differentiable parameters as an nn.Module:

```python
class SimulationParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_compute_growth_rate = nn.Parameter(torch.tensor(0.693))
        self.training_compute_growth = nn.Parameter(torch.tensor(0.5))
        self.base_sw_rate = nn.Parameter(torch.tensor(0.5))
        self.ai_sw_multiplier = nn.Parameter(torch.tensor(5.0))
        self.automation_midpoint = nn.Parameter(torch.tensor(8.0))
        self.automation_slope = nn.Parameter(torch.tensor(0.3))
```

## Running the Simulation

The main entry point is the `AIFuturesSimulator` class:

```python
from ai_futures_simulator import AIFuturesSimulator
from parameters.simulation_parameters import ModelParameters

# Load parameters from YAML
model_params = ModelParameters.from_yaml("parameters/default_parameters.yaml")

# Create simulator
simulator = AIFuturesSimulator(model_params)

# Run a single simulation
result = simulator.run_simulation()

# Or run Monte Carlo simulations
results = simulator.run_simulations(num_samples=100)
```

Or run directly:
```bash
python ai_futures_simulator.py
```
