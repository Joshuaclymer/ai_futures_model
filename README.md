# AI Futures Model

This is a model of AI futures. The model is designed to incorporate many components together into a unified framework. The simulation is differentiable, allowing gradient-based optimization of parameters.

## Status of Components

✅ AI software R&D
- `world_updaters/software_r_and_d.py`

⏳ Phases of AI slowdown agreements and export controls: need to test code
- `world_updaters/policy_effects.py`

⏳ Dark compute: I need to copy this over from another repo
- `world_updaters/covert_compute.py`

⏳ Human power grab and AI takeover risk: I need to copy this over from another repo
- `world_updaters/takeover_risks.py`

⏳ AI infrastructure, AI hardware R&D, construction R&D, and economic output: I sketched a model here: https://docs.google.com/document/d/1RFDA30q1N21aKPH4aKZcDm2H44joTMWWppryQV053Kk/edit?usp=sharing. But I haven't written any code yet.

❌ Attacks (cyber / kinetic / bio): I haven't worked on this yet.

❌ Government awareness of AI natsec importance and takeover risks: I haven't worked on this yet.

## Design Principles

1. **Separate *modeling decisions* from *parameter values***. Modeling decisions are represented as code in the `/world_updaters` directory. Parameter values are stored in `/parameters` as csv files.

2. **Separate *model state* from *metrics***. *State* includes the minimal set of variables needed to determine how the world changes at a given time. *Metrics* are derived quantities computed from state. For example, since all AI R&D metrics (time horizon, speedup, etc) can be calculated from OOMs of effective compute and research stock, OOMs of effective compute and research stock are the only state variables. State and metrics live in the same dataclass, but state fields are marked with `metadata={'is_state': True}` so they can be identified programmatically.

3. **Separate *data structures* from *object instances* and *logic***. Define all data structures in `world_classes/`. This makes it easier to keep track of what kinds of data the model represents.

4. **Organize components by *what part of the state they affect***. Each module in `/world_updaters` can contribute to `d(state)/dt` or set state/metric values directly. We want it to be easy to determine *what code* affects a given part of the world state.

5. **Differentiable by default**. Avoid discrete logic to preserve differentiability when possible.

6. **Document components**. Document components with a README.md file that includes:
   - State the component reads
   - State the component writes (which derivatives it contributes to)
   - An explanation of the code, which ideally links to an interactive visualization in /app.

The rest of this document explains the high level structure of the model.

## Model Primitives

The model accepts `SimulationParameters` (an `nn.Module` containing differentiable parameters) and an initial `World`. It produces a trajectory by integrating the ODE system:

```
d(state)/dt = f(world, t, params)
```

Each `StateUpdater` module can contribute to this derivative. The ODE solver integrates state variables over time. Metrics are recomputed from state after integration.

For discrete changes to state (e.g., policy enacted), the simulation breaks into segments with event handling between them.

For Monte Carlo runs, `ModelParameters` defines distributions over `SimulationParameters`. Each sampled `SimulationParameters` produces one trajectory.

## Directory Structure

```
ai_futures_simulator/
├── world_classes/
│   ├── world.py              # World dataclass (state + metrics)
│   └── state_derivative.py   # StateDerivative wrapper
│
├── parameters/
│   ├── simulation_parameters.py  # SimulationParameters (nn.Module)
│   └── model_parameters.py       # ModelParameters (distributions for Monte Carlo)
│
├── state_initialization/
│   └── world_initializers.py     # Functions to create initial World states
│
├── world_updaters/
│   ├── base.py               # StateUpdater base class
│   ├── combined.py           # Orchestrates all state updaters
│   ├── software_r_and_d.py   # AI software R&D
│   ├── compute_growth.py     # Compute growth
│   └── ...                   # Add more modules here
│
├── ai_futures_simulator.py
```

## Core Classes

### World

`World` is a dataclass containing nested entities (coalitions, nations, AI projects, etc.) along with global state and metrics. State fields are marked with `metadata={'is_state': True}`:

```python
@dataclass
class World(TensorDataclass):
    current_time: Tensor = field(metadata={'is_state': True})

    # Nested entities
    coalitions: dict[str, Coalition] = field(default_factory=dict)
    nations: dict[str, Nation] = field(default_factory=dict)
    ai_projects: dict[str, AIProject] = field(default_factory=dict)
    ai_policies: dict[str, AIPolicy] = field(default_factory=dict)

@dataclass
class AIProject(TensorDataclass):
    id: str

    # State variables (integrated by ODE solver)
    log_compute: Tensor = field(metadata={'is_state': True})
    progress: Tensor = field(metadata={'is_state': True})
    research_stock: Tensor = field(metadata={'is_state': True})

    # Metrics (derived from state, recomputed after integration)
    automation_fraction: Tensor = field(metadata={'is_state': False})
    time_horizon: Tensor = field(metadata={'is_state': False})
    software_progress_rate: Tensor = field(metadata={'is_state': False})
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
class AIProject(TensorDataclass):
    # Existing fields...

    # New state variable
    alignment_tax: Tensor = field(metadata={'is_state': True})

    # New metric
    effective_compute: Tensor = field(metadata={'is_state': False})
```

The `TensorDataclass` base class recursively traverses nested entities to collect all `is_state=True` fields. No separate derivative class to update.

## StateUpdater

Each module extends `StateUpdater` and can implement three methods:

```python
class StateUpdater(nn.Module):
    def contribute_state_derivatives(self, t: Tensor, world: World) -> StateDerivative:
        """Continuous contribution to d(state)/dt. Differentiable."""
        raise NotImplementedError

    def set_state_attributes(self, t: Tensor, world: World) -> World:
        """Discrete changes to state. Called between integration segments.
        NOT differentiable through the change."""
        return world  # Default: no changes

    def set_metric_attributes(self, t: Tensor, world: World) -> World:
        """Compute derived metrics from state. Can be called anytime."""
        return world  # Default: no changes
```

### Example: SoftwareRAndD

```python
class SoftwareRAndD(StateUpdater):
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

### Combined

Orchestrates all state updaters:

```python
class Combined(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.software = SoftwareRAndD(params)
        self.compute = ComputeGrowth(params)
        self.policy = PolicyEffects(params)

        self.updaters = [self.software, self.compute, self.policy]

    def forward(self, t: Tensor, state_tensor: Tensor) -> Tensor:
        """Called by ODE solver."""
        world = World.from_state_tensor(state_tensor)

        d_state = StateDerivative.zeros()
        for updater in self.updaters:
            d_state += updater.contribute_state_derivatives(t, world)

        return d_state.to_state_tensor()

    def apply_discrete_changes(self, t: Tensor, world: World) -> World:
        """Called between integration segments at event times."""
        for updater in self.updaters:
            world = updater.set_state_attributes(t, world)
        return world

    def compute_metrics(self, t: Tensor, world: World) -> World:
        """Called after integration to compute derived metrics."""
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

```bash
python ai_futures_simulator.py
```
