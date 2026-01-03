# Flat Tensor Optimization Plan

## Problem

The ODE-based simulation is slow (~1.1s per simulation) compared to the reference discrete model (~0.1-0.2s). Profiling revealed the bottleneck:

- **0.38s** spent on `_clone()` - deep copying entire World dataclass on every arithmetic operation
- **0.38s** spent on `_get_state_fields()` - Python reflection to iterate through nested dataclass fields
- **62 derivative function calls** for 40 output time points (RK overhead)

Every `__add__` and `__mul__` on the World/StateDerivative dataclass clones the entire nested structure. Runge-Kutta does ~10 arithmetic ops per step, causing massive overhead.

## Solution: Flat Tensor-Backed State

Keep the nice Python object interface for updaters, but store state in a single flat tensor. Arithmetic operations become fast tensor ops instead of nested dataclass cloning.

### Key Design

1. **Dynamic schema generation**: At startup, recurse through World's state fields to build an index mapping:
   ```python
   STATE_SCHEMA = {
       "compute.prc_stock": 0,
       "compute.us_stock": 1,
       "black_projects.bp1.operating_compute": 2,
       ...
   }
   ```

2. **Tensor-backed properties**: State attributes become properties that read/write from the flat tensor:
   ```python
   @property
   def prc_compute_stock(self) -> Tensor:
       return self._state[self._schema["compute.prc_stock"]]
   ```

3. **Updaters unchanged**: They still write `deriv.prc_compute_stock = world.prc_compute_stock * rate`, but under the hood it's pure tensor indexing.

4. **Fast ODE integration**: The solver works directly on the flat `_state` tensor. Arithmetic is just `Tensor + float * Tensor`.

## Implementation Steps

### Step 1: Create Schema Generator

File: `ai_futures_simulator/classes/world/state_schema.py`

```python
def generate_state_schema(template_world: World) -> dict[str, int]:
    """
    Recursively walk through World's state fields and assign indices.

    Returns dict mapping "path.to.field" -> index in flat tensor.
    Only includes fields marked with metadata={'is_state': True}.
    """
    schema = {}
    index = [0]  # Use list for mutable counter in recursion

    def recurse(obj, prefix=""):
        for f in fields(obj):
            value = getattr(obj, f.name)
            path = f"{prefix}{f.name}" if prefix else f.name
            is_state = f.metadata.get('is_state', False)

            if is_state and isinstance(value, Tensor):
                schema[path] = index[0]
                index[0] += 1
            elif is_tensor_dataclass(value):
                recurse(value, f"{path}.")
            elif isinstance(value, dict):
                for key, item in value.items():
                    if is_tensor_dataclass(item):
                        recurse(item, f"{path}[{key}].")

    recurse(template_world)
    return schema
```

### Step 2: Create FlatWorld and FlatStateDerivative Classes

File: `ai_futures_simulator/classes/world/flat_world.py`

```python
class FlatWorld:
    """World state backed by a flat tensor for fast ODE integration."""

    def __init__(self, state_tensor: Tensor, schema: dict, metadata: dict):
        self._state = state_tensor
        self._schema = schema
        self._metadata = metadata  # Non-state fields (year, params, etc.)

    @classmethod
    def from_world(cls, world: World) -> 'FlatWorld':
        """Convert nested World to flat representation."""
        schema = generate_state_schema(world)
        state_tensor = torch.zeros(len(schema))

        for path, idx in schema.items():
            value = get_nested_attr(world, path)
            state_tensor[idx] = value

        metadata = extract_non_state_fields(world)
        return cls(state_tensor, schema, metadata)

    def to_world(self) -> World:
        """Convert back to nested World structure."""
        world = create_world_from_metadata(self._metadata)
        for path, idx in self._schema.items():
            set_nested_attr(world, path, self._state[idx])
        return world

    def __getattr__(self, name: str):
        """Allow attribute access like world.compute_prc_stock."""
        # Convert attribute name to schema path and return tensor slice
        ...

    # Fast arithmetic (no cloning!)
    def __add__(self, other: 'FlatWorld') -> 'FlatWorld':
        return FlatWorld(self._state + other._state, self._schema, self._metadata)

    def __mul__(self, scalar: float) -> 'FlatWorld':
        return FlatWorld(self._state * scalar, self._schema, self._metadata)
```

### Step 3: Modify ODE Integration

File: `ai_futures_simulator/ai_futures_simulator.py`

```python
def run_simulation(self, params):
    # Initialize
    initial_world = initialize_world_for_year(params, start_year)

    # Convert to flat representation
    flat_world = FlatWorld.from_world(initial_world)

    def ode_func(t, state_tensor):
        # Update the flat world's state tensor (no copy, just reference)
        flat_world._state = state_tensor

        # Create derivative tensor
        deriv_tensor = torch.zeros_like(state_tensor)
        flat_deriv = FlatStateDerivative(deriv_tensor, flat_world._schema)

        # Call updaters (they use the same nice interface)
        for updater in self.updaters:
            updater.contribute_state_derivatives(flat_world, flat_deriv, t)

        return deriv_tensor

    # ODE solver works on flat tensor directly
    solution = odeint(ode_func, flat_world._state, t_eval)

    # Convert back to World objects only at save points
    trajectory = []
    for i, t in enumerate(t_eval):
        flat_world._state = solution[i]
        trajectory.append(flat_world.to_world())

    return trajectory
```

### Step 4: Add Attribute Access Bridge

The updaters currently access nested attributes like:
- `world.compute.prc_stock`
- `world.black_projects["bp1"].operating_compute`
- `deriv.compute.prc_stock`

We need to support this interface on FlatWorld. Options:

**Option A**: Create proxy objects that map attribute access to tensor indices
```python
class FlatWorld:
    @property
    def compute(self):
        return ComputeProxy(self._state, self._schema, prefix="compute.")

class ComputeProxy:
    @property
    def prc_stock(self):
        return self._state[self._schema["compute.prc_stock"]]
```

**Option B**: Keep World dataclass structure but replace state Tensors with views into flat tensor
```python
# world.compute.prc_stock is actually a view: flat_tensor[0:1]
```

**Option C**: Modify updaters to use flat path strings (more invasive)

Recommend **Option A** for minimal updater changes.

### Step 5: Test Performance

```python
# Before: ~1.1s per simulation
# After: ~0.2s per simulation (expected 5x speedup)

python -c "
from ai_futures_simulator import AIFuturesSimulator
import time

sim = AIFuturesSimulator(...)
t0 = time.time()
results = sim.run_simulations(10)
print(f'Time: {(time.time()-t0)/10:.3f}s per sim')
"
```

### Step 6: Verify Correctness

```bash
python scripts/compare_datacenter_capacity.py
# Should show alignment with reference model
```

## Files to Modify

1. **New files:**
   - `ai_futures_simulator/classes/world/state_schema.py` - Schema generator
   - `ai_futures_simulator/classes/world/flat_world.py` - FlatWorld class

2. **Modify:**
   - `ai_futures_simulator/ai_futures_simulator.py` - Use FlatWorld in ODE integration
   - `ai_futures_simulator/classes/world/world.py` - Add from_flat/to_flat methods

3. **Unchanged:**
   - All updaters in `world_updaters/` - They keep using the same interface
   - `parameters/` - No changes needed

## Expected Performance

| Metric | Before | After |
|--------|--------|-------|
| Time per sim | 1.1s | ~0.2s |
| Clone operations | ~60/step | 0 |
| Field reflection | ~23,000 calls | ~80 (only at boundaries) |
| Speedup | baseline | ~5x |

## Rollback Plan

If issues arise, the changes are isolated:
- Keep original World/TensorDataclass classes
- FlatWorld is a separate class
- Can switch between them by changing one line in `run_simulation()`

## Testing Checklist

- [ ] Schema generation correctly identifies all state fields
- [ ] FlatWorld.from_world() and to_world() are inverse operations
- [ ] Updaters work with FlatWorld proxy objects
- [ ] ODE integration produces same results as before
- [ ] Performance improved (~5x faster)
- [ ] `compare_datacenter_capacity.py` shows alignment with reference
