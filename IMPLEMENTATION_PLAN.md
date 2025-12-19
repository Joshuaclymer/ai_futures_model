# Black Project Migration Plan

## Overview

Migrate the discrete-time black_project_backend to the continuous ODE-based ai_futures_simulator.

## Key Differences

| Aspect | black_project_backend | ai_futures_simulator |
|--------|----------------------|---------------------|
| Time | Discrete steps (0.1 years) | Continuous ODE integration |
| State updates | `stock(t+dt) = stock(t) + rate * dt` | `d(stock)/dt = rate` |
| Attrition | Survival rate per cohort | Continuous hazard rate |
| Events | All discrete | Discrete events trigger ODE restart |

## Implementation Steps

### Phase 1: Parameters and Data Structures

1. **Create `parameters/black_project_parameters.py`**
   - `BlackProjectParameters` dataclass with:
     - `agreement_year`: When black project starts
     - `proportion_of_initial_compute_to_divert`: Fraction of PRC stock diverted
     - `run_black_project`: Enable/disable flag
   - `BlackFabParameters` dataclass with:
     - `construction_labor`, `operating_labor`
     - `process_node_strategy`
     - `proportion_of_scanners_devoted`
   - `BlackDatacenterParameters` dataclass with:
     - `construction_labor`
     - `years_before_agreement_to_start`
     - `max_proportion_of_prc_energy`

2. **Extend `classes/world/assets.py`**
   - Add `BlackFab` dataclass (extends `Fabs`)
   - Add `BlackDatacenters` dataclass (extends `Datacenters`)

3. **Extend `classes/world/entities.py`**
   - Modify `Nation` to have compute stock state:
     - `log_compute_stock: Tensor = field(metadata={'is_state': True})`
   - Extend `AIBlackProject` with:
     - `log_compute_stock: Tensor = field(metadata={'is_state': True})`
     - `fab: Optional[BlackFab]`
     - `datacenters: BlackDatacenters`
     - `fab_operational: bool` (set as discrete event)

### Phase 2: World Updaters

1. **Create `world_updaters/nation_compute.py`**
   - `NationComputeUpdater` class:
     - Models continuous compute growth for US and PRC
     - `d(log_compute)/dt = growth_rate * ln(10)`

2. **Create `world_updaters/black_project.py`**
   - `BlackProjectUpdater` class:
   - Continuous dynamics:
     - Fab production: `d(log_compute)/dt = production_rate / compute` (when operational)
     - Survival: `d(compute)/dt = -hazard_rate * compute`
     - Combined: `d(log_compute)/dt = (production - hazard * stock) / stock`
   - Datacenter capacity growth: `d(capacity)/dt = construction_rate`
   - Discrete events:
     - Fab becomes operational at `agreement_year + construction_duration`

### Phase 3: Integration

1. **Update `world_updaters/combined_updater.py`**
   - Add `NationComputeUpdater` to default updaters
   - Add `BlackProjectUpdater` to default updaters

2. **Update `initialize_world_history/`**
   - Initialize PRC compute stock at historical values
   - Initialize US compute stock at historical values

## Mathematical Model

### Compute Stock Dynamics (Continuous)

For the black project compute stock with continuous production and hazard:

```
d(S)/dt = P(t) - H(t) * S

where:
  S = compute stock (H100e)
  P(t) = production rate (H100e/year) = fab_production if fab operational, else 0
  H(t) = hazard rate = initial_hazard + increase_per_year * (t - t_added)
```

For log-space (used for ODE stability):
```
d(log(S))/dt = P(t)/S - H(t)
```

### Fab Production Rate

```
production_rate = wafer_starts_per_month * 12 * chips_per_wafer * h100e_per_chip

where:
  wafer_starts_per_month = min(labor_capacity, scanner_capacity)
  h100e_per_chip = transistor_density_ratio * architecture_efficiency
```

### Datacenter Capacity

```
d(capacity)/dt = construction_rate if t >= start_time and capacity < max_capacity, else 0

where:
  construction_rate = construction_labor * GW_per_worker_per_year
  max_capacity = max_proportion * total_prc_energy
```

## Files to Create

1. `parameters/black_project_parameters.py` - Parameter definitions
2. `classes/world/black_project.py` - Black project state classes
3. `world_updaters/nation_compute.py` - Nation compute updater
4. `world_updaters/black_project.py` - Black project updater

## Files to Modify

1. `classes/world/entities.py` - Add compute state to Nation
2. `classes/world/world.py` - Add black_project field
3. `world_updaters/combined_updater.py` - Include new updaters
4. `parameters/simulation_parameters.py` - Add black project params
