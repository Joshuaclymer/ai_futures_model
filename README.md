# AI Futures model

## Main classes

### Simulation Runs 

The model accepts as input a distribution of 'model parameters' and outputs a list of 'simulation runs.' Each simulation run is a trajectory of 'world states.'

```
world state class
```

### Entities

Entities are legal entities, which can have assets and be subject to state or international policies.

```python
class Entity:
    name : str
    history_of_policies_entity_is_party_to: Dict[Time, AIPolicy]
```

There are four types of entities:
- Coalition
- State
- AISoftwareDeveloper
- AIBlackProject

Only some entities are named - many others are nameless, and are meant to represent a distribution of actors rather than particular ones.

Named entities include:
- The USA
- The PRC
- US Allies (which can receive AI exports)
- US Rivals (which are export controlled)
- Slowdown Cooperators (which cooperate with an AI slowdown policy)
- Slowdown Holdouts (which don't cooperate with an AI slowdown policy)

### Assets

Entities can have *assets,* which include [list]

### AI Policies

An AI policy is a set of rules that restrict entities. Here are supported AI policies:

[todo update these]
```python
class AIPolicy:
    compute_export_blacklist = List[Entity]
    SME_export_blacklist = List[Entity]
    stock_reporting_requirements: bool = True
    experiment_compute_cap: Optional[float] = None  # in H100e TPP
    experiment_plus_training_compute_cap: Optional[float] = None  # in H100e TPP
    shut_down_chips_not_under_verification: bool = True
    ai_researcher_headcount_cap: Optional[int] = None  # number of researchers
    ai_capability_cap: Optional[Capability] = None #
    compute_production_cap: Optional[float] = None  # units of H100e TPP per month
    compute_production_capacity_cap: Optional[float] = None # Can states keep building fabs?
    alignment_to_international_controllability_standard: bool = False
```

The supported AI policies include:
[list them]

### Initialize

### Update 
