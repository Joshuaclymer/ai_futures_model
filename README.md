# AI Futures model

This is a model of AI futures. The model is designed to incorporate many components together into a unified framework.

Status of components:
✅ AI software R&D: Needs to be updated.
- `update_model_state/update_ai_software_progress`
⏳ Phases of AI slowdown agreements and export controls: need to test code
- `update_model_state/update_policies`
⏳ Dark compute: I need to copy this over from another repo 
- `update_model_state/update_perception_of_covert_compute`
⏳ Human power grab and AI takeover risk: I need to copy this over from another repo
- `update_model_state/update_takeover_risks`
⏳ AI infrastructure, AI hardware R&D, construction R&D, and economic output: I sketched a model here: https://docs.google.com/document/d/1RFDA30q1N21aKPH4aKZcDm2H44joTMWWppryQV053Kk/edit?usp=sharing. But I haven't written any code yet.
❌ Attacks (cyber / kinetic / bio): I haven't worked on this yet.
❌ Government awareness of AI natsec importance and takeover risks: I haven't worked on this yet.

Design principles:

1. **Separate *modeling decisions* from *parameter values***. Modeling decisions are represented as code in the `/update_model_state` directory. Parameter values are stored separately as csv files in `parameter_inputs`.

3. **Separate *model state* from *metrics***. *State* includes variables that affect how the trajectory evolves. Please save state inside of World.state and save other metrics you want to trace into World.metrics. This makes it easier to keep track of the core variables of the model.

2. **Separate *data structures* from *object instances* and *logic***. Define all data structures in `/state_classes` and `/metric_classes`. This makes it easier to keep track of what kinds of data the model represents.

4. **Organize components by *what part of the world state they update***. We want it to be easy to determine *what code* updates a given part of the world state. So please organize model components by what part of the state they modify.

3. **Document components**. Document components with a README.md file that includes:
- State the component reads
- State the component writes
- An explanation of the code (ideally a link to an interactive webpage in `/app`. Claude can generate interactive webpages pretty well.)

The rest of this document explains the high level structure of the model.

## Model primitives

The model accepts a 'ModelParameters' as input along with the number of trajectories to generate. For each trajectory, the model samples a 'SimulationParameters' instance. Simulation parameters specify a particular way the world could be, while ModelParameters incorporate uncertainty.

For each trajectory, the model generates a SimulationRun given the SimulationParameters. Each SimulationRun is a sequence of 'World.' objects which include state and metrics. World state affects how the trajectory evolves, and metrics don't.

## Core model State classes 

### World State

WorldState is the high level class - all other state is embedded in the attributes of this object.

```python
class Time:
    year: float

class WorldState:
    current_time: Time
    coalitions: dict[str, Coalition]  # id -> all coalitions of states in the world at this time
    states: dict[str, State]  # id -> all states in the world at this time
    ai_software_developers: dict[str, AISoftwareDeveloper]  # id -> all AI software developers in the world at this time
    ai_policies: dict[str, AIPolicy]  # id -> all AI policies in effect at this time
```

### Entities

Entities are legal entities, which can have assets and be subject to state or international policies.

```python
class Entity:
    id: str
    verification_capacity: VerificationCapacity | None = None
    policies_entity_is_verifying: list[AIPolicy] | None = None
    policies_entity_is_subject_to: list[AIPolicy] | None = None
    assets_under_ownership: list[Assets] | None = None
    perceptions: Perceptions | None = None
    utilities: Utilities | None = None
    current_attacks: list[Attack] | None = None
```

There are four types of entities:
- Coalition
- State
- AISoftwareDeveloper
- AIBlackProject

```python
class Coalition(Entity):
    id: str
    member_states: list[State]

class State(Entity):
    id: str
    leading_ai_software_developer: AISoftwareDeveloper | None = None
    black_project: BlackProject | None = None
    all_entities_under_jurisdiction: list[Entity]
    economy: StateEconomy
    technologies_unlocked: list[Technologies]
    division_of_political_influence: list[Entity]
    proportion_of_GDP_spent_on_kinetic_offense: float
    proportion_of_GDP_spent_on_kinetic_defense: float
    proportion_of_population_defended_from_novel_bio_attack: float
    proportion_of_population_defended_from_superhuman_influence_operations: float

class AISoftwareDeveloper(Entity):
    id: str
    is_primarily_controlled_by_misaligned_AI: bool
    compute_in_use: list[Compute]
    compute_allocation: list[ComputeAllocation]
    ai_software_progress: AISoftwareProgress
    ai_alignment_status: AIAlignmentStatus
    software_security_level: SoftwareSecurityLevel
    human_ai_capability_researchers: int

class AIBlackProject(AISoftwareDeveloper):
    parent_entity: Entity
    human_datacenter_construction_labor: int
    human_datacenter_operating_labor: int
    human_fab_construction_labor: int
    human_fab_operating_labor: int
    human_ai_capability_researchers: int
    proportion_of_compute_taken_from_parent_entity_initially: float | None
    proportion_of_SME_taken_from_parent_entity_initially: float | None
    proportion_of_unconcealed_datacenters_taken_from_parent_entity_initially: float | None
    proportion_of_researcher_headcount_taken_from_parent_entity_initially: float | None
    max_proportion_of_energy_taken_from_parent_entity_ongoingly: float | None
```

Only some entities are named - many others are nameless, and are meant to represent a distribution of actors rather than particular ones.

Named entities include:
- The USA
- The PRC
- US Allies (which can receive AI exports)
- US Rivals (which are export controlled)
- Slowdown Cooperators (which cooperate with an AI slowdown policy)
- Slowdown Holdouts (which don't cooperate with an AI slowdown policy)

### Assets

Entities can have *assets,* which include:
- Compute
- Datacenters
- Fabs
- Energy generation
- Robots
- Robot factories
etc

```python
class Assets:
    asset_status: list[AssetStatus]
    under_verification_measures: VerificationMeasure | None = None

class AssetStatus(Enum):
    OPERATING = "operating"
    NOT_OPERATING = "not_operating"
    DESTROYED = "destroyed"
    DEGRADED = "degraded"
    ONEARTH = "on_earth"
    InSpace = "in_space"

class Compute(Assets):
    """Represents the stock of a single type of chip."""
    total_tpp_h100e: float
    total_energy_requirements_watts: float
    number_of_chips: int | None = None
    inter_chip_bandwidth_gbps: float | None = None
    intra_chip_bandwidth_gbps: float | None = None

class Datacenters(Assets):
    data_center_capacity_gw: float

class Fabs(Assets):
    monthly_production: Compute
    production_method: ProductionTechnology
    tsmc_process_node_equivalent_in_nm: float | None
    number_of_lithography_scanners: int | None
    h100_sized_chips_per_wafer: float | None
    wafers_per_month_during_operation: float | None

class EnergyGeneration(Assets):
    continuous_power_generation_GW: float
    production_method: ProductionTechnology
    energy_generation_method: EnergyGenerationMethod

class Robots(Assets):
    embodied_ai_labor_in_human_equivalents: int

class RobotFactories(Assets):
    monthly_production: Robots
    production_method: ProductionTechnology

class UnmannedWeapons(Assets):
    type: UnmannedWeaponType
    metric_tons_of_explosives: float | None

class UnmannedWeaponsFactory(Assets):
    type: UnmannedWeaponType
    monthly_production: Robots
    production_method: ProductionTechnology
```

### AI Policies

An AI policy is a set of rules that restrict entities. Here are the provisions that an AI policy can have:

```python
class AIPolicy:
    template_id: str
    entities_subject_to_policy: list[Entity]
    entities_verifying_compliance: list[Entity]
    verification_protocol: VerificationProtocol | None

    # Export controls
    compute_export_blacklist: list[Entity] = []
    SME_export_blacklist: list[Entity] = []

    # Location reporting (e.g. for trust building or to help enforce export controls)
    compute_location_reporting: bool = False

    # AI researcher headcount caps
    ai_researcher_headcount_cap: int | None = None

    # AI R&D caps
    non_inference_compute_cap_tpp_h100e: float | None = None
    experiment_compute_cap_tpp_h100e: float | None = None
    ai_capability_cap: AISoftwareCapabilityCap | None = None

    # Compute production caps
    compute_production_cap_monthly_tpp_h100e: float | None = None
    compute_production_capacity_cap_monthly_tpp_h100e: float | None = None

    # Alignment standard
    international_alignment_standard: bool = False
```

Supported AI policy templates include:
- `US_EXPORT_CONTROLS` - US export controls on compute and SME
- `INTERNATIONAL_AI_RND_SLOWDOWN` - International AI R&D slowdown agreement
- `INTERNATIONAL_ALIGNMENT_STANDARD` - International alignment standard

There are many other classes (see `model_state_classes`), but these are the core ones.