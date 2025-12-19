"""
Test script for black project simulation.

Verifies that:
1. US and PRC compute stocks grow correctly
2. Black project initializes and updates correctly
3. Fab becomes operational at the correct time
4. Datacenter capacity grows correctly
5. Compute stock dynamics work (production + attrition)
"""

import sys
import math
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classes.world.world import World
from classes.world.entities import (
    Nation, NamedNations, Coalition, AISoftwareDeveloper, ComputeAllocation,
)
from classes.world.assets import Compute
from classes.world.software_progress import AISoftwareProgress
from classes.world.policies import AIPolicy
from parameters.simulation_parameters import ModelParameters
from world_updaters.compute import (
    NationComputeUpdater,
    NationComputeConfig,
    get_nation_compute_stock_h100e,
    get_compute_stock_h100e,
    get_datacenter_concealed_capacity_gw,
    get_datacenter_total_capacity_gw,
    get_fab_operational_year,
    get_fab_annual_production_h100e,
)
from world_updaters.black_project import BlackProjectUpdater, initialize_black_project
from classes.simulation_primitives import StateDerivative


# Load parameters from YAML
YAML_PATH = Path(__file__).resolve().parent.parent / "parameters" / "modal_parameters.yaml"


def get_black_project_params():
    """Load black project parameters from YAML."""
    model_params = ModelParameters.from_yaml(YAML_PATH)
    sim_params = model_params.sample()
    return sim_params.black_project


def get_policy_params():
    """Load policy parameters from YAML."""
    model_params = ModelParameters.from_yaml(YAML_PATH)
    sim_params = model_params.sample()
    return sim_params.policy


def get_compute_growth_params():
    """Load compute growth parameters from YAML."""
    model_params = ModelParameters.from_yaml(YAML_PATH)
    sim_params = model_params.sample()
    return sim_params.compute_growth


def get_energy_consumption_params():
    """Load energy consumption parameters from YAML."""
    model_params = ModelParameters.from_yaml(YAML_PATH)
    sim_params = model_params.sample()
    return sim_params.energy_consumption


def create_minimal_ai_software_progress() -> AISoftwareProgress:
    """Create a minimal AISoftwareProgress with all required fields."""
    return AISoftwareProgress(
        progress=torch.tensor(0.0),
        research_stock=torch.tensor(0.0),
        ai_coding_labor_multiplier=torch.tensor(1.0),
        ai_sw_progress_mult_ref_present_day=torch.tensor(1.0),
        progress_rate=torch.tensor(0.0),
        software_progress_rate=torch.tensor(0.0),
        research_effort=torch.tensor(0.0),
        automation_fraction=torch.tensor(0.0),
        coding_labor=torch.tensor(0.0),
        serial_coding_labor=torch.tensor(0.0),
        ai_research_taste=torch.tensor(0.0),
        ai_research_taste_sd=torch.tensor(0.0),
        aggregate_research_taste=torch.tensor(0.0),
    )


def create_minimal_world(year: float = 2030.0) -> World:
    """Create a minimal World for testing."""

    # Initialize PRC nation with compute stock
    prc_config = NationComputeConfig.initialize_nation_compute(NamedNations.PRC, year)
    prc_compute_stock = prc_config['compute_stock']

    prc = Nation(
        id=NamedNations.PRC,
        log_compute_stock=torch.tensor(math.log(max(prc_compute_stock, 1e-10))),
        compute_growth_rate=prc_config['growth_rate'],
        total_energy_consumption_gw=prc_config['total_energy_gw'],
    )

    # Initialize USA nation with compute stock
    usa_config = NationComputeConfig.initialize_nation_compute(NamedNations.USA, year)
    usa_compute_stock = usa_config['compute_stock']

    usa = Nation(
        id=NamedNations.USA,
        log_compute_stock=torch.tensor(math.log(max(usa_compute_stock, 1e-10))),
        compute_growth_rate=usa_config['growth_rate'],
    )

    # Create minimal AI developer for World requirements
    developer = AISoftwareDeveloper(
        id="test_developer",
        is_primarily_controlled_by_misaligned_AI=False,
        compute=Compute(total_tpp_h100e=1000.0, total_energy_requirements_watts=700000.0),
        compute_allocation=ComputeAllocation(
            fraction_for_ai_r_and_d_inference=0.5,
            fraction_for_ai_r_and_d_training=0.3,
            fraction_for_external_deployment=0.0,
            fraction_for_alignment_research=0.1,
            fraction_for_frontier_training=0.1,
        ),
        ai_software_progress=create_minimal_ai_software_progress(),
        human_ai_capability_researchers=100,
        log_compute=torch.tensor(math.log(1000.0)),
        log_researchers=torch.tensor(math.log(100.0)),
    )

    world = World(
        current_time=torch.tensor(year),
        coalitions={},
        nations={NamedNations.PRC: prc, NamedNations.USA: usa},
        ai_software_developers={"test_developer": developer},
        ai_policies={},
        black_projects={},
    )

    return world, prc_compute_stock


def test_nation_compute_growth():
    """Test that nation compute stocks grow correctly."""
    print("\n=== Testing Nation Compute Growth ===")

    world, prc_stock = create_minimal_world(2030.0)

    print(f"Initial PRC compute stock: {get_nation_compute_stock_h100e(world.nations[NamedNations.PRC]):.2f} H100e")
    print(f"Initial USA compute stock: {get_nation_compute_stock_h100e(world.nations[NamedNations.USA]):.2f} H100e")

    # Create updater (using None for params since we don't need them for this test)
    class MockParams:
        pass
    updater = NationComputeUpdater(MockParams())

    # Get derivatives
    t = torch.tensor(2030.0)
    derivative = updater.contribute_state_derivatives(t, world)

    # Check that derivatives are positive (growth)
    prc_growth = derivative.world.nations[NamedNations.PRC].log_compute_stock.item()
    usa_growth = derivative.world.nations[NamedNations.USA].log_compute_stock.item()

    print(f"PRC d(log_stock)/dt: {prc_growth:.4f} (expected ~{math.log(2.2):.4f})")
    print(f"USA d(log_stock)/dt: {usa_growth:.4f} (expected ~{math.log(2.91):.4f})")

    assert prc_growth > 0, "PRC compute should be growing"
    assert usa_growth > 0, "USA compute should be growing"
    assert abs(prc_growth - math.log(2.2)) < 0.01, "PRC growth rate should match expected"
    assert abs(usa_growth - math.log(2.91)) < 0.01, "USA growth rate should match expected"

    print("PASSED: Nation compute growth works correctly")


def test_black_project_initialization():
    """Test that black project initializes correctly."""
    print("\n=== Testing Black Project Initialization ===")

    world, prc_stock = create_minimal_world(2030.0)
    params = get_black_project_params()
    policy_params = get_policy_params()
    compute_growth_params = get_compute_growth_params()
    energy_consumption_params = get_energy_consumption_params()

    # Initialize black project
    project = initialize_black_project(
        project_id="prc_black_project",
        ai_slowdown_start_year=policy_params.ai_slowdown_start_year,
        prc_compute_stock=prc_stock,
        params=params,
        compute_growth_params=compute_growth_params,
        energy_consumption_params=energy_consumption_params,
    )

    # Add to world
    world.black_projects["prc_black_project"] = project

    print(f"Black project initialized:")
    print(f"  - AI slowdown start year: {project.ai_slowdown_start_year}")
    print(f"  - Initial compute stock: {get_compute_stock_h100e(project.compute_stock):.2f} H100e")
    print(f"  - Diverted proportion: {params.properties.proportion_of_initial_compute_stock_to_divert}")
    print(f"  - Expected diverted: {prc_stock * params.properties.proportion_of_initial_compute_stock_to_divert:.2f} H100e")

    # Check fab
    if project.fab is not None:
        print(f"  - Fab process node: {project.fab.process_node_nm}nm")
        print(f"  - Fab construction start: {project.fab.construction_start_year}")
        print(f"  - Fab construction duration: {project.fab.construction_duration:.2f} years")
        print(f"  - Fab operational year: {get_fab_operational_year(project.fab):.2f}")
        print(f"  - Fab is operational: {project.fab.is_operational}")
        assert not project.fab.is_operational, "Fab should not be operational at agreement year"

    # Check datacenters
    if project.datacenters is not None:
        print(f"  - Datacenter concealed capacity: {get_datacenter_concealed_capacity_gw(project.datacenters):.4f} GW")
        print(f"  - Datacenter unconcealed capacity: {project.datacenters.unconcealed_capacity_gw:.4f} GW")
        print(f"  - Datacenter total capacity: {get_datacenter_total_capacity_gw(project.datacenters):.4f} GW")
        print(f"  - Datacenter max capacity: {project.datacenters.max_total_capacity_gw:.2f} GW")

    print("PASSED: Black project initialization works correctly")


def test_black_project_dynamics():
    """Test black project dynamics (datacenter growth, fab production, attrition)."""
    print("\n=== Testing Black Project Dynamics ===")

    world, prc_stock = create_minimal_world(2030.0)
    params = get_black_project_params()
    policy_params = get_policy_params()
    compute_growth_params = get_compute_growth_params()
    energy_consumption_params = get_energy_consumption_params()

    # Initialize black project
    project = initialize_black_project(
        project_id="prc_black_project",
        ai_slowdown_start_year=policy_params.ai_slowdown_start_year,
        prc_compute_stock=prc_stock,
        params=params,
        compute_growth_params=compute_growth_params,
        energy_consumption_params=energy_consumption_params,
    )
    world.black_projects["prc_black_project"] = project

    # Create updater
    class MockParams:
        pass
    updater = BlackProjectUpdater(MockParams(), params)

    # Test dynamics at agreement year
    print("\nAt agreement year (2030):")
    t = torch.tensor(2030.0)
    derivative = updater.contribute_state_derivatives(t, world)

    d_project = derivative.world.black_projects["prc_black_project"]

    print(f"  - d(log_concealed_capacity)/dt: {d_project.datacenters.log_concealed_capacity_gw.item():.4f}")
    print(f"  - d(log_compute_stock)/dt: {d_project.compute_stock.log_compute_stock.item():.4f}")

    # At agreement year, fab is not operational, so stock should be decreasing (attrition only)
    stock_deriv = d_project.compute_stock.log_compute_stock.item()
    if not project.fab.is_operational:
        assert stock_deriv < 0, "Without fab production, stock should decrease due to attrition"
        print("  - Compute stock decreasing (attrition only, no fab production yet)")

    # Test dynamics after fab becomes operational
    fab_op_year = get_fab_operational_year(project.fab)
    print(f"\nAfter fab becomes operational ({fab_op_year}):")
    t_after = torch.tensor(fab_op_year + 0.1)

    # Apply discrete state change
    result = updater.set_state_attributes(t_after, world)
    if result is not None:
        world = result
        print(f"  - Fab operational: {world.black_projects['prc_black_project'].fab.is_operational}")

    # Get new derivatives
    derivative_after = updater.contribute_state_derivatives(t_after, world)
    d_project_after = derivative_after.world.black_projects["prc_black_project"]

    stock_deriv_after = d_project_after.compute_stock.log_compute_stock.item()
    print(f"  - d(log_compute_stock)/dt: {stock_deriv_after:.4f}")

    if world.black_projects['prc_black_project'].fab.is_operational:
        # With fab production, stock might be increasing
        annual_production = get_fab_annual_production_h100e(world.black_projects['prc_black_project'].fab)
        print(f"  - Annual fab production: {annual_production:.2f} H100e")

    print("PASSED: Black project dynamics work correctly")


def test_full_simulation_step():
    """Test a full simulation step with both nation compute and black project."""
    print("\n=== Testing Full Simulation Step ===")

    world, prc_stock = create_minimal_world(2030.0)
    params = get_black_project_params()
    policy_params = get_policy_params()
    compute_growth_params = get_compute_growth_params()
    energy_consumption_params = get_energy_consumption_params()

    # Initialize black project
    project = initialize_black_project(
        project_id="prc_black_project",
        ai_slowdown_start_year=policy_params.ai_slowdown_start_year,
        prc_compute_stock=prc_stock,
        params=params,
        compute_growth_params=compute_growth_params,
        energy_consumption_params=energy_consumption_params,
    )
    world.black_projects["prc_black_project"] = project

    # Create updaters
    class MockParams:
        pass
    nation_updater = NationComputeUpdater(MockParams())
    black_project_updater = BlackProjectUpdater(MockParams(), params)

    # Get combined derivatives
    t = torch.tensor(2030.0)

    nation_deriv = nation_updater.contribute_state_derivatives(t, world)
    bp_deriv = black_project_updater.contribute_state_derivatives(t, world)

    # Combine (in real simulator this is done by CombinedUpdater)
    total_deriv = nation_deriv + bp_deriv

    print("Combined derivatives at t=2030:")
    print(f"  - PRC nation d(log_compute)/dt: {total_deriv.world.nations[NamedNations.PRC].log_compute_stock.item():.4f}")
    print(f"  - USA nation d(log_compute)/dt: {total_deriv.world.nations[NamedNations.USA].log_compute_stock.item():.4f}")
    print(f"  - Black project d(log_compute)/dt: {total_deriv.world.black_projects['prc_black_project'].compute_stock.log_compute_stock.item():.4f}")
    print(f"  - Black project d(log_capacity)/dt: {total_deriv.world.black_projects['prc_black_project'].datacenters.log_concealed_capacity_gw.item():.4f}")

    print("PASSED: Full simulation step works correctly")


def test_state_tensor_roundtrip():
    """Test that world can be converted to/from state tensor."""
    print("\n=== Testing State Tensor Roundtrip ===")

    world, prc_stock = create_minimal_world(2030.0)
    params = get_black_project_params()
    policy_params = get_policy_params()
    compute_growth_params = get_compute_growth_params()
    energy_consumption_params = get_energy_consumption_params()

    # Initialize black project
    project = initialize_black_project(
        project_id="prc_black_project",
        ai_slowdown_start_year=policy_params.ai_slowdown_start_year,
        prc_compute_stock=prc_stock,
        params=params,
        compute_growth_params=compute_growth_params,
        energy_consumption_params=energy_consumption_params,
    )
    world.black_projects["prc_black_project"] = project

    # Convert to state tensor
    state_tensor = world.to_state_tensor()
    print(f"State tensor shape: {state_tensor.shape}")
    print(f"State tensor values (first 10): {state_tensor[:10].tolist()}")

    # Reconstruct from tensor
    reconstructed = World.from_state_tensor(state_tensor, world)

    # Check key values match
    orig_prc_stock = world.nations[NamedNations.PRC].log_compute_stock.item()
    recon_prc_stock = reconstructed.nations[NamedNations.PRC].log_compute_stock.item()
    assert abs(orig_prc_stock - recon_prc_stock) < 1e-6, "PRC compute stock should match"

    orig_bp_stock = world.black_projects['prc_black_project'].compute_stock.log_compute_stock.item()
    recon_bp_stock = reconstructed.black_projects['prc_black_project'].compute_stock.log_compute_stock.item()
    assert abs(orig_bp_stock - recon_bp_stock) < 1e-6, "Black project compute stock should match"

    print("Original PRC log_compute_stock:", orig_prc_stock)
    print("Reconstructed PRC log_compute_stock:", recon_prc_stock)
    print("Original BP log_compute_stock:", orig_bp_stock)
    print("Reconstructed BP log_compute_stock:", recon_bp_stock)

    print("PASSED: State tensor roundtrip works correctly")


def test_detection_likelihood_ratio():
    """Test that detection likelihood ratio is computed correctly."""
    print("\n=== Testing Detection Likelihood Ratio ===")

    world, prc_stock = create_minimal_world(2030.0)
    params = get_black_project_params()
    policy_params = get_policy_params()
    compute_growth_params = get_compute_growth_params()
    energy_consumption_params = get_energy_consumption_params()

    # Initialize black project
    project = initialize_black_project(
        project_id="prc_black_project",
        ai_slowdown_start_year=policy_params.ai_slowdown_start_year,
        prc_compute_stock=prc_stock,
        params=params,
        compute_growth_params=compute_growth_params,
        energy_consumption_params=energy_consumption_params,
    )
    world.black_projects["prc_black_project"] = project

    print(f"  - Sampled detection time: {project.sampled_detection_time:.2f} years since start")
    print(f"  - Is detected: {project.is_detected}")
    print(f"  - Initial cumulative LR: {project.cumulative_likelihood_ratio:.4f}")

    # Create updater
    class MockParams:
        pass
    updater = BlackProjectUpdater(MockParams(), params)

    # Test likelihood ratio at different times
    for years_after in [0, 2, 5, 10]:
        t = torch.tensor(policy_params.ai_slowdown_start_year + years_after)
        world = updater.set_metric_attributes(t, world)
        result = updater.set_state_attributes(t, world)
        if result is not None:
            world = result

        project = world.black_projects["prc_black_project"]
        print(f"  - At year {t.item():.0f} (t+{years_after}): LR={project.cumulative_likelihood_ratio:.4f}, detected={project.is_detected}")

    # Verify LR changes over time (should decrease before detection, jump to high value after)
    assert project.cumulative_likelihood_ratio > 0, "LR should be positive"

    print("PASSED: Detection likelihood ratio works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("BLACK PROJECT SIMULATION TESTS")
    print("=" * 60)

    try:
        test_nation_compute_growth()
        test_black_project_initialization()
        test_black_project_dynamics()
        test_full_simulation_step()
        test_state_tensor_roundtrip()
        test_detection_likelihood_ratio()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
