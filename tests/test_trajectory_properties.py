"""
Trajectory property tests for AI Futures Model.

These tests verify invariants that should always hold for any valid simulation,
regardless of golden data. They test mathematical and physical constraints
of the model.
"""

import numpy as np
import pytest

from tests.utils import (
    extract_trajectory_field,
    assert_monotonic,
    assert_bounded,
    assert_finite,
    NUMERICAL_ZERO,
    FLOAT32_ATOL,
    TIME_SYNC_ATOL,
    YEAR_RANGE_ATOL,
    MONOTONICITY_ATOL,
    COMPLIANCE_THRESHOLD,
    REPRODUCIBILITY_DECIMALS,
    SHORT_SIMULATION_EVAL_POINTS,
)


class TestTrajectoryInvariants:
    """Test invariants that should always hold for valid simulations."""

    @pytest.fixture
    def simulation_result(self, simulator):
        """Run a single simulation and return the trajectory."""
        return simulator.run_modal_simulation()

    def test_times_are_monotonic(self, simulation_result):
        """Time values should be monotonically increasing."""
        times = simulation_result.times.numpy()
        assert_monotonic(times, field_name="times", non_decreasing=True)

    def test_progress_is_monotonic(self, simulation_result):
        """Progress should be monotonically non-decreasing."""
        progress = extract_trajectory_field(simulation_result, "progress")
        assert_monotonic(progress, field_name="progress", non_decreasing=True)

    def test_automation_fraction_bounded(self, simulation_result):
        """Automation fraction should be in [0, 1]."""
        af = extract_trajectory_field(simulation_result, "automation_fraction")
        assert_bounded(af, 0.0, 1.0, field_name="automation_fraction")

    def test_ai_research_taste_positive(self, simulation_result):
        """AI research taste should be positive."""
        art = extract_trajectory_field(simulation_result, "ai_research_taste")
        # Note: ai_research_taste is not bounded to [0,1] - it can grow large
        assert np.all(art >= -NUMERICAL_ZERO), f"AI research taste should be positive. Min: {np.min(art)}"

    def test_research_stock_positive(self, simulation_result):
        """Research stock should be non-negative."""
        rs = extract_trajectory_field(simulation_result, "research_stock")
        assert np.all(rs >= -NUMERICAL_ZERO), f"Research stock should be non-negative. Min: {np.min(rs)}"

    def test_progress_finite(self, simulation_result):
        """All progress values should be finite."""
        progress = extract_trajectory_field(simulation_result, "progress")
        assert_finite(progress, field_name="progress")

    def test_automation_fraction_finite(self, simulation_result):
        """All automation fraction values should be finite."""
        af = extract_trajectory_field(simulation_result, "automation_fraction")
        assert_finite(af, field_name="automation_fraction")

    def test_research_stock_finite(self, simulation_result):
        """All research stock values should be finite."""
        rs = extract_trajectory_field(simulation_result, "research_stock")
        assert_finite(rs, field_name="research_stock")

    def test_software_progress_rate_non_negative(self, simulation_result):
        """Software progress rate should be non-negative."""
        spr = extract_trajectory_field(simulation_result, "software_progress_rate")
        # Filter out NaN values
        spr_valid = spr[~np.isnan(spr)]
        assert np.all(spr_valid >= -NUMERICAL_ZERO), (
            f"Software progress rate should be non-negative. Min: {np.min(spr_valid)}"
        )

    def test_trajectory_has_expected_length(self, simulation_result):
        """Trajectory should have the expected number of points."""
        n_points = len(simulation_result.trajectory)
        n_times = len(simulation_result.times)
        assert n_points == n_times, (
            f"Trajectory length ({n_points}) should match times length ({n_times})"
        )


class TestTrajectoryConsistency:
    """Test consistency relationships between fields."""

    @pytest.fixture
    def simulation_result(self, simulator):
        """Run a single simulation and return the trajectory."""
        return simulator.run_modal_simulation()

    def test_automation_increases_with_progress(self, simulation_result):
        """Automation fraction should generally increase as progress increases."""
        progress = extract_trajectory_field(simulation_result, "progress")
        af = extract_trajectory_field(simulation_result, "automation_fraction")

        # Both should increase monotonically (or at least not decrease significantly)
        progress_diff = np.diff(progress)
        af_diff = np.diff(af)

        # Where progress is increasing, automation should usually not decrease
        progress_increasing = progress_diff >= -MONOTONICITY_ATOL
        af_changes_where_progress_increases = af_diff[progress_increasing]

        # Allow some tolerance - automation shouldn't decrease when progress increases
        if len(af_changes_where_progress_increases) > 0:
            compliance_rate = np.mean(af_changes_where_progress_increases >= -FLOAT32_ATOL)
            assert compliance_rate > COMPLIANCE_THRESHOLD, (
                f"Automation should generally increase with progress. "
                f"Compliance rate: {compliance_rate:.2%}"
            )

    def test_times_cover_expected_range(self, simulation_result):
        """Times should cover the configured simulation range."""
        times = simulation_result.times.numpy()
        params = simulation_result.params

        if params.settings is not None:
            start_year = params.settings.simulation_start_year
            end_year = params.settings.simulation_end_year

            assert abs(times[0] - start_year) < YEAR_RANGE_ATOL, (
                f"First time point should be near start year. "
                f"Got {times[0]}, expected {start_year}"
            )
            assert abs(times[-1] - end_year) < YEAR_RANGE_ATOL, (
                f"Last time point should be near end year. "
                f"Got {times[-1]}, expected {end_year}"
            )


class TestEdgeCases:
    """Test behavior at edge cases and boundary conditions."""

    def test_short_simulation(self, model_parameters):
        """Test a very short simulation doesn't crash."""
        from ai_futures_simulator import AIFuturesSimulator

        # Modify to run a short simulation
        if model_parameters.params.settings is not None:
            model_parameters.params.settings.simulation_end_year = (
                model_parameters.params.settings.simulation_start_year + 1
            )
            model_parameters.params.settings.n_eval_points = SHORT_SIMULATION_EVAL_POINTS

        simulator = AIFuturesSimulator(model_parameters=model_parameters)
        result = simulator.run_modal_simulation()

        # Should still produce valid output
        assert len(result.trajectory) > 0
        progress = extract_trajectory_field(result, "progress")
        assert_finite(progress, field_name="progress")

    def test_simulation_reproducibility(self, model_parameters):
        """Running with same parameters should produce identical results."""
        from ai_futures_simulator import AIFuturesSimulator

        simulator1 = AIFuturesSimulator(model_parameters=model_parameters)
        result1 = simulator1.run_modal_simulation()

        simulator2 = AIFuturesSimulator(model_parameters=model_parameters)
        result2 = simulator2.run_modal_simulation()

        progress1 = extract_trajectory_field(result1, "progress")
        progress2 = extract_trajectory_field(result2, "progress")

        np.testing.assert_array_almost_equal(
            progress1, progress2,
            decimal=REPRODUCIBILITY_DECIMALS,
            err_msg="Modal simulation should be reproducible"
        )


class TestWorldStateIntegrity:
    """Test that World objects maintain internal consistency."""

    @pytest.fixture
    def simulation_result(self, simulator):
        """Run a single simulation and return the trajectory."""
        return simulator.run_modal_simulation()

    def test_all_worlds_have_current_time(self, simulation_result):
        """Every World in the trajectory should have a valid current_time."""
        for i, world in enumerate(simulation_result.trajectory):
            assert hasattr(world, "current_time"), f"World at index {i} missing current_time"
            time_val = world.current_time
            assert time_val is not None, f"World at index {i} has None current_time"

    def test_all_worlds_have_software_progress(self, simulation_result):
        """Every World in the trajectory should have software_progress via ai_software_developers or black_projects."""
        for i, world in enumerate(simulation_result.trajectory):
            has_software_progress = False

            # Check ai_software_developers
            if hasattr(world, "ai_software_developers") and world.ai_software_developers:
                for dev_id, developer in world.ai_software_developers.items():
                    if hasattr(developer, "ai_software_progress") and developer.ai_software_progress is not None:
                        has_software_progress = True
                        break

            # Check black_projects
            if not has_software_progress and hasattr(world, "black_projects") and world.black_projects:
                for proj_id, project in world.black_projects.items():
                    if hasattr(project, "ai_software_progress") and project.ai_software_progress is not None:
                        has_software_progress = True
                        break

            assert has_software_progress, (
                f"World at index {i} has no software_progress in ai_software_developers or black_projects"
            )

    def test_times_match_world_current_times(self, simulation_result):
        """Times array should match current_time in each World."""
        times = simulation_result.times.numpy()

        for i, world in enumerate(simulation_result.trajectory):
            world_time = float(
                world.current_time.item()
                if hasattr(world.current_time, "item")
                else world.current_time
            )
            expected_time = times[i]
            # Use tolerance that accounts for ODE solver adaptive stepping and float32 precision
            assert abs(world_time - expected_time) < TIME_SYNC_ATOL, (
                f"Time mismatch at index {i}. "
                f"World.current_time={world_time}, times[{i}]={expected_time}"
            )
