"""
Golden trajectory tests for AI Futures Model.

These tests compare the PyTorch implementation against golden data
fetched from the production model at aifuturesmodel.com.

Run `python -m scripts.generate_golden_data` before running these tests.
"""

import numpy as np
import pytest

from tests.conftest import requires_golden_data
from tests.utils import (
    load_golden_data,
    get_golden_field,
    get_golden_years,
    get_golden_time_series,
    extract_trajectory_field,
    assert_arrays_close,
    TRAJECTORY_RTOL,
    STRICT_RTOL,
    FIELD_MAPPING,
    has_golden_data,
    EXPECTED_INPUT_DATA_START_YEAR,
    EXPECTED_INPUT_DATA_END_YEAR,
)


@requires_golden_data
class TestDefaultTrajectory:
    """Compare PyTorch model against default golden trajectory from production API."""

    def test_golden_data_exists(self, golden_trajectory):
        """Verify golden data is properly loaded."""
        assert golden_trajectory is not None
        assert "time_series" in golden_trajectory
        assert "success" in golden_trajectory
        assert golden_trajectory["success"] is True

    def test_time_series_has_data(self, golden_trajectory):
        """Verify time series contains data points."""
        time_series = get_golden_time_series(golden_trajectory)
        assert len(time_series) > 0, "Time series should not be empty"

    def test_years_are_sequential(self, golden_trajectory):
        """Verify years in golden data are monotonically increasing."""
        years = get_golden_years(golden_trajectory)
        diffs = np.diff(years)
        assert np.all(diffs > 0), "Years should be monotonically increasing"

    def test_progress_values_exist(self, golden_trajectory):
        """Verify progress values exist in golden data."""
        progress = get_golden_field(golden_trajectory, "progress")
        assert len(progress) > 0
        assert not np.any(np.isnan(progress)), "Progress should not contain NaN"

    @pytest.mark.slow
    def test_progress_matches_production(self, golden_trajectory, computed_trajectory):
        """
        Compare progress values between PyTorch model and production API.

        This is the key test - if progress matches, the core ODE dynamics are correct.
        """
        golden_progress = get_golden_field(golden_trajectory, "progress")
        computed_progress = extract_trajectory_field(computed_trajectory, "progress")

        # Align arrays by length (they may have different sampling rates)
        min_len = min(len(golden_progress), len(computed_progress))
        if min_len < len(golden_progress):
            # Interpolate golden data to match computed trajectory
            golden_years = get_golden_years(golden_trajectory)
            computed_years = computed_trajectory.times.numpy()
            golden_progress = np.interp(computed_years, golden_years, golden_progress)

        assert_arrays_close(
            computed_progress[:min_len],
            golden_progress[:min_len],
            rtol=TRAJECTORY_RTOL,
            field_name="progress",
        )

    @pytest.mark.slow
    def test_automation_fraction_matches_production(self, golden_trajectory, computed_trajectory):
        """Compare automation fraction values."""
        golden_af = get_golden_field(golden_trajectory, "automationFraction")
        computed_af = extract_trajectory_field(computed_trajectory, "automation_fraction")

        golden_years = get_golden_years(golden_trajectory)
        computed_years = computed_trajectory.times.numpy()
        golden_af_interp = np.interp(computed_years, golden_years, golden_af)

        assert_arrays_close(
            computed_af,
            golden_af_interp,
            rtol=TRAJECTORY_RTOL,
            field_name="automation_fraction",
        )

    @pytest.mark.slow
    def test_research_stock_matches_production(self, golden_trajectory, computed_trajectory):
        """Compare research stock values."""
        golden_rs = get_golden_field(golden_trajectory, "researchStock")
        computed_rs = extract_trajectory_field(computed_trajectory, "research_stock")

        golden_years = get_golden_years(golden_trajectory)
        computed_years = computed_trajectory.times.numpy()
        golden_rs_interp = np.interp(computed_years, golden_years, golden_rs)

        assert_arrays_close(
            computed_rs,
            golden_rs_interp,
            rtol=TRAJECTORY_RTOL,
            field_name="research_stock",
        )

    @pytest.mark.slow
    def test_ai_research_taste_matches_production(self, golden_trajectory, computed_trajectory):
        """Compare AI research taste values."""
        golden_art = get_golden_field(golden_trajectory, "aiResearchTaste")
        computed_art = extract_trajectory_field(computed_trajectory, "ai_research_taste")

        golden_years = get_golden_years(golden_trajectory)
        computed_years = computed_trajectory.times.numpy()
        golden_art_interp = np.interp(computed_years, golden_years, golden_art)

        assert_arrays_close(
            computed_art,
            golden_art_interp,
            rtol=TRAJECTORY_RTOL,
            field_name="ai_research_taste",
        )

    @pytest.mark.slow
    def test_software_progress_rate_matches_production(self, golden_trajectory, computed_trajectory):
        """Compare software progress rate values."""
        golden_spr = get_golden_field(golden_trajectory, "softwareProgressRate")
        computed_spr = extract_trajectory_field(computed_trajectory, "software_progress_rate")

        golden_years = get_golden_years(golden_trajectory)
        computed_years = computed_trajectory.times.numpy()
        golden_spr_interp = np.interp(computed_years, golden_years, golden_spr)

        assert_arrays_close(
            computed_spr,
            golden_spr_interp,
            rtol=TRAJECTORY_RTOL,
            field_name="software_progress_rate",
        )


@requires_golden_data
class TestInputDataConsistency:
    """Tests to verify input data consistency between local and production."""

    def test_input_data_file_exists(self):
        """Verify local input_data.csv exists."""
        from pathlib import Path

        input_data_path = (
            Path(__file__).parent.parent
            / "ai_futures_simulator"
            / "parameters"
            / "input_data.csv"
        )
        assert input_data_path.exists(), f"Input data file not found: {input_data_path}"

    def test_input_data_has_expected_columns(self):
        """Verify input_data.csv has the expected columns."""
        import csv
        from pathlib import Path

        input_data_path = (
            Path(__file__).parent.parent
            / "ai_futures_simulator"
            / "parameters"
            / "input_data.csv"
        )

        with open(input_data_path, "r") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames

        expected_columns = {"time", "L_HUMAN", "inference_compute", "experiment_compute", "training_compute"}
        assert expected_columns.issubset(set(columns)), (
            f"Missing expected columns. Expected: {expected_columns}, Got: {set(columns)}"
        )

    def test_input_data_time_range(self):
        """Verify input_data.csv covers expected time range."""
        import csv
        from pathlib import Path

        input_data_path = (
            Path(__file__).parent.parent
            / "ai_futures_simulator"
            / "parameters"
            / "input_data.csv"
        )

        with open(input_data_path, "r") as f:
            reader = csv.DictReader(f)
            times = [float(row["time"]) for row in reader]

        assert min(times) == EXPECTED_INPUT_DATA_START_YEAR, (
            f"Expected start year {EXPECTED_INPUT_DATA_START_YEAR}, got {min(times)}"
        )
        assert max(times) == EXPECTED_INPUT_DATA_END_YEAR, (
            f"Expected end year {EXPECTED_INPUT_DATA_END_YEAR}, got {max(times)}"
        )


@requires_golden_data
class TestScenarioTrajectories:
    """Test various parameter scenarios against golden data."""

    @pytest.fixture
    def fast_progress_golden(self):
        """Load fast progress scenario golden data."""
        try:
            return load_golden_data("trajectory_fast_progress.json")
        except FileNotFoundError:
            pytest.skip("Fast progress golden data not available")

    @pytest.fixture
    def slow_progress_golden(self):
        """Load slow progress scenario golden data."""
        try:
            return load_golden_data("trajectory_slow_progress.json")
        except FileNotFoundError:
            pytest.skip("Slow progress golden data not available")

    def test_fast_progress_data_valid(self, fast_progress_golden):
        """Verify fast progress golden data is valid."""
        assert fast_progress_golden is not None
        assert fast_progress_golden.get("success") is True
        time_series = get_golden_time_series(fast_progress_golden)
        assert len(time_series) > 0

    def test_slow_progress_data_valid(self, slow_progress_golden):
        """Verify slow progress golden data is valid."""
        assert slow_progress_golden is not None
        assert slow_progress_golden.get("success") is True
        time_series = get_golden_time_series(slow_progress_golden)
        assert len(time_series) > 0

    def test_fast_vs_slow_progress_ordering(self, fast_progress_golden, slow_progress_golden):
        """
        Verify fast progress scenario achieves higher progress than slow.

        This is a sanity check that the scenarios are meaningfully different.
        """
        fast_progress = get_golden_field(fast_progress_golden, "progress")
        slow_progress = get_golden_field(slow_progress_golden, "progress")

        # Compare final progress values
        final_fast = fast_progress[-1]
        final_slow = slow_progress[-1]

        assert final_fast > final_slow, (
            f"Fast progress scenario should achieve higher final progress. "
            f"Fast: {final_fast:.4f}, Slow: {final_slow:.4f}"
        )


@requires_golden_data
class TestGoldenDataMetadata:
    """Test golden data metadata and generation info."""

    def test_metadata_present(self, golden_trajectory):
        """Verify golden data includes generation metadata."""
        assert "_metadata" in golden_trajectory
        metadata = golden_trajectory["_metadata"]
        assert "source" in metadata
        assert "time_range" in metadata

    def test_source_is_production(self, golden_trajectory):
        """Verify golden data came from production API."""
        metadata = golden_trajectory["_metadata"]
        assert "aifuturesmodel.com" in metadata["source"]

    def test_time_range_valid(self, golden_trajectory):
        """Verify time range in metadata is valid."""
        metadata = golden_trajectory["_metadata"]
        time_range = metadata["time_range"]
        assert len(time_range) == 2
        assert time_range[0] < time_range[1]
