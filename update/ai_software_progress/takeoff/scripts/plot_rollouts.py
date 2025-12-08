#!/usr/bin/env python3
"""
Plot utilities for analyzing batch rollout results.

Currently supported:
- Plot distribution (histogram) of AC times from a rollouts.jsonl file
- Plot distribution (histogram) of horizon length at AC across rollouts
- Plot distribution (histogram) of arrival time for arbitrary milestones
- Plot distribution (histogram) of effective compute (OOMs) required to achieve milestones

Usage examples:
  python scripts/plot_rollouts.py --run-dir outputs/20250813_020347 \
    --out outputs/20250813_020347/aa_time_hist.png

  python scripts/plot_rollouts.py --rollouts outputs/20250813_020347/rollouts.jsonl

  python scripts/plot_rollouts.py --rollouts outputs/20250813_020347/rollouts.jsonl \
    --mode milestone_compute_hist --milestone AC
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr, norm
from datetime import datetime, timedelta
import yaml

# Use monospace font for all text elements
matplotlib.rcParams["font.family"] = "monospace"

# Import utilities
from plotting_utils.kde import make_gaussian_kde
from plotting_utils.rollouts_reader import RolloutsReader
from plotting_utils.helpers import (
    decimal_year_to_date_string,
    format_time_duration,
    format_years_value,
    get_year_tick_values_and_labels,
    now_decimal_year,
    get_time_tick_values_and_labels,
    load_metr_p80_points,
    simplify_milestone_name,
)
from plotting.histograms import (
    plot_aa_time_histogram,
    plot_aa_time_histogram_cdf,
    plot_horizon_at_sc_histogram,
    plot_milestone_effective_compute_histogram,
    plot_milestone_time_histogram,
    plot_milestone_time_histogram_cdf,
    plot_m_over_beta_histogram,
    plot_x_years_in_1_year_histogram,
)
from plotting.trajectories import (
    plot_horizon_trajectories,
    plot_uplift_trajectories,
    plot_metric_trajectories,
)
from sensitivity_analysis import plot_all_correlations
from plotting.scatter import plot_milestone_scatter
from plotting.boxplots import plot_milestone_transition_boxplot
from milestone_pdfs import plot_milestone_pdfs_overlay, _load_milestone_batch


def _resolve_rollouts_path(run_dir: Optional[str], rollouts_path: Optional[str]) -> Path:
    if rollouts_path:
        p = Path(rollouts_path)
        if not p.exists():
            raise FileNotFoundError(f"rollouts file does not exist: {p}")
        return p
    if not run_dir:
        raise ValueError("Either --run-dir or --rollouts must be provided")
    d = Path(run_dir)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"run directory does not exist or is not a directory: {d}")
    p = d / "rollouts.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"rollouts.jsonl not found in run directory: {d}")
    return p


# Removed duplicate functions - now using imports from plotting_utils.helpers:
# - _decimal_year_to_date_string -> decimal_year_to_date_string
# - _now_decimal_year -> now_decimal_year
# - _format_time_duration -> format_time_duration
# - _get_time_tick_values_and_labels -> get_time_tick_values_and_labels
# - _load_metr_p80_points -> load_metr_p80_points


def _read_m_over_beta_values(rollouts_file: Path) -> Tuple[List[float], int]:
    """Read m/beta (automation_fraction_a / automation_fraction_b) values from rollouts.

    Returns:
        m_over_beta_values: list of m/beta ratio values
        num_skipped: count of rollouts where r_software was missing
    """
    m_over_beta_values: List[float] = []
    num_skipped: int = 0

    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            params = rec.get("parameters")
            if not isinstance(params, dict):
                continue

            # Extract beta_software (this is automation_fraction_b, i.e., β)
            beta_software = params.get("beta_software")
            # Extract r_software (used to compute automation_fraction_a, i.e., m)
            # m = 1 / (1 + exp(-r_software))
            r_software = params.get("r_software")

            if r_software is None:
                num_skipped += 1
                continue

            try:
                r_soft = float(r_software)
                beta_soft = float(beta_software) if beta_software is not None else 0.0

                # Compute m from r_software (sigmoid)
                m = 1.0 / (1.0 + np.exp(-r_soft))

                # Compute m/β
                if beta_soft > 0 and np.isfinite(m) and np.isfinite(beta_soft):
                    m_over_beta = m / beta_soft
                    if np.isfinite(m_over_beta):
                        m_over_beta_values.append(m_over_beta)
                    else:
                        num_skipped += 1
                else:
                    num_skipped += 1
            except (ValueError, TypeError):
                num_skipped += 1
                continue

    return m_over_beta_values, num_skipped


# Removed wrapper functions - use RolloutsReader.read_metric_trajectories() directly:
# - _read_horizon_trajectories -> reader.read_metric_trajectories("horizon_lengths")
# - _read_uplift_trajectories -> reader.read_metric_trajectories("ai_sw_progress_mult_ref_present_day", ...)
# - _read_metric_trajectories -> reader.read_metric_trajectories(metric_name)
# - _read_horizon_at_sc -> reader.read_metric_at_milestone()


def _list_milestone_names(rollouts_file: Path) -> List[str]:
    names = set()
    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            results = rec.get("results")
            if not isinstance(results, dict):
                continue
            milestones = results.get("milestones")
            if isinstance(milestones, dict):
                for k in milestones.keys():
                    names.add(str(k))
    return sorted(names)


# Removed: _read_milestone_times - now using RolloutsReader.read_milestone_times()


# Removed: _read_milestone_effective_compute - now using RolloutsReader.read_milestone_compute()


def _parse_milestone_pairs(pairs_arg: Optional[str]) -> List[Tuple[str, str]]:
    """Parse a pairs string like "SC:SAR,SAR:SIAR" into [("SC","SAR"), ...]."""
    if not pairs_arg:
        return []
    pairs: List[Tuple[str, str]] = []
    for chunk in pairs_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            continue
        left, right = chunk.split(":", 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            pairs.append((left, right))
    return pairs




def _compute_x_years_in_1_year(
    times: np.ndarray,
    progress: np.ndarray,
    current_time_idx: int = 0,
) -> float:
    """Compute maximum 'X years in 1 year' metric for a trajectory.

    Finds the 1-year window with the maximum OOMs crossed and compares it to
    the OOMs crossed in the present year (starting at current_time_idx).

    The metric answers: "If the max year had X times more progress than the
    present year, what is X?" Result should always be >= 1.

    Args:
        times: array of time points (decimal years)
        progress: array of progress values (in OOMs or log10 scale)
        current_time_idx: index representing the "present" time

    Returns:
        X value such that "X years in 1 year" happened (ratio of max to present)
    """
    if len(times) < 2 or len(progress) < 2:
        return np.nan

    # Find OOMs crossed in each rolling 1-year window (looking forward from each point)
    max_ooms_per_year = 0.0

    for i in range(len(times)):
        # Find the point approximately 1 year after times[i]
        target_time = times[i] + 1.0

        # Find the index closest to target_time
        end_idx = None
        for j in range(i + 1, len(times)):
            if times[j] >= target_time:
                end_idx = j
                break

        if end_idx is None:
            # Use the last point if we don't reach 1 year ahead
            if i < len(times) - 1:
                end_idx = len(times) - 1
            else:
                continue

        # Calculate OOMs crossed in this window
        ooms_in_window = progress[end_idx] - progress[i]
        max_ooms_per_year = max(max_ooms_per_year, float(ooms_in_window))

    # Get current year OOMs (1 year window starting from current_time_idx)
    if current_time_idx >= len(times) - 1:
        return np.nan

    target_time = times[current_time_idx] + 1.0
    current_end_idx = None
    for j in range(current_time_idx + 1, len(times)):
        if times[j] >= target_time:
            current_end_idx = j
            break

    if current_end_idx is None:
        current_end_idx = len(times) - 1

    current_ooms = progress[current_end_idx] - progress[current_time_idx]

    if current_ooms <= 0:
        return np.nan

    # This ratio should always be >= 1 since max includes current as a candidate
    return float(max_ooms_per_year / current_ooms)


def _read_x_years_in_1_year(rollouts_file: Path) -> List[float]:
    """Read progress trajectories and compute 'X years in 1 year' metric for each rollout.

    Only includes rollouts where AC (aa_time) is achieved.

    Returns:
        List of X values (one per rollout that has valid progress data and achieved AC)
    """
    x_values: List[float] = []

    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            results = rec.get("results")
            if not isinstance(results, dict):
                continue

            # Check if AC was achieved
            aa_time = results.get("aa_time")
            try:
                aa_t = float(aa_time) if aa_time is not None else np.nan
            except (TypeError, ValueError):
                aa_t = np.nan
            if not np.isfinite(aa_t):
                # Skip rollouts where AC was not achieved
                continue

            times = results.get("times")
            progress = results.get("progress")

            if times is None or progress is None:
                continue

            try:
                times_arr = np.asarray(times, dtype=float)
                progress_arr = np.asarray(progress, dtype=float)
            except Exception:
                continue

            if times_arr.ndim != 1 or progress_arr.ndim != 1 or times_arr.size != progress_arr.size:
                continue

            if times_arr.size < 2:
                continue

            # Assume index 0 is the "present" time
            x_val = _compute_x_years_in_1_year(times_arr, progress_arr, current_time_idx=0)

            if np.isfinite(x_val) and x_val > 0:
                x_values.append(x_val)

    return x_values


# Removed/refactored functions - all now use RolloutsReader methods directly:
# Functions moved to other modules:
# - plot_milestone_scatter -> plotting/scatter.py
# - plot_milestone_transition_boxplot -> plotting/boxplots.py
# - _format_years_value -> plotting_utils/helpers.py (format_years_value)
# - _get_year_tick_values_and_labels -> plotting_utils/helpers.py (get_year_tick_values_and_labels)
# - _simplify_milestone_name -> plotting_utils/helpers.py (simplify_milestone_name)
#
# Wrapper functions removed - use RolloutsReader methods directly:
# - _read_milestone_transition_durations -> reader.read_multiple_transition_durations(pairs, ...)
# - _read_milestone_scatter_data -> reader.read_transition_data(from, to, ...)
# - _read_horizon_trajectories -> reader.read_metric_trajectories("horizon_lengths")
# - _read_uplift_trajectories -> reader.read_metric_trajectories("ai_sw_progress_mult_ref_present_day", ...)
# - _read_metric_trajectories -> reader.read_metric_trajectories(metric_name)

def batch_plot_all(rollouts_file: Path, output_dir: Path) -> None:
    """Generate all standard plots for a batch rollout.

    Creates:
    - Milestone time histograms for key milestones
    - Milestone transition boxplot for key transitions
    - Overlaid PDF plot showing arrival distributions for key milestones
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load present_day from run directory for gamma kernel KDE
    from plotting_utils.helpers import load_present_day
    present_day = load_present_day(output_dir)
    print(f"Present day: {present_day}")

    # Create milestone_time_hists subfolder
    milestone_hist_dir = output_dir / "milestone_time_hists"
    milestone_hist_dir.mkdir(parents=True, exist_ok=True)

    # Create a single RolloutsReader instance to reuse (avoids reloading cache)
    print(f"Loading rollouts data from {rollouts_file}...")
    reader = RolloutsReader(rollouts_file)
    print("Rollouts data loaded.")

    # Milestone time histograms (both PDF and CDF style)
    milestones = [
        "AC",
        "AIR-5x",
        "AI2027-SC",
        "SAR-level-experiment-selection-skill",
        "AIR-25x",
        "SIAR-level-experiment-selection-skill",
        "AIR-250x"
    ]

    for milestone in milestones:
        # Create safe filename from milestone name
        safe_name = milestone.replace("(Expensive, threshold only considers taste) ", "").replace("-", "_").replace("x", "x")

        # Reuse the reader instance
        times, num_not_achieved, sim_end = reader.read_milestone_times(milestone)

        if not times and num_not_achieved == 0:
            print(f"Warning: No data found for milestone '{milestone}', skipping histograms")
            continue

        # PDF-style histogram - 2050 cutoff version
        out_path_pdf = milestone_hist_dir / f"milestone_time_hist_{safe_name}.png"
        plot_milestone_time_histogram(
            times,
            num_not_achieved,
            out_path_pdf,
            bins=50,
            title=f"Distribution of {milestone} Times",
            sim_end=sim_end,
            max_year=2050,
            present_day=present_day
        )
        print(f"Saved {out_path_pdf}")

        # PDF-style histogram - full range version
        out_path_pdf_full = milestone_hist_dir / f"milestone_time_hist_{safe_name}_full.png"
        plot_milestone_time_histogram(
            times,
            num_not_achieved,
            out_path_pdf_full,
            bins=50,
            title=f"Distribution of {milestone} Times (Full Range)",
            sim_end=sim_end,
            max_year=None,
            present_day=present_day
        )
        print(f"Saved {out_path_pdf_full}")

        # CDF-style histogram - 2050 cutoff version
        out_path_cdf = milestone_hist_dir / f"milestone_time_hist_{safe_name}_cdf.png"
        plot_milestone_time_histogram_cdf(
            times,
            num_not_achieved,
            out_path_cdf,
            bins=50,
            title=f"Cumulative Distribution of {milestone} Times",
            sim_end=sim_end,
            max_year=2050
        )
        print(f"Saved {out_path_cdf}")

        # CDF-style histogram - full range version
        out_path_cdf_full = milestone_hist_dir / f"milestone_time_hist_{safe_name}_cdf_full.png"
        plot_milestone_time_histogram_cdf(
            times,
            num_not_achieved,
            out_path_cdf_full,
            bins=50,
            title=f"Cumulative Distribution of {milestone} Times (Full Range)",
            sim_end=sim_end,
            max_year=None
        )
        print(f"Saved {out_path_cdf_full}")

        # Effective compute histogram - save in milestone_time_hists subfolder
        compute_ooms, num_compute_not_achieved = reader.read_milestone_compute(milestone)
        if compute_ooms or num_compute_not_achieved > 0:
            out_path_compute = milestone_hist_dir / f"milestone_compute_hist_{safe_name}.png"
            plot_milestone_effective_compute_histogram(
                compute_ooms,
                num_compute_not_achieved,
                out_path_compute,
                bins=50,
                title=f"Distribution of Effective Compute at {milestone}",
                milestone_label=milestone
            )
            print(f"Saved {out_path_compute}")

    # Milestone transition boxplot
    pairs_str = "AC:SAR-level-experiment-selection-skill,SAR-level-experiment-selection-skill:SIAR-level-experiment-selection-skill,SIAR-level-experiment-selection-skill:STRAT-AI,STRAT-AI:TED-AI,TED-AI:ASI"
    pairs = _parse_milestone_pairs(pairs_str)
    out_path = output_dir / "milestone_transition_box.png"

    # Reuse the reader instance
    labels, durations, durations_censored, num_b_not_achieved, num_b_before_a, total_per_pair, typical_max, simulation_cutoff = reader.read_multiple_transition_durations(
        pairs,
        filter_milestone=None,
        filter_by_year=None
    )

    if labels:
        plot_milestone_transition_boxplot(
            labels,
            durations,
            durations_censored,
            num_b_not_achieved,
            num_b_before_a,
            total_per_pair,
            out_path,
            ymin_years=None,
            ymax_years=None,
            exclude_inf_from_stats=False,
            inf_years_cap=typical_max,
            inf_years_display=100.0,
            title="Milestone Transition Durations",
            simulation_cutoff=simulation_cutoff
        )
        print(f"Saved {out_path}")
    else:
        print("Warning: No transition data found, skipping boxplot")

    # Overlaid PDF plot for key milestones
    overlay_milestones = [
        "AC",
        "AI2027-SC",
        "SAR-level-experiment-selection-skill",
        "SIAR-level-experiment-selection-skill"
    ]
    out_path = output_dir / "milestone_pdfs_overlay.png"

    # Preload milestone data using the existing reader to avoid reloading cache
    milestone_batch = _load_milestone_batch(reader, overlay_milestones)

    plot_milestone_pdfs_overlay(
        rollouts_file,
        overlay_milestones,
        out_path,
        title="Milestone Arrival Time Distributions",
        milestone_cache=milestone_batch  # Pass preloaded data
    )
    print(f"Saved {out_path}")

    # Post-SAR milestone PDFs (SIAR, TED-AI, ASI)
    print("\nGenerating post-SAR milestone PDFs (SIAR, TED-AI, ASI)...")
    try:
        import subprocess
        import sys
        post_sar_milestones_script = Path(__file__).parent / "plot_post_sar_milestone_pdfs.py"
        subprocess.run(
            [sys.executable, str(post_sar_milestones_script), "--run-dir", str(output_dir)],
            check=True
        )
    except Exception as e:
        print(f"Warning: Could not generate post-SAR milestone PDFs: {e}")

    # Horizon trajectory plots with MSE filtering
    print("\nGenerating horizon trajectory plots with MSE filtering...")
    mse_thresholds = [10.0, 1.5, 1.0, 0.7, 0.5]

    # Create subfolder for horizon trajectories
    horizon_dir = output_dir / "horizon_trajectories"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Read horizon data once (reuse the reader instance)
        times, trajectories, aa_times, mse_values = reader.read_metric_trajectories("horizon_lengths")

        # First generate a plot with all trajectories and MSE shading (no filtering)
        out_path = horizon_dir / "horizon_trajectories_all_shaded.png"
        try:
            plot_horizon_trajectories(
                times,
                trajectories,
                out_path,
                max_trajectories=1000,
                stop_at_sc=False,
                aa_times=aa_times,
                mse_values=mse_values,
                shade_by_mse=True,
                mse_threshold=None,  # No filtering, just shading
                cutoff_year=2040.0
            )
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Warning: Could not generate all trajectories with shading: {e}")

        for mse_threshold in mse_thresholds:
            out_path = horizon_dir / f"horizon_trajectories_mse_{mse_threshold:.1f}.png"
            try:
                plot_horizon_trajectories(
                    times,
                    trajectories,
                    out_path,
                    max_trajectories=1000,
                    stop_at_sc=False,
                    aa_times=aa_times,
                    mse_values=mse_values,
                    shade_by_mse=True,
                    mse_threshold=mse_threshold,
                    cutoff_year=2040.0
                )
                print(f"Saved {out_path}")
            except Exception as e:
                print(f"Warning: Could not generate horizon trajectories for MSE <= {mse_threshold}: {e}")
    except Exception as e:
        print(f"Warning: Could not load horizon trajectories: {e}")

    # Uplift trajectory plots
    print("\nGenerating AI software R&D uplift trajectory plots...")

    # Create subfolder for uplift trajectories
    uplift_dir = output_dir / "uplift_trajectories"
    uplift_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Read uplift data (all trajectories)
        reader = RolloutsReader(rollouts_file)
        times, trajectories, aa_times, _ = reader.read_metric_trajectories(
            "ai_sw_progress_mult_ref_present_day",
            include_mse=False
        )

        # Generate basic uplift trajectory plot
        out_path = uplift_dir / "uplift_trajectories.png"
        try:
            plot_uplift_trajectories(
                times,
                trajectories,
                out_path,
                max_trajectories=1000,
                stop_at_sc=False,
                aa_times=aa_times,
            )
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Warning: Could not generate uplift trajectories: {e}")

        # Generate variant with 2050 cutoff
        out_path = uplift_dir / "uplift_trajectories_2050.png"
        try:
            plot_uplift_trajectories(
                times,
                trajectories,
                out_path,
                max_trajectories=1000,
                stop_at_sc=False,
                aa_times=aa_times,
                max_year=2050,
            )
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Warning: Could not generate 2050 uplift trajectories: {e}")
    except Exception as e:
        print(f"Warning: Could not load uplift trajectories: {e}")

    # Uplift trajectories filtered for SAR in 2027
    try:
        # Read uplift data filtered for SAR-level-experiment-selection-skill in 2027
        reader = RolloutsReader(rollouts_file)
        times, trajectories, aa_times, _ = reader.read_metric_trajectories(
            "ai_sw_progress_mult_ref_present_day",
            include_mse=False,
            filter_milestone="SAR-level-experiment-selection-skill",
            filter_year=2027.0
        )

        # Generate variant with 2050 cutoff and SAR filter
        out_path = uplift_dir / "uplift_trajectories_2050_sar2027.png"
        try:
            plot_uplift_trajectories(
                times,
                trajectories,
                out_path,
                max_trajectories=1000,
                stop_at_sc=False,
                aa_times=aa_times,
                max_year=2050,
                title="AI Software R&D Uplift Trajectories\n(Filtered for SAR-level in 2027)"
            )
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Warning: Could not generate SAR 2027 uplift trajectories: {e}")
    except Exception as e:
        print(f"Warning: Could not load SAR 2027 uplift trajectories: {e}")

    # Parameter sensitivity analysis for milestone transitions
    try:
        import sys
        sys.path.insert(0, str(rollouts_file.parent.parent / "scripts"))
        from sensitivity_analysis import analyze_milestone_transitions

        # Sensitivity analysis: AC to SAR-level
        print("Running parameter sensitivity analysis for AC to SAR-level-experiment-selection-skill (both achieved only)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("AC", "SAR-level-experiment-selection-skill"),
            include_censored=False
        )

        print("Running parameter sensitivity analysis for AC to SAR-level-experiment-selection-skill (including censored)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("AC", "SAR-level-experiment-selection-skill"),
            include_censored=True
        )

        # Sensitivity analysis: SAR-level to SIAR-level
        print("Running parameter sensitivity analysis for SAR-level-experiment-selection-skill to SIAR-level-experiment-selection-skill (both achieved only)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("SAR-level-experiment-selection-skill", "SIAR-level-experiment-selection-skill"),
            include_censored=False
        )

        print("Running parameter sensitivity analysis for SAR-level-experiment-selection-skill to SIAR-level-experiment-selection-skill (including censored)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("SAR-level-experiment-selection-skill", "SIAR-level-experiment-selection-skill"),
            include_censored=True
        )

        # Sensitivity analysis: AC to SIAR-level
        print("Running parameter sensitivity analysis for AC to SIAR-level-experiment-selection-skill (both achieved only)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("AC", "SIAR-level-experiment-selection-skill"),
            include_censored=False
        )

        print("Running parameter sensitivity analysis for AC to SIAR-level-experiment-selection-skill (including censored)...")
        analyze_milestone_transitions(
            rollouts_file,
            output_dir,
            transition_pair=("AC", "SIAR-level-experiment-selection-skill"),
            include_censored=True
        )

    except Exception as e:
        print(f"Warning: Could not run parameter sensitivity analysis: {e}")

    # X years in 1 year distribution
    out_path = output_dir / "x_years_in_1_year_hist.png"
    x_values = _read_x_years_in_1_year(rollouts_file)

    if len(x_values) > 0:
        plot_x_years_in_1_year_histogram(
            x_values,
            out_path,
            bins=50
        )
        # Print statistics
        arr = np.asarray(x_values, dtype=float)
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
        print(f"Saved {out_path}")
        print(f"  X years in 1 year - P10/Median/P90: {q10:.1f}x / {q50:.1f}x / {q90:.1f}x")
    else:
        print(f"Warning: No valid X values found, skipping X years in 1 year plot")

    # m/beta distribution
    out_path = output_dir / "m_over_beta_hist.png"
    m_over_beta_values, num_skipped = _read_m_over_beta_values(rollouts_file)

    if len(m_over_beta_values) > 0:
        plot_m_over_beta_histogram(
            m_over_beta_values,
            num_skipped,
            out_path,
            bins=50,
            title="Distribution of m/β"
        )
        # Print statistics
        arr = np.asarray(m_over_beta_values, dtype=float)
        pct_above_one = 100.0 * np.sum(arr > 1) / len(arr)
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
        print(f"Saved {out_path}")
        print(f"  m/β - Percentage > 1: {pct_above_one:.1f}% | P10/Median/P90: {q10:.2f} / {q50:.2f} / {q90:.2f}")
        if num_skipped > 0:
            print(f"  Note: {num_skipped} rollouts skipped due to missing r_software")
    else:
        print(f"Warning: No valid m/β values found, skipping m/β plot")
        if num_skipped > 0:
            print(f"  (All {num_skipped} rollouts were missing r_software - run a new Monte Carlo with updated code)")

    # Generate short timelines analysis (AC and SAR)
    print("\nGenerating short timelines analysis (AC and SAR)...")
    try:
        import sys
        import subprocess
        scripts_dir = rollouts_file.parent.parent / "scripts"
        python_exe = sys.executable
        subprocess.run(
            [python_exe, str(scripts_dir / "short_timelines_analysis.py"), "--rollouts", str(rollouts_file)],
            check=True
        )
        print("Short timelines analysis complete!")
    except Exception as e:
        print(f"Warning: Could not generate short timelines analysis: {e}")

    # Generate slack summary table for all MSE thresholds
    print("\nGenerating slack summary table for all MSE thresholds...")
    try:
        import sys
        sys.path.insert(0, str(rollouts_file.parent.parent / "scripts"))
        from slack_summary import compute_slack_summary, format_year
        import csv

        mse_thresholds_for_summary = [10.0, 1.5, 1.0, 0.7, 0.5]
        milestone = "AC"

        # Collect results
        summary_results = []
        for mse_threshold in mse_thresholds_for_summary:
            try:
                result = compute_slack_summary(rollouts_file, mse_threshold, milestone)
                summary_results.append(result)
            except Exception as e:
                print(f"Warning: Could not compute summary for MSE <= {mse_threshold}: {e}")

        if summary_results:
            # Write to CSV
            csv_path = output_dir / "slack_summary.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow([
                    "MSE Threshold",
                    "N Trajectories",
                    f"{milestone} Median",
                    f"{milestone} P10",
                    f"{milestone} P90",
                    "10y Horizon Median",
                    "10y Horizon P10",
                    "10y Horizon P90",
                    "100y Horizon Median",
                    "100y Horizon P10",
                    "100y Horizon P90"
                ])
                # Rows
                for result in summary_results:
                    writer.writerow([
                        f"<= {result['mse_threshold']}",
                        result['num_trajectories'],
                        format_year(result['milestone_median']) if result['milestone_median'] is not None else "N/A",
                        format_year(result['milestone_p10']) if result['milestone_p10'] is not None else "N/A",
                        format_year(result['milestone_p90']) if result['milestone_p90'] is not None else "N/A",
                        format_year(result['horizon_10y_median']) if result['horizon_10y_median'] is not None else "N/A",
                        format_year(result['horizon_10y_p10']) if result['horizon_10y_p10'] is not None else "N/A",
                        format_year(result['horizon_10y_p90']) if result['horizon_10y_p90'] is not None else "N/A",
                        format_year(result['horizon_100y_median']) if result['horizon_100y_median'] is not None else "N/A",
                        format_year(result['horizon_100y_p10']) if result['horizon_100y_p10'] is not None else "N/A",
                        format_year(result['horizon_100y_p90']) if result['horizon_100y_p90'] is not None else "N/A",
                    ])
            print(f"Saved slack summary table to {csv_path}")

            # Also print to console
            print("\n=== Slack Summary Table ===")
            for result in summary_results:
                print(f"\nMSE <= {result['mse_threshold']} ({result['num_trajectories']} trajectories):")
                if result['milestone_median'] is not None:
                    print(f"  {milestone}: {format_year(result['milestone_median'])} [{format_year(result['milestone_p10'])}, {format_year(result['milestone_p90'])}]")
                else:
                    print(f"  {milestone}: N/A")
                if result['horizon_10y_median'] is not None:
                    print(f"  10y horizons: {format_year(result['horizon_10y_median'])} [{format_year(result['horizon_10y_p10'])}, {format_year(result['horizon_10y_p90'])}]")
                else:
                    print(f"  10y horizons: N/A")
                if result['horizon_100y_median'] is not None:
                    print(f"  100y horizons: {format_year(result['horizon_100y_median'])} [{format_year(result['horizon_100y_p10'])}, {format_year(result['horizon_100y_p90'])}]")
                else:
                    print(f"  100y horizons: N/A")
    except Exception as e:
        print(f"Warning: Could not generate slack summary table: {e}")

    # Fast takeoff analysis
    try:
        print("\nGenerating fast takeoff analysis (SAR transitions)...")
        import subprocess
        import sys
        from pathlib import Path as PathLib
        python_exe = sys.executable
        # Get the repository root (where scripts/ is located)
        repo_root = PathLib(__file__).resolve().parent.parent
        scripts_dir = repo_root / "scripts"
        subprocess.run(
            [python_exe, str(scripts_dir / "fast_takeoff_analysis.py"), "--rollouts", str(rollouts_file)],
            cwd=str(repo_root),
            check=True
        )
        print("Fast takeoff analysis complete!")
    except Exception as e:
        print(f"Warning: Could not generate fast takeoff analysis: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tools for batch rollout results")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=str, default=None, help="Path to a single rollout run directory containing rollouts.jsonl")
    g.add_argument("--rollouts", type=str, default=None, help="Path directly to a rollouts.jsonl file")
    parser.add_argument("--out", type=str, default=None, help="Output image path (PNG). Defaults vary by --mode")
    parser.add_argument("--mode", type=str, choices=["sc_hist", "horizon_trajectories", "horizon_at_sc_hist", "milestone_time_hist", "milestone_compute_hist", "milestone_transition_box", "milestone_scatter", "correlations", "m_over_beta", "metric_trajectories"], default="sc_hist", help="Which plot to generate")
    parser.add_argument("--batch-all", action="store_true", help="Generate all standard plots (ignores --mode and --out)")
    parser.add_argument("--plot-correlations", action="store_true", help="Generate correlation scatter plots for all parameter pairs with non-zero correlation")
    # Histogram options
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins for sc_hist mode")
    parser.add_argument("--cdf-hist", action="store_true", help="Generate CDF-style histograms where bar heights show cumulative counts (rollouts reaching milestone BY that year)")
    # Milestone options
    parser.add_argument("--milestone", type=str, default=None, help="Milestone name for milestone_time_hist mode (e.g., 'AC')")
    parser.add_argument("--list-milestones", action="store_true", help="List milestone names found in the rollouts file and exit (works for milestone_time_hist and milestone_transition_box)")
    # Metric trajectories options
    parser.add_argument("--metric", type=str, default=None, help="Metric name for metric_trajectories mode (e.g., 'ai_coding_labor_mult_ref_present_day')")
    # Milestone transition boxplot options
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated milestone pairs for transition durations, formatted as FROM:TO (e.g., 'SC:SAR,SAR:SIAR,SIAR:ASI')")
    parser.add_argument("--ymin-years", type=float, default=None, help="Minimum y-axis (years, log scale) for transition boxplot")
    parser.add_argument("--ymax-years", type=float, default=None, help="Maximum y-axis (years, log scale) for transition boxplot")
    parser.add_argument("--exclude-inf-from-stats", action="store_true", help="Exclude 'not achieved' (treated as +inf) from the stats panel and the plot")
    parser.add_argument("--inf-years", type=float, default=100.0, help="Where to plot 'not achieved' as points on the y-axis (years)")
    parser.add_argument("--filter-milestone", type=str, default=None, help="Only include rollouts where this milestone was achieved by --filter-by-year (e.g., 'AIR-5x')")
    parser.add_argument("--filter-by-year", type=float, default=None, help="Decimal year cutoff for --filter-milestone (e.g., 2029.5)")
    # Milestone scatter/heatmap options
    parser.add_argument("--scatter-pair", type=str, default=None, help="Pair for scatter/heatmap FROM:TO (e.g., 'AIR-5x:AIR-2000x')")
    parser.add_argument("--scatter-kind", type=str, choices=["hex", "hist2d", "scatter"], default="hex", help="Density visualization type")
    parser.add_argument("--gridsize", type=int, default=50, help="Grid size for hex/hist2d density")
    parser.add_argument("--point-size", type=float, default=8.0, help="Point size for scatter overlay")
    parser.add_argument("--no-scatter-overlay", action="store_true", help="Disable scatter overlay when using hex/hist2d")
    # Horizon trajectories options
    parser.add_argument("--current-horizon-minutes", type=float, default=15.0, help="Horizontal reference line for current horizon in minutes")
    parser.add_argument("--alpha", type=float, default=0.08, help="Transparency for individual trajectories")
    parser.add_argument("--max-trajectories", type=int, default=2000, help="Maximum number of trajectories to draw")
    parser.add_argument("--no-metr", action="store_true", help="Disable overlay of METR p80 benchmark points")
    parser.add_argument("--stop-at-sc", action="store_true", help="Mask each trajectory after its own aa_time")
    parser.add_argument("--shade-by-mse", action="store_true", help="Color trajectories by their METR MSE (low MSE=green, high MSE=red)")
    parser.add_argument("--mse-threshold", type=float, default=None, help="Exclude trajectories with METR MSE above this threshold")
    parser.add_argument("--cutoff-year", type=float, default=None, help="Cut off all trajectories at this year")
    parser.add_argument("--yscale", type=str, choices=["auto", "log", "linear"], default="auto", help="Y-axis scale: auto (default), log, or linear")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Handle correlations mode or flag (doesn't need rollouts_path)
    if args.mode == "correlations" or args.plot_correlations:
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            # If --rollouts was provided, use its parent directory
            rollouts_path = Path(args.rollouts)
            run_dir = rollouts_path.parent
        
        plot_all_correlations(run_dir)
        return
    
    rollouts_path = _resolve_rollouts_path(args.run_dir, args.rollouts)

    default_dir = rollouts_path.parent

    # Handle batch-all mode
    if args.batch_all:
        print(f"Generating all standard plots from {rollouts_path}")
        batch_plot_all(rollouts_path, default_dir)
        print("Batch plotting complete.")
        return

    if args.out is not None:
        out_path = Path(args.out)
    else:
        if args.mode == "sc_hist":
            out_path = default_dir / ("aa_time_hist_cdf.png" if args.cdf_hist else "aa_time_hist.png")
        elif args.mode == "horizon_at_sc_hist":
            out_path = default_dir / "horizon_at_sc_hist.png"
        elif args.mode == "milestone_time_hist":
            name = args.milestone or "milestone"
            safe = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in name])
            suffix = "_cdf" if args.cdf_hist else ""
            out_path = default_dir / f"milestone_{safe}_hist{suffix}.png"
        elif args.mode == "milestone_compute_hist":
            name = args.milestone or "milestone"
            safe = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in name])
            out_path = default_dir / f"milestone_{safe}_compute_hist.png"
        elif args.mode == "milestone_scatter":
            out_path = default_dir / "milestone_scatter.png"
        elif args.mode == "m_over_beta":
            out_path = default_dir / "m_over_beta_hist.png"
        elif args.mode == "metric_trajectories":
            metric_name = args.metric or "metric"
            safe = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in metric_name])
            suffix = "_stop_at_sc" if args.stop_at_sc else ""
            out_path = default_dir / f"{safe}_trajectories{suffix}.png"
        else:
            if args.mode == "milestone_transition_box":
                out_path = default_dir / "milestone_transition_box.png"
            else:
                # Create horizon_trajectories subfolder for horizon plots
                horizon_dir = default_dir / "horizon_trajectories"
                horizon_dir.mkdir(parents=True, exist_ok=True)
                out_path = horizon_dir / ("horizon_trajectories_stop_at_sc.png" if args.stop_at_sc else "horizon_trajectories.png")

    if args.mode == "sc_hist":
        reader = RolloutsReader(rollouts_path)
        aa_times, num_no_sc, sim_end = reader.read_aa_times()
        if (len(aa_times) + num_no_sc) > 0 and len(aa_times) > 0:
            arr = np.asarray(aa_times, dtype=float)
            print(f"Loaded {len(arr)} finite AC times (+{num_no_sc} No AC) from {rollouts_path}")
            print(f"Finite Min/Median/Max: {arr.min():.3f} / {np.median(arr):.3f} / {arr.max():.3f}")
            with_inf = np.concatenate([arr, np.full(int(num_no_sc), np.inf)]) if num_no_sc > 0 else arr
            q10, q50, q90 = np.quantile(with_inf, [0.1, 0.5, 0.9])
            def _fmt_q(qv: float) -> str:
                return f"{qv:.3f}" if np.isfinite(qv) else "No AC"
            print(f"P10/Median/P90 (incl. No AC): {_fmt_q(q10)} / {_fmt_q(q50)} / {_fmt_q(q90)}")
        elif (len(aa_times) + num_no_sc) > 0:
            print(f"Loaded 0 finite AC times (+{num_no_sc} No AC) from {rollouts_path}")

        if args.cdf_hist:
            plot_aa_time_histogram_cdf(aa_times, num_no_sc=num_no_sc, out_path=out_path, bins=int(args.bins), sim_end=sim_end)
        else:
            plot_aa_time_histogram(aa_times, num_no_sc=num_no_sc, out_path=out_path, bins=int(args.bins), sim_end=sim_end)
        print(f"Saved histogram to: {out_path}")
        return

    if args.mode == "horizon_at_sc_hist":
        reader = RolloutsReader(rollouts_path)
        # Read horizon lengths at AC time with appropriate clipping
        cap = float(120000 * 52 * 40 * 60)  # plotting cap to avoid absurd tails
        values = reader.read_metric_at_milestone("horizon_lengths", "aa_time", clip_min=0.001, clip_max=cap)
        if len(values) > 0:
            arr = np.asarray(values, dtype=float)
            q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
            print(f"Loaded {len(arr)} horizon_at_sc values from {rollouts_path}")
            print(f"P10/Median/P90: {format_time_duration(q10)} / {format_time_duration(q50)} / {format_time_duration(q90)}")
        plot_horizon_at_sc_histogram(values, out_path=out_path, bins=int(args.bins))
        print(f"Saved histogram to: {out_path}")
        return

    if args.mode == "milestone_time_hist":
        if args.list_milestones:
            names = _list_milestone_names(rollouts_path)
            if names:
                print("Milestones found:")
                for n in names:
                    print(f" - {n}")
            else:
                print("No milestones found in file.")
            return
        if not args.milestone:
            raise ValueError("--milestone is required for milestone_time_hist mode (use --list-milestones to inspect names)")

        # Load present_day for gamma kernel KDE
        from plotting_utils.helpers import load_present_day
        present_day = load_present_day(default_dir)

        reader = RolloutsReader(rollouts_path)
        times, num_na, sim_end = reader.read_milestone_times(args.milestone)
        if (len(times) + num_na) > 0 and len(times) > 0:
            arr = np.asarray(times, dtype=float)
            print(f"Loaded {len(arr)} finite '{args.milestone}' arrival times (+{num_na} Not achieved) from {rollouts_path}")
            print(f"Finite Min/Median/Max: {arr.min():.3f} / {np.median(arr):.3f} / {arr.max():.3f}")
            with_inf = np.concatenate([arr, np.full(int(num_na), np.inf)]) if num_na > 0 else arr
            q10, q50, q90 = np.quantile(with_inf, [0.1, 0.5, 0.9])
            def _fmt_q(qv: float) -> str:
                return f"{qv:.3f}" if np.isfinite(qv) else "Not achieved"
            print(f"P10/Median/P90 (incl. Not achieved): {_fmt_q(q10)} / {_fmt_q(q50)} / {_fmt_q(q90)}")
        elif (len(times) + num_na) > 0:
            print(f"Loaded 0 finite '{args.milestone}' arrival times (+{num_na} Not achieved) from {rollouts_path}")

        if args.cdf_hist:
            plot_milestone_time_histogram_cdf(times, num_not_achieved=num_na, out_path=out_path, bins=int(args.bins), milestone_label=args.milestone, sim_end=sim_end)
        else:
            plot_milestone_time_histogram(times, num_not_achieved=num_na, out_path=out_path, bins=int(args.bins), milestone_label=args.milestone, sim_end=sim_end, present_day=present_day)
        print(f"Saved histogram to: {out_path}")
        return

    if args.mode == "milestone_compute_hist":
        if args.list_milestones:
            names = _list_milestone_names(rollouts_path)
            if names:
                print("Milestones found:")
                for n in names:
                    print(f" - {n}")
            else:
                print("No milestones found in file.")
            return
        if not args.milestone:
            raise ValueError("--milestone is required for milestone_compute_hist mode (use --list-milestones to inspect names)")
        reader = RolloutsReader(rollouts_path)
        compute_ooms, num_na = reader.read_milestone_compute(args.milestone)
        if (len(compute_ooms) + num_na) > 0 and len(compute_ooms) > 0:
            arr = np.asarray(compute_ooms, dtype=float)
            print(f"Loaded {len(arr)} effective compute values for '{args.milestone}' (+{num_na} Not achieved) from {rollouts_path}")
            print(f"Min/Median/Max: {arr.min():.2f} / {np.median(arr):.2f} / {arr.max():.2f} log(2025-FLOP)")
            q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
            print(f"P10/Median/P90: {q10:.2f} / {q50:.2f} / {q90:.2f} log(2025-FLOP)")
        elif (len(compute_ooms) + num_na) > 0:
            print(f"Loaded 0 effective compute values for '{args.milestone}' (+{num_na} Not achieved) from {rollouts_path}")

        plot_milestone_effective_compute_histogram(compute_ooms, num_not_achieved=num_na, out_path=out_path, bins=int(args.bins), milestone_label=args.milestone)
        print(f"Saved histogram to: {out_path}")
        return

    if args.mode == "milestone_transition_box":
        if args.list_milestones:
            names = _list_milestone_names(rollouts_path)
            if names:
                print("Milestones found:")
                for n in names:
                    print(f" - {n}")
            else:
                print("No milestones found in file.")
            return
        pairs = _parse_milestone_pairs(args.pairs)
        if not pairs:
            raise ValueError("--pairs is required for milestone_transition_box mode (use --list-milestones to inspect names)")
        reader = RolloutsReader(rollouts_path)
        labels, durations, durations_censored, num_b_not_achieved, num_b_before_a, total_per_pair, typical_max, simulation_cutoff = reader.read_multiple_transition_durations(
            pairs,
            filter_milestone=(args.filter_milestone if args.filter_milestone else None),
            filter_by_year=(float(args.filter_by_year) if args.filter_by_year is not None else None),
        )
        for lbl, arr, n_not_achieved, n_before, total_a in zip(labels, durations, num_b_not_achieved, num_b_before_a, total_per_pair):
            arr_np = np.asarray(arr, dtype=float)
            if arr_np.size:
                q10, q50, q90 = np.quantile(arr_np, [0.1, 0.5, 0.9])
                print(f"{lbl}: n={arr_np.size} achieved in order | P10/Median/P90 (years): {q10:.3f} / {q50:.3f} / {q90:.3f}")
                if n_not_achieved > 0:
                    milestone_b = lbl.split(" to ")[1] if " to " in lbl else "B"
                    print(f"  +{n_not_achieved}/{total_a} {milestone_b} not achieved")
                if n_before > 0:
                    print(f"  +{n_before}/{total_a} out of order")
            else:
                print(f"{lbl}: n=0 achieved in order (total where A achieved: {total_a})")
        plot_milestone_transition_boxplot(
            labels,
            durations,
            durations_censored,
            num_b_not_achieved,
            num_b_before_a,
            total_per_pair,
            out_path=out_path,
            title=None,
            ymin_years=(float(args.ymin_years) if args.ymin_years is not None else None),
            ymax_years=(float(args.ymax_years) if args.ymax_years is not None else None),
            exclude_inf_from_stats=bool(args.exclude_inf_from_stats),
            inf_years_cap=typical_max,
            inf_years_display=float(args.inf_years),
            condition_text=(f"{args.filter_milestone} achieved by {args.filter_by_year}" if args.filter_milestone and args.filter_by_year is not None else None),
            simulation_cutoff=simulation_cutoff,
        )
        print(f"Saved transition boxplot to: {out_path}")
        return

    if args.mode == "milestone_scatter":
        if not args.scatter_pair:
            raise ValueError("--scatter-pair is required for milestone_scatter mode (e.g., 'AIR-5x:AIR-2000x')")
        if ":" not in args.scatter_pair:
            raise ValueError("--scatter-pair must be formatted as FROM:TO")
        from_name, to_name = [s.strip() for s in args.scatter_pair.split(":", 1)]
        reader = RolloutsReader(rollouts_path)
        xs, ys = reader.read_transition_data(
            from_name,
            to_name,
            include_censored=(not args.exclude_inf_from_stats),
            inf_years_cap=float(args.inf_years) if not args.exclude_inf_from_stats else None,
            return_arrays=True
        )
        print(f"Loaded {xs.size} points for scatter {from_name} -> {to_name} from {rollouts_path}")
        plot_milestone_scatter(
            xs,
            ys,
            out_path=out_path,
            kind=str(args.scatter_kind),
            gridsize=int(args.gridsize),
            point_size=float(args.point_size),
            scatter_overlay=(not args.no_scatter_overlay),
            ymin_years=(float(args.ymin_years) if args.ymin_years is not None else None),
            ymax_years=(float(args.ymax_years) if args.ymax_years is not None else None),
            condition_text=None,
        )
        print(f"Saved milestone scatter to: {out_path}")
        return

    if args.mode == "m_over_beta":
        m_over_beta_values, num_skipped = _read_m_over_beta_values(rollouts_path)
        if len(m_over_beta_values) > 0:
            plot_m_over_beta_histogram(
                m_over_beta_values,
                num_skipped,
                out_path,
                bins=args.bins,
                title="Distribution of m/β"
            )
            # Print statistics
            arr = np.asarray(m_over_beta_values, dtype=float)
            pct_above_one = 100.0 * np.sum(arr > 1) / len(arr)
            q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
            print(f"Saved m/β histogram to: {out_path}")
            print(f"  Percentage m/β > 1: {pct_above_one:.1f}%")
            print(f"  P10/Median/P90: {q10:.2f} / {q50:.2f} / {q90:.2f}")
            if num_skipped > 0:
                print(f"  Note: {num_skipped} rollouts skipped due to missing data")
        else:
            print(f"Warning: No valid m/β values found")
            if num_skipped > 0:
                print(f"  (All {num_skipped} rollouts were missing required data)")
        return

    if args.mode == "metric_trajectories":
        if not args.metric:
            raise ValueError("--metric is required for metric_trajectories mode")
        reader = RolloutsReader(rollouts_path)
        times, trajectories, aa_times, mse_values = reader.read_metric_trajectories(args.metric)
        print(f"Loaded {len(trajectories)} {args.metric} trajectories with {len(times)} time points each from {rollouts_path}")
        plot_metric_trajectories(
            times,
            trajectories,
            out_path=out_path,
            metric_name=args.metric,
            alpha=float(args.alpha),
            max_trajectories=int(args.max_trajectories),
            stop_at_sc=bool(args.stop_at_sc),
            aa_times=aa_times,
            mse_values=mse_values,
            shade_by_mse=bool(args.shade_by_mse),
            mse_threshold=args.mse_threshold,
            cutoff_year=args.cutoff_year,
            yscale=args.yscale,
        )
        print(f"Saved {args.metric} trajectories to: {out_path}")
        return

    # horizon_trajectories mode
    reader = RolloutsReader(rollouts_path)
    times, trajectories, aa_times, mse_values = reader.read_metric_trajectories("horizon_lengths")
    print(f"Loaded {len(trajectories)} trajectories with {len(times)} time points each from {rollouts_path}")
    plot_horizon_trajectories(
        times,
        trajectories,
        out_path=out_path,
        current_horizon_minutes=float(args.current_horizon_minutes),
        alpha=float(args.alpha),
        max_trajectories=int(args.max_trajectories),
        overlay_metr=(not args.no_metr),
        stop_at_sc=bool(args.stop_at_sc),
        aa_times=aa_times,
        mse_values=mse_values,
        shade_by_mse=bool(args.shade_by_mse),
        mse_threshold=args.mse_threshold,
        cutoff_year=args.cutoff_year,
    )
    print(f"Saved horizon trajectories to: {out_path}")


if __name__ == "__main__":
    main()
