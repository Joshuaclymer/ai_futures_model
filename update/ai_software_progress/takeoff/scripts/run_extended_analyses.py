#!/usr/bin/env python3
"""
Run extended analyses including:
1. Median splits for taste/exp selection parameters
2. Continuous plots showing how outcomes vary with parameters

Usage:
  python scripts/run_extended_analyses.py --run-dir outputs/251109_eli_2200
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

# Add scripts directory to path for imports
REPO_ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

from plot_rollouts import plot_milestone_pdfs_overlay


def split_by_median_and_plot(
    run_dir: Path,
    parameter_name: str,
    comparison_name: str,
    include_takeoff_analysis: bool = False
) -> float:
    """Split rollouts by median of a parameter and generate comparison plots."""
    print(f"\n{'='*60}")
    print(f"Splitting by median: {parameter_name}")
    print(f"{'='*60}")

    rollouts_file = run_dir / "rollouts.jsonl"

    # First pass: compute median
    values = []
    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                params = rec.get("parameters", {})
                ts_params = rec.get("time_series_parameters", {})
                results = rec.get("results", {})
                all_params = {**params, **ts_params, **results}

                value = all_params.get(parameter_name)
                if value is not None and isinstance(value, (int, float)) and np.isfinite(value):
                    values.append(float(value))
            except Exception:
                continue

    if not values:
        print(f"No valid values found for parameter: {parameter_name}")
        return np.nan

    median = float(np.median(values))
    print(f"Median {parameter_name}: {median:.4f}")

    # Create output directory
    safe_param_name = parameter_name.replace('_', '-')
    output_dir = run_dir / "parameter_splits" / f"{safe_param_name}_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split rollouts
    below_file = output_dir / f"rollouts_below_median.jsonl"
    above_file = output_dir / f"rollouts_above_median.jsonl"

    count_below = 0
    count_above = 0

    with rollouts_file.open("r", encoding="utf-8") as f_in, \
         below_file.open("w", encoding="utf-8") as f_below, \
         above_file.open("w", encoding="utf-8") as f_above:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                params = rec.get("parameters", {})
                ts_params = rec.get("time_series_parameters", {})
                results = rec.get("results", {})
                all_params = {**params, **ts_params, **results}

                value = all_params.get(parameter_name)
                if value is None:
                    continue

                if float(value) < median:
                    f_below.write(line + "\n")
                    count_below += 1
                else:
                    f_above.write(line + "\n")
                    count_above += 1
            except Exception:
                continue

    print(f"Split complete: {count_below} below median, {count_above} above median")

    # Generate milestone PDF plots
    milestone_names = ["AC", "SAR-level-experiment-selection-skill", "TED-AI", "ASI"]
    plot_milestone_pdfs_overlay(
        below_file,
        milestone_names,
        output_dir / "milestone_pdfs_below_median.png"
    )
    plot_milestone_pdfs_overlay(
        above_file,
        milestone_names,
        output_dir / "milestone_pdfs_above_median.png"
    )
    print(f"Saved milestone PDF plots to: {output_dir}")

    # Generate takeoff analysis if requested
    if include_takeoff_analysis:
        print(f"Generating takeoff analysis for {parameter_name} split...")

        # Create output directories
        below_outputs = output_dir / "below_median_fast_takeoff"
        above_outputs = output_dir / "above_median_fast_takeoff"
        below_outputs.mkdir(parents=True, exist_ok=True)
        above_outputs.mkdir(parents=True, exist_ok=True)

        # Below median
        print(f"\nAnalyzing below median ({parameter_name} < {median:.4f})...")
        try:
            subprocess.run([
                sys.executable,
                str(SCRIPTS_DIR / "fast_takeoff_analysis.py"),
                "--rollouts", str(below_file)
            ], check=True)

            # Move below median outputs
            src_below = output_dir / "fast_takeoff_outputs"
            if src_below.exists():
                import shutil
                for item in src_below.iterdir():
                    shutil.move(str(item), str(below_outputs / item.name))
                src_below.rmdir()
        except subprocess.CalledProcessError as e:
            print(f"Warning: Fast takeoff analysis failed for below median {parameter_name}: {e}")

        # Above median
        print(f"\nAnalyzing above median ({parameter_name} >= {median:.4f})...")
        try:
            subprocess.run([
                sys.executable,
                str(SCRIPTS_DIR / "fast_takeoff_analysis.py"),
                "--rollouts", str(above_file)
            ], check=True)

            # Move above median outputs
            src_above = output_dir / "fast_takeoff_outputs"
            if src_above.exists():
                import shutil
                for item in src_above.iterdir():
                    shutil.move(str(item), str(above_outputs / item.name))
                src_above.rmdir()
        except subprocess.CalledProcessError as e:
            print(f"Warning: Fast takeoff analysis failed for above median {parameter_name}: {e}")

        print(f"Saved takeoff analysis to: {below_outputs} and {above_outputs}")

    return median


def create_continuous_plots(
    run_dir: Path,
    parameter_name: str,
    outcome_metrics: List[Tuple[str, str]],
    bins: int = 20
) -> None:
    """Create continuous plots showing how outcomes vary with a parameter.

    Args:
        run_dir: Run directory
        parameter_name: Name of parameter to vary on x-axis
        outcome_metrics: List of (metric_name, display_name) tuples for y-axis
        bins: Number of bins for grouping parameter values
    """
    print(f"\n{'='*60}")
    print(f"Creating continuous plots for: {parameter_name}")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "monospace"

    rollouts_file = run_dir / "rollouts.jsonl"

    # Create output directory
    safe_param_name = parameter_name.replace('_', '-')
    output_dir = run_dir / "continuous_plots" / safe_param_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    param_values = []
    outcome_data = {metric: [] for metric, _ in outcome_metrics}

    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                params = rec.get("parameters", {})
                ts_params = rec.get("time_series_parameters", {})
                results = rec.get("results", {})
                all_params = {**params, **ts_params, **results}

                param_val = all_params.get(parameter_name)
                if param_val is None or not isinstance(param_val, (int, float)) or not np.isfinite(param_val):
                    continue

                param_values.append(float(param_val))

                # Extract outcome metrics
                for metric, _ in outcome_metrics:
                    if metric.startswith("p_1yr_takeoff_"):
                        # Calculate from transition durations
                        milestone_from = metric.split("_")[3]  # e.g., "AC" from "p_1yr_takeoff_AC"
                        milestone_to = metric.split("_to_")[1] if "_to_" in metric else "TED-AI"

                        milestones = results.get("milestones", {})
                        from_data = milestones.get(milestone_from)
                        to_data = milestones.get(milestone_to)

                        if from_data and to_data:
                            from_time = from_data.get("time")
                            to_time = to_data.get("time")
                            if from_time is not None and to_time is not None:
                                duration = float(to_time) - float(from_time)
                                outcome_data[metric].append(1.0 if duration <= 1.0 else 0.0)
                            else:
                                outcome_data[metric].append(np.nan)
                        else:
                            outcome_data[metric].append(np.nan)
                    else:
                        # Direct metric from results
                        val = all_params.get(metric)
                        if val is not None and isinstance(val, (int, float)):
                            outcome_data[metric].append(float(val))
                        else:
                            outcome_data[metric].append(np.nan)

            except Exception as e:
                continue

    if len(param_values) < 10:
        print(f"Insufficient data for {parameter_name}: only {len(param_values)} samples")
        return

    param_values = np.array(param_values)

    # Create plots for each outcome metric
    for metric, display_name in outcome_metrics:
        outcome_vals = np.array(outcome_data[metric])

        # Filter out NaN values
        mask = np.isfinite(param_values) & np.isfinite(outcome_vals)
        if mask.sum() < 10:
            print(f"Insufficient data for {metric}: only {mask.sum()} valid samples")
            continue

        x = param_values[mask]
        y = outcome_vals[mask]

        # Bin the parameter values
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(x, percentiles)
        bin_centers = []
        bin_means = []
        bin_stds = []
        bin_counts = []

        for i in range(len(bin_edges) - 1):
            if i == len(bin_edges) - 2:
                # Last bin includes right edge
                bin_mask = (x >= bin_edges[i]) & (x <= bin_edges[i + 1])
            else:
                bin_mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])

            if bin_mask.sum() > 0:
                bin_centers.append(float(np.mean(x[bin_mask])))
                bin_means.append(float(np.mean(y[bin_mask])))
                bin_stds.append(float(np.std(y[bin_mask])))
                bin_counts.append(int(bin_mask.sum()))

        if len(bin_centers) < 3:
            print(f"Insufficient bins for {metric}")
            continue

        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        bin_counts = np.array(bin_counts)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # Scatter plot with transparency
        ax.scatter(x, y, alpha=0.1, s=10, color='gray', label='Individual rollouts')

        # Line plot with error bars
        if metric.startswith("p_1yr_takeoff_"):
            # For probability metrics, convert to percentage
            ax.errorbar(bin_centers, bin_means * 100, yerr=bin_stds * 100 / np.sqrt(bin_counts),
                       fmt='o-', linewidth=2, markersize=8, capsize=5,
                       color='tab:blue', label='Mean ± SE per bin')
            ax.set_ylabel(f"{display_name} (%)", fontsize=12)
            ax.set_ylim(-5, 105)
        else:
            ax.errorbar(bin_centers, bin_means, yerr=bin_stds / np.sqrt(bin_counts),
                       fmt='o-', linewidth=2, markersize=8, capsize=5,
                       color='tab:blue', label='Mean ± SE per bin')
            ax.set_ylabel(display_name, fontsize=12)

        ax.set_xlabel(parameter_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f"{display_name} vs {parameter_name.replace('_', ' ').title()}",
                    fontsize=14, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Save plot
        safe_metric_name = metric.replace('_', '-')
        plot_path = output_dir / f"{safe_metric_name}_vs_{safe_param_name}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Run extended analyses")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print("\n" + "="*60)
    print("EXTENDED ANALYSES")
    print("="*60)

    # Part 1: Median splits with both timelines and takeoff plots
    print("\n### PART 1: Median Splits ###\n")

    # All parameters to analyze (all varying numeric parameters)
    parameters_to_split = [
        # Taste/experiment selection parameters
        ("ai_research_taste_at_coding_automation_anchor_sd", "Taste/Exp Selection at AC"),
        ("median_to_top_taste_multiplier", "Top Taste Multiplier"),
        ("ai_research_taste_slope", "Taste/Exp Selection Slope"),

        # Time horizon parameters
        ("ac_time_horizon_minutes", "AC Time Horizon"),
        ("pre_gap_ac_time_horizon", "Pre-Gap AC Time Horizon"),
        ("gap_years", "Gap Years"),

        # Doubling time parameters
        ("present_doubling_time", "Present Doubling Time"),
        ("doubling_difficulty_growth_factor", "Doubling Difficulty Growth Factor"),

        # Coding automation parameters
        ("coding_automation_efficiency_slope", "Coding Automation Efficiency Slope"),
        ("swe_multiplier_at_present_day", "SWE Multiplier at Present"),
        ("max_serial_coding_labor_multiplier", "Max Serial Coding Labor Multiplier"),

        # Experiment capacity parameters
        ("inf_compute_asymptote", "Inf Compute Asymptote"),
        ("inf_labor_asymptote", "Inf Labor Asymptote"),
        ("inv_compute_anchor_exp_cap", "Inv Compute Anchor Exp Cap"),
        ("parallel_penalty", "Parallel Penalty"),

        # Software/training compute parameters
        ("software_progress_rate_at_reference_year", "Software Progress Rate"),
        ("r_software", "R Software"),
        ("slowdown_year", "Slowdown Year"),
        ("constant_training_compute_growth_rate", "Constant Training Compute Growth"),
        ("post_slowdown_training_compute_growth_rate", "Post-Slowdown Training Compute Growth"),

        # Other
        ("rho_coding_labor", "Rho Coding Labor"),
    ]

    for param_name, display_name in parameters_to_split:
        split_by_median_and_plot(
            run_dir,
            param_name,
            display_name,
            include_takeoff_analysis=True
        )

    # Part 2: Continuous plots for all split parameters
    print("\n### PART 2: Continuous Plots ###\n")

    # Define outcome metrics
    outcome_metrics = [
        ("aa_time", "AC Arrival Time (years)"),
        ("p_1yr_takeoff_AC", "P(≤1yr AC→TED-AI takeoff)"),
        ("p_1yr_takeoff_AC_to_ASI", "P(≤1yr AC→ASI takeoff)"),
    ]

    # Parameters to analyze
    parameters = [
        "doubling_difficulty_growth_factor",
        "present_doubling_time",
        "ai_research_taste_at_coding_automation_anchor_sd",
        "median_to_top_taste_multiplier",
        "ai_research_taste_slope",
    ]

    for param in parameters:
        create_continuous_plots(run_dir, param, outcome_metrics, bins=20)

    print("\n" + "="*60)
    print("EXTENDED ANALYSES COMPLETE")
    print("="*60)
    print(f"\nOutputs organized in:")
    print(f"  - {run_dir}/parameter_splits/")
    print(f"  - {run_dir}/continuous_plots/")


if __name__ == "__main__":
    main()
