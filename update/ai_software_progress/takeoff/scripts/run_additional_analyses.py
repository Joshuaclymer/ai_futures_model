#!/usr/bin/env python3
"""
Run additional analyses requested:
(a) Spearman sensitivity for SAR arrival date
(b) Spearman for AC->SAR (already done)
(c) Spearman for SAR->TED-AI, SAR->ASI
(d) Plot AC timelines with doubling difficulty growth factor >1 vs. <1
(e) Plot AC timelines with vs. without gap
(f) Plot AC timelines with present doubling time below or above the median for that run
(g) Plot all of the takeoff outcomes in the corresponding folder for m/beta>1 vs. m/beta<1

Usage:
  python scripts/run_additional_analyses.py --run-dir outputs/251109_eli_2200
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Add scripts directory to path for imports
REPO_ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

from plot_rollouts import (
    plot_milestone_pdfs_overlay,
    plot_horizon_trajectories,
    _read_horizon_trajectories
)


def run_milestone_time_sensitivity(run_dir: Path, milestone_name: str) -> None:
    """Run sensitivity analysis for a milestone arrival time."""
    print(f"\n=== Running sensitivity analysis for {milestone_name} arrival time ===")

    rollouts_file = run_dir / "rollouts.jsonl"
    sensitivity_dir = run_dir / "sensitivity_analysis"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary script to extract milestone times and run sensitivity
    import json
    import numpy as np
    from scipy import stats
    from pathlib import Path

    # Read rollouts and extract milestone times
    milestone_times = []
    parameters_list = []

    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                results = rec.get("results")
                if not results or rec.get("error"):
                    continue

                milestones = results.get("milestones", {})
                milestone_data = milestones.get(milestone_name)
                if milestone_data and milestone_data.get("time") is not None:
                    t = float(milestone_data["time"])
                    if np.isfinite(t):
                        milestone_times.append(t)
                        # Merge parameters and time_series_parameters
                        params = rec.get("parameters", {})
                        ts_params = rec.get("time_series_parameters", {})
                        all_params = {**params, **ts_params}
                        parameters_list.append(all_params)
            except Exception as e:
                continue

    if len(milestone_times) < 10:
        print(f"Insufficient data for {milestone_name}: only {len(milestone_times)} samples")
        return

    y = np.array(milestone_times)

    # Get parameter names
    all_param_names = sorted({k for params in parameters_list for k in params.keys() if k != 'r_software'})

    # Compute Spearman correlations for numeric parameters
    correlations = []
    for param_name in all_param_names:
        param_values = []
        for params in parameters_list:
            v = params.get(param_name)
            if isinstance(v, (int, float)) and np.isfinite(v):
                param_values.append(float(v))
            else:
                param_values.append(np.nan)

        param_array = np.array(param_values)
        mask = np.isfinite(param_array) & np.isfinite(y)

        if mask.sum() >= 3:
            try:
                rho, p = stats.spearmanr(param_array[mask], y[mask])
                if np.isfinite(rho):
                    correlations.append((param_name, float(rho), float(p)))
            except Exception:
                pass

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Save results
    output_file = sensitivity_dir / f"sensitivity_{milestone_name.replace(' ', '_')}_arrival.json"
    result = {
        "milestone": milestone_name,
        "num_samples": len(milestone_times),
        "spearman_correlations": [
            {"parameter": name, "rho": rho, "p_value": p}
            for name, rho, p in correlations[:30]
        ]
    }

    with output_file.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved sensitivity analysis to: {output_file}")
    print(f"Top 10 parameters by Spearman correlation:")
    for name, rho, p in correlations[:10]:
        print(f"  {name:40s}  rho={rho:+.3f}  (p={p:.4f})")

    # Generate plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "monospace"

    if correlations:
        names, rhos, _ = zip(*correlations[:30])
        plt.figure(figsize=(13, max(4, len(names) * 0.35)))
        y_pos = np.arange(len(names))
        plt.barh(y_pos, rhos, align='center')
        plt.yticks(y_pos, names, fontsize=9)
        plt.xlabel(f'Spearman rho with {milestone_name} arrival time')
        plt.title(f'Parameter sensitivity for {milestone_name} arrival', pad=15, fontsize=10)
        plt.gca().invert_yaxis()
        plot_path = sensitivity_dir / f'sensitivity_spearman_{milestone_name.replace(" ", "_")}_arrival.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to: {plot_path}")


def run_transition_sensitivity(run_dir: Path, from_milestone: str, to_milestone: str) -> None:
    """Run sensitivity analysis for milestone transitions."""
    print(f"\n=== Running sensitivity for {from_milestone} -> {to_milestone} ===")

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "sensitivity_analysis.py"),
        "--run-dir", str(run_dir),
        "--transition-pair", f"{from_milestone}:{to_milestone}",
        "--plot"
    ]

    subprocess.run(cmd, check=True)


def split_and_plot_by_parameter(
    run_dir: Path,
    parameter_name: str,
    threshold: float,
    comparison_name: str,
    plot_type: str = "milestone_pdfs"
) -> None:
    """Split rollouts by a parameter threshold and generate comparison plots."""
    print(f"\n=== Generating plots for {comparison_name} ===")

    rollouts_file = run_dir / "rollouts.jsonl"
    output_dir = run_dir / f"{parameter_name.replace('_', '-')}_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split rollouts
    below_file = output_dir / f"rollouts_below_{threshold}.jsonl"
    above_file = output_dir / f"rollouts_above_{threshold}.jsonl"

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
                all_params = {**params, **ts_params}

                value = all_params.get(parameter_name)
                if value is None:
                    continue

                if float(value) < threshold:
                    f_below.write(line + "\n")
                    count_below += 1
                else:
                    f_above.write(line + "\n")
                    count_above += 1
            except Exception:
                continue

    print(f"Split complete: {count_below} below threshold, {count_above} above threshold")

    # Generate plots for each subset
    if plot_type == "milestone_pdfs":
        milestone_names = ["AC", "SAR-level-experiment-selection-skill", "TED-AI", "ASI"]
        plot_milestone_pdfs_overlay(
            below_file,
            milestone_names,
            output_dir / f"milestone_pdfs_below_{threshold}.png"
        )
        plot_milestone_pdfs_overlay(
            above_file,
            milestone_names,
            output_dir / f"milestone_pdfs_above_{threshold}.png"
        )
        print(f"Saved milestone PDF plots to: {output_dir}")


def split_by_median(run_dir: Path, parameter_name: str, comparison_name: str) -> None:
    """Split rollouts by median of a parameter and generate comparison plots."""
    print(f"\n=== Generating plots for {comparison_name} (split by median) ===")

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
                all_params = {**params, **ts_params}

                value = all_params.get(parameter_name)
                if value is not None and isinstance(value, (int, float)) and np.isfinite(value):
                    values.append(float(value))
            except Exception:
                continue

    if not values:
        print(f"No valid values found for parameter: {parameter_name}")
        return

    median = float(np.median(values))
    print(f"Median {parameter_name}: {median:.4f}")

    # Split by median
    split_and_plot_by_parameter(
        run_dir,
        parameter_name,
        median,
        comparison_name,
        plot_type="milestone_pdfs"
    )


def split_by_categorical(
    run_dir: Path,
    parameter_name: str,
    value1: str,
    value2: str,
    comparison_name: str
) -> None:
    """Split rollouts by categorical parameter values and generate comparison plots."""
    print(f"\n=== Generating plots for {comparison_name} ===")

    rollouts_file = run_dir / "rollouts.jsonl"
    output_dir = run_dir / f"{parameter_name.replace('_', '-')}_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split rollouts
    file1 = output_dir / f"rollouts_{value1.replace(' ', '_')}.jsonl"
    file2 = output_dir / f"rollouts_{value2.replace(' ', '_')}.jsonl"

    count1 = 0
    count2 = 0

    with rollouts_file.open("r", encoding="utf-8") as f_in, \
         file1.open("w", encoding="utf-8") as f1, \
         file2.open("w", encoding="utf-8") as f2:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                params = rec.get("parameters", {})
                ts_params = rec.get("time_series_parameters", {})
                all_params = {**params, **ts_params}

                value = all_params.get(parameter_name)
                if value == value1:
                    f1.write(line + "\n")
                    count1 += 1
                elif value == value2:
                    f2.write(line + "\n")
                    count2 += 1
            except Exception:
                continue

    print(f"Split complete: {count1} with {value1}, {count2} with {value2}")

    # Generate plots for each subset
    milestone_names = ["AC", "SAR-level-experiment-selection-skill", "TED-AI", "ASI"]
    plot_milestone_pdfs_overlay(
        file1,
        milestone_names,
        output_dir / f"milestone_pdfs_{value1.replace(' ', '_')}.png"
    )
    plot_milestone_pdfs_overlay(
        file2,
        milestone_names,
        output_dir / f"milestone_pdfs_{value2.replace(' ', '_')}.png"
    )
    print(f"Saved milestone PDF plots to: {output_dir}")


def plot_takeoff_by_m_over_beta(run_dir: Path) -> None:
    """Plot takeoff outcomes split by m/beta > 1 vs < 1."""
    print("\n=== Generating takeoff outcome plots by m/beta ===")

    rollouts_file = run_dir / "rollouts.jsonl"
    fast_takeoff_dir = run_dir / "fast_takeoff_outputs"
    output_dir = fast_takeoff_dir / "m_over_beta_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split rollouts by m/beta
    below_file = output_dir / "rollouts_m_over_beta_below_1.jsonl"
    above_file = output_dir / "rollouts_m_over_beta_above_1.jsonl"

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

                m = params.get("doubling_difficulty_growth_factor")
                beta = params.get("ai_research_taste_slope")

                if m is None or beta is None:
                    continue
                if beta == 0:
                    continue

                m_over_beta = float(m) / abs(float(beta))

                if m_over_beta < 1.0:
                    f_below.write(line + "\n")
                    count_below += 1
                else:
                    f_above.write(line + "\n")
                    count_above += 1
            except Exception:
                continue

    print(f"Split complete: {count_below} with m/beta < 1, {count_above} with m/beta >= 1")

    # Generate takeoff analysis plots for each subset
    from fast_takeoff_analysis import main as run_fast_takeoff_analysis

    # Run for m/beta < 1
    print("\nGenerating plots for m/beta < 1...")
    subprocess.run([
        sys.executable,
        str(SCRIPTS_DIR / "fast_takeoff_analysis.py"),
        "--rollouts", str(below_file),
        "--output-dir", str(output_dir / "m_over_beta_below_1")
    ])

    # Run for m/beta >= 1
    print("\nGenerating plots for m/beta >= 1...")
    subprocess.run([
        sys.executable,
        str(SCRIPTS_DIR / "fast_takeoff_analysis.py"),
        "--rollouts", str(above_file),
        "--output-dir", str(output_dir / "m_over_beta_above_1")
    ])

    print(f"Saved takeoff outcome plots to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run additional analyses")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # (a) Spearman sensitivity for SAR arrival date
    run_milestone_time_sensitivity(run_dir, "SAR-level-experiment-selection-skill")

    # (b) AC->SAR already done via sensitivity_analysis.py
    print("\n(b) AC->SAR Spearman already exists in sensitivity_analysis folder")

    # (c) Spearman for SAR->TED-AI, SAR->ASI
    run_transition_sensitivity(run_dir, "SAR-level-experiment-selection-skill", "TED-AI")
    run_transition_sensitivity(run_dir, "SAR-level-experiment-selection-skill", "ASI")

    # (d) Plot AC timelines with doubling difficulty growth factor >1 vs. <1
    split_and_plot_by_parameter(
        run_dir,
        "doubling_difficulty_growth_factor",
        1.0,
        "AC timelines by growth factor",
        plot_type="milestone_pdfs"
    )

    # (e) Plot AC timelines with vs. without gap
    split_by_categorical(
        run_dir,
        "include_gap",
        "gap",
        "no gap",
        "AC timelines by gap presence"
    )

    # (f) Plot AC timelines with present doubling time below or above median
    split_by_median(
        run_dir,
        "present_doubling_time",
        "AC timelines by present doubling time"
    )

    # (g) Plot all takeoff outcomes for m/beta>1 vs. m/beta<1
    plot_takeoff_by_m_over_beta(run_dir)

    print("\n=== All analyses complete! ===")


if __name__ == "__main__":
    main()
