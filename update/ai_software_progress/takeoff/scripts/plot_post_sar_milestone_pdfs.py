#!/usr/bin/env python3
"""
Generate PDF overlay plot for post-SAR milestones: SIAR, TED-AI, and ASI.

This script reads rollout results and creates a probability density function overlay plot
showing when SIAR, TED-AI, and ASI milestones are reached across all rollouts.

Usage:
  # Generate overlay plot for a specific run directory
  python scripts/plot_post_sar_milestone_pdfs.py --run-dir outputs/251110_eli_2200

  # Specify custom output path
  python scripts/plot_post_sar_milestone_pdfs.py --run-dir outputs/251110_eli_2200 \
    --out-path outputs/251110_eli_2200/custom_milestones.png
"""

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use monospace font for all text elements
matplotlib.rcParams["font.family"] = "monospace"

# Import utilities
from plotting_utils.kde import make_gamma_kernel_kde
from plotting_utils.rollouts_reader import RolloutsReader
from plotting_utils.helpers import load_present_day


def plot_post_sar_milestones_overlay(
    rollouts_file: Path,
    out_path: Path,
    max_year: float = 2050,
    present_day: Optional[float] = None,
) -> None:
    """Plot overlaid PDFs for post-SAR milestones: SIAR, TED-AI, and ASI.

    Args:
        rollouts_file: path to rollouts.jsonl
        out_path: output file path
        max_year: maximum year to plot (default: 2050)
        present_day: lower bound for the gamma KDE support (defaults to loading from rollouts_file's parent directory)
    """
    # Load present_day if not provided
    if present_day is None:
        present_day = load_present_day(Path(rollouts_file).parent)

    # Post-SAR milestones to plot
    milestone_names = [
        'SIAR-level-experiment-selection-skill',
        'TED-AI',
        'ASI'
    ]

    # Display names (cleaned up)
    display_names = {
        'SIAR-level-experiment-selection-skill': 'SIAR',
        'TED-AI': 'TED-AI',
        'ASI': 'ASI'
    }

    # Colors for each milestone
    colors = {
        'SIAR-level-experiment-selection-skill': 'tab:blue',
        'TED-AI': 'tab:orange',
        'ASI': 'tab:red'
    }

    plt.figure(figsize=(14, 8))

    stats_lines = []
    all_kde_data = {}  # milestone_name -> (xs, pdf_values)
    reader = RolloutsReader(rollouts_file)

    for milestone_name in milestone_names:
        times, num_not_achieved, sim_end = reader.read_milestone_times(milestone_name)

        if len(times) < 2:
            print(f"Warning: Not enough data for {milestone_name} to create KDE, skipping")
            continue

        raw_data = np.asarray(times, dtype=float)

        # Filter out times before present_day
        boundary = float(present_day)
        valid_mask = raw_data >= boundary
        data = raw_data[valid_mask]
        dropped = raw_data.size - data.size
        if dropped:
            print(
                f"  {milestone_name}: dropped {dropped} samples before present_day={boundary:.3f}."
            )
        if data.size < 2:
            print(f"Warning: Not enough data for {milestone_name} after enforcing present_day boundary, skipping.")
            continue

        # Calculate percentiles including not achieved as sim_end
        if num_not_achieved > 0 and sim_end is not None:
            combined_data = np.concatenate([raw_data, np.full(num_not_achieved, sim_end)])
        else:
            combined_data = raw_data

        q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

        # Calculate achievement probability
        total_runs = len(times) + num_not_achieved
        prob_achieved = len(times) / total_runs if total_runs > 0 else 0.0

        # Create KDE
        try:
            kde = make_gamma_kernel_kde(data, lower_bound=present_day)

            # Get bandwidth from gamma kernel KDE
            bw = float(kde.bandwidth)

            # Compute evaluation range respecting present_day lower bound
            min_eval = max(present_day, float(np.min(data) - 5.0 * bw))
            max_eval = max(float(np.max(data) + 5.0 * bw), min_eval + bw)

            # Use 512 evenly-spaced points
            xs = np.linspace(min_eval, max_eval, 512)

            # Scale PDF by probability of achievement and convert to percentage per year
            pdf_values = kde(xs) * prob_achieved * 100

            # Cut off plotting at max_year
            plot_mask = xs <= max_year
            xs_plot = xs[plot_mask]
            pdf_plot = pdf_values[plot_mask]

            # Plot the curve
            display_name = display_names[milestone_name]
            color = colors[milestone_name]
            plt.plot(xs_plot, pdf_plot, linewidth=2.5, label=display_name, color=color)

            # Find mode (peak of KDE)
            mode_idx = np.argmax(pdf_values)
            mode = xs[mode_idx]

            # Store KDE data
            all_kde_data[milestone_name] = (xs, pdf_values)

            # Calculate percentage after 2050
            num_after_2050 = sum(1 for t in data if t > 2050) + num_not_achieved
            pct_after_2050 = (num_after_2050 / total_runs * 100) if total_runs > 0 else 0

            stats_lines.append(
                f"{display_name}: Mode={mode:.1f}, P10={q10:.1f}, P50={q50:.1f}, P90={q90:.1f}, {pct_after_2050:.0f}% > 2050"
            )
        except Exception as e:
            print(f"Warning: Could not create KDE for {milestone_name}: {e}")
            continue

    plt.xlabel("Arrival Time (decimal year)", fontsize=12)
    plt.ylabel("Probability Density (% per year)", fontsize=12)
    plt.title("Post-SAR Milestone Arrival Time Distributions: SIAR, TED-AI, ASI", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=11)
    plt.xlim(right=max_year)

    # Add statistics text in top right
    if stats_lines:
        stats_text = "\n".join(stats_lines)
        plt.text(0.98, 0.98, stats_text,
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
                 family='monospace')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved post-SAR milestones PDF plot to: {out_path}")

    # Save CSV with combined distribution data
    if all_kde_data:
        csv_path = out_path.parent / (out_path.stem + "_distributions.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header: time_decimal_year, then one column per milestone
            header = ["time_decimal_year"] + [display_names[m] for m in milestone_names if m in all_kde_data]
            writer.writerow(header)

            # Find all unique time points and sort them
            all_times = set()
            for xs, _ in all_kde_data.values():
                all_times.update(xs)
            sorted_times = sorted(all_times)

            # For each time point, interpolate PDF value for each milestone
            for t in sorted_times:
                row = [f"{t:.6f}"]
                for milestone_name in milestone_names:
                    if milestone_name not in all_kde_data:
                        continue
                    xs, pdf_vals = all_kde_data[milestone_name]
                    # Interpolate
                    pdf_val = float(np.interp(t, xs, pdf_vals, left=0.0, right=0.0))
                    row.append(f"{pdf_val:.10f}")
                writer.writerow(row)

        print(f"Saved distribution data to: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PDF overlay plot for post-SAR milestones (SIAR, TED-AI, ASI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory containing rollouts.jsonl"
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Output path for plot (default: <run-dir>/post_sar_milestone_pdfs.png)"
    )
    parser.add_argument(
        "--max-year",
        type=float,
        default=2050,
        help="Maximum year to plot (default: 2050)"
    )
    parser.add_argument(
        "--parameter-split",
        action="store_true",
        help="Process parameter split directory (looks for above/below median files)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    rollouts_file = run_dir / "rollouts.jsonl"
    if not rollouts_file.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_file}")

    # Determine output path
    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = run_dir / "post_sar_milestone_pdfs.png"

    # Generate plot
    plot_post_sar_milestones_overlay(
        rollouts_file,
        out_path,
        max_year=args.max_year
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
