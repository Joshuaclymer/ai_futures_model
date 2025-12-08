#!/usr/bin/env python3
"""
Plot PDFs (probability density functions) for AC and SAR milestone arrival times.

This script reads rollout results and creates a combined plot showing the
probability density functions for AC and SAR milestone times.

Usage examples:
  # Plot for a specific run directory
  python scripts/plot_ac_sar_pdfs.py --run-dir outputs/251110_eli_2200

  # Plot with custom output path
  python scripts/plot_ac_sar_pdfs.py --rollouts outputs/251110_eli_2200/rollouts.jsonl \
    --output custom_ac_sar_pdfs.png

  # Plot with custom x-axis range
  python scripts/plot_ac_sar_pdfs.py --run-dir outputs/251110_eli_2200 \
    --x-min 2025 --x-max 2060
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

# Use a non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use monospace font for all text elements
matplotlib.rcParams["font.family"] = "monospace"

from plotting_utils.helpers import load_present_day
from plotting_utils.kde import (
    make_gamma_kernel_kde,
    make_lower_bounded_kde,
)
from plotting_utils.rollouts_reader import RolloutsReader


def _with_suffix(path: Path, suffix: str) -> Path:
    """Return path with suffix inserted before extension."""
    if not suffix:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.with_name(f"{path.name}{suffix}")


def load_milestone_times(
    rollouts_file: Path,
    milestones: List[str]
) -> Dict[str, List[float]]:
    """Load milestone arrival times via the shared RolloutsReader."""
    reader = RolloutsReader(rollouts_file)
    times_map, not_achieved_map, _, total_rollouts = reader.read_milestone_times_batch(milestones)
    milestone_data: Dict[str, List[float]] = {}
    summaries = []

    for milestone_name in milestones:
        times = times_map.get(milestone_name, [])
        num_not_achieved = not_achieved_map.get(milestone_name, 0)
        milestone_data[milestone_name] = times
        candidate_total = len(times) + num_not_achieved
        total_rollouts = max(total_rollouts, candidate_total)
        pct = (len(times) / candidate_total * 100) if candidate_total else 0.0
        summaries.append((milestone_name, len(times), pct))

    print(f"\nTotal rollouts: {total_rollouts}")
    for milestone_name, count, pct in summaries:
        print(f"  {milestone_name}: {count} ({pct:.1f}%)")

    return milestone_data


def plot_ac_sar_pdfs(
    milestone_data: Dict[str, List[float]],
    out_path: Path,
    x_min: float,
    x_max: float,
    *,
    filter_to_range: bool = False,
    present_day: float,
    kde_mode: str = "gamma",
) -> None:
    """Plot PDFs for AC and SAR milestones on the same plot.

    Args:
        milestone_data: Dictionary mapping milestone name to list of arrival times
        out_path: Output file path
        x_min: Minimum x-axis value (year)
        x_max: Maximum x-axis value (year)
    """
    # Filter out milestones with insufficient data
    valid_milestones = {
        name: times for name, times in milestone_data.items()
        if len(times) >= 2
    }

    if not valid_milestones:
        print("Error: Not enough data for any milestone to create KDE")
        return

    # Color scheme for milestones
    colors = {
        "AC": "tab:blue",
        "SAR-level-experiment-selection-skill": "tab:green",
    }

    # Display names for milestones
    display_names = {
        "AC": "AC",
        "SAR-level-experiment-selection-skill": "SAR",
    }

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each milestone
    eps = 1e-6
    support_start = max(x_min, present_day + eps)

    if support_start >= x_max:
        raise ValueError(
            f"x_max ({x_max}) must be greater than present_day-aware minimum ({support_start})."
        )

    for milestone_name, times in valid_milestones.items():
        data = np.asarray(times, dtype=float)
        pre_filter_size = data.size
        data = data[data >= present_day]
        dropped = pre_filter_size - data.size
        if dropped:
            print(
                f"  {milestone_name}: dropped {dropped} samples before present_day={present_day:.3f}."
            )
        if filter_to_range:
            window_mask = (data >= x_min) & (data <= x_max)
            data = data[window_mask]

        if data.size < 2:
            reason = "within plotting range" if filter_to_range else "available overall"
            print(f"Skipping {milestone_name}: fewer than 2 samples {reason}.")
            continue

        color = colors.get(milestone_name, "tab:gray")
        display_name = display_names.get(milestone_name, milestone_name)

        try:
            if kde_mode == "gamma":
                kde = make_gamma_kernel_kde(
                    data,
                    lower_bound=present_day,
                )
            else:
                kde = make_lower_bounded_kde(data, lower_bound=present_day)

            xs = np.linspace(
                support_start,
                x_max,
                1024,
            )

            pdf_values = kde(xs)

            # Plot PDF
            ax.plot(xs, pdf_values, linewidth=2.5, color=color, label=display_name)
            ax.fill_between(xs, pdf_values, alpha=0.2, color=color)

            # Calculate and display statistics
            q10, q50, q90 = np.quantile(data, [0.1, 0.5, 0.9])

            # Add subtle median line
            y_max = np.max(pdf_values)
            ax.axvline(q50, color=color, linestyle='--', linewidth=1.5, alpha=0.5)

            print(f"\n{display_name} statistics:")
            print(f"  P10: {q10:.1f}")
            print(f"  Median: {q50:.1f}")
            print(f"  P90: {q90:.1f}")
            print(f"  Mean: {np.mean(data):.1f}")

        except Exception as e:
            print(f"Error creating KDE for {milestone_name}: {e}")
            continue

    # Labels and title
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Probability Density", fontsize=13)
    ax.set_title("AC and SAR Arrival Time Distributions", fontsize=15, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(x_min, x_max)
    label = "Filtered to x-axis range" if filter_to_range else "Full sample"
    ax.text(
        0.02,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Ensure y-axis starts at 0
    y_min, y_max_current = ax.get_ylim()
    ax.set_ylim(0, y_max_current)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved AC/SAR PDFs plot to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PDFs for AC and SAR milestone arrival times",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--rollouts",
        type=str,
        help="Path to rollouts.jsonl file"
    )
    group.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory (will use <run-dir>/rollouts.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: <run-dir>/ac_sar_pdfs.png)"
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=2025,
        help="Minimum x-axis value (year) (default: 2025)"
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=2050,
        help="Maximum x-axis value (year) (default: 2050)"
    )
    parser.add_argument(
        "--filter-to-range",
        action="store_true",
        help="Only include milestone times within [x-min, x-max] when building KDEs.",
    )
    parser.add_argument(
        "--present-day",
        type=float,
        default=None,
        help=(
            "Override the present_day value used for lower-bounded KDE support. "
            "Defaults to the value stored in <run-dir>/model_config_snapshot.*"
        ),
    )
    parser.add_argument(
        "--kde-mode",
        choices=("lower-bounded", "gamma"),
        default="gamma",
        help=(
            "Select the KDE strategy: 'gamma' (positive-support gamma kernels) or "
            "'lower-bounded' (log transform with CV bandwidth)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine rollouts file path
    if args.rollouts:
        rollouts_path = Path(args.rollouts)
        run_dir = rollouts_path.parent
    else:
        run_dir = Path(args.run_dir)
        rollouts_path = run_dir / "rollouts.jsonl"

    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")

    # Milestones to plot
    milestones = [
        "AC",
        "SAR-level-experiment-selection-skill"
    ]

    present_day = (
        float(args.present_day)
        if args.present_day is not None
        else load_present_day(run_dir)
    )
    print(f"Using present_day = {present_day:.3f}")

    # Read milestone times
    milestone_data = load_milestone_times(rollouts_path, milestones)

    # Determine output path
    base_output = Path(args.output) if args.output else run_dir / "ac_sar_pdfs.png"
    out_path = _with_suffix(base_output, "_filtered") if args.filter_to_range else base_output

    # Generate plot
    plot_ac_sar_pdfs(
        milestone_data,
        out_path,
        x_min=args.x_min,
        x_max=args.x_max,
        filter_to_range=args.filter_to_range,
        present_day=present_day,
        kde_mode=args.kde_mode,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
