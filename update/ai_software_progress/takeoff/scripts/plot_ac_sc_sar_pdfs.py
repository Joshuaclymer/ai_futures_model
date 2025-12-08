#!/usr/bin/env python3
"""
Plot PDFs (probability density functions) for AC, SC, and SAR milestone arrival times.

This script reads rollout results and creates a combined plot showing the
probability density functions for AC, SC (AI2027-SC), and SAR milestone times.

Usage examples:
  # Plot for a specific run directory
  python scripts/plot_ac_sc_sar_pdfs.py --run-dir outputs/251110_eli_2200

  # Plot with custom output path
  python scripts/plot_ac_sc_sar_pdfs.py --rollouts outputs/251110_eli_2200/rollouts.jsonl \
    --output custom_ac_sc_sar_pdfs.png
"""

import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np

# Use a non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use monospace font for all text elements
matplotlib.rcParams["font.family"] = "monospace"

# Import utilities
from plotting_utils.kde import make_gaussian_kde
from plotting_utils.rollouts_reader import RolloutsReader


def read_milestone_times(
    rollouts_file: Path,
    milestones: List[str]
) -> Dict[str, List[float]]:
    """Read arrival times for specified milestones.

    Args:
        rollouts_file: Path to rollouts.jsonl
        milestones: List of milestone names to extract

    Returns:
        Dictionary mapping milestone name to list of arrival times (years)
    """
    reader = RolloutsReader(rollouts_file)
    milestone_data = reader.read_multiple_milestone_times(milestones)

    # Print summary
    total_rollouts = reader.count_rollouts()
    print(f"\nTotal rollouts: {total_rollouts}")
    for milestone_name in milestones:
        count = len(milestone_data[milestone_name])
        pct = (count / total_rollouts * 100) if total_rollouts > 0 else 0
        print(f"  {milestone_name}: {count} ({pct:.1f}%)")

    return milestone_data


def plot_ac_sc_sar_pdfs(
    milestone_data: Dict[str, List[float]],
    out_path: Path,
    x_min: float = 2025,
    x_max: float = 2050,
) -> None:
    """Plot PDFs for AC, SC, and SAR milestones on the same plot.

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
        "AI2027-SC": "tab:orange",
        "SAR-level-experiment-selection-skill": "tab:green",
    }

    # Display names for milestones
    display_names = {
        "AC": "AC",
        "AI2027-SC": "SC",
        "SAR-level-experiment-selection-skill": "SAR",
    }

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each milestone
    for milestone_name, times in valid_milestones.items():
        data = np.asarray(times, dtype=float)
        color = colors.get(milestone_name, "tab:gray")
        display_name = display_names.get(milestone_name, milestone_name)

        try:
            # Create KDE
            kde = make_gaussian_kde(data)

            # Extend range beyond data to capture full KDE tails
            bw = kde.factor * data.std()
            xs = np.linspace(
                max(x_min, data.min() - 3 * bw),
                min(x_max, data.max() + 3 * bw),
                512
            )

            # Calculate PDF values
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
    ax.set_title("Milestone Arrival Time Distributions", fontsize=15, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(x_min, x_max)

    # Ensure y-axis starts at 0
    y_min, y_max_current = ax.get_ylim()
    ax.set_ylim(0, y_max_current)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved AC/SC/SAR PDFs plot to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PDFs for key milestone arrival times (AC, SC, SAR)",
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
        help="Output file path (default: <run-dir>/ac_sc_sar_pdfs.png)"
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
        "AI2027-SC",
        "SAR-level-experiment-selection-skill"
    ]

    # Read milestone times
    milestone_data = read_milestone_times(rollouts_path, milestones)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = run_dir / "ac_sc_sar_pdfs.png"

    # Generate plot
    plot_ac_sc_sar_pdfs(
        milestone_data,
        out_path,
        x_min=args.x_min,
        x_max=args.x_max
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
