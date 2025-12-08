#!/usr/bin/env python3
"""
Generate PDF plots for milestone arrival times using gamma-kernel KDEs anchored at present_day.

This script reads rollout results and creates probability density function plots
showing when each milestone is reached across all rollouts. It can generate
either individual plots for each milestone or a combined overlay plot.

Additionally, it generates CSV files with:
- Summary statistics for all milestones
- KDE distribution data (x values and PDF values)

Usage examples:
  # Generate individual PDFs for all milestones
  python scripts/milestone_pdfs.py --rollouts outputs/20250813_020347/rollouts.jsonl

  # Generate overlay plot for all milestones
  python scripts/milestone_pdfs.py --rollouts outputs/20250813_020347/rollouts.jsonl --overlay

  # Generate overlay for specific milestones only
  python scripts/milestone_pdfs.py --rollouts outputs/20250813_020347/rollouts.jsonl \
    --overlay --milestones "AC,SAR-level-experiment-selection-skill"

  # Use monthly sampling (one point per month) for smaller CSV files
  python scripts/milestone_pdfs.py --rollouts outputs/20250813_020347/rollouts.jsonl \
    --overlay --monthly

  # Customize output directory
  python scripts/milestone_pdfs.py --rollouts outputs/20250813_020347/rollouts.jsonl \
    --out-dir outputs/20250813_020347/milestone_pdfs
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

# Use a non-interactive backend in case this runs on a headless server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use monospace font for all text elements
matplotlib.rcParams["font.family"] = "monospace"

# Import utilities
from plotting_utils.helpers import load_present_day
from plotting_utils.kde import make_gamma_kernel_kde
from plotting_utils.rollouts_reader import RolloutsReader


@dataclass(frozen=True)
class MilestoneBatch:
    times_map: Dict[str, List[float]]
    not_achieved_map: Dict[str, int]
    typical_sim_end: Optional[float]
    total_rollouts: int


OVERLAY_ALLOWED_MILESTONES = (
    "AC",
    "AI2027-SC",
    "SAR-level-experiment-selection-skill",
    "SIAR-level-experiment-selection-skill",
)


def _filter_overlay_milestones(milestone_names: Sequence[str]) -> List[str]:
    allowed = set(OVERLAY_ALLOWED_MILESTONES)
    return [name for name in milestone_names if name in allowed]


def _load_milestone_batch(
    reader: RolloutsReader,
    milestone_names: Sequence[str],
) -> MilestoneBatch:
    if not milestone_names:
        return MilestoneBatch({}, {}, None, 0)
    times_map, not_achieved_map, typical_sim_end, total_rollouts = reader.read_milestone_times_batch(milestone_names)
    return MilestoneBatch(times_map, not_achieved_map, typical_sim_end, total_rollouts)


def _list_milestone_names(rollouts_file: Path) -> List[str]:
    """List all unique milestone names found in the rollouts file."""
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


def save_kde_distribution_csv(
    xs: np.ndarray,
    pdf_values: np.ndarray,
    milestone_name: str,
    out_path: Path,
) -> None:
    """Save KDE distribution data to CSV.

    Args:
        xs: x values (time points)
        pdf_values: PDF values at each x
        milestone_name: name of the milestone
        out_path: output CSV file path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_decimal_year", "probability_density"])
        for x, pdf in zip(xs, pdf_values):
            writer.writerow([f"{x:.6f}", f"{pdf:.10f}"])
    
    print(f"Saved KDE distribution data to: {out_path}")


def save_milestone_stats_csv(
    stats: List[Dict[str, any]],
    out_path: Path,
) -> None:
    """Save summary statistics for all milestones to CSV.

    Args:
        stats: list of dictionaries containing milestone statistics
        out_path: output CSV file path
    """
    if not stats:
        return
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "milestone_name",
        "num_achieved",
        "num_not_achieved",
        "total_rollouts",
        "achievement_rate_pct",
        "mode",
        "p10",
        "p50",
        "p90",
    ]
    
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
    
    print(f"Saved milestone statistics to: {out_path}")


def _prepare_gamma_pdf(
    data: np.ndarray,
    *,
    present_day: float,
    monthly_sampling: bool,
    scale_factor: float,
    plot_max_year: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit a gamma-kernel KDE and sample it on a grid suitable for plotting and CSV export.

    Returns:
        Tuple of (xs_full, pdf_full, xs_plot, pdf_plot, mode_year)
    """
    if data.size < 2:
        raise ValueError("Need at least two samples to build KDE.")

    kde = make_gamma_kernel_kde(data, lower_bound=present_day)

    bandwidth = float(kde.bandwidth)

    min_eval = max(present_day, float(np.min(data) - 5.0 * bandwidth))
    max_eval = max(float(np.max(data) + 5.0 * bandwidth), min_eval + bandwidth)

    if monthly_sampling:
        min_year = max(present_day, np.floor(min_eval * 12.0) / 12.0)
        max_year_full = np.ceil(max_eval * 12.0) / 12.0
        steps = max(int(np.ceil((max_year_full - min_year) * 12.0)) + 1, 2)
        xs = np.linspace(min_year, max_year_full, steps)
    else:
        xs = np.linspace(min_eval, max_eval, 2048)

    pdf_values = kde(xs) * scale_factor
    if plot_max_year is not None:
        mask = xs <= plot_max_year
        if not np.any(mask):
            xs_plot = xs
            pdf_plot = pdf_values
        else:
            xs_plot = xs[mask]
            pdf_plot = pdf_values[mask]
    else:
        xs_plot = xs
        pdf_plot = pdf_values

    mode_idx = int(np.argmax(pdf_values))
    mode = float(xs[mode_idx])
    return xs, pdf_values, xs_plot, pdf_plot, mode


def plot_single_milestone_pdf(
    times: List[float],
    num_not_achieved: int,
    milestone_name: str,
    out_path: Path,
    sim_end: Optional[float] = None,
    title: Optional[str] = None,
    save_csv: bool = True,
    monthly_sampling: bool = False,
    max_year: float = 2050,
    present_day: Optional[float] = None,
) -> Optional[Dict[str, any]]:
    """Plot PDF for a single milestone using the gamma-kernel KDE.

    Args:
        times: list of arrival times (achieved only)
        num_not_achieved: count of rollouts that didn't achieve this milestone
        milestone_name: name of the milestone for labeling
        out_path: output file path
        sim_end: simulation end time for reference
        title: optional custom title
        save_csv: if True, save KDE distribution to CSV
        monthly_sampling: if True, sample at monthly intervals; else use 2048 points
        max_year: maximum year to plot (default: 2050)
        present_day: lower bound for the gamma KDE support (defaults to loading from out_path's parent directory)

    Returns:
        Dictionary with milestone statistics, or None if insufficient data
    """
    # Helper function to clean milestone names for display
    def clean_milestone_name(name: str) -> str:
        """Remove 'level' and 'skill' from milestone names."""
        return name.replace("-level", "").replace("-skill", "")

    if len(times) < 2:
        print(f"Warning: Not enough data for {milestone_name} to create KDE (need at least 2 points)")
        return None

    # Load present_day if not provided
    if present_day is None:
        present_day = load_present_day(out_path.parent)

    raw_data = np.asarray(times, dtype=float)
    boundary = float(present_day)
    valid_mask = raw_data >= boundary
    data = raw_data[valid_mask]
    dropped = raw_data.size - data.size
    if dropped:
        print(
            f"  {milestone_name}: dropped {dropped} samples before present_day={boundary:.3f}."
        )
    if data.size < 2:
        print(f"Warning: Not enough data for {milestone_name} after enforcing present_day boundary.")
        return None

    total_runs = len(times) + num_not_achieved
    prob_achieved = len(times) / total_runs if total_runs > 0 else 0.0

    # Create KDE
    try:
        xs, pdf_values, xs_plot, pdf_plot, mode = _prepare_gamma_pdf(
            data,
            present_day=boundary,
            monthly_sampling=monthly_sampling,
            scale_factor=prob_achieved * 100.0,
            plot_max_year=max_year,
        )
    except Exception as e:
        print(f"Warning: Could not create KDE for {milestone_name}: {e}")
        return None

    # Calculate percentiles including not achieved (treating them as sim_end)
    if num_not_achieved > 0 and sim_end is not None:
        combined_data = np.concatenate([raw_data, np.full(num_not_achieved, sim_end)])
    else:
        combined_data = raw_data

    q10, q50, q90 = np.quantile(combined_data, [0.1, 0.5, 0.9])

    # Save CSV if requested (save full range, not just plot range)
    if save_csv:
        csv_path = out_path.parent / (out_path.stem + "_distribution.csv")
        save_kde_distribution_csv(xs, pdf_values, milestone_name, csv_path)

    # Create plot
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Plot PDF (only up to max_year)
    plt.plot(xs_plot, pdf_plot, linewidth=2.5, color='tab:blue', label='PDF (scaled by achievement probability)')

    # Add percentile lines
    plt.axvline(q10, color='tab:gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P10: {q10:.1f}')
    plt.axvline(q50, color='tab:green', linestyle='-', linewidth=2, alpha=0.7, label=f'P50: {q50:.1f}')
    plt.axvline(q90, color='tab:gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P90: {q90:.1f}')
    plt.axvline(mode, color='tab:red', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Mode: {mode:.1f}')

    # Add statistics text
    pct_achieved = prob_achieved * 100
    stats_text = (
        f"Achievement rate: {pct_achieved:.1f}%\n"
        f"Mode: {mode:.1f}\n"
        f"P10: {q10:.1f}\n"
        f"P50: {q50:.1f}\n"
        f"P90: {q90:.1f}\n"
    )

    plt.text(0.98, 0.98, stats_text,
             transform=ax.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
             family='monospace')

    clean_name = clean_milestone_name(milestone_name)
    plt.xlabel("Arrival Time (decimal year)", fontsize=12)
    plt.ylabel("Probability Density (% per year)", fontsize=12)
    plt.title(title or f"Arrival Time Distribution: {clean_name}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=9)
    plt.xlim(right=max_year)  # Set x-axis limit to max_year

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved PDF plot for {milestone_name} to: {out_path}")
    
    # Return statistics dictionary
    pct_achieved = prob_achieved * 100
    return {
        "milestone_name": milestone_name,
        "num_achieved": len(times),
        "num_not_achieved": num_not_achieved,
        "total_rollouts": total_runs,
        "achievement_rate_pct": f"{pct_achieved:.2f}",
        "mode": f"{mode:.4f}",
        "p10": f"{q10:.4f}",
        "p50": f"{q50:.4f}",
        "p90": f"{q90:.4f}",
    }


def plot_milestone_pdfs_overlay(
    rollouts_file: Path,
    milestone_names: List[str],
    out_path: Path,
    title: Optional[str] = None,
    show_achieved_only_stats: bool = False,
    save_csv: bool = True,
    monthly_sampling: bool = False,
    max_year: Optional[float] = 2050,
    present_day: Optional[float] = None,
    milestone_cache: Optional[MilestoneBatch] = None,
) -> List[Dict[str, any]]:
    """Plot overlaid PDFs for multiple milestones.

    Args:
        rollouts_file: path to rollouts.jsonl
        milestone_names: list of milestone names to plot
        out_path: output file path
        title: optional custom title
        show_achieved_only_stats: if True, show separate stats for achieved-only runs
        save_csv: if True, save statistics and distribution data to CSV
        monthly_sampling: if True, sample at monthly intervals; else use 2048 points
        max_year: maximum year to plot (default: 2050), or None for no cutoff
        present_day: lower bound for the gamma KDE support (defaults to loading from rollouts_file's parent directory)
        milestone_cache: Optional preloaded milestone data to avoid rereading rollouts

    Returns:
        List of statistics dictionaries for each milestone
    """
    milestone_names = _filter_overlay_milestones(milestone_names)

    if not milestone_names:
        print("Warning: No milestones matching AC, SC, SAR, or SIAR found")
        return []

    # Load present_day if not provided
    if present_day is None:
        present_day = load_present_day(Path(rollouts_file).parent)

    batch = milestone_cache
    if batch is None:
        reader = RolloutsReader(rollouts_file)
        batch = _load_milestone_batch(reader, milestone_names)

    plt.figure(figsize=(14, 8))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    stats_lines = []
    stats_dicts = []
    all_kde_data = {}  # milestone_name -> (xs, pdf_values)

    # Helper function to clean milestone names for display
    def clean_milestone_name(name: str) -> str:
        """Remove 'level' and 'skill' from milestone names."""
        return name.replace("-level", "").replace("-skill", "")

    for idx, milestone_name in enumerate(milestone_names):
        times = batch.times_map.get(milestone_name, [])
        num_not_achieved = batch.not_achieved_map.get(milestone_name, 0)
        sim_end = batch.typical_sim_end

        if len(times) < 2:
            print(f"Warning: Not enough data for {milestone_name} to create KDE, skipping")
            continue

        raw_data = np.asarray(times, dtype=float)
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
            xs, pdf_values, xs_plot, pdf_plot, mode = _prepare_gamma_pdf(
                data,
                present_day=boundary,
                monthly_sampling=monthly_sampling,
                scale_factor=prob_achieved * 100.0,
                plot_max_year=max_year,
            )

            color = colors[idx % len(colors)]
            clean_name = clean_milestone_name(milestone_name)
            plt.plot(xs_plot, pdf_plot, linewidth=2.5, label=clean_name, color=color)

            # Store KDE data for CSV output
            all_kde_data[milestone_name] = (xs, pdf_values)

            # Store stats for display
            pct_achieved = prob_achieved * 100

            # Store statistics dictionary
            total_runs = len(times) + num_not_achieved
            stats_dicts.append({
                "milestone_name": milestone_name,
                "num_achieved": len(times),
                "num_not_achieved": num_not_achieved,
                "total_rollouts": total_runs,
                "achievement_rate_pct": f"{pct_achieved:.2f}",
                "mode": f"{mode:.4f}",
                "p10": f"{q10:.4f}",
                "p50": f"{q50:.4f}",
                "p90": f"{q90:.4f}",
            })

            # Calculate percentage after 2050 (includes both those achieved after 2050 and not achieved at all)
            num_after_2050 = sum(1 for t in raw_data if t > 2050) + num_not_achieved
            pct_after_2050 = (num_after_2050 / total_runs * 100) if total_runs > 0 else 0

            stats_lines.append(
                f"{clean_name}: Mode={mode:.1f}, P10={q10:.1f}, P50={q50:.1f}, P90={q90:.1f}, {pct_after_2050:.0f}% > 2050"
            )
        except Exception as e:
            print(f"Warning: Could not create KDE for {milestone_name}: {e}")
            continue

    plt.xlabel("Arrival Time (decimal year)", fontsize=12)
    plt.ylabel("Probability Density (% per year)", fontsize=12)
    plt.title(title or "Milestone Arrival Time Distributions", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=9)
    if max_year is not None:
        plt.xlim(right=max_year)  # Set x-axis limit to max_year

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
    print(f"Saved overlay PDF plot to: {out_path}")
    
    # Save CSV files if requested
    if save_csv:
        # Save summary statistics
        stats_csv_path = out_path.parent / (out_path.stem + "_statistics.csv")
        save_milestone_stats_csv(stats_dicts, stats_csv_path)
        
        # Save combined KDE distribution data
        if all_kde_data:
            dist_csv_path = out_path.parent / (out_path.stem + "_distributions.csv")
            dist_csv_path.parent.mkdir(parents=True, exist_ok=True)

            with dist_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header: time_decimal_year, then one column per milestone
                header = ["time_decimal_year"] + list(all_kde_data.keys())
                writer.writerow(header)

                # Create a common time grid by finding min/max across all milestones
                # and using the first milestone's xs array as the canonical grid
                first_xs, _ = next(iter(all_kde_data.values()))
                common_times = first_xs

                # For each time point, interpolate PDF values for all milestones
                for t in common_times:
                    row = [f"{t:.6f}"]
                    for milestone_name in header[1:]:
                        xs, pdf_vals = all_kde_data[milestone_name]
                        # Interpolate to the common time grid
                        pdf_val = float(np.interp(t, xs, pdf_vals, left=0.0, right=0.0))
                        row.append(f"{pdf_val:.10f}")
                    writer.writerow(row)
            
            print(f"Saved combined distribution data to: {dist_csv_path}")
    
    return stats_dicts


def generate_all_milestone_pdfs(
    rollouts_file: Path,
    out_dir: Path,
    overlay: bool = False,
    milestone_filter: Optional[List[str]] = None,
    monthly_sampling: bool = False,
    generate_both_cutoffs: bool = False,
    present_day: float = None,
) -> None:
    """Generate PDF plots for all milestones (or specified subset).

    Args:
        rollouts_file: path to rollouts.jsonl
        out_dir: output directory for plots
        overlay: if True, create single overlay plot; if False, create individual plots
        milestone_filter: if provided, only process these milestones
        monthly_sampling: if True, sample at monthly intervals; else use 2048 points
        generate_both_cutoffs: if True, generate both 2050 and full-range versions
        present_day: Lower bound for the gamma KDE support
    """
    # Discover all milestones
    all_milestones = _list_milestone_names(rollouts_file)

    if not all_milestones:
        print("No milestones found in rollouts file.")
        return

    # Filter milestones if requested
    if milestone_filter:
        milestones = [m for m in milestone_filter if m in all_milestones]
        missing = [m for m in milestone_filter if m not in all_milestones]
        if missing:
            print(f"Warning: Requested milestones not found: {missing}")
        if not milestones:
            print("No matching milestones found.")
            return
    else:
        milestones = all_milestones

    print(f"Processing {len(milestones)} milestones...")

    reader = RolloutsReader(rollouts_file)

    if overlay:
        overlay_targets = _filter_overlay_milestones(milestones)
        if not overlay_targets:
            print("Warning: No milestones matching AC, SC, SAR, or SIAR found")
            return
        overlay_batch = _load_milestone_batch(reader, overlay_targets)

        # Create single overlay plot (2050 cutoff)
        out_path = out_dir / "milestone_pdfs_overlay.png"
        stats = plot_milestone_pdfs_overlay(
            rollouts_file,
            milestones,
            out_path,
            show_achieved_only_stats=True,
            save_csv=True,
            monthly_sampling=monthly_sampling,
            present_day=present_day,
            milestone_cache=overlay_batch,
        )

        # If requested, also create full-range version
        if generate_both_cutoffs:
            out_path_full = out_dir / "milestone_pdfs_overlay_full.png"
            plot_milestone_pdfs_overlay(
                rollouts_file,
                milestones,
                out_path_full,
                show_achieved_only_stats=True,
                save_csv=False,  # Don't duplicate CSV files
                monthly_sampling=monthly_sampling,
                max_year=None,  # No cutoff
                present_day=present_day,
                milestone_cache=overlay_batch,
            )
    else:
        # Create individual plots
        out_dir.mkdir(parents=True, exist_ok=True)
        stats = []

        batch = _load_milestone_batch(reader, milestones)

        for milestone_name in milestones:
            times = batch.times_map.get(milestone_name, [])
            num_not_achieved = batch.not_achieved_map.get(milestone_name, 0)
            sim_end = batch.typical_sim_end

            if len(times) < 2:
                print(f"Skipping {milestone_name}: insufficient data (need at least 2 achieved rollouts)")
                continue

            # Create safe filename
            safe_name = milestone_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")

            # Generate 2050 cutoff version
            out_path = out_dir / f"milestone_pdf_{safe_name}.png"
            stat = plot_single_milestone_pdf(
                times,
                num_not_achieved,
                milestone_name,
                out_path,
                sim_end=sim_end,
                save_csv=True,
                monthly_sampling=monthly_sampling,
                max_year=2050,
                present_day=present_day,
            )
            if stat:
                stats.append(stat)

            # If requested, also generate full-range version
            if generate_both_cutoffs:
                out_path_full = out_dir / f"milestone_pdf_{safe_name}_full.png"
                # Determine appropriate max_year from data
                if times:
                    max_data_year = max(times)
                    full_max_year = max(2200, max_data_year * 1.1)  # At least 2200 or 110% of max
                else:
                    full_max_year = 2200

                plot_single_milestone_pdf(
                    times,
                    num_not_achieved,
                    milestone_name,
                    out_path_full,
                    sim_end=sim_end,
                    save_csv=False,  # Don't duplicate CSV files
                    monthly_sampling=monthly_sampling,
                    max_year=full_max_year,
                    present_day=present_day,
                )

        # Save combined statistics CSV for individual plots mode
        if stats:
            stats_csv_path = out_dir / "milestone_statistics_summary.csv"
            save_milestone_stats_csv(stats, stats_csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PDF plots for milestone arrival times using gamma-kernel KDEs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--rollouts",
        type=str,
        required=True,
        help="Path to rollouts.jsonl file"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same directory as rollouts file)"
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Create single overlay plot instead of individual plots for each milestone"
    )
    parser.add_argument(
        "--milestones",
        type=str,
        default=None,
        help="Comma-separated list of specific milestones to plot (default: all milestones)"
    )
    parser.add_argument(
        "--list-milestones",
        action="store_true",
        help="List all available milestones and exit"
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Sample KDE at monthly intervals instead of 512 evenly-spaced points (reduces CSV file size)"
    )
    parser.add_argument(
        "--both-cutoffs",
        action="store_true",
        help="Generate both 2050 cutoff and full-range versions of plots"
    )
    parser.add_argument(
        "--present-day",
        type=float,
        default=None,
        help="Override the present_day boundary for the gamma KDE support (defaults to value saved with the run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rollouts_path = Path(args.rollouts)
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")

    # List milestones and exit if requested
    if args.list_milestones:
        milestones = _list_milestone_names(rollouts_path)
        if milestones:
            print(f"Found {len(milestones)} milestones:")
            for m in milestones:
                print(f"  - {m}")
        else:
            print("No milestones found in rollouts file.")
        return

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        if args.overlay:
            out_dir = rollouts_path.parent
        else:
            out_dir = rollouts_path.parent / "milestone_pdfs"

    present_day = (
        float(args.present_day)
        if args.present_day is not None
        else load_present_day(rollouts_path.parent)
    )
    print(f"Using present_day = {present_day:.3f}")

    # Parse milestone filter if provided
    milestone_filter = None
    if args.milestones:
        milestone_filter = [m.strip() for m in args.milestones.split(",")]
        print(f"Filtering to {len(milestone_filter)} milestone(s)")

    # Generate plots
    generate_all_milestone_pdfs(
        rollouts_path,
        out_dir,
        overlay=args.overlay,
        milestone_filter=milestone_filter,
        monthly_sampling=args.monthly,
        generate_both_cutoffs=args.both_cutoffs,
        present_day=present_day,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
