#!/usr/bin/env python3
"""
Analyze correlation between timelines to SAR and takeoff speeds from SAR onward.

WARNING: This script contains bespoke data loading code (custom timeline_and_takeoff_data
extraction) that doesn't follow the project's plotting utility conventions. It should be
refactored to use utilities from scripts/plotting_utils/.

This script generates visualizations to help understand why these two metrics are correlated:
1. Dual-axis plots showing how each parameter affects both timeline and takeoff speed
2. Scatter plot showing parameter effects: timeline sensitivity vs takeoff speed sensitivity
3. Joint distribution plots for timeline and takeoff speed

Usage:
  python scripts/timeline_takeoff_correlation_analysis.py --run-dir outputs/20250813_020347
  python scripts/timeline_takeoff_correlation_analysis.py --rollouts outputs/20250813_020347/rollouts.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import yaml

# Use a non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d

# Use monospace font
matplotlib.rcParams["font.family"] = "monospace"


def load_rollouts(rollouts_file: Path) -> List[Dict]:
    """Load all rollouts from a jsonl file."""
    rollouts = []
    with rollouts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rollouts.append(rec)
            except json.JSONDecodeError:
                continue
    return rollouts


def extract_timeline_and_takeoff_data(rollouts: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Extract SAR timeline and takeoff speed data from rollouts.

    Returns:
        timeline: Array of SAR arrival times (years)
        takeoff_speed: Array of SAR->ASI durations (years)
        valid_rollouts: List of rollouts with both metrics available
    """
    timeline = []
    takeoff_speed = []
    valid_rollouts = []

    for rollout in rollouts:
        results = rollout.get("results")
        if not isinstance(results, dict):
            continue
        milestones = results.get("milestones", {})

        # Get SAR arrival time
        sar_info = milestones.get("SAR-level-experiment-selection-skill", {})
        sar_time = sar_info.get("time")
        if sar_time is None or not np.isfinite(float(sar_time)):
            continue

        # Get ASI arrival time
        asi_info = milestones.get("ASI", {})
        asi_time = asi_info.get("time")
        if asi_time is None or not np.isfinite(float(asi_time)):
            continue

        sar_t = float(sar_time)
        asi_t = float(asi_time)

        if asi_t > sar_t:
            timeline.append(sar_t)
            takeoff_speed.append(asi_t - sar_t)
            valid_rollouts.append(rollout)

    return np.array(timeline), np.array(takeoff_speed), valid_rollouts


def extract_ac_timeline_and_takeoff_data(rollouts: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Extract AC timeline and takeoff speed data from rollouts.

    Returns:
        ac_timeline: Array of AC arrival times (years)
        takeoff_speed: Array of AC->ASI durations (years)
        valid_rollouts: List of rollouts with both metrics available
    """
    ac_timeline = []
    takeoff_speed = []
    valid_rollouts = []

    for rollout in rollouts:
        results = rollout.get("results")
        if not isinstance(results, dict):
            continue
        milestones = results.get("milestones", {})

        # Get AC arrival time
        ac_info = milestones.get("AC", {})
        ac_time = ac_info.get("time")
        if ac_time is None or not np.isfinite(float(ac_time)):
            continue

        # Get ASI arrival time
        asi_info = milestones.get("ASI", {})
        asi_time = asi_info.get("time")
        if asi_time is None or not np.isfinite(float(asi_time)):
            continue

        ac_t = float(ac_time)
        asi_t = float(asi_time)

        if asi_t > ac_t:
            ac_timeline.append(ac_t)
            takeoff_speed.append(asi_t - ac_t)
            valid_rollouts.append(rollout)

    return np.array(ac_timeline), np.array(takeoff_speed), valid_rollouts


def compute_parameter_sensitivities(
    rollouts: List[Dict],
    timeline: np.ndarray,
    takeoff_speed: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compute how each parameter affects timeline and takeoff speed.

    For each parameter, compute:
    - Correlation with SAR timeline
    - Correlation with takeoff speed (SAR->TED-AI duration)
    - Rank correlation (Spearman)

    Returns:
        Dict mapping parameter name to dict of metrics
    """
    sensitivities = {}

    # Extract all parameter names from first rollout
    if not rollouts:
        return sensitivities

    sample_params = rollouts[0].get("parameters", {})
    param_names = list(sample_params.keys())

    for param_name in param_names:
        # Extract parameter values
        param_values = []
        for rollout in rollouts:
            params = rollout.get("parameters", {})
            value = params.get(param_name)
            if value is not None:
                try:
                    param_values.append(float(value))
                except (ValueError, TypeError):
                    # Skip non-numeric parameters
                    param_values.append(np.nan)
            else:
                param_values.append(np.nan)

        param_values = np.array(param_values)

        # Filter out NaNs
        valid_mask = np.isfinite(param_values)
        if np.sum(valid_mask) < 10:  # Need at least 10 valid points
            continue

        param_vals = param_values[valid_mask]
        timeline_vals = timeline[valid_mask]
        takeoff_vals = takeoff_speed[valid_mask]

        # Compute correlations
        try:
            timeline_corr, timeline_pval = pearsonr(param_vals, timeline_vals)
            takeoff_corr, takeoff_pval = pearsonr(param_vals, takeoff_vals)
            timeline_rank_corr, _ = spearmanr(param_vals, timeline_vals)
            takeoff_rank_corr, _ = spearmanr(param_vals, takeoff_vals)
        except:
            continue

        sensitivities[param_name] = {
            'timeline_corr': timeline_corr,
            'timeline_pval': timeline_pval,
            'takeoff_corr': takeoff_corr,
            'takeoff_pval': takeoff_pval,
            'timeline_rank_corr': timeline_rank_corr,
            'takeoff_rank_corr': takeoff_rank_corr,
            'n_samples': int(np.sum(valid_mask))
        }

    return sensitivities


def plot_dual_axis_parameter_effects(
    rollouts: List[Dict],
    timeline: np.ndarray,
    takeoff_speed: np.ndarray,
    output_dir: Path,
    top_n: int = 20,
    title_suffix: str = "",
    filename_suffix: str = ""
) -> None:
    """Create dual-axis plots showing how each parameter affects both metrics.

    For top N parameters by total correlation magnitude, create plots showing:
    - Left axis: Effect on timeline
    - Right axis: Effect on takeoff speed
    """
    print(f"\nGenerating dual-axis parameter effect plots{filename_suffix}...")

    # Compute sensitivities
    sensitivities = compute_parameter_sensitivities(rollouts, timeline, takeoff_speed)

    # Sort parameters by total correlation magnitude
    param_scores = []
    for param_name, metrics in sensitivities.items():
        total_corr = abs(metrics['timeline_corr']) + abs(metrics['takeoff_corr'])
        param_scores.append((param_name, total_corr, metrics))

    param_scores.sort(key=lambda x: x[1], reverse=True)
    top_params = param_scores[:top_n]

    # Create plots
    for param_name, _, metrics in top_params:
        # Extract parameter values
        param_values = []
        for rollout in rollouts:
            params = rollout.get("parameters", {})
            value = params.get(param_name)
            if value is not None:
                param_values.append(float(value))
            else:
                param_values.append(np.nan)

        param_values = np.array(param_values)

        # Filter out NaNs
        valid_mask = np.isfinite(param_values)
        param_vals = param_values[valid_mask]
        timeline_vals = timeline[valid_mask]
        takeoff_vals = takeoff_speed[valid_mask]

        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot timeline on left axis
        color1 = 'tab:blue'
        ax1.set_xlabel(param_name, fontsize=11)
        ax1.set_ylabel('Timeline (years)', color=color1, fontsize=11)
        ax1.scatter(param_vals, timeline_vals, alpha=0.3, s=20, color=color1, label='Timeline')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        # Add trend line for timeline
        if len(param_vals) > 1:
            sort_idx = np.argsort(param_vals)
            sorted_params = param_vals[sort_idx]
            sorted_timeline = timeline_vals[sort_idx]

            # Use rolling average for trend
            window = max(len(sorted_params) // 20, 5)
            if len(sorted_params) >= window:
                trend_timeline = np.convolve(sorted_timeline, np.ones(window)/window, mode='valid')
                trend_params = sorted_params[window//2:-(window//2)+1][:len(trend_timeline)]
                ax1.plot(trend_params, trend_timeline, color=color1, linewidth=2, alpha=0.7)

        # Plot takeoff speed on right axis
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Takeoff Speed (years)', color=color2, fontsize=11)
        ax2.scatter(param_vals, takeoff_vals, alpha=0.3, s=20, color=color2, label='Takeoff Speed')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Add trend line for takeoff
        if len(param_vals) > 1:
            sorted_takeoff = takeoff_vals[sort_idx]
            if len(sorted_params) >= window:
                trend_takeoff = np.convolve(sorted_takeoff, np.ones(window)/window, mode='valid')
                ax2.plot(trend_params, trend_takeoff, color=color2, linewidth=2, alpha=0.7)

        # Add title with correlations
        title = (f"{param_name} {title_suffix}\n"
                f"Timeline corr: {metrics['timeline_corr']:.3f} (p={metrics['timeline_pval']:.3e})  |  "
                f"Takeoff corr: {metrics['takeoff_corr']:.3f} (p={metrics['takeoff_pval']:.3e})")
        plt.title(title, fontsize=10)

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

        fig.tight_layout()

        # Save with sanitized filename
        safe_name = param_name.replace('/', '_').replace('\\', '_')
        output_path = output_dir / f"dual_axis_{safe_name}{filename_suffix}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Generated {len(top_params)} dual-axis plots{filename_suffix}")


def plot_parameter_sensitivity_scatter(
    rollouts: List[Dict],
    timeline: np.ndarray,
    takeoff_speed: np.ndarray,
    output_dir: Path,
    x_label: str = "Effect on SAR Timeline (Pearson correlation)",
    y_label: str = "Effect on Takeoff Speed SAR→ASI (Pearson correlation)",
    title: str = "Parameter Effects: Timeline vs Takeoff Speed",
    filename_suffix: str = ""
) -> None:
    """Create scatter plot of parameter effects: timeline vs takeoff speed.

    Each point represents a parameter, with:
    - X-axis: Effect on timeline (correlation)
    - Y-axis: Effect on takeoff speed (correlation)
    """
    print(f"\nGenerating parameter sensitivity scatter plot{filename_suffix}...")

    # Compute sensitivities
    sensitivities = compute_parameter_sensitivities(rollouts, timeline, takeoff_speed)

    if not sensitivities:
        print("  No sensitivities computed, skipping scatter plot")
        return

    # Extract correlations
    param_names = list(sensitivities.keys())
    timeline_corrs = [sensitivities[p]['timeline_corr'] for p in param_names]
    takeoff_corrs = [sensitivities[p]['takeoff_corr'] for p in param_names]

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color points by magnitude of total correlation
    total_corrs = [abs(tc) + abs(to) for tc, to in zip(timeline_corrs, takeoff_corrs)]
    scatter = ax.scatter(timeline_corrs, takeoff_corrs,
                        c=total_corrs, cmap='viridis',
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('|Timeline Corr| + |Takeoff Corr|', fontsize=11)

    # Add diagonal line (equal correlation)
    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
              abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, linewidth=1, label='Equal correlation')

    # Add zero lines
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Label significant parameters (at least 0.1 magnitude on either axis, or r_software)
    for i, param_name in enumerate(param_names):
        should_label = (abs(timeline_corrs[i]) >= 0.1 or
                       abs(takeoff_corrs[i]) >= 0.1 or
                       param_name == 'r_software')
        if should_label:
            # Add small offset to avoid overlap
            offset_x = 0.02 * (1 if timeline_corrs[i] > 0 else -1)
            offset_y = 0.02 * (1 if takeoff_corrs[i] > 0 else -1)
            ax.annotate(param_name,
                       (timeline_corrs[i] + offset_x, takeoff_corrs[i] + offset_y),
                       fontsize=7, alpha=0.8)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_path = output_dir / f"parameter_sensitivity_scatter{filename_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")


def plot_joint_distribution(
    timeline: np.ndarray,
    takeoff_speed: np.ndarray,
    output_dir: Path,
    title_suffix: str = "",
    filename_suffix: str = ""
) -> None:
    """Plot joint distribution of timeline and takeoff speed."""
    print(f"\nGenerating joint distribution plot{filename_suffix}...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot with density coloring
    from scipy.stats import gaussian_kde

    # Compute density
    xy = np.vstack([timeline, takeoff_speed])
    try:
        z = gaussian_kde(xy)(xy)
    except:
        z = np.ones(len(timeline))

    # Sort points by density so densest points are plotted last
    idx = z.argsort()
    timeline_sorted = timeline[idx]
    takeoff_sorted = takeoff_speed[idx]
    z_sorted = z[idx]

    scatter = ax1.scatter(timeline_sorted, takeoff_sorted,
                         c=z_sorted, cmap='viridis',
                         s=20, alpha=0.5, edgecolors='none')

    ax1.set_xlabel('Timeline (years)', fontsize=12)
    ax1.set_ylabel('Takeoff Speed (years)', fontsize=12)
    ax1.set_title(f'Joint Distribution: Timeline vs Takeoff Speed {title_suffix}', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Density', fontsize=11)

    # Add correlation text
    corr, pval = pearsonr(timeline, takeoff_speed)
    rank_corr, _ = spearmanr(timeline, takeoff_speed)
    ax1.text(0.05, 0.95,
            f'Pearson r = {corr:.3f} (p = {pval:.3e})\nSpearman ρ = {rank_corr:.3f}',
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hexbin plot
    hexbin = ax2.hexbin(timeline, takeoff_speed,
                       gridsize=30, cmap='viridis',
                       mincnt=1, alpha=0.8)

    ax2.set_xlabel('Timeline (years)', fontsize=12)
    ax2.set_ylabel('Takeoff Speed (years)', fontsize=12)
    ax2.set_title(f'Hexbin Plot: Timeline vs Takeoff Speed {title_suffix}', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add colorbar
    cbar2 = plt.colorbar(hexbin, ax=ax2)
    cbar2.set_label('Count', fontsize=11)

    plt.tight_layout()
    output_path = output_dir / f"joint_distribution{filename_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")


def plot_binned_takeoff_vs_timeline(
    timeline: np.ndarray,
    takeoff_speed: np.ndarray,
    output_dir: Path,
    n_bins: int = 10,
    title_suffix: str = "",
    filename_suffix: str = ""
) -> None:
    """Plot average takeoff speed in bins of timeline.

    This helps visualize the relationship more clearly than a scatter plot.
    """
    print(f"\nGenerating binned takeoff vs timeline plot{filename_suffix}...")

    # Create bins based on timeline percentiles
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(timeline, percentiles)

    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_medians = []
    bin_counts = []

    for i in range(n_bins):
        mask = (timeline >= bin_edges[i]) & (timeline < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (timeline >= bin_edges[i]) & (timeline <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(np.mean(takeoff_speed[mask]))
            bin_stds.append(np.std(takeoff_speed[mask]))
            bin_medians.append(np.median(takeoff_speed[mask]))
            bin_counts.append(np.sum(mask))

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_medians = np.array(bin_medians)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot scatter in background
    ax.scatter(timeline, takeoff_speed, alpha=0.1, s=10, color='gray', label='Individual rollouts')

    # Plot binned statistics
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
               fmt='o-', linewidth=2, markersize=8,
               capsize=5, capthick=2,
               color='tab:red', label='Mean ± Std', zorder=10)

    ax.plot(bin_centers, bin_medians,
           's--', linewidth=2, markersize=8,
           color='tab:blue', label='Median', zorder=10)

    ax.set_xlabel('Timeline (years)', fontsize=12)
    ax.set_ylabel('Takeoff Speed (years)', fontsize=12)
    ax.set_title(f'Takeoff Speed vs Timeline (Binned) {title_suffix}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    output_path = output_dir / f"binned_takeoff_vs_timeline{filename_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")


def write_sensitivity_summary(
    sensitivities: Dict[str, Dict[str, float]],
    output_dir: Path,
    filename_suffix: str = ""
) -> None:
    """Write summary table of parameter sensitivities to CSV."""
    print(f"\nWriting sensitivity summary table{filename_suffix}...")

    import csv

    # Sort by total correlation magnitude
    sorted_params = sorted(sensitivities.items(),
                          key=lambda x: abs(x[1]['timeline_corr']) + abs(x[1]['takeoff_corr']),
                          reverse=True)

    output_path = output_dir / f"parameter_sensitivities{filename_suffix}.csv"
    with output_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Parameter',
            'Timeline Corr',
            'Timeline P-val',
            'Takeoff Corr',
            'Takeoff P-val',
            'Timeline Rank Corr',
            'Takeoff Rank Corr',
            'N Samples'
        ])

        for param_name, metrics in sorted_params:
            writer.writerow([
                param_name,
                f"{metrics['timeline_corr']:.4f}",
                f"{metrics['timeline_pval']:.4e}",
                f"{metrics['takeoff_corr']:.4f}",
                f"{metrics['takeoff_pval']:.4e}",
                f"{metrics['timeline_rank_corr']:.4f}",
                f"{metrics['takeoff_rank_corr']:.4f}",
                metrics['n_samples']
            ])

    print(f"  Saved to {output_path}")


def analyze_timeline_takeoff_correlation(
    rollouts_file: Path,
    output_dir: Path,
    run_label: str = ""
) -> None:
    """Run complete timeline-takeoff correlation analysis.

    Args:
        rollouts_file: Path to rollouts.jsonl
        output_dir: Output directory (run directory)
        run_label: Optional label to add to plot titles (e.g., "no correlation")
    """

    # Create output directory
    corr_dir = output_dir / "timeline_takeoff_correlation"
    corr_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TIMELINE-TAKEOFF CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"Rollouts: {rollouts_file}")
    print(f"Output: {corr_dir}")
    if run_label:
        print(f"Label: {run_label}")

    # Load rollouts
    print("\nLoading rollouts...")
    rollouts = load_rollouts(rollouts_file)
    print(f"  Loaded {len(rollouts)} rollouts")

    # ========== SAR-BASED ANALYSIS ==========
    print("\n" + "=" * 60)
    print("SAR-BASED ANALYSIS (SAR → ASI)")
    print("=" * 60)

    # Extract timeline and takeoff data
    print("\nExtracting SAR timeline and takeoff speed data...")
    sar_timeline, sar_takeoff_speed, sar_valid_rollouts = extract_timeline_and_takeoff_data(rollouts)
    print(f"  Found {len(sar_valid_rollouts)} rollouts with both SAR and ASI")

    if len(sar_valid_rollouts) < 10:
        print("ERROR: Not enough valid rollouts for SAR analysis")
    else:
        # Compute basic correlation
        corr, pval = pearsonr(sar_timeline, sar_takeoff_speed)
        rank_corr, _ = spearmanr(sar_timeline, sar_takeoff_speed)
        print(f"\nSAR → ASI Overall correlation:")
        print(f"  Pearson r = {corr:.4f} (p = {pval:.4e})")
        print(f"  Spearman ρ = {rank_corr:.4f}")

        # Build title suffix with run label
        sar_title_suffix = f"(SAR → ASI){' - ' + run_label if run_label else ''}"

        # Generate visualizations
        plot_joint_distribution(sar_timeline, sar_takeoff_speed, corr_dir,
                               title_suffix=sar_title_suffix, filename_suffix="_sar")
        plot_binned_takeoff_vs_timeline(sar_timeline, sar_takeoff_speed, corr_dir,
                                       title_suffix=sar_title_suffix, filename_suffix="_sar")

        # Compute parameter sensitivities
        print("\nComputing SAR parameter sensitivities...")
        sar_sensitivities = compute_parameter_sensitivities(sar_valid_rollouts, sar_timeline, sar_takeoff_speed)
        print(f"  Analyzed {len(sar_sensitivities)} parameters")

        # Write sensitivity summary
        write_sensitivity_summary(sar_sensitivities, corr_dir, filename_suffix="_sar")

        # Generate parameter effect plots
        plot_dual_axis_parameter_effects(sar_valid_rollouts, sar_timeline, sar_takeoff_speed, corr_dir,
                                        top_n=20, title_suffix=sar_title_suffix, filename_suffix="_sar")
        plot_parameter_sensitivity_scatter(sar_valid_rollouts, sar_timeline, sar_takeoff_speed, corr_dir,
                                          x_label="Effect on SAR Timeline",
                                          y_label="Effect on Takeoff Speed SAR→ASI",
                                          title=f"Parameter Effects: SAR Timeline vs Takeoff Speed (SAR→ASI){' - ' + run_label if run_label else ''}",
                                          filename_suffix="_sar")

    # ========== AC-BASED ANALYSIS ==========
    print("\n" + "=" * 60)
    print("AC-BASED ANALYSIS (AC → ASI)")
    print("=" * 60)

    # Extract AC timeline and takeoff data
    print("\nExtracting AC timeline and takeoff speed data...")
    ac_timeline, ac_takeoff_speed, ac_valid_rollouts = extract_ac_timeline_and_takeoff_data(rollouts)
    print(f"  Found {len(ac_valid_rollouts)} rollouts with both AC and ASI")

    if len(ac_valid_rollouts) < 10:
        print("ERROR: Not enough valid rollouts for AC analysis")
    else:
        # Compute basic correlation
        corr, pval = pearsonr(ac_timeline, ac_takeoff_speed)
        rank_corr, _ = spearmanr(ac_timeline, ac_takeoff_speed)
        print(f"\nAC → ASI Overall correlation:")
        print(f"  Pearson r = {corr:.4f} (p = {pval:.4e})")
        print(f"  Spearman ρ = {rank_corr:.4f}")

        # Build title suffix with run label
        ac_title_suffix = f"(AC → ASI){' - ' + run_label if run_label else ''}"

        # Generate visualizations
        plot_joint_distribution(ac_timeline, ac_takeoff_speed, corr_dir,
                               title_suffix=ac_title_suffix, filename_suffix="_ac")
        plot_binned_takeoff_vs_timeline(ac_timeline, ac_takeoff_speed, corr_dir,
                                       title_suffix=ac_title_suffix, filename_suffix="_ac")

        # Compute parameter sensitivities
        print("\nComputing AC parameter sensitivities...")
        ac_sensitivities = compute_parameter_sensitivities(ac_valid_rollouts, ac_timeline, ac_takeoff_speed)
        print(f"  Analyzed {len(ac_sensitivities)} parameters")

        # Write sensitivity summary
        write_sensitivity_summary(ac_sensitivities, corr_dir, filename_suffix="_ac")

        # Generate parameter effect plots
        plot_dual_axis_parameter_effects(ac_valid_rollouts, ac_timeline, ac_takeoff_speed, corr_dir,
                                        top_n=20, title_suffix=ac_title_suffix, filename_suffix="_ac")
        plot_parameter_sensitivity_scatter(ac_valid_rollouts, ac_timeline, ac_takeoff_speed, corr_dir,
                                          x_label="Effect on AC Timeline",
                                          y_label="Effect on Takeoff Speed AC→ASI",
                                          title=f"Parameter Effects: AC Timeline vs Takeoff Speed (AC→ASI){' - ' + run_label if run_label else ''}",
                                          filename_suffix="_ac")

    print(f"\n✓ Analysis complete! All outputs saved to: {corr_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze timeline-takeoff correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory (will use rollouts.jsonl inside)"
    )
    group.add_argument(
        "--rollouts",
        type=str,
        help="Path to rollouts.jsonl file"
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label to add to plot titles (e.g., 'no correlation')"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine rollouts file and output directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        rollouts_file = run_dir / "rollouts.jsonl"
        output_dir = run_dir
    else:
        rollouts_file = Path(args.rollouts)
        output_dir = rollouts_file.parent

    if not rollouts_file.exists():
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_file}")

    # Run analysis
    analyze_timeline_takeoff_correlation(rollouts_file, output_dir, run_label=args.label)


if __name__ == "__main__":
    main()
