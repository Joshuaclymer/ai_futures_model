"""
Trajectory plotting functions for rollout analysis.

This module contains functions for plotting time series trajectories including
horizon length trajectories, uplift trajectories, and generic metric trajectories.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np

# Use a non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use monospace font for all text elements
matplotlib.rcParams["font.family"] = "monospace"

# Import utilities
import sys
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from plotting_utils.helpers import (
    now_decimal_year,
    get_time_tick_values_and_labels,
    load_metr_p80_points,
)


def plot_horizon_trajectories(
    times: np.ndarray,
    trajectories: List[np.ndarray],
    out_path: Path,
    current_horizon_minutes: float = 15.0,
    alpha: float = 0.08,
    max_trajectories: int = 2000,
    overlay_metr: bool = True,
    title: Optional[str] = None,
    stop_at_sc: bool = False,
    aa_times: Optional[List[Optional[float]]] = None,
    mse_values: Optional[List[Optional[float]]] = None,
    shade_by_mse: bool = False,
    mse_threshold: Optional[float] = None,
    cutoff_year: Optional[float] = None,
) -> None:
    """Render horizon length trajectories similar to the reference figure.

    Assumes times are decimal years and horizons are in minutes.
    """
    if len(trajectories) == 0:
        raise ValueError("No trajectories provided")

    # Filter trajectories by MSE threshold if specified
    if mse_threshold is not None and mse_values is not None:
        filtered_indices = []
        for i, mse in enumerate(mse_values):
            if mse is None or mse <= mse_threshold:
                filtered_indices.append(i)
        trajectories = [trajectories[i] for i in filtered_indices]
        if aa_times is not None:
            aa_times = [aa_times[i] for i in filtered_indices]
        if mse_values is not None:
            mse_values = [mse_values[i] for i in filtered_indices]
        print(f"Filtered to {len(trajectories)} trajectories with MSE <= {mse_threshold}")

    if len(trajectories) == 0:
        raise ValueError("No trajectories remain after MSE filtering")

    # Clip extremely large values to avoid distorting the log scale
    min_horizon_minutes = 0.001
    # Set a very high cap that won't artificially bunch trajectories
    # Cap at 10 million work-years instead of 120,000 to avoid bunching
    max_work_year_minutes = float(10_000_000 * 52 * 40 * 60)
    # Replace non-positive and non-finite values with NaN so they don't draw lines to zero
    cleaned: List[np.ndarray] = []
    for idx, t in enumerate(trajectories):
        arr = t.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        arr[arr <= 0] = np.nan
        arr = np.clip(arr, min_horizon_minutes, max_work_year_minutes)
        # If requested, mask values after this rollout's aa_time
        if stop_at_sc and aa_times is not None and idx < len(aa_times) and aa_times[idx] is not None:
            sc = aa_times[idx]
            if sc is not None:
                arr = arr.copy()
                arr[times > float(sc)] = np.nan
        cleaned.append(arr)

    # Optionally interpolate trajectories for smoother plotting
    # Skip interpolation if many trajectories are at the cap (causes visual artifacts)
    num_at_cap = sum(1 for arr in cleaned if np.any(arr >= max_work_year_minutes * 0.99))
    skip_interpolation = num_at_cap > len(cleaned) * 0.5  # Skip if >50% hit cap

    if not skip_interpolation:
        if cutoff_year is not None:
            # Only interpolate the visible range
            time_range = cutoff_year - times[0]
            num_interp_points = min(len(times) * 10, int(time_range * 50))  # ~50 points per year in visible range
            times_interp = np.linspace(times[0], cutoff_year, num_interp_points)
            print(f"Interpolating from {len(times)} to {num_interp_points} points ({num_interp_points/time_range:.1f} pts/year)")
        else:
            num_interp_points = len(times) * 10
            times_interp = np.linspace(times[0], times[-1], num_interp_points)
            print(f"Interpolating from {len(times)} to {num_interp_points} points")

        from scipy.interpolate import interp1d
        cleaned_interp: List[np.ndarray] = []
        for arr in cleaned:
            # Only interpolate over valid (non-NaN) regions
            valid_mask = ~np.isnan(arr)
            if np.sum(valid_mask) > 1:  # Need at least 2 points to interpolate
                try:
                    f = interp1d(times[valid_mask], arr[valid_mask], kind='linear',
                               bounds_error=False, fill_value=np.nan)
                    arr_interp = f(times_interp)
                except Exception:
                    # If interpolation fails, fall back to original
                    arr_interp = np.interp(times_interp, times, arr)
            else:
                arr_interp = np.full(len(times_interp), np.nan)
            cleaned_interp.append(arr_interp)

        # Use interpolated data for plotting
        cleaned = cleaned_interp
        times = times_interp
    else:
        print(f"Skipping interpolation ({num_at_cap}/{len(cleaned)} trajectories at cap - would cause artifacts)")

    # Compute median trajectory across rollouts (align by index; the batch process uses a shared grid)
    stacked = np.vstack([t for t in cleaned[:max_trajectories]])
    median_traj = np.nanmedian(stacked, axis=0)

    plt.figure(figsize=(14, 8))

    # Determine colors for trajectories
    if shade_by_mse and mse_values is not None:
        # Compute color mapping based on MSE values
        # Filter out None values for computing percentiles
        valid_mse = [mse for mse in mse_values if mse is not None]
        if len(valid_mse) > 0:
            mse_min = np.percentile(valid_mse, 5)  # Use 5th percentile as min
            mse_max = np.percentile(valid_mse, 95)  # Use 95th percentile as max
            print(f"MSE range for coloring: {mse_min:.4f} to {mse_max:.4f}")
        else:
            mse_min = 0.0
            mse_max = 1.0
    else:
        valid_mse = []

    # Draw all trajectories
    num_plot = min(len(cleaned), max_trajectories)
    for i in range(num_plot):
        if shade_by_mse and mse_values is not None and i < len(mse_values) and mse_values[i] is not None:
            # Color based on MSE: green (low) to red (high)
            mse = mse_values[i]
            # Normalize MSE to [0, 1]
            if len(valid_mse) > 0 and mse_max > mse_min:
                norm_mse = (mse - mse_min) / (mse_max - mse_min)
                norm_mse = np.clip(norm_mse, 0.0, 1.0)
            else:
                norm_mse = 0.5
            # Green (0.0) to Red (1.0) via Yellow
            # RGB: (0,1,0) -> (1,1,0) -> (1,0,0)
            if norm_mse < 0.5:
                # Green to Yellow
                r = norm_mse * 2
                g = 1.0
                b = 0.0
            else:
                # Yellow to Red
                r = 1.0
                g = 2.0 * (1.0 - norm_mse)
                b = 0.0
            color = (r, g, b, alpha)
        else:
            # Default blue color
            color = (0.2, 0.5, 0.7, alpha)
        plt.plot(times, cleaned[i], color=color, linewidth=1.0)

    # Central trajectory
    plt.plot(times, median_traj, color="tab:green", linestyle="--", linewidth=2.0, label="Central Trajectory")

    # Horizontal line for current horizon
    plt.axhline(current_horizon_minutes, color="red", linewidth=2.0, label=f"Current Horizon ({int(current_horizon_minutes)} min)")

    # Vertical line for current time
    now_year = now_decimal_year()
    plt.axvline(now_year, color="tab:blue", linestyle="--", linewidth=1.75, label="Current Time")

    # Optional METR p80 scatter
    if overlay_metr:
        points = load_metr_p80_points()
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.scatter(xs, ys, color="black", s=18, label="External Benchmarks (p80)", zorder=10)

    # Axes and formatting
    plt.yscale("log")
    tick_values, tick_labels = get_time_tick_values_and_labels()
    plt.yticks(tick_values, tick_labels)
    # Expand y-limit to cover data range
    finite_max = np.nanmax(stacked)
    if np.isfinite(finite_max):
        ymin = min(tick_values)
        ymax = min(max(finite_max, max(tick_values)), max_work_year_minutes)
        plt.ylim(ymin, ymax)

    # Set x-axis limit if cutoff_year is specified
    if cutoff_year is not None:
        plt.xlim(times[0], cutoff_year)

    plt.xlabel("Year")
    plt.ylabel("Time Horizon")

    # Update title to indicate MSE shading/filtering
    base_title = title or "Complete Time Horizon Extension Trajectories\n(Historical development and future projections)"
    base_title += f"\n(n={len(trajectories)} trajectories"
    if mse_threshold is not None:
        base_title += f", MSE ≤ {mse_threshold:.3f}"
    base_title += ")"
    if shade_by_mse and len(valid_mse) > 0:
        base_title += f"\nColor: Green (low MSE) → Yellow → Red (high MSE)"
    plt.title(base_title)

    plt.grid(True, which="both", axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_uplift_trajectories(
    times: np.ndarray,
    trajectories: List[np.ndarray],
    out_path: Path,
    alpha: float = 0.08,
    max_trajectories: int = 2000,
    title: Optional[str] = None,
    stop_at_sc: bool = False,
    aa_times: Optional[List[Optional[float]]] = None,
    max_year: Optional[float] = None,
) -> None:
    """Render AI software R&D uplift (ai_sw_progress_mult_ref_present_day) trajectories.

    Assumes times are decimal years and uplift values are dimensionless multipliers.

    Args:
        max_year: Optional maximum year for x-axis. Trajectories continue to edge if they extend beyond.
    """
    if len(trajectories) == 0:
        raise ValueError("No trajectories provided")

    # Replace non-positive and non-finite values with NaN
    min_uplift = 0.1
    max_uplift = 1e6
    cleaned: List[np.ndarray] = []
    for idx, t in enumerate(trajectories):
        arr = t.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        arr[arr <= 0] = np.nan
        arr = np.clip(arr, min_uplift, max_uplift)
        # If requested, mask values after this rollout's aa_time
        if stop_at_sc and aa_times is not None and idx < len(aa_times) and aa_times[idx] is not None:
            sc = aa_times[idx]
            if sc is not None:
                arr = arr.copy()
                arr[times > float(sc)] = np.nan
        cleaned.append(arr)

    # Compute median trajectory across rollouts
    stacked = np.vstack([t for t in cleaned[:max_trajectories]])
    median_traj = np.nanmedian(stacked, axis=0)

    plt.figure(figsize=(14, 8))

    # Draw all trajectories in blue
    num_plot = min(len(cleaned), max_trajectories)
    for i in range(num_plot):
        color = (0.2, 0.5, 0.7, alpha)
        plt.plot(times, cleaned[i], color=color, linewidth=1.0)

    # Central trajectory
    plt.plot(times, median_traj, color="tab:green", linestyle="--", linewidth=2.0, label="Central Trajectory")

    # Horizontal line for current uplift (assuming ~1x at present day)
    plt.axhline(1.0, color="red", linewidth=2.0, label="Current Uplift (1x)")

    # Vertical line for current time
    now_year = now_decimal_year()
    plt.axvline(now_year, color="tab:blue", linestyle="--", linewidth=1.75, label="Current Time")

    # Axes and formatting
    plt.yscale("log")
    plt.xlabel("Year")
    plt.ylabel("AI Software R&D Uplift (Multiplier)")

    # Set x-axis limits if max_year is specified
    if max_year is not None:
        plt.xlim(right=max_year)

    # Update title
    base_title = title or "AI Software R&D Uplift Trajectories\n(Historical development and future projections)"
    base_title += f"\n(n={len(trajectories)} trajectories)"
    plt.title(base_title)

    plt.grid(True, which="both", axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_trajectories(
    times: np.ndarray,
    trajectories: List[np.ndarray],
    out_path: Path,
    metric_name: str,
    alpha: float = 0.08,
    max_trajectories: int = 2000,
    title: Optional[str] = None,
    stop_at_sc: bool = False,
    aa_times: Optional[List[Optional[float]]] = None,
    mse_values: Optional[List[Optional[float]]] = None,
    shade_by_mse: bool = False,
    mse_threshold: Optional[float] = None,
    cutoff_year: Optional[float] = None,
    yscale: str = "auto",
) -> None:
    """Render arbitrary metric trajectories over time.

    Args:
        times: Array of decimal years
        trajectories: List of metric value arrays (one per rollout)
        out_path: Output PNG path
        metric_name: Name of the metric being plotted
        alpha: Transparency for individual trajectories
        max_trajectories: Maximum number of trajectories to draw
        title: Optional custom title
        stop_at_sc: Mask each trajectory after its own aa_time
        aa_times: List of aa_time values (one per rollout)
        mse_values: List of METR MSE values (one per rollout)
        shade_by_mse: Color trajectories by their METR MSE
        mse_threshold: Exclude trajectories with METR MSE above this threshold
        cutoff_year: Cut off x-axis at this year
        yscale: Y-axis scale - "auto" (default), "log", or "linear"
    """
    if len(trajectories) == 0:
        raise ValueError("No trajectories provided")

    # Filter trajectories by MSE threshold if specified
    if mse_threshold is not None and mse_values is not None:
        filtered_indices = []
        for i, mse in enumerate(mse_values):
            if mse is None or mse <= mse_threshold:
                filtered_indices.append(i)
        trajectories = [trajectories[i] for i in filtered_indices]
        if aa_times is not None:
            aa_times = [aa_times[i] for i in filtered_indices]
        if mse_values is not None:
            mse_values = [mse_values[i] for i in filtered_indices]
        print(f"Filtered to {len(trajectories)} trajectories with MSE <= {mse_threshold}")

    if len(trajectories) == 0:
        raise ValueError("No trajectories remain after MSE filtering")

    # Clean trajectories: replace non-finite values with NaN
    cleaned: List[np.ndarray] = []
    for idx, t in enumerate(trajectories):
        arr = t.astype(float)
        arr[~np.isfinite(arr)] = np.nan
        # If requested, mask values after this rollout's aa_time
        if stop_at_sc and aa_times is not None and idx < len(aa_times) and aa_times[idx] is not None:
            sc = aa_times[idx]
            if sc is not None:
                arr = arr.copy()
                arr[times > float(sc)] = np.nan
        cleaned.append(arr)

    # Compute median trajectory across rollouts
    stacked = np.vstack([t for t in cleaned[:max_trajectories]])
    median_traj = np.nanmedian(stacked, axis=0)

    plt.figure(figsize=(14, 8))

    # Determine colors for trajectories
    if shade_by_mse and mse_values is not None:
        # Compute color mapping based on MSE values
        valid_mse = [mse for mse in mse_values if mse is not None]
        if len(valid_mse) > 0:
            mse_min = np.percentile(valid_mse, 5)
            mse_max = np.percentile(valid_mse, 95)
            print(f"MSE range for coloring: {mse_min:.4f} to {mse_max:.4f}")
        else:
            mse_min = 0.0
            mse_max = 1.0
    else:
        valid_mse = []

    # Draw all trajectories
    num_plot = min(len(cleaned), max_trajectories)
    for i in range(num_plot):
        if shade_by_mse and mse_values is not None and i < len(mse_values) and mse_values[i] is not None:
            # Color based on MSE: green (low) to red (high)
            mse = mse_values[i]
            if len(valid_mse) > 0 and mse_max > mse_min:
                norm_mse = (mse - mse_min) / (mse_max - mse_min)
                norm_mse = np.clip(norm_mse, 0.0, 1.0)
            else:
                norm_mse = 0.5
            # Green (0.0) to Red (1.0) via Yellow
            if norm_mse < 0.5:
                r = norm_mse * 2
                g = 1.0
                b = 0.0
            else:
                r = 1.0
                g = 2.0 * (1.0 - norm_mse)
                b = 0.0
            color = (r, g, b, alpha)
        else:
            # Default blue color
            color = (0.2, 0.5, 0.7, alpha)
        plt.plot(times, cleaned[i], color=color, linewidth=1.0)

    # Central trajectory
    plt.plot(times, median_traj, color="tab:green", linestyle="--", linewidth=2.0, label="Median Trajectory")

    # Vertical line for current time
    now_year = now_decimal_year()
    plt.axvline(now_year, color="tab:blue", linestyle="--", linewidth=1.75, label="Current Time")

    # Determine if we should use log scale
    if yscale == "log":
        use_log_scale = True
    elif yscale == "linear":
        use_log_scale = False
    else:  # auto
        finite_vals = stacked[np.isfinite(stacked)]
        use_log_scale = False
        if len(finite_vals) > 0:
            val_min = np.min(finite_vals)
            val_max = np.max(finite_vals)
            # Use log scale if data spans more than 2 orders of magnitude and all values are positive
            if val_min > 0 and val_max / val_min > 100:
                use_log_scale = True

    if use_log_scale:
        plt.yscale("log")

    plt.xlabel("Year")
    plt.ylabel(metric_name.replace("_", " ").title())

    # Build title
    base_title = title or f"{metric_name.replace('_', ' ').title()} Trajectories"
    base_title += f"\n(n={len(trajectories)} trajectories"
    if mse_threshold is not None:
        base_title += f", MSE ≤ {mse_threshold:.3f}"
    base_title += ")"
    if shade_by_mse and len(valid_mse) > 0:
        base_title += f"\nColor: Green (low MSE) → Yellow → Red (high MSE)"
    plt.title(base_title)

    plt.grid(True, which="both", alpha=0.25)
    plt.legend(loc="best")

    # Set x-axis limit if cutoff_year is specified
    if cutoff_year is not None:
        plt.xlim(times[0], cutoff_year)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
