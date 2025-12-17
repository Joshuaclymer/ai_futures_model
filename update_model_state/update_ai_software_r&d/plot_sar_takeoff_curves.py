#!/usr/bin/env python3
"""
Plot AI R&D progress multiplier (takeoff curves) for different human labor drop-off scenarios.

This script runs the progress model with CSV files where human labor
drops to 5 after different dates:
- input_data_human_labor_drops_2026.csv: Human labor drops to 5 in November 2026
- input_data_human_labor_drops_2027.csv: Human labor drops to 5 in November 2027
- input_data_human_labor_drops_2028.csv: Human labor drops to 5 in November 2028
- input_data_human_labor_drops_2029.csv: Human labor drops to 5 in November 2029
- input_data_human_labor_drops_2030.csv: Human labor drops to 5 in September 2030

It plots the ai_sw_progress_mult_ref_present_day metric (AI R&D progress multiplier
referenced to present day resources).
"""

import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import model_config as cfg
from progress_model import ProgressModel, Parameters, TimeSeriesData


def load_time_series(csv_path: str) -> TimeSeriesData:
    """Load time series from CSV."""
    df = pd.read_csv(csv_path)
    return TimeSeriesData(
        time=df['time'].values,
        L_HUMAN=df['L_HUMAN'].values,
        inference_compute=df['inference_compute'].values,
        experiment_compute=df['experiment_compute'].values,
        training_compute_growth_rate=df['training_compute_growth_rate'].values
    )


def run_model(csv_path: str, time_range: list) -> tuple:
    """
    Run the progress model with the given CSV file.

    Returns:
        tuple: (times, ai_sw_progress_mult_ref_present_day)
    """
    print(f"Loading time series from {csv_path}...")
    time_series = load_time_series(csv_path)

    # Create parameters (use linear interpolation)
    params_dict = copy.deepcopy(cfg.DEFAULT_PARAMETERS)
    params_dict['automation_interp_type'] = 'linear'
    params = Parameters(**{k: v for k, v in params_dict.items() if k in Parameters.__dataclass_fields__})

    # Create and run model
    print(f"Running model for {csv_path}...")
    model = ProgressModel(params, time_series)
    times, progress, rs = model.compute_progress_trajectory(time_range, initial_progress=0.0)

    # Get the AI R&D progress multiplier
    results = model.results
    ai_sw_progress_mult = np.array(results.get('ai_sw_progress_mult_ref_present_day', np.ones_like(times)))

    return times, ai_sw_progress_mult


def main():
    # CSV files and their labels
    scenarios = [
        ("input_data_human_labor_drops_2026.csv", "Human labor drops to 5 in Nov 2026", "#d62728"),
        ("input_data_human_labor_drops_2027.csv", "Human labor drops to 5 in Nov 2027", "#9467bd"),
        ("input_data_human_labor_drops_2028.csv", "Human labor drops to 5 in Nov 2028", "#1f77b4"),
        ("input_data_human_labor_drops_2029.csv", "Human labor drops to 5 in Nov 2029", "#ff7f0e"),
        ("input_data_human_labor_drops_2030.csv", "Human labor drops to 5 in Sep 2030", "#2ca02c"),
        ("input_data_human_labor_never_drops.csv", "Human labor never drops", "#7f7f7f"),
    ]

    # Time range for simulation
    time_range = [2018.0, 2040.0]

    # Create Plotly figure
    fig = go.Figure()

    for csv_file, label, color in scenarios:
        try:
            times, ai_sw_progress_mult = run_model(csv_file, time_range)

            fig.add_trace(go.Scatter(
                x=times,
                y=ai_sw_progress_mult,
                mode='lines',
                name=label,
                line=dict(color=color, width=2),
            ))
            print(f"Successfully ran model for {label}")
        except Exception as e:
            print(f"Error running model for {csv_file}: {e}")

    # Update layout
    fig.update_layout(
        title="AI R&D Progress Multiplier by Human Labor Drop Date",
        xaxis_title="Year",
        yaxis_title="AI R&D Progress Multiplier (ref present day)",
        yaxis_type="log",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template="plotly_white",
        height=600,
        width=1000,
    )

    # Add horizontal line at y=1 (baseline)
    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)

    # Save as HTML
    output_file = "sar_takeoff_curves.html"
    fig.write_html(output_file)
    print(f"\nPlot saved to {output_file}")

    # Also show the plot
    fig.show()


if __name__ == "__main__":
    main()
