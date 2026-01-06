"""
Generate construction time vs labor plot using Plotly with consistent styling.
Shows how construction time varies with labor supply for different fab capacities.
"""

import numpy as np
import plotly.graph_objects as go
import sys
sys.path.insert(0, '../..')
from plotly_style import STYLE, apply_common_layout, save_plot

# Constants from the model
WORKERS_PER_BILLION_USD = 100  # Concurrent workers needed per $1B USD
COST_PER_CAPACITY = 0.000141  # B$/wafer/month from cost_of_fab_plot.py

# Regression coefficients from construction_time_plot.py (with concealment)
SLOPE_CONCEALMENT = 0.773
INTERCEPT_CONCEALMENT = -1.456

def estimate_cost(capacity_wafers_per_month):
    """Estimate fab cost in billions USD based on production capacity"""
    return COST_PER_CAPACITY * capacity_wafers_per_month

def required_labor(capacity_wafers_per_month):
    """Calculate required concurrent workers for standard construction timeline"""
    cost_billion = estimate_cost(capacity_wafers_per_month)
    return WORKERS_PER_BILLION_USD * cost_billion

def baseline_construction_time(capacity_wafers_per_month):
    """Calculate baseline construction time in years based on capacity"""
    log_capacity = np.log10(capacity_wafers_per_month)
    return SLOPE_CONCEALMENT * log_capacity + INTERCEPT_CONCEALMENT

def construction_time(capacity_wafers_per_month, actual_labor):
    """Calculate construction time in years based on labor supply."""
    baseline_time = baseline_construction_time(capacity_wafers_per_month)
    req_labor = required_labor(capacity_wafers_per_month)

    if actual_labor < req_labor:
        return baseline_time * (req_labor / actual_labor)
    else:
        return baseline_time

# Define three different production capacities to plot
capacities = [5000, 25000, 120000]  # wafers per month
capacity_labels = ['5k wafers/month', '25k wafers/month', '120k wafers/month']
colors = [STYLE['blue'], STYLE['purple'], STYLE['teal']]

# Create labor range
max_capacity = max(capacities)
max_required = required_labor(max_capacity)
labor_range = np.linspace(1, 3 * max_required, 500)

# Create figure
fig = go.Figure()

# Plot construction time vs labor for each capacity
for capacity, label, color in zip(capacities, capacity_labels, colors):
    req_labor = required_labor(capacity)
    times = [construction_time(capacity, labor) for labor in labor_range]

    fig.add_trace(go.Scatter(
        x=labor_range, y=times,
        mode='lines',
        name=label,
        line=dict(color=color, width=STYLE['line_width']),
        hovertemplate='Workers: %{x:.0f}<br>Time: %{y:.2f} years<extra></extra>'
    ))

apply_common_layout(
    fig,
    xaxis_title='Construction Labor',
    yaxis_title='Construction Time (Years)',
    legend_position='top_right',
    show_legend=True
)

fig.update_xaxes(rangemode='tozero')
fig.update_yaxes(range=[0, 15])

save_plot(fig, 'construction_time_vs_labor.png')
