"""
Generate construction time plots using Plotly with consistent styling.
Shows fab construction time vs capacity with and without concealment adjustment.
"""

import numpy as np
import plotly.graph_objects as go
import sys
sys.path.insert(0, '../..')
from plotly_style import STYLE, apply_common_layout, save_plot

# Data
capacity_wspm_construction = [
    55000, 200000, 83000, 60000, 62500, 100000, 140000, 150000, 5000, 80000,
    4000, 4000, 20000, 20000, 30000, 30000
]
construction_months = [
    30, 26, 24, 36, 24, 30, 24, 26, 12, 27, 12, 11, 16, 19, 11, 15
]
country_construction = [
    'Singapore', 'South Korea', 'Taiwan', 'USA', 'USA', 'South Korea',
    'Singapore', 'South Korea', 'China', 'Europe', 'Europe', 'Taiwan',
    'USA', 'China', 'China', 'China'
]

construction_years = [m / 12 for m in construction_months]

# Calculate regression
log_capacity = np.log10(capacity_wspm_construction)
z = np.polyfit(log_capacity, construction_years, 1)
p = np.poly1d(z)
x_range_log = np.logspace(np.log10(min(capacity_wspm_construction)), np.log10(max(capacity_wspm_construction)), 100)
y_range_log = p(np.log10(x_range_log))

# ========== FIGURE 1: Original plot ==========
fig1 = go.Figure()

# Plot all points in single color (no country labels)
fig1.add_trace(go.Scatter(
    x=capacity_wspm_construction, y=construction_years,
    mode='markers',
    marker=dict(size=STYLE['marker_size'], color=STYLE['blue']),
    showlegend=False,
    hovertemplate='Capacity: %{x:,.0f}<br>Time: %{y:.2f} years<extra></extra>'
))

# Regression line
fig1.add_trace(go.Scatter(
    x=x_range_log.tolist(), y=y_range_log.tolist(),
    mode='lines',
    name=f'Fit: y = {z[0]:.3f}*log10(x) + {z[1]:.3f}',
    line=dict(color=STYLE['blue'], width=STYLE['line_width'], dash='dash')
))

apply_common_layout(
    fig1,
    xaxis_title='Fab Production Capacity (Wafers per Month)',
    yaxis_title='Construction Time (Years)',
    xaxis_log=True,
    legend_position='top_left',
    show_legend=True
)

fig1.update_xaxes(
    tickvals=[5000, 10000, 20000, 50000, 100000, 200000],
    ticktext=['5K', '10K', '20K', '50K', '100K', '200K']
)
fig1.update_yaxes(rangemode='tozero')

save_plot(fig1, 'construction_time_plot.png')

# ========== FIGURE 2: Adjusted plot ==========
construction_years_transformed = [t * 1.5 for t in construction_years]
z_concealment = np.polyfit(log_capacity, construction_years_transformed, 1)
p_concealment = np.poly1d(z_concealment)
y_range_log_concealment = p_concealment(np.log10(x_range_log))

fig2 = go.Figure()

# Original points in gray
fig2.add_trace(go.Scatter(
    x=capacity_wspm_construction, y=construction_years,
    mode='markers',
    name='Original data',
    marker=dict(size=STYLE['marker_size'], color=STYLE['gray'])
))

# Transformed points in red
fig2.add_trace(go.Scatter(
    x=capacity_wspm_construction, y=construction_years_transformed,
    mode='markers',
    name='1.5x construction time',
    marker=dict(size=STYLE['marker_size'], color=STYLE['red'])
))

# Original regression line
fig2.add_trace(go.Scatter(
    x=x_range_log.tolist(), y=y_range_log.tolist(),
    mode='lines',
    name=f'Fit: y = {z[0]:.3f}*log10(x) + {z[1]:.3f}',
    line=dict(color=STYLE['gray'], width=STYLE['line_width'], dash='dash')
))

# Concealment regression line
fig2.add_trace(go.Scatter(
    x=x_range_log.tolist(), y=y_range_log_concealment.tolist(),
    mode='lines',
    name=f'With concealment: y = {z_concealment[0]:.3f}*log10(x) + {z_concealment[1]:.3f}',
    line=dict(color=STYLE['red'], width=STYLE['line_width'])
))

apply_common_layout(
    fig2,
    xaxis_title='Fab Production Capacity (Wafers per Month)',
    yaxis_title='Construction Time (Years)',
    xaxis_log=True,
    legend_position='top_left',
    show_legend=True
)

fig2.update_xaxes(
    tickvals=[5000, 10000, 20000, 50000, 100000, 200000],
    ticktext=['5K', '10K', '20K', '50K', '100K', '200K']
)
fig2.update_yaxes(rangemode='tozero')

save_plot(fig2, 'construction_time_plot_adjusted.png')

print(f"\nConstruction Time Regression:")
print(f"  Slope: {z[0]:.3f}")
print(f"  Intercept: {z[1]:.3f}")
