"""
Generate H100 equivalents per wafer vs process node plot using Plotly.
Shows how compute output scales with transistor density improvements.
"""

import numpy as np
import plotly.graph_objects as go
from scipy import stats
import sys
sys.path.insert(0, '../..')
from plotly_style import STYLE, apply_common_layout, save_plot

# Process node data
process_node = [3, 4, 5, 6, 7, 10, 12, 16, 22, 28, 40, 65, 90, 130, 180]
transistor_density = [215.6, 145.86, 137.6, 106.96, 90.64, 60.3, 33.8, 28.88, 16.50, 14.44, 7.22, 3.61, 1.80, 0.90, 0.45]

# H100 specs
h100_node = 4
h100_tpp = 63328
dies_per_wafer = 28

# Adjust for 2030 chip architectures (1.23x improvement per year from 2022)
architectural_improvement_factor = 1.23 ** 8  # ~5.596
TPP_per_transistor_density = 644 * architectural_improvement_factor

# Calculate TPP and H100 equivalents per wafer for each node
tpp = [td * TPP_per_transistor_density for td in transistor_density]
h100_equiv_per_wafer = [(dies_per_wafer * tpp_val) / h100_tpp for tpp_val in tpp]

# Power law fit: log(y) = a * log(x) + b
log_nodes = np.log10(process_node)
log_equiv = np.log10(h100_equiv_per_wafer)
slope, intercept, r_value, p_value, std_err = stats.linregress(log_nodes, log_equiv)

# Generate fit line
x_fit = np.logspace(np.log10(min(process_node)), np.log10(max(process_node)), 100)
y_fit = 10 ** (slope * np.log10(x_fit) + intercept)

# Create figure
fig = go.Figure()

# Data line (connecting points)
fig.add_trace(go.Scatter(
    x=process_node, y=h100_equiv_per_wafer,
    mode='lines',
    name='H100 equivalents',
    line=dict(color=STYLE['blue'], width=STYLE['line_width']),
))

# Data points
fig.add_trace(go.Scatter(
    x=process_node, y=h100_equiv_per_wafer,
    mode='markers',
    name='Process nodes',
    marker=dict(size=STYLE['marker_size'], color=STYLE['blue'], line=dict(color='white', width=STYLE['marker_line_width'])),
    hovertemplate='%{x}nm: %{y:.2f} H100e/wafer<extra></extra>'
))

# Power law fit line
fig.add_trace(go.Scatter(
    x=x_fit, y=y_fit,
    mode='lines',
    name=f'Power law fit (R2={r_value**2:.3f})',
    line=dict(color=STYLE['blue'], width=STYLE['line_width'], dash='dash'),
))

apply_common_layout(
    fig,
    xaxis_title='TSMC Process Node (nm)',
    yaxis_title='H100 Equivalents per Wafer',
    xaxis_log=True,
    yaxis_log=True,
    legend_position='top_left',
    show_legend=True
)

# Reverse x-axis (larger nodes on left, smaller on right)
fig.update_xaxes(autorange='reversed')

# Set specific x-axis tick values
fig.update_xaxes(
    tickmode='array',
    tickvals=[3, 4, 5, 6, 7, 10, 12, 16, 22, 28, 40, 65, 90, 130, 180],
    ticktext=['3', '4', '5', '6', '7', '10', '12', '16', '22', '28', '40', '65', '90', '130', '180']
)

save_plot(fig, 'h100_equiv_per_year_vs_process_node.png')

print(f"Power law exponent: {slope:.3f}")
print(f"R²: {r_value**2:.3f}")
