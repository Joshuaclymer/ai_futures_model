"""
Generate cost of fab plot using Plotly with consistent styling.
Shows fab cost vs production capacity by process node.
"""

import numpy as np
import plotly.graph_objects as go
import sys
sys.path.insert(0, '..')
from plotly_style import STYLE, apply_common_layout, save_plot, save_html

# Color scheme for process nodes
NODE_COLORS = {
    "d7nm": STYLE['purple'],
    "10-14nm": STYLE['blue'],
    "16-28nm": STYLE['teal'],
    "e32nm": STYLE['colors']['accent'],
}

# Data
fabs = [
    {"cost_billion_usd": 10.0, "capacity_wafers": 70000, "node_nm": 14},
    {"cost_billion_usd": 3.59, "capacity_wafers": 41000, "node_nm": 28},
    {"cost_billion_usd": 6.7, "capacity_wafers": 60000, "node_nm": 12},
    {"cost_billion_usd": 17.0, "capacity_wafers": 120000, "node_nm": 5},
    {"cost_billion_usd": 9.3, "capacity_wafers": 100000, "node_nm": 20},
    {"cost_billion_usd": 3.75, "capacity_wafers": 45500, "node_nm": 16},
    {"cost_billion_usd": 6.0, "capacity_wafers": 40000, "node_nm": 20},
    {"cost_billion_usd": 3.1, "capacity_wafers": 55000, "node_nm": 16},
    {"cost_billion_usd": 3.6, "capacity_wafers": 35000, "node_nm": 22},
]

costs = [fab["cost_billion_usd"] for fab in fabs]
capacities = [fab["capacity_wafers"] for fab in fabs]
process_nodes = [fab["node_nm"] for fab in fabs]

def categorize_node(node_nm):
    if node_nm <= 7:
        return "d7nm"
    elif node_nm <= 14:
        return "10-14nm"
    elif node_nm <= 28:
        return "16-28nm"
    else:
        return "e32nm"

node_categories = [categorize_node(node) for node in process_nodes]

# Calculate regression through origin
alpha = np.sum(np.array(capacities) * np.array(costs)) / np.sum(np.array(capacities)**2)
x_range = np.linspace(0, max(capacities), 100)
y_range = alpha * x_range

# Create figure
fig = go.Figure()

# Plot points by category
for category in set(node_categories):
    mask = [c == category for c in node_categories]
    caps = [cap for cap, m in zip(capacities, mask) if m]
    cost = [c for c, m in zip(costs, mask) if m]
    fig.add_trace(go.Scatter(
        x=caps, y=cost,
        mode='markers',
        name=category,
        marker=dict(size=STYLE['marker_size'], color=NODE_COLORS.get(category, STYLE['gray']))
    ))

# Regression line
fig.add_trace(go.Scatter(
    x=x_range.tolist(), y=y_range.tolist(),
    mode='lines',
    name=f'Fit: y = {alpha:.6f}*x',
    line=dict(color=STYLE['gray'], width=STYLE['line_width'], dash='dash')
))

apply_common_layout(
    fig,
    xaxis_title='Fab Production Capacity (Wafers per Month)',
    yaxis_title='Cost (Billions USD)',
    legend_position='top_left',
    show_legend=True
)

fig.update_xaxes(rangemode='tozero')
fig.update_yaxes(rangemode='tozero')

save_plot(fig, 'cost_of_fab_plot.png')
save_html(fig, 'cost_of_fab_plot.html')

print(f"Cost per capacity (alpha): {alpha:.6f} B$/wafer/month")
