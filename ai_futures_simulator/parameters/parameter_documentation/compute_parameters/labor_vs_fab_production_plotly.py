"""
Generate labor vs fab production plot using Plotly with consistent styling.
Shows relationship between number of workers and wafer production.
"""

import numpy as np
import plotly.graph_objects as go
import sys
sys.path.insert(0, '..')
from plotly_style import STYLE, apply_common_layout, save_plot, save_html

# Data
wafers_per_month = [4000, 55000, 31500, 800000, 83000, 20000, 33000, 62500, 100000, 140000, 450000]
employees = [80, 1500, 3950, 31000, 11300, 2200, 2750, 3000, 3000, 8500, 10000]

# Calculate regression through origin
slope = np.sum(np.array(employees) * np.array(wafers_per_month)) / np.sum(np.array(employees)**2)
x_range = np.linspace(min(employees), max(employees), 100)
y_range = slope * x_range

# Create figure
fig = go.Figure()

# Scatter points
fig.add_trace(go.Scatter(
    x=employees, y=wafers_per_month,
    mode='markers',
    name='Operating Employees',
    marker=dict(size=STYLE['marker_size'], color=STYLE['blue'], line=dict(color='white', width=1))
))

# Regression line
fig.add_trace(go.Scatter(
    x=x_range, y=y_range,
    mode='lines',
    name=f'Operating fit (y = {slope:.1f}*x)',
    line=dict(color=STYLE['blue'], width=STYLE['line_width'], dash='dash')
))

apply_common_layout(
    fig,
    xaxis_title='Number of Workers',
    yaxis_title='Wafers per Month',
    xaxis_log=True,
    yaxis_log=True,
    legend_position='top_left',
    show_legend=True
)

save_plot(fig, 'labor_vs_production.png')
save_html(fig, 'labor_vs_production.html')

print(f"Slope (wafers per worker): {slope:.1f}")
