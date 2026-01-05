"""
Generate lithography sales plot using Plotly with consistent styling.
Compares DUV vs EUV scanner sales trajectories.
"""

import numpy as np
import plotly.graph_objects as go
import sys
sys.path.insert(0, '..')
from plotly_style import STYLE, apply_common_layout, save_plot, save_html

# DUV Immersion Systems Data
duv_start_year = 2006
duv_years = [2006, 2007, 2008, 2009, 2010, 2011]
duv_sales = [23, 35, 56, 35, 90, 101]
duv_years_relative = [year - duv_start_year for year in duv_years]

# EUV Systems Data
euv_start_year = 2019
euv_years = [2019, 2020, 2021, 2022, 2023, 2024]
euv_sales = [26, 31, 42, 40, 53, 44]
euv_years_relative = [year - euv_start_year for year in euv_years]

# Estimated PRC Rampup Data
prc_years_relative = [0, 5]
prc_sales = [20, 100]
slope = (prc_sales[1] - prc_sales[0]) / (prc_years_relative[1] - prc_years_relative[0])
intercept = prc_sales[0]

# Create figure
fig = go.Figure()

# DUV line
fig.add_trace(go.Scatter(
    x=duv_years_relative, y=duv_sales,
    mode='lines+markers',
    name=f'DUV Immersion Systems (starting {duv_start_year})',
    line=dict(color=STYLE['blue'], width=STYLE['line_width']),
    marker=dict(size=STYLE['marker_size'], color=STYLE['blue'])
))

# EUV line
fig.add_trace(go.Scatter(
    x=euv_years_relative, y=euv_sales,
    mode='lines+markers',
    name=f'EUV Systems (starting {euv_start_year})',
    line=dict(color=STYLE['purple'], width=STYLE['line_width']),
    marker=dict(size=STYLE['marker_size'], symbol='square', color=STYLE['purple'])
))

# PRC rampup estimate
fig.add_trace(go.Scatter(
    x=prc_years_relative, y=prc_sales,
    mode='lines',
    name=f'Estimated PRC Rampup (y = {slope:.1f}x + {intercept:.1f})',
    line=dict(color=STYLE['teal'], width=STYLE['line_width'], dash='dash')
))

apply_common_layout(
    fig,
    xaxis_title='Years After First High Volume Production',
    yaxis_title='Lithography Scanners Sold by ASML',
    legend_position='top_left',
    show_legend=True
)

save_plot(fig, 'lithography_sales_plot.png')
save_html(fig, 'lithography_sales_plot.html')
