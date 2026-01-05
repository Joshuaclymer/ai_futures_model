"""
Generate transistor density over time plot using Plotly with consistent styling.
Shows transistor density evolution with regression line.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import sys
sys.path.insert(0, '..')
from plotly_style import STYLE, apply_common_layout, save_plot, save_html

# Read data
df = pd.read_csv('epoch_data.csv')

# Filter and process
df_filtered = df[['Name of the hardware', 'Release year', 'Number of transistors in million', 'Die Size in mm^2']].copy()
df_filtered.columns = ['name', 'year', 'transistors_million', 'die_size']
df_filtered = df_filtered.dropna()
df_filtered['density'] = df_filtered['transistors_million'] / df_filtered['die_size']
df_filtered['log_density'] = np.log10(df_filtered['density'])

# Regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df_filtered['year'], df_filtered['log_density'])
years_range = np.linspace(df_filtered['year'].min(), df_filtered['year'].max(), 100)
log_regression_line = slope * years_range + intercept
regression_line = 10 ** log_regression_line
annual_rate = 10 ** slope

# Create figure
fig = go.Figure()

# Data points
fig.add_trace(go.Scatter(
    x=df_filtered['year'], y=df_filtered['density'],
    mode='markers',
    name='Actual Data',
    marker=dict(size=STYLE['marker_size'], color=STYLE['blue'], line=dict(color='white', width=STYLE['marker_line_width'])),
    text=df_filtered['name'],
    hovertemplate='%{text}<br>Year: %{x}<br>Density: %{y:.2f} M/mm2<extra></extra>'
))

# Regression line
fig.add_trace(go.Scatter(
    x=years_range, y=regression_line,
    mode='lines',
    name=f'Regression (R2={r_value**2:.3f})',
    line=dict(color=STYLE['blue'], width=STYLE['line_width'], dash='dash')
))

# Annotate A100 and H100
for idx, row in df_filtered.iterrows():
    if 'A100' in row['name'] or 'H100' in row['name']:
        fig.add_annotation(
            x=row['year'], y=row['density'],
            text=row['name'],
            showarrow=True,
            arrowhead=2,
            arrowsize=0.5,
            arrowwidth=1,
            ax=20, ay=-20,
            font=dict(size=STYLE['font_size_annotation'])
        )

# Add equation annotation
fig.add_annotation(
    x=0.05, y=0.95,
    xref='paper', yref='paper',
    text=f'log10(density) = {slope:.4f}*year + {intercept:.4f}<br>Annual increase: {annual_rate:.3f}x',
    showarrow=False,
    bgcolor='rgba(255, 255, 255, 0.9)',
    bordercolor=STYLE['axis_line_color'], borderwidth=1,
    font=dict(size=STYLE['font_size_tick']),
    align='left'
)

apply_common_layout(
    fig,
    xaxis_title='Year',
    yaxis_title='Transistor Density (M transistors/mm²)',
    yaxis_log=True,
    legend_position='bottom_right',
    show_legend=True
)

fig.update_yaxes(range=[np.log10(1), np.log10(200)])

save_plot(fig, 'transistor_density_over_time.png')
save_html(fig, 'transistor_density_over_time.html')

print(f"Annual rate of increase: {annual_rate:.3f}x")
print(f"R²: {r_value**2:.4f}")
