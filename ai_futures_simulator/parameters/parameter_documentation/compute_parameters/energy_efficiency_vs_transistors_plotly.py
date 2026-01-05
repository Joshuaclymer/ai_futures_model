"""
Generate energy efficiency vs transistors plot using Plotly with consistent styling.
Shows energy per TPP vs transistor density with pre/post Dennard scaling regression.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
sys.path.insert(0, '..')
from plotly_style import STYLE, apply_common_layout, save_plot, save_html

# Read data
df = pd.read_csv('epoch_data.csv')

# Read additional older chips for pre-Dennard data
try:
    additional = pd.read_csv('additional_older_chips.csv', sep='\t')

    # Extrapolate transistor counts based on year
    year_transistor_data = df[['Release year', 'Number of transistors in million']].dropna()
    log_transistors = np.log10(year_transistor_data['Number of transistors in million'])
    years = year_transistor_data['Release year']
    z = np.polyfit(years, log_transistors, 1)

    # Clean up the additional data
    additional['Power (W)'] = additional['Power (W)'].str.replace('~', '').astype(float)
    additional['FP32 Performance (TFLOPS)'] = additional['FP32 Performance (TFLOPS)'].str.replace('~', '').str.replace('*', '').astype(float)

    # Predict transistor counts for older chips
    additional['Number of transistors in million'] = 10**(z[0] * additional['Year'] + z[1])

    # Extrapolate die sizes based on year
    die_size_data = df[['Release year', 'Die Size in mm^2']].dropna()
    log_die_size = np.log10(die_size_data['Die Size in mm^2'])
    years_die = die_size_data['Release year']
    z_die = np.polyfit(years_die, log_die_size, 1)
    additional['Die Size in mm^2'] = 10**(z_die[0] * additional['Year'] + z_die[1])

    # Convert to match main dataframe format
    additional['TDP in W'] = additional['Power (W)']
    additional['FP32 Performance (FLOP/s)'] = additional['FP32 Performance (TFLOPS)'] * 1e12
    additional['FP16 Performance (FLOP/s)'] = np.nan  # These old chips don't have FP16
    additional['Name of the hardware'] = additional['Chip Name']
    additional['Release year'] = additional['Year']

    # Combine with main data
    df = pd.concat([df, additional[['Number of transistors in million', 'Die Size in mm^2', 'TDP in W', 'FP32 Performance (FLOP/s)',
                                     'FP16 Performance (FLOP/s)', 'Name of the hardware', 'Release year']]], ignore_index=True)
    print(f"Added {len(additional)} older chips with extrapolated transistor counts")
except FileNotFoundError:
    print("No additional_older_chips.csv found, using only main dataset")

# Process data
data = df[['Number of transistors in million', 'Die Size in mm^2', 'TDP in W',
           'FP16 Performance (FLOP/s)', 'Name of the hardware', 'Release year']].copy()

if 'FP32 Performance (FLOP/s)' in df.columns:
    data.loc[:, 'FP16 Performance (FLOP/s)'] = data['FP16 Performance (FLOP/s)'].fillna(df['FP32 Performance (FLOP/s)'])

data_clean = data.dropna().copy()
data_clean['Transistor Density (M/mm^2)'] = data_clean['Number of transistors in million'] / data_clean['Die Size in mm^2']
data_clean = data_clean[~data_clean['Name of the hardware'].str.contains('T4', na=False)].copy()

# Determine bit length
fp16_col = df['FP16 Performance (FLOP/s)']

def get_bit_length(idx):
    if idx in fp16_col.index and pd.notna(fp16_col.loc[idx]):
        return 16
    return 32

data_clean = data_clean.copy()
data_clean['bit_length'] = data_clean.index.map(get_bit_length)

# Calculate TPP and W/TPP
data_clean['TPP (TPP/s)'] = data_clean['FP16 Performance (FLOP/s)'] * data_clean['bit_length']
data_clean['Energy per TPP (W/TPP)'] = (data_clean['TDP in W'] / data_clean['TPP (TPP/s)']) * 1e12

# Create figure
fig = go.Figure()

# Separate by bit length
fp16_mask = data_clean['bit_length'] == 16
fp32_mask = data_clean['bit_length'] == 32

# FP16 points
fig.add_trace(go.Scatter(
    x=data_clean[fp16_mask]['Transistor Density (M/mm^2)'],
    y=data_clean[fp16_mask]['Energy per TPP (W/TPP)'],
    mode='markers',
    name='FP16',
    marker=dict(size=STYLE['marker_size_small'], color=STYLE['purple']),
    text=data_clean[fp16_mask]['Name of the hardware'],
    hovertemplate='%{text}<br>Density: %{x:.2f}<br>W/TPP: %{y:.4f}<extra></extra>'
))

# FP32 points
fig.add_trace(go.Scatter(
    x=data_clean[fp32_mask]['Transistor Density (M/mm^2)'],
    y=data_clean[fp32_mask]['Energy per TPP (W/TPP)'],
    mode='markers',
    name='FP32',
    marker=dict(size=STYLE['marker_size_small'], color=STYLE['blue']),
    text=data_clean[fp32_mask]['Name of the hardware'],
    hovertemplate='%{text}<br>Density: %{x:.2f}<br>W/TPP: %{y:.4f}<extra></extra>'
))

# Regression lines
outlier_mask = (data_clean['Number of transistors in million'] > 1000) & (data_clean['Energy per TPP (W/TPP)'] > 40)

# Pre-Dennard (<=2006)
pre_dennard = data_clean[(data_clean['Release year'] <= 2006) & ~outlier_mask]
if len(pre_dennard) > 1:
    log_density_pre = np.log10(pre_dennard['Transistor Density (M/mm^2)'])
    log_efficiency_pre = np.log10(pre_dennard['Energy per TPP (W/TPP)'])
    z_pre = np.polyfit(log_density_pre, log_efficiency_pre, 1)
    p_pre = np.poly1d(z_pre)

    x_trend_pre = np.logspace(np.log10(pre_dennard['Transistor Density (M/mm^2)'].min()),
                               np.log10(pre_dennard['Transistor Density (M/mm^2)'].max()), 100)
    y_trend_pre = 10**(p_pre(np.log10(x_trend_pre)))

    fig.add_trace(go.Scatter(
        x=x_trend_pre, y=y_trend_pre,
        mode='lines',
        name='Before Dennard end: y ~ x^' + f'{z_pre[0]:.2f}',
        line=dict(color=STYLE['teal'], width=STYLE['line_width'], dash='dash')
    ))

# Post-Dennard (>2006)
post_dennard = data_clean[(data_clean['Release year'] > 2006) & ~outlier_mask]
if len(post_dennard) > 1:
    log_density_post = np.log10(post_dennard['Transistor Density (M/mm^2)'])
    log_efficiency_post = np.log10(post_dennard['Energy per TPP (W/TPP)'])
    z_post = np.polyfit(log_density_post, log_efficiency_post, 1)
    p_post = np.poly1d(z_post)

    x_trend_post = np.logspace(np.log10(post_dennard['Transistor Density (M/mm^2)'].min()),
                                np.log10(post_dennard['Transistor Density (M/mm^2)'].max()), 100)
    y_trend_post = 10**(p_post(np.log10(x_trend_post)))

    fig.add_trace(go.Scatter(
        x=x_trend_post, y=y_trend_post,
        mode='lines',
        name='After Dennard end: y ~ x^' + f'{z_post[0]:.2f}',
        line=dict(color=STYLE['dark_teal'], width=STYLE['line_width'], dash='dash')
    ))

# Add vertical line for Dennard scaling end (2006)
year_density_data = df[['Release year', 'Number of transistors in million', 'Die Size in mm^2']].dropna()
year_density_data = year_density_data.copy()
year_density_data['Transistor Density (M/mm^2)'] = year_density_data['Number of transistors in million'] / year_density_data['Die Size in mm^2']
log_density = np.log10(year_density_data['Transistor Density (M/mm^2)'])
years = year_density_data['Release year']
z_year = np.polyfit(years, log_density, 1)
density_2006 = 10**(z_year[0] * 2006 + z_year[1])

# Calculate paper position for annotation (x ranges from log10(0.4) to log10(200))
x_min_log = np.log10(0.4)
x_max_log = np.log10(200)
x_paper = (np.log10(density_2006) - x_min_log) / (x_max_log - x_min_log)

fig.add_vline(x=density_2006, line=dict(color='gray', width=1.5, dash='dash'), opacity=0.6)
fig.add_annotation(
    x=x_paper, y=0.15,
    xref='paper', yref='paper',
    text='End of Dennard<br>Scaling (2006)',
    showarrow=False,
    bgcolor='rgba(200, 200, 200, 0.8)',
    borderpad=3,
    font=dict(size=STYLE['font_size_annotation'])
)

# Annotate H100
h100_data = data_clean[data_clean['Name of the hardware'].str.contains('H100', na=False)]
if len(h100_data) > 0:
    h100 = h100_data.iloc[0]
    fig.add_annotation(
        x=h100['Transistor Density (M/mm^2)'],
        y=h100['Energy per TPP (W/TPP)'],
        text='H100',
        showarrow=True,
        arrowhead=2, arrowsize=0.5, arrowwidth=1,
        ax=20, ay=-20,
        bgcolor='lightgreen',
        font=dict(size=STYLE['font_size_annotation'])
    )

apply_common_layout(
    fig,
    xaxis_title='Transistor Density (M transistors/mm²)',
    yaxis_title='Energy per TPP (W/TPP)',
    xaxis_log=True,
    yaxis_log=True,
    legend_position='top_right',
    show_legend=True
)

fig.update_xaxes(range=[np.log10(0.4), np.log10(200)])
fig.update_yaxes(range=[np.log10(0.01), np.log10(200)])

save_plot(fig, 'energy_efficiency_vs_transistors.png')
save_html(fig, 'energy_efficiency_vs_transistors.html')
