"""
Generate TPP per die area plot using Plotly with consistent styling.
Shows TPP per die area over time with state-of-the-art trend.
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

# Performance columns ordered from lowest to highest bit length
perf_columns_ordered = [
    ('INT4 Tensor', 'INT4 Tensor Performance (OP/s)', 4),
    ('INT8 Tensor', 'INT8 Tensor Performance (OP/s)', 8),
    ('FP8 Tensor', 'FP8 Tensor Performance (FLOP/s)', 8),
    ('FP16', 'FP16 Performance (FLOP/s)', 16),
    ('FP16 Tensor', 'FP16 Tensor Performance (FLOP/s)', 16),
    ('INT16', 'INT16 Performance (OP/s)', 16),
    ('FP/TF32 Tensor', 'FP/TF32 Tensor Performance (FLOP/s)', 32),
    ('FP32', 'FP32 Performance (FLOP/s)', 32),
    ('FP64', 'FP64 Performance (FLOP/s)', 64),
]

die_size_col = 'Die Size in mm^2'

def get_lowest_bitlength_tpp(row):
    fp32_perf = row.get('FP32 Performance (FLOP/s)', None)
    for perf_type, perf_col, bit_length in perf_columns_ordered:
        if perf_col not in row:
            continue
        perf_value = row[perf_col]
        if pd.notna(perf_value) and perf_value > 0:
            if perf_type == 'FP16' and pd.notna(fp32_perf) and fp32_perf > 0:
                if perf_value < 0.1 * fp32_perf:
                    continue
            tpp = 2 * perf_value * bit_length / 1e12
            return tpp, bit_length, perf_type
    return None, None, None

# Get necessary columns
columns_needed = ['Name of the hardware', 'Release year', die_size_col]
columns_needed += [col for _, col, _ in perf_columns_ordered if col in df.columns]
valid_data = df[columns_needed].dropna(subset=[die_size_col, 'Release year'])

# Process each chip
results = []
for i, row in valid_data.iterrows():
    tpp, bit_length, perf_type = get_lowest_bitlength_tpp(row)
    if tpp is None:
        continue
    tpp_per_area = tpp / row[die_size_col]
    results.append({
        'name': row['Name of the hardware'],
        'year': row['Release year'],
        'tpp_per_area': tpp_per_area,
        'bit_length': bit_length,
        'perf_type': perf_type
    })

results_df = pd.DataFrame(results)

# Find state-of-the-art chips
sota_chips = []
for year in sorted(results_df['year'].unique()):
    chips_this_year = results_df[results_df['year'] == year]
    chips_previous_years = results_df[results_df['year'] < year]
    max_tpp_previous = chips_previous_years['tpp_per_area'].max() if len(chips_previous_years) > 0 else 0
    max_tpp_this_year = chips_this_year['tpp_per_area'].max()
    if max_tpp_this_year > max_tpp_previous:
        sota_this_year = chips_this_year[chips_this_year['tpp_per_area'] == max_tpp_this_year]
        sota_chips.extend(sota_this_year.index.tolist())

sota_df = results_df.loc[sota_chips]

# Create figure
fig = go.Figure()

# All chips
fig.add_trace(go.Scatter(
    x=results_df['year'], y=results_df['tpp_per_area'],
    mode='markers',
    name='All chips',
    marker=dict(size=STYLE['marker_size_small'], color=STYLE['blue']),
    text=results_df['name'],
    hovertemplate='%{text}<br>Year: %{x}<br>TPP/Area: %{y:.2f}<extra></extra>'
))

# State-of-the-art chips
fig.add_trace(go.Scatter(
    x=sota_df['year'], y=sota_df['tpp_per_area'],
    mode='markers',
    name='State-of-the-art',
    marker=dict(size=STYLE['marker_size_small'], color=STYLE['teal']),
    text=sota_df['name'],
    hovertemplate='%{text}<br>Year: %{x}<br>TPP/Area: %{y:.2f}<extra></extra>'
))

# Regression through SOTA chips
if len(sota_df) > 1:
    log_tpp = np.log10(sota_df['tpp_per_area'])
    years = sota_df['year'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, log_tpp)
    annual_multiplier = 10 ** slope
    year_range = np.linspace(years.min(), years.max(), 100)
    log_tpp_fit = slope * year_range + intercept
    tpp_fit = 10 ** log_tpp_fit

    fig.add_trace(go.Scatter(
        x=year_range, y=tpp_fit,
        mode='lines',
        name=f'Trend ({annual_multiplier:.2f}x per year, R2={r_value**2:.3f})',
        line=dict(color=STYLE['teal'], width=STYLE['line_width'], dash='dash')
    ))

apply_common_layout(
    fig,
    xaxis_title='Release Year',
    yaxis_title='TPP/Die Area (Tera-ops/mm2)',
    yaxis_log=True,
    legend_position='top_left',
    show_legend=True
)

save_plot(fig, 'tpp_per_die_area.png')
save_html(fig, 'tpp_per_die_area.html')

if len(sota_df) > 1:
    print(f"Annual multiplier: {annual_multiplier:.2f}x")
    print(f"R2: {r_value**2:.4f}")
