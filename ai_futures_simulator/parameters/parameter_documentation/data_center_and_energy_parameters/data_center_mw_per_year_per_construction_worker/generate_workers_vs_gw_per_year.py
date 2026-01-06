import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
sys.path.insert(0, '../..')
from plotly_style import STYLE, apply_common_layout, save_plot

# Read the workers vs build rate data (contains both workers and GW/year)
workers_df = pd.read_csv('data_center_workers_vs_build_rate.csv')

# Read the timelines data (for construction type lookup)
df = pd.read_csv('data_centers/data_center_timelines.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Create a mapping from workers CSV names to timeline names
name_mapping = {
    'Amazon Canton Mississippi (Project Atlas)': 'Amazon Canton Mississippi',
    'Amazon Ridgeland Mississippi (Project Atlas)': 'Amazon Ridgeland Mississippi',
    'Anthropic-Amazon Project Rainier New Carlisle Indiana': 'Anthropic-Amazon Project Rainier New Carlisle Indiana',
    'Crusoe Goodnight Texas (Goodnight AI Campus)': 'Crusoe Goodnight Texas',
    'Google New Albany Ohio': 'Google New Albany Ohio',
    'Meta Hyperion Holly Ridge Louisiana': 'Meta Hyperion Holly Ridge Louisiana',
    'Meta Prometheus New Albany Ohio': 'Meta Prometheus New Albany Ohio',
    'Microsoft Fairwater Mount Pleasant Wisconsin': 'Microsoft Fairwater Mount Pleasant Wisconsin',
    'Microsoft Project Firecracker Rome Georgia': 'Microsoft Fayetteville Georgia',
    'OpenAI-Oracle Stargate Abilene Texas': 'OpenAI-Oracle Stargate Abilene Texas',
    'xAI Colossus 1 Memphis Tennessee': 'xAI Colossus 1 Memphis Tennessee',
    'xAI Colossus 2 Memphis Tennessee': 'xAI Colossus 2 Memphis Tennessee',
    'ORNL Frontier Supercomputer Oak Ridge Tennessee': 'ORNL Frontier Supercomputer Oak Ridge Tennessee',
    'LLNL El Capitan Phase 2 Livermore California': 'LLNL El Capitan Phase 2 Livermore California'
}

# Build results from the CSV
results = []

for idx, row in workers_df.iterrows():
    worker_name = row['Data Center Name']
    worker_count = row['Number of Concurrent Workers']
    gw_per_year = row['Gigawatts per Year']

    # Skip if no data
    if pd.isna(worker_count) or pd.isna(gw_per_year):
        continue
    if worker_count == "Can't find data" or worker_count == "No specific data available" or worker_count == "":
        continue
    if gw_per_year == "Can't find data" or gw_per_year == "" or gw_per_year == "No specific data available":
        continue

    try:
        # Handle approximate values like "~700"
        worker_count_str = str(worker_count).replace('~', '').replace(',', '').strip()
        worker_count = int(worker_count_str)
        gw_per_year = float(gw_per_year)
        # Convert GW/year to MW/year
        mw_per_year = gw_per_year * 1000
    except (ValueError, TypeError):
        continue

    # Get timeline name from mapping
    timeline_name = name_mapping.get(worker_name, worker_name)

    # Get timeline data to determine construction type
    dc_data = df[df['Data center'] == timeline_name].copy()

    # Determine construction type
    construction_type = 'Traditional'
    if len(dc_data) > 0:
        if 'xAI Colossus' in timeline_name:
            construction_type = 'Retrofit'
        elif 'Meta Prometheus' in timeline_name or 'Meta Hyperion' in timeline_name:
            # Check for tents in construction status
            tent_mentions = dc_data[dc_data['Construction status'].str.contains('tent', case=False, na=False)]
            if len(tent_mentions) > 0:
                construction_type = 'Tent'

    results.append({
        'Data Center': timeline_name,
        'Workers': worker_count,
        'MW per Year': mw_per_year,
        'Construction Type': construction_type
    })

# Create dataframe
results_df = pd.DataFrame(results)

print("Data Centers with both worker and timeline data:")
print(results_df.to_string(index=False))

# Create scatter plot
fig = go.Figure()

# Add all scatter points at once (simpler style matching other plots)
fig.add_trace(go.Scatter(
    x=results_df['Workers'],
    y=results_df['MW per Year'],
    mode='markers',
    marker=dict(size=STYLE['marker_size'], color=STYLE['blue']),
    showlegend=False,
    text=results_df['Data Center'],
    hovertemplate='<b>%{text}</b><br>Workers: %{x:,.0f}<br>MW/year: %{y:.1f}<extra></extra>'
))

# Calculate and add regression line (forcing through origin)
if len(results_df) > 1:
    X = results_df['Workers'].values
    y = results_df['MW per Year'].values

    # Fit linear regression through origin (intercept = 0)
    slope = np.sum(X * y) / np.sum(X ** 2)
    intercept = 0

    # Calculate relative sigma for prediction intervals
    y_pred_points = slope * X
    residuals = y - y_pred_points

    # Calculate relative residuals
    relative_residuals = residuals / y_pred_points

    # Standard deviation of relative residuals (inherent scatter)
    relative_sigma_residuals = np.std(relative_residuals, ddof=1)

    # Standard error of the slope estimate
    residual_std = np.std(residuals, ddof=1)
    se_slope = residual_std / np.sqrt(np.sum(X ** 2))

    # Relative uncertainty from slope
    relative_sigma_slope = se_slope / slope

    # Combined relative sigma for prediction interval
    relative_sigma = np.sqrt(relative_sigma_residuals**2 + relative_sigma_slope**2)

    # Generate regression line from origin
    x_range = np.linspace(0, X.max(), 100)
    y_pred = slope * x_range

    # Calculate R² for regression through origin
    y_pred_points = slope * X
    ss_res = np.sum((y - y_pred_points) ** 2)
    ss_tot = np.sum(y ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        line=dict(color=STYLE['blue'], width=STYLE['line_width'], dash='dash'),
        name=f'Fit: y = {slope:.4f}x',
        showlegend=True
    ))

    print(f"\n\nRegression Results (through origin):")
    print(f"Slope (m): {slope:.6f} MW/year per worker")
    print(f"Standard error of slope: {se_slope:.6f}")
    print(f"Relative uncertainty in slope: {relative_sigma_slope:.4f} ({relative_sigma_slope*100:.2f}%)")
    print(f"\nRelative residual scatter: {relative_sigma_residuals:.4f} ({relative_sigma_residuals*100:.2f}%)")
    print(f"\nCombined Relative Sigma (for prediction intervals): {relative_sigma:.4f} ({relative_sigma*100:.2f}%)")
    print(f"  - This is sqrt(residual_scatter² + slope_uncertainty²)")
    print(f"\nSample size: {len(X)}")
    print(f"Intercept: {intercept:.6f} MW/year (forced to 0)")
    print(f"R²: {r2:.4f}")

# Apply common layout
apply_common_layout(
    fig,
    xaxis_title='Number of Concurrent Construction Workers',
    yaxis_title='Build Rate (MW per Year)',
    legend_position='top_left',
    show_legend=True
)


# Add rangemode for axes to start at zero
fig.update_xaxes(rangemode='tozero')
fig.update_yaxes(rangemode='tozero')

# Save to PNG and HTML
save_plot(fig, 'workers_vs_gw_per_year.png')

# For responsive HTML, remove fixed width/height and set to 100%
fig.update_layout(width=None, height=None, autosize=True)
fig.write_html(
    'workers_vs_gw_per_year.html',
    include_plotlyjs='cdn',
    full_html=True,
    config={'responsive': True},
    default_width='100%',
    default_height='100%'
)
print("Saved: workers_vs_gw_per_year.html")
