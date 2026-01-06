import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Read the capabilities data
with open('capabilities.json', 'r') as f:
    data = json.load(f)

# Get China's data
china_data = data['china']

# Get all tool types and nm nodes
tool_types = ['DUV_scanner', 'Photomask_writer', 'Photoresist', 'Overlay_metrology',
              'Ion_implanter', 'Electroplating', 'deposition', 'etch']
nm_nodes = ['180nm', '90nm', '45nm', '28nm', '14nm', '7nm']

# Column display names (edit these to customize the table headers)
column_display_names = {
    'DUV_scanner': 'DUV\nscanner',
    'Photomask_writer': 'Photomask\nwriter',
    'Photoresist': 'Photoresist',
    'Overlay_metrology': 'Overlay\nmetrology',
    'Ion_implanter': 'Ion\nimplanter',
    'Electroplating': 'Electro-\nplating',
    'deposition': 'Deposition',
    'etch': 'Etch'
}

# Reverse nm_nodes so 7nm is first
nm_nodes_reversed = nm_nodes[::-1]

# First, find the most advanced node where each tool/capability first appears
# and the earliest date for each capability type
first_appearance = {}
earliest_dates = {}
for tool in tool_types:
    first_appearance[tool] = {'HVM': None, 'Pilot': None}
    earliest_dates[tool] = {'HVM': None, 'Pilot': None}

    # Check from most advanced to least advanced for first appearance at most advanced node
    for nm in nm_nodes_reversed:
        hvm_value = china_data['hvm'].get(nm, {}).get(tool)
        pilot_value = china_data['pilot'].get(nm, {}).get(tool)

        if hvm_value is not None and first_appearance[tool]['HVM'] is None:
            first_appearance[tool]['HVM'] = (nm, hvm_value)
        if pilot_value is not None and first_appearance[tool]['Pilot'] is None:
            first_appearance[tool]['Pilot'] = (nm, pilot_value)

    # Find earliest dates across all nodes
    for nm in nm_nodes:
        hvm_value = china_data['hvm'].get(nm, {}).get(tool)
        pilot_value = china_data['pilot'].get(nm, {}).get(tool)

        if hvm_value is not None:
            if earliest_dates[tool]['HVM'] is None or hvm_value < earliest_dates[tool]['HVM']:
                earliest_dates[tool]['HVM'] = hvm_value
        if pilot_value is not None:
            if earliest_dates[tool]['Pilot'] is None or pilot_value < earliest_dates[tool]['Pilot']:
                earliest_dates[tool]['Pilot'] = pilot_value

# Create the table data with capability propagation and dates
table_data = []
for nm in nm_nodes_reversed:
    row = {'nm': nm}
    for tool in tool_types:
        # Start with current node capability
        hvm_value = china_data['hvm'].get(nm, {}).get(tool)
        pilot_value = china_data['pilot'].get(nm, {}).get(tool)

        if hvm_value is not None:
            capability = 'Production'
            hvm_date = hvm_value
        elif pilot_value is not None:
            capability = 'Pilot'
            hvm_date = None
        else:
            capability = 'None'
            hvm_date = None

        pilot_date = pilot_value

        # Check more advanced nodes (smaller nm values)
        current_nm_value = int(nm.replace('nm', ''))
        for check_nm in nm_nodes:
            check_nm_value = int(check_nm.replace('nm', ''))
            # Only check nodes that are more advanced (smaller nm)
            if check_nm_value < current_nm_value:
                check_hvm = china_data['hvm'].get(check_nm, {}).get(tool)
                check_pilot = china_data['pilot'].get(check_nm, {}).get(tool)

                # If more advanced node has Production, this should be at least Production
                if check_hvm is not None:
                    capability = 'Production'
                    hvm_date = None  # Don't show date for propagated capabilities
                    pilot_date = None
                    break
                # If more advanced node has Pilot and current is None, upgrade to Pilot
                elif check_pilot is not None and capability == 'None':
                    capability = 'Pilot'
                    pilot_date = None  # Don't show date for propagated capabilities

        # Format the cell text with dates
        if capability == 'Production':
            # Check if this is where Production first appears
            is_first_hvm = (first_appearance[tool]['HVM'] and
                           first_appearance[tool]['HVM'][0] == nm)

            if is_first_hvm and hvm_date is not None:
                row[tool] = f'Production\n({int(hvm_date)})'
            else:
                row[tool] = 'Production'
        elif capability == 'Pilot':
            is_first_pilot = (first_appearance[tool]['Pilot'] and
                             first_appearance[tool]['Pilot'][0] == nm)
            if is_first_pilot and pilot_date is not None:
                row[tool] = f'Pilot\n({int(pilot_date)})'
            else:
                row[tool] = 'Pilot'
        else:
            row[tool] = 'None'

    table_data.append(row)

# Create DataFrame
df = pd.DataFrame(table_data)

# Set nm as index
df.set_index('nm', inplace=True)

# Print the table
print("\nChina's Domestic Semiconductor Manufacturing Equipment Capabilities")
print("=" * 100)
print(df.to_string())
print("\n")

# Also save as CSV
df.to_csv('china_capabilities_table.csv')
print("Table saved to china_capabilities_table.csv")

# Create color-coded visualization
fig, ax = plt.subplots(figsize=(7, 3))
fig.patch.set_alpha(0)  # Transparent figure background
ax.patch.set_alpha(0)   # Transparent axes background
ax.axis('tight')
ax.axis('off')

# Define colors for each capability level (softer colors)
color_map = {
    'Production': '#a8e6cf',    # Soft green
    'Pilot': '#ffd3b6',  # Soft orange
    'None': '#ffaaa5'    # Soft red
}

# Create cell colors based on values (extract capability from text)
cell_colors = []
for i in range(len(df)):
    row_colors = []
    for j in range(len(df.columns)):
        value = df.iloc[i, j]
        # Extract capability type (Production, Pilot, or None) from the cell text
        if 'Production' in value:
            row_colors.append(color_map['Production'])
        elif 'Pilot' in value:
            row_colors.append(color_map['Pilot'])
        else:
            row_colors.append(color_map['None'])
    cell_colors.append(row_colors)

# Format column names using the display names dictionary
col_labels = [column_display_names.get(col, col) for col in df.columns]

# Create the table
table = ax.table(cellText=df.values,
                rowLabels=df.index,
                colLabels=col_labels,
                cellColours=cell_colors,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.5)

# Style header and row labels
for i in range(len(col_labels)):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

for i in range(len(df)):
    cell = table[(i+1, -1)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Create legend
legend_elements = [
    mpatches.Patch(facecolor=color_map['Production'], label='Production (High Volume Manufacturing)'),
    mpatches.Patch(facecolor=color_map['Pilot'], label='Pilot (Limited Production)'),
    mpatches.Patch(facecolor=color_map['None'], label='None (No Capability)')
]
# ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.05), ncol=3, fontsize=7)

plt.tight_layout()
plt.savefig('china_capabilities_table.png', dpi=300, bbox_inches='tight', transparent=True)
print("Color-coded table saved to china_capabilities_table.png")
