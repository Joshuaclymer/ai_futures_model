"""
Centralized Plotly style configuration for consistent plot styling across all analysis scripts.

Usage:
    from analysis.plotly_style import STYLE, apply_common_layout
    # or if running from within analysis folder:
    import sys
    sys.path.insert(0, '..')
    from plotly_style import STYLE, apply_common_layout

All plots should use these constants for consistency.
"""

# =============================================================================
# PLOT DIMENSIONS (5x3 inches at 100 DPI base, scaled 2x for high-res export)
# =============================================================================
PLOT_WIDTH = 500   # 5 inches * 100 DPI
PLOT_HEIGHT = 300  # 3 inches * 100 DPI
EXPORT_SCALE = 2   # Scale factor for high-resolution PNG export

# For wider plots (e.g., subplots side by side)
WIDE_PLOT_WIDTH = 800
WIDE_PLOT_HEIGHT = 400

# =============================================================================
# COLORS - Consistent color scheme matching frontend
# =============================================================================
COLORS = {
    'primary': '#5B8DBE',      # Blue - main data points
    'secondary': '#9B72B0',    # Purple - regression lines, fits
    'tertiary': '#5AA89B',     # Teal - alternative data series
    'dark_teal': '#2D6B61',    # Dark teal - secondary fits
    'accent': '#74B3A8',       # Light teal - highlights
    'warning': '#E74C3C',      # Red - warnings, transformed data
    'gray': '#888888',         # Gray - reference lines, original data
    'light_gray': '#cccccc',   # Light gray - borders
}

# Convenient aliases
BLUE = COLORS['primary']
PURPLE = COLORS['secondary']
TEAL = COLORS['tertiary']
DARK_TEAL = COLORS['dark_teal']
RED = COLORS['warning']
GRAY = COLORS['gray']

# =============================================================================
# MARKER STYLES
# =============================================================================
MARKER_SIZE = 8                # Standard marker size for scatter plots
MARKER_SIZE_SMALL = 6          # Smaller markers for dense plots
LINE_WIDTH = 2                 # Standard line width
LINE_WIDTH_THIN = 1            # Thin lines for reference/grid
MARKER_LINE_WIDTH = 0.5        # Border around markers

# =============================================================================
# FONT SETTINGS - Matching the parameter documentation page styling
# =============================================================================
# Font colors to match the page (stone color palette from Tailwind)
FONT_COLOR = '#57534e'         # stone-600 - body text color
FONT_COLOR_LIGHT = '#78716c'   # stone-500 - secondary text
FONT_COLOR_DARK = '#44403c'    # stone-700 - headings

# Font family matching the page (system fonts)
FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

# Font sizes
FONT_SIZE_BASE = 11            # Base font size
FONT_SIZE_TITLE = 14           # Plot titles
FONT_SIZE_AXIS_TITLE = 12      # Axis titles
FONT_SIZE_TICK = 10            # Tick labels
FONT_SIZE_LEGEND = 9           # Legend text
FONT_SIZE_ANNOTATION = 9       # Annotation text

# =============================================================================
# LAYOUT MARGINS
# =============================================================================
MARGIN = dict(l=60, r=20, t=30, b=60)
MARGIN_WITH_TITLE = dict(l=60, r=20, t=50, b=60)

# =============================================================================
# GRID AND AXIS STYLING
# =============================================================================
GRID_COLOR = 'rgba(128, 128, 128, 0.2)'
AXIS_LINE_COLOR = '#ccc'

# =============================================================================
# HOVER LABEL STYLING
# =============================================================================
# NOTE: Hover label visual styling is defined in frontend/styles.css using CSS
# !important rules to ensure consistency across all plots. The CSS selectors are:
#   .hoverlayer .hovertext path { fill, stroke }
#   .hoverlayer .hovertext text { fill }
# This ensures a single source of truth for hover styling.

# =============================================================================
# COMBINED STYLE DICTIONARY
# =============================================================================
STYLE = {
    # Dimensions
    'width': PLOT_WIDTH,
    'height': PLOT_HEIGHT,
    'wide_width': WIDE_PLOT_WIDTH,
    'wide_height': WIDE_PLOT_HEIGHT,
    'export_scale': EXPORT_SCALE,

    # Colors
    'colors': COLORS,
    'blue': BLUE,
    'purple': PURPLE,
    'teal': TEAL,
    'dark_teal': DARK_TEAL,
    'red': RED,
    'gray': GRAY,

    # Markers and lines
    'marker_size': MARKER_SIZE,
    'marker_size_small': MARKER_SIZE_SMALL,
    'line_width': LINE_WIDTH,
    'line_width_thin': LINE_WIDTH_THIN,
    'marker_line_width': MARKER_LINE_WIDTH,

    # Fonts
    'font_size_base': FONT_SIZE_BASE,
    'font_size_title': FONT_SIZE_TITLE,
    'font_size_axis_title': FONT_SIZE_AXIS_TITLE,
    'font_size_tick': FONT_SIZE_TICK,
    'font_size_legend': FONT_SIZE_LEGEND,
    'font_size_annotation': FONT_SIZE_ANNOTATION,

    # Layout
    'margin': MARGIN,
    'margin_with_title': MARGIN_WITH_TITLE,
    'grid_color': GRID_COLOR,
    'axis_line_color': AXIS_LINE_COLOR,
}


def get_common_layout(title=None, wide=False):
    """
    Returns a dictionary of common layout settings for Plotly figures.

    Args:
        title: Optional title for the plot
        wide: If True, use wider dimensions for subplots

    Returns:
        dict: Layout settings to be passed to fig.update_layout()
    """
    layout = {
        'width': WIDE_PLOT_WIDTH if wide else PLOT_WIDTH,
        'height': WIDE_PLOT_HEIGHT if wide else PLOT_HEIGHT,
        'template': 'none',  # Disable default Plotly template to avoid style conflicts
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': dict(size=FONT_SIZE_BASE, color=FONT_COLOR),
        'margin': MARGIN_WITH_TITLE if title else MARGIN,
        'hovermode': 'closest',
        'hoverlabel': dict(
            bgcolor='#fffff8',  # Cream background matching page
            bordercolor='#fffff8',  # No visible border (same as background)
            font=dict(size=FONT_SIZE_BASE, color=FONT_COLOR),
        ),
    }

    if title:
        layout['title'] = dict(text=title, font=dict(size=FONT_SIZE_TITLE))

    return layout


def get_axis_style(title, log_scale=False):
    """
    Returns common axis styling settings.

    Args:
        title: Axis title text
        log_scale: If True, set axis type to 'log'

    Returns:
        dict: Axis settings
    """
    style = {
        'title': dict(text=title, font=dict(size=FONT_SIZE_AXIS_TITLE)),
        'tickfont': dict(size=FONT_SIZE_TICK),
        'gridcolor': GRID_COLOR,
        'linecolor': AXIS_LINE_COLOR,
        'showline': True,
    }

    if log_scale:
        style['type'] = 'log'

    return style


def get_legend_style(position='top_left'):
    """
    Returns legend styling settings.

    Args:
        position: One of 'top_left', 'top_right', 'bottom_left', 'bottom_right'

    Returns:
        dict: Legend settings
    """
    positions = {
        'top_left': dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        'top_right': dict(x=0.98, y=0.98, xanchor='right', yanchor='top'),
        'bottom_left': dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom'),
        'bottom_right': dict(x=0.98, y=0.02, xanchor='right', yanchor='bottom'),
    }

    pos = positions.get(position, positions['top_left'])

    return {
        **pos,
        'bgcolor': 'rgba(0, 0, 0, 0)',
        'bordercolor': 'rgba(0, 0, 0, 0)',
        'borderwidth': 0,
        'font': dict(size=FONT_SIZE_LEGEND),
    }


def apply_common_layout(fig, title=None, wide=False,
                        xaxis_title=None, yaxis_title=None,
                        xaxis_log=False, yaxis_log=False,
                        legend_position='top_left', show_legend=True):
    """
    Apply common layout settings to a Plotly figure.

    Args:
        fig: Plotly figure object
        title: Optional plot title
        wide: If True, use wider dimensions
        xaxis_title: X-axis title
        yaxis_title: Y-axis title
        xaxis_log: If True, use log scale for x-axis
        yaxis_log: If True, use log scale for y-axis
        legend_position: Legend position ('top_left', 'top_right', etc.)
        show_legend: Whether to show the legend

    Returns:
        fig: The modified figure
    """
    layout = get_common_layout(title=title, wide=wide)

    if xaxis_title:
        layout['xaxis'] = get_axis_style(xaxis_title, log_scale=xaxis_log)

    if yaxis_title:
        layout['yaxis'] = get_axis_style(yaxis_title, log_scale=yaxis_log)

    layout['showlegend'] = show_legend
    if show_legend:
        layout['legend'] = get_legend_style(legend_position)

    fig.update_layout(**layout)
    return fig


def save_plot(fig, filename, wide=False):
    """
    Save a Plotly figure with consistent settings (both PNG and HTML).

    Args:
        fig: Plotly figure object
        filename: Output filename (should end in .png)
        wide: If True, use wider dimensions
    """
    width = WIDE_PLOT_WIDTH if wide else PLOT_WIDTH
    height = WIDE_PLOT_HEIGHT if wide else PLOT_HEIGHT

    fig.write_image(filename, width=width, height=height, scale=EXPORT_SCALE)
    print(f"Saved: {filename}")

    # Also save as HTML for better quality in web
    html_filename = filename.replace('.png', '.html')
    save_html(fig, html_filename)


def save_html(fig, filename):
    """
    Save a Plotly figure as responsive HTML for embedding.

    Args:
        fig: Plotly figure object
        filename: Output filename (should end in .html)
    """
    # Make a copy of the figure to avoid modifying the original
    fig_html = fig
    fig_html.update_layout(width=None, height=None, autosize=True)

    # Generate HTML content
    html_content = fig_html.to_html(
        include_plotlyjs='cdn',
        full_html=True,
        config={'responsive': True},
        default_width='100%',
        default_height='100%'
    )

    # Inject CSS to override Plotly's default hover label styling
    # This ensures cream background and no border regardless of trace colors
    hover_css = """
    <style>
    .hoverlayer .hovertext path {
        fill: #fffff8 !important;
        stroke: #fffff8 !important;
    }
    .hoverlayer .hovertext text {
        fill: #57534e !important;
    }
    </style>
    """

    # Insert CSS before closing </head> tag
    html_content = html_content.replace('</head>', hover_css + '</head>')

    # Write the modified HTML
    with open(filename, 'w') as f:
        f.write(html_content)

    print(f"Saved: {filename}")
