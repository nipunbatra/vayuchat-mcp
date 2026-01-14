"""
VayuChat MCP Server - Natural language data analysis for air quality data.

Pre-loaded datasets:
- air_quality: Hourly PM2.5, PM10, NO2, etc. for Delhi & Bangalore
- funding: Government funding for air quality initiatives by city/year
- city_info: City metadata (population, vehicles, industries, etc.)
"""

import base64
import io
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("VayuChat")

# Global state
_dataframes: dict[str, pd.DataFrame] = {}
_plots: dict[str, str] = {}  # Store generated plots

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _setup_plot_style():
    """Set up dark theme for plots."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#1a1a1a',
        'axes.facecolor': '#1a1a1a',
        'axes.edgecolor': '#444',
        'axes.labelcolor': '#ccc',
        'text.color': '#ccc',
        'xtick.color': '#999',
        'ytick.color': '#999',
        'grid.color': '#333',
        'legend.facecolor': '#1a1a1a',
        'legend.edgecolor': '#444',
    })


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
               facecolor='#1a1a1a', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _auto_load_datasets():
    """Auto-load datasets from data directory on startup."""
    if not DATA_DIR.exists():
        return

    # Define the datasets to load with their names
    datasets = {
        'air_quality': 'sample_air_quality.csv',
        'funding': 'funding_data.csv',
        'city_info': 'city_info.csv',
    }

    for name, filename in datasets.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            try:
                _dataframes[name] = pd.read_csv(filepath)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")


# Auto-load on module import
_auto_load_datasets()


# =============================================================================
# DATA EXPLORATION TOOLS
# =============================================================================

@mcp.tool()
def list_tables() -> str:
    """
    List all available tables/dataframes with their descriptions.

    Returns:
        Summary of all available tables.
    """
    if not _dataframes:
        return "No tables loaded."

    descriptions = {
        'air_quality': 'Hourly air quality readings (PM2.5, PM10, NO2, SO2, CO, O3) for Delhi & Bangalore',
        'funding': 'Government funding for air quality initiatives by city and year',
        'city_info': 'City metadata - population, vehicles, industries, green cover',
    }

    lines = ["# Available Tables", ""]
    for name, df in _dataframes.items():
        desc = descriptions.get(name, "No description")
        lines.append(f"## {name}")
        lines.append(f"- **Description:** {desc}")
        lines.append(f"- **Rows:** {len(df):,}")
        lines.append(f"- **Columns:** {', '.join(df.columns[:8])}")
        if len(df.columns) > 8:
            lines.append(f"  ... and {len(df.columns) - 8} more")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def show_table(name: str, rows: int = 10, columns: list[str] | None = None) -> str:
    """
    Display rows from a table.

    Args:
        name: Table name (air_quality, funding, city_info)
        rows: Number of rows to show (default: 10)
        columns: Optional list of columns to display

    Returns:
        Formatted table data.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys())
        return f"Table '{name}' not found. Available: {available}"

    df = _dataframes[name]
    if columns:
        df = df[columns]

    return f"## {name} (showing {min(rows, len(df))} of {len(df)} rows)\n\n{df.head(rows).to_markdown(index=False)}"


@mcp.tool()
def describe_table(name: str) -> str:
    """
    Get detailed statistics for a table.

    Args:
        name: Table name

    Returns:
        Statistical summary and column info.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys())
        return f"Table '{name}' not found. Available: {available}"

    df = _dataframes[name]

    lines = [f"# {name} - Detailed Info", ""]
    lines.append(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    lines.append("")
    lines.append("## Columns")

    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        unique = df[col].nunique()

        lines.append(f"\n### {col}")
        lines.append(f"- Type: {dtype}")
        lines.append(f"- Non-null: {non_null:,} ({non_null/len(df)*100:.1f}%)")
        lines.append(f"- Unique: {unique:,}")

        if pd.api.types.is_numeric_dtype(df[col]):
            lines.append(f"- Range: {df[col].min():.2f} to {df[col].max():.2f}")
            lines.append(f"- Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")
        elif df[col].dtype == 'object' and unique <= 10:
            lines.append(f"- Values: {', '.join(map(str, df[col].unique()))}")

    return "\n".join(lines)


@mcp.tool()
def query_table(name: str, query: str) -> str:
    """
    Filter a table using pandas query syntax.

    Args:
        name: Table name
        query: Pandas query (e.g., "city == 'Delhi' and PM2.5 > 200")

    Returns:
        Filtered results.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys())
        return f"Table '{name}' not found. Available: {available}"

    df = _dataframes[name]

    try:
        result = df.query(query)
        n_results = len(result)

        if n_results == 0:
            return f"No rows match query: {query}"

        lines = [f"## Query Results ({n_results:,} rows)", f"Query: `{query}`", ""]

        if n_results <= 20:
            lines.append(result.to_markdown(index=False))
        else:
            lines.append(result.head(20).to_markdown(index=False))
            lines.append(f"\n*...and {n_results - 20:,} more rows*")

        return "\n".join(lines)

    except Exception as e:
        return f"Query error: {e}"


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

@mcp.tool()
def compare_weekday_weekend(value_column: str, group_by: str | None = None,
                            table: str = "air_quality") -> str:
    """
    Compare weekday vs weekend values for a metric.

    Args:
        value_column: Column to compare (e.g., 'PM2.5', 'PM10')
        group_by: Optional grouping column (e.g., 'city')
        table: Table name (default: air_quality)

    Returns:
        Comparison statistics.
    """
    if table not in _dataframes:
        return f"Table '{table}' not found."

    df = _dataframes[table].copy()

    if 'day_of_week' not in df.columns:
        return "Table doesn't have 'day_of_week' column"

    weekend_days = ['Saturday', 'Sunday']
    df['_day_type'] = df['day_of_week'].apply(lambda x: 'Weekend' if x in weekend_days else 'Weekday')

    lines = [f"# Weekday vs Weekend: {value_column}", ""]

    if group_by and group_by in df.columns:
        pivot = df.pivot_table(values=value_column, index=group_by,
                               columns='_day_type', aggfunc='mean').round(2)
        pivot['Change'] = pivot['Weekend'] - pivot['Weekday']
        pivot['Change %'] = ((pivot['Weekend'] - pivot['Weekday']) / pivot['Weekday'] * 100).round(1)

        lines.append(pivot.to_markdown())
    else:
        stats = df.groupby('_day_type')[value_column].agg(['mean', 'std', 'count']).round(2)
        lines.append(stats.to_markdown())

        diff = stats.loc['Weekend', 'mean'] - stats.loc['Weekday', 'mean']
        pct = diff / stats.loc['Weekday', 'mean'] * 100
        lines.append(f"\n**Change:** {diff:.2f} ({pct:.1f}%)")

    return "\n".join(lines)


@mcp.tool()
def compare_cities(value_column: str, cities: list[str] | None = None,
                   table: str = "air_quality") -> str:
    """
    Compare a metric across cities.

    Args:
        value_column: Column to compare (e.g., 'PM2.5')
        cities: Optional list of cities to compare
        table: Table name (default: air_quality)

    Returns:
        City comparison statistics.
    """
    if table not in _dataframes:
        return f"Table '{table}' not found."

    df = _dataframes[table]

    if 'city' not in df.columns:
        return "Table doesn't have 'city' column"

    if cities:
        df = df[df['city'].isin(cities)]

    stats = df.groupby('city')[value_column].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(2).sort_values('mean', ascending=False)

    lines = [f"# {value_column} by City", "", stats.to_markdown()]

    return "\n".join(lines)


@mcp.tool()
def analyze_correlation(columns: list[str] | None = None, target: str | None = None,
                        table: str = "air_quality") -> str:
    """
    Analyze correlations between numeric columns.

    Args:
        columns: Optional list of columns to analyze
        target: Optional target column to show correlations with
        table: Table name (default: air_quality)

    Returns:
        Correlation analysis.
    """
    if table not in _dataframes:
        return f"Table '{table}' not found."

    df = _dataframes[table]

    if columns:
        numeric_df = df[columns].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])

    corr = numeric_df.corr().round(3)

    if target and target in corr.columns:
        target_corr = corr[target].drop(target).sort_values(key=abs, ascending=False)
        lines = [f"# Correlations with {target}", ""]
        for col, val in target_corr.items():
            strength = "strong" if abs(val) > 0.7 else "moderate" if abs(val) > 0.4 else "weak"
            lines.append(f"- **{col}:** {val:.3f} ({strength})")
        return "\n".join(lines)
    else:
        return f"# Correlation Matrix\n\n{corr.to_markdown()}"


@mcp.tool()
def analyze_funding(city: str | None = None, year: int | None = None) -> str:
    """
    Analyze air quality funding data.

    Args:
        city: Optional city to filter by
        year: Optional year to filter by

    Returns:
        Funding analysis.
    """
    if 'funding' not in _dataframes:
        return "Funding table not loaded."

    df = _dataframes['funding']

    if city:
        df = df[df['city'] == city]
    if year:
        df = df[df['year'] == year]

    if len(df) == 0:
        return "No data matches the filters."

    lines = ["# Funding Analysis", ""]

    if city and not year:
        lines.append(f"## {city} - Yearly Breakdown")
        lines.append(df.to_markdown(index=False))

        total = df['total_budget_cr'].sum()
        growth = (df.iloc[-1]['total_budget_cr'] / df.iloc[0]['total_budget_cr'] - 1) * 100
        lines.append(f"\n**Total (all years):** ₹{total:.1f} Cr")
        lines.append(f"**Growth:** {growth:.1f}%")

    elif year and not city:
        lines.append(f"## Year {year} - All Cities")
        lines.append(df.to_markdown(index=False))

        total = df['total_budget_cr'].sum()
        lines.append(f"\n**Total allocation:** ₹{total:.1f} Cr")

    else:
        by_city = df.groupby('city')['total_budget_cr'].sum().sort_values(ascending=False)
        by_year = df.groupby('year')['total_budget_cr'].sum()

        lines.append("## Total Budget by City")
        lines.append(by_city.to_markdown())
        lines.append("\n## Total Budget by Year")
        lines.append(by_year.to_markdown())

    return "\n".join(lines)


@mcp.tool()
def get_city_profile(city: str) -> str:
    """
    Get comprehensive profile for a city including all available data.

    Args:
        city: City name (Delhi, Bangalore, Mumbai, etc.)

    Returns:
        City profile with air quality, funding, and metadata.
    """
    lines = [f"# {city} - City Profile", ""]

    # City info
    if 'city_info' in _dataframes:
        info = _dataframes['city_info']
        city_row = info[info['city'] == city]
        if len(city_row) > 0:
            row = city_row.iloc[0]
            lines.append("## Demographics & Infrastructure")
            lines.append(f"- **Population:** {row['population_millions']:.1f} million")
            lines.append(f"- **Area:** {row['area_sq_km']:,} sq km")
            lines.append(f"- **Vehicles:** {row['vehicles_lakhs']} lakhs")
            lines.append(f"- **Industries:** {row['industries']:,}")
            lines.append(f"- **Green Cover:** {row['green_cover_pct']}%")
            lines.append(f"- **Days exceeding WHO limit:** {row['who_limit_days_exceeded']}")
            lines.append("")

    # Air quality summary
    if 'air_quality' in _dataframes:
        aq = _dataframes['air_quality']
        city_aq = aq[aq['city'] == city]
        if len(city_aq) > 0:
            lines.append("## Air Quality Summary")
            for col in ['PM2.5', 'PM10', 'NO2', 'AQI_category']:
                if col in city_aq.columns:
                    if col == 'AQI_category':
                        top_cat = city_aq[col].value_counts().head(3)
                        lines.append(f"- **AQI Categories:** {', '.join(f'{k}({v})' for k,v in top_cat.items())}")
                    else:
                        lines.append(f"- **{col}:** Mean={city_aq[col].mean():.1f}, Max={city_aq[col].max():.1f}")
            lines.append("")

    # Funding summary
    if 'funding' in _dataframes:
        fund = _dataframes['funding']
        city_fund = fund[fund['city'] == city]
        if len(city_fund) > 0:
            lines.append("## Funding Summary")
            total = city_fund['total_budget_cr'].sum()
            latest = city_fund[city_fund['year'] == city_fund['year'].max()]['total_budget_cr'].iloc[0]
            lines.append(f"- **Total Budget (all years):** ₹{total:.1f} Cr")
            lines.append(f"- **Latest Year Budget:** ₹{latest:.1f} Cr")

    return "\n".join(lines)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

@mcp.tool()
def plot_comparison(value_column: str, group_column: str = "city",
                    chart_type: str = "bar", table: str = "air_quality",
                    title: str | None = None) -> str:
    """
    Create a comparison chart.

    Args:
        value_column: Column to plot (e.g., 'PM2.5')
        group_column: Grouping column (default: 'city')
        chart_type: 'bar', 'horizontal_bar', or 'box'
        table: Table name
        title: Optional title

    Returns:
        Base64 encoded plot.
    """
    if table not in _dataframes:
        return f"Table '{table}' not found."

    df = _dataframes[table]
    _setup_plot_style()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == 'box':
        groups = df[group_column].unique()
        data = [df[df[group_column] == g][value_column].dropna() for g in groups]
        bp = ax.boxplot(data, labels=groups, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    else:
        grouped = df.groupby(group_column)[value_column].mean().sort_values(ascending=False)
        colors = plt.cm.Set2(np.linspace(0, 1, len(grouped)))

        if chart_type == 'horizontal_bar':
            ax.barh(grouped.index, grouped.values, color=colors)
            ax.set_xlabel(value_column)
        else:
            bars = ax.bar(grouped.index, grouped.values, color=colors)
            ax.set_ylabel(value_column)
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
            plt.xticks(rotation=45, ha='right')

    ax.set_title(title or f'{value_column} by {group_column}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    img_base64 = _fig_to_base64(fig)
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


@mcp.tool()
def plot_time_series(value_column: str, group_by: str | None = None,
                     table: str = "air_quality", title: str | None = None) -> str:
    """
    Create a time series plot.

    Args:
        value_column: Column to plot
        group_by: Optional column for separate lines (e.g., 'city')
        table: Table name
        title: Optional title

    Returns:
        Base64 encoded plot.
    """
    if table not in _dataframes:
        return f"Table '{table}' not found."

    df = _dataframes[table].copy()

    if 'date' not in df.columns:
        return "Table doesn't have 'date' column"

    df['_date'] = pd.to_datetime(df['date'])
    _setup_plot_style()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(12, 6))

    if group_by and group_by in df.columns:
        groups = df[group_by].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for group, color in zip(groups, colors):
            group_df = df[df[group_by] == group]
            daily = group_df.groupby('_date')[value_column].mean()
            ax.plot(daily.index, daily.values, label=group, color=color, linewidth=2)
        ax.legend()
    else:
        daily = df.groupby('_date')[value_column].mean()
        ax.plot(daily.index, daily.values, color='#3498db', linewidth=2)

    ax.set_xlabel('Date')
    ax.set_ylabel(value_column)
    ax.set_title(title or f'{value_column} Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    img_base64 = _fig_to_base64(fig)
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


@mcp.tool()
def plot_weekday_weekend(value_column: str, group_by: str | None = "city",
                         table: str = "air_quality", title: str | None = None) -> str:
    """
    Create weekday vs weekend comparison chart.

    Args:
        value_column: Column to compare
        group_by: Grouping column (default: 'city')
        table: Table name
        title: Optional title

    Returns:
        Base64 encoded plot.
    """
    if table not in _dataframes:
        return f"Table '{table}' not found."

    df = _dataframes[table].copy()
    _setup_plot_style()
    plt.close('all')

    weekend_days = ['Saturday', 'Sunday']
    df['_day_type'] = df['day_of_week'].apply(lambda x: 'Weekend' if x in weekend_days else 'Weekday')

    fig, ax = plt.subplots(figsize=(10, 6))

    if group_by and group_by in df.columns:
        groups = df[group_by].unique()
        x = np.arange(len(groups))
        width = 0.35

        weekday_vals = [df[(df[group_by] == g) & (df['_day_type'] == 'Weekday')][value_column].mean() for g in groups]
        weekend_vals = [df[(df[group_by] == g) & (df['_day_type'] == 'Weekend')][value_column].mean() for g in groups]

        bars1 = ax.bar(x - width/2, weekday_vals, width, label='Weekday', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, weekend_vals, width, label='Weekend', color='#3498db', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(groups)

        for bar in bars1:
            ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
        for bar in bars2:
            ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

    ax.set_ylabel(value_column)
    ax.set_title(title or f'Weekday vs Weekend: {value_column}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    img_base64 = _fig_to_base64(fig)
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


@mcp.tool()
def plot_funding_trend(cities: list[str] | None = None, title: str | None = None) -> str:
    """
    Plot funding trends over years by city.

    Args:
        cities: Optional list of cities to include
        title: Optional title

    Returns:
        Base64 encoded plot.
    """
    if 'funding' not in _dataframes:
        return "Funding table not loaded."

    df = _dataframes['funding']

    if cities:
        df = df[df['city'].isin(cities)]

    _setup_plot_style()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(12, 6))

    for city in df['city'].unique():
        city_df = df[df['city'] == city].sort_values('year')
        ax.plot(city_df['year'], city_df['total_budget_cr'], marker='o',
               linewidth=2, markersize=8, label=city)

    ax.set_xlabel('Year')
    ax.set_ylabel('Total Budget (₹ Cr)')
    ax.set_title(title or 'Air Quality Funding by City', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    img_base64 = _fig_to_base64(fig)
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


@mcp.tool()
def plot_hourly_pattern(value_column: str, group_by: str | None = "city",
                        table: str = "air_quality", title: str | None = None) -> str:
    """
    Plot hourly patterns.

    Args:
        value_column: Column to plot
        group_by: Optional grouping column
        table: Table name
        title: Optional title

    Returns:
        Base64 encoded plot.
    """
    if table not in _dataframes:
        return f"Table '{table}' not found."

    df = _dataframes[table]
    _setup_plot_style()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(12, 6))

    if group_by and group_by in df.columns:
        for city in df[group_by].unique():
            city_df = df[df[group_by] == city]
            hourly = city_df.groupby('hour')[value_column].mean()
            ax.plot(hourly.index, hourly.values, marker='o', linewidth=2,
                   markersize=4, label=city)
        ax.legend()
    else:
        hourly = df.groupby('hour')[value_column].mean()
        ax.plot(hourly.index, hourly.values, marker='o', linewidth=2,
               markersize=6, color='#3498db')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(value_column)
    ax.set_title(title or f'{value_column} by Hour', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    img_base64 = _fig_to_base64(fig)
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


@mcp.tool()
def execute_code(code: str) -> str:
    """
    Execute custom Python code for advanced analysis.

    Available variables:
    - air_quality, funding, city_info: DataFrames
    - pd, np, plt: Libraries

    Args:
        code: Python code to execute

    Returns:
        Output from code execution.
    """
    _setup_plot_style()

    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    namespace: dict[str, Any] = {
        'pd': pd, 'np': np, 'plt': plt,
        **_dataframes
    }

    result_parts = []

    try:
        plt.close('all')
        exec(code, namespace)

        text_output = captured_output.getvalue()
        if text_output.strip():
            result_parts.append(text_output)

        # Capture figures
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            img_base64 = _fig_to_base64(fig)
            result_parts.append(f"data:image/png;base64,{img_base64}")
            plt.close(fig)

        return "\n".join(result_parts) if result_parts else "Code executed successfully."

    except Exception as e:
        return f"Error: {traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout
        plt.close('all')


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
