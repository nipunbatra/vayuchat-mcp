"""
VayuChat MCP Server - Natural language data analysis with pandas and matplotlib.

This MCP server provides tools for:
- Loading CSV files into pandas DataFrames
- Exploring and querying data
- Executing Python code for data analysis
- Generating visualizations with matplotlib
"""

import base64
import io
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("VayuChat")

# Global state to store loaded dataframes
_dataframes: dict[str, pd.DataFrame] = {}


@mcp.tool()
def load_csv(file_path: str, name: str | None = None) -> str:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file to load
        name: Optional name for the dataframe. If not provided, uses the filename without extension.

    Returns:
        Summary of the loaded dataframe including shape, columns, and dtypes.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return f"Error: File not found: {path}"

    if not path.suffix.lower() == '.csv':
        return f"Error: File must be a CSV file, got: {path.suffix}"

    try:
        df = pd.read_csv(path)
        df_name = name or path.stem

        # Clean up the name to be a valid Python identifier
        df_name = df_name.replace('-', '_').replace(' ', '_')

        _dataframes[df_name] = df

        # Generate summary
        summary = [
            f"Successfully loaded '{df_name}'",
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            "",
            "Columns and types:",
        ]

        for col, dtype in df.dtypes.items():
            summary.append(f"  - {col}: {dtype}")

        summary.extend([
            "",
            "First 5 rows:",
            df.head().to_string(),
        ])

        return "\n".join(summary)

    except Exception as e:
        return f"Error loading CSV: {str(e)}"


@mcp.tool()
def list_dataframes() -> str:
    """
    List all currently loaded dataframes with their basic info.

    Returns:
        Summary of all loaded dataframes.
    """
    if not _dataframes:
        return "No dataframes loaded. Use load_csv() to load a CSV file."

    lines = [f"Loaded dataframes ({len(_dataframes)}):"]

    for name, df in _dataframes.items():
        lines.append(f"\n{name}:")
        lines.append(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        lines.append(f"  Columns: {', '.join(df.columns[:10])}")
        if len(df.columns) > 10:
            lines.append(f"    ... and {len(df.columns) - 10} more columns")
        lines.append(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    return "\n".join(lines)


@mcp.tool()
def get_dataframe_info(name: str) -> str:
    """
    Get detailed information about a specific dataframe.

    Args:
        name: Name of the dataframe to inspect

    Returns:
        Detailed info including columns, dtypes, sample data, and statistics.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]

    lines = [
        f"DataFrame: {name}",
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        "",
        "Column Information:",
    ]

    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        unique = df[col].nunique()

        lines.append(f"\n  {col}:")
        lines.append(f"    Type: {dtype}")
        lines.append(f"    Non-null: {non_null} ({100 - null_pct:.1f}%)")
        lines.append(f"    Unique values: {unique}")

        if pd.api.types.is_numeric_dtype(df[col]):
            lines.append(f"    Range: {df[col].min()} to {df[col].max()}")
            lines.append(f"    Mean: {df[col].mean():.2f}")
        elif pd.api.types.is_object_dtype(df[col]):
            top_values = df[col].value_counts().head(3)
            lines.append(f"    Top values: {', '.join(f'{v}({c})' for v, c in top_values.items())}")

    lines.extend([
        "",
        "Sample data (first 5 rows):",
        df.head().to_string(),
    ])

    return "\n".join(lines)


@mcp.tool()
def execute_code(code: str) -> str:
    """
    Execute Python code for data analysis.

    The code has access to:
    - All loaded dataframes by their names (e.g., 'air_quality', 'sales_data')
    - pandas as 'pd'
    - numpy as 'np'
    - matplotlib.pyplot as 'plt'

    For visualizations, use plt.savefig() or the code will automatically
    capture any open figures as base64 PNG images.

    Args:
        code: Python code to execute

    Returns:
        Output from the code execution, including any print statements
        and base64-encoded images for any generated plots.
    """
    # Set up dark theme for matplotlib
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

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    # Build the execution namespace with all dataframes
    namespace: dict[str, Any] = {
        'pd': pd,
        'np': np,
        'plt': plt,
        '_dataframes': _dataframes,
    }

    # Add each dataframe to the namespace
    for name, df in _dataframes.items():
        namespace[name] = df

    result_parts = []
    images = []

    try:
        # Close any existing figures
        plt.close('all')

        # Execute the code
        exec(code, namespace)

        # Capture any text output
        text_output = captured_output.getvalue()
        if text_output.strip():
            result_parts.append("Output:")
            result_parts.append(text_output)

        # Capture any open figures
        fig_nums = plt.get_fignums()
        for fig_num in fig_nums:
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                       facecolor='#1a1a1a', edgecolor='none')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            images.append(img_base64)
            plt.close(fig)

        if images:
            result_parts.append(f"\nGenerated {len(images)} plot(s):")
            for i, img in enumerate(images, 1):
                result_parts.append(f"\n[Plot {i}]")
                result_parts.append(f"data:image/png;base64,{img}")

        if not result_parts:
            result_parts.append("Code executed successfully (no output)")

        return "\n".join(result_parts)

    except Exception as e:
        error_msg = traceback.format_exc()
        return f"Error executing code:\n{error_msg}"

    finally:
        sys.stdout = old_stdout
        plt.close('all')


@mcp.tool()
def query_dataframe(name: str, query: str) -> str:
    """
    Run a pandas query on a dataframe and return results.

    This is a convenience method for simple queries. For complex analysis,
    use execute_code() instead.

    Args:
        name: Name of the dataframe to query
        query: Pandas query string (e.g., "column > 100" or "city == 'Delhi'")

    Returns:
        Filtered dataframe results.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]

    try:
        result = df.query(query)

        lines = [
            f"Query: {query}",
            f"Result: {len(result)} rows (from {len(df)} total)",
            "",
        ]

        if len(result) <= 20:
            lines.append(result.to_string())
        else:
            lines.append("First 20 rows:")
            lines.append(result.head(20).to_string())
            lines.append(f"\n... and {len(result) - 20} more rows")

        return "\n".join(lines)

    except Exception as e:
        return f"Error executing query: {str(e)}"


@mcp.tool()
def describe_dataframe(name: str, columns: list[str] | None = None) -> str:
    """
    Get statistical summary of a dataframe.

    Args:
        name: Name of the dataframe
        columns: Optional list of specific columns to describe. If not provided, describes all numeric columns.

    Returns:
        Statistical summary including count, mean, std, min, max, and quartiles.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]

    try:
        if columns:
            subset = df[columns]
        else:
            subset = df

        description = subset.describe(include='all')

        return f"Statistical Summary of '{name}':\n\n{description.to_string()}"

    except Exception as e:
        return f"Error describing dataframe: {str(e)}"


@mcp.tool()
def unload_dataframe(name: str) -> str:
    """
    Unload a dataframe from memory.

    Args:
        name: Name of the dataframe to unload

    Returns:
        Confirmation message.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    del _dataframes[name]
    return f"Dataframe '{name}' has been unloaded."


@mcp.tool()
def sample_dataframe(name: str, n: int = 10, random: bool = False) -> str:
    """
    Get a sample of rows from a dataframe.

    Args:
        name: Name of the dataframe
        n: Number of rows to return (default: 10)
        random: If True, return random sample. If False, return first n rows.

    Returns:
        Sample rows from the dataframe.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]
    n = min(n, len(df))

    if random:
        sample = df.sample(n=n)
        method = "Random"
    else:
        sample = df.head(n)
        method = "First"

    return f"{method} {n} rows of '{name}':\n\n{sample.to_string()}"


# =============================================================================
# PREDEFINED ANALYSIS FUNCTIONS
# These allow the MCP to perform common analyses without writing code
# =============================================================================

@mcp.tool()
def compare_weekday_weekend(name: str, value_column: str, date_column: str = "date",
                            day_of_week_column: str = "day_of_week",
                            group_by: str | None = None) -> str:
    """
    Compare weekday vs weekend values for a metric.

    Args:
        name: Name of the dataframe
        value_column: Column containing the values to compare (e.g., 'PM2.5', 'sales')
        date_column: Column containing dates (default: 'date')
        day_of_week_column: Column containing day names (default: 'day_of_week')
        group_by: Optional column to group by (e.g., 'city', 'station')

    Returns:
        Comparison of weekday vs weekend averages with statistics.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name].copy()

    # Determine day type
    weekend_days = ['Saturday', 'Sunday']
    if day_of_week_column in df.columns:
        df['_day_type'] = df[day_of_week_column].apply(
            lambda x: 'Weekend' if x in weekend_days else 'Weekday'
        )
    elif date_column in df.columns:
        df['_date'] = pd.to_datetime(df[date_column])
        df['_day_type'] = df['_date'].dt.dayofweek.apply(
            lambda x: 'Weekend' if x >= 5 else 'Weekday'
        )
    else:
        return f"Error: Need either '{day_of_week_column}' or '{date_column}' column"

    lines = ["# Weekday vs Weekend Comparison", ""]

    if group_by and group_by in df.columns:
        # Group comparison
        comparison = df.groupby([group_by, '_day_type'])[value_column].agg(['mean', 'std', 'count'])
        comparison = comparison.round(2)

        pivot = df.pivot_table(values=value_column, index=group_by, columns='_day_type', aggfunc='mean').round(2)
        pivot['Difference'] = (pivot['Weekend'] - pivot['Weekday']).round(2)
        pivot['% Change'] = ((pivot['Weekend'] - pivot['Weekday']) / pivot['Weekday'] * 100).round(2)

        lines.append(f"## By {group_by}")
        lines.append("")
        lines.append(pivot.to_string())
        lines.append("")
        lines.append("## Detailed Statistics")
        lines.append(comparison.to_string())
    else:
        # Overall comparison
        comparison = df.groupby('_day_type')[value_column].agg(['mean', 'std', 'count']).round(2)
        diff = comparison.loc['Weekend', 'mean'] - comparison.loc['Weekday', 'mean']
        pct_change = (diff / comparison.loc['Weekday', 'mean'] * 100)

        lines.append(f"## {value_column} Summary")
        lines.append(comparison.to_string())
        lines.append("")
        lines.append(f"**Difference:** {diff:.2f} ({pct_change:.1f}%)")

    return "\n".join(lines)


@mcp.tool()
def compare_groups(name: str, value_column: str, group_column: str,
                   groups: list[str] | None = None) -> str:
    """
    Compare a metric across different groups (e.g., cities, categories).

    Args:
        name: Name of the dataframe
        value_column: Column containing values to compare
        group_column: Column containing groups (e.g., 'city', 'category')
        groups: Optional list of specific groups to compare. If None, uses all groups.

    Returns:
        Statistical comparison across groups.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]

    if groups:
        df = df[df[group_column].isin(groups)]

    comparison = df.groupby(group_column)[value_column].agg([
        'count', 'mean', 'std', 'min', 'max',
        ('median', 'median'),
        ('25%', lambda x: x.quantile(0.25)),
        ('75%', lambda x: x.quantile(0.75))
    ]).round(2)

    comparison = comparison.sort_values('mean', ascending=False)

    lines = [
        f"# {value_column} Comparison by {group_column}",
        "",
        comparison.to_string(),
        "",
        "## Key Insights:",
        f"- Highest avg: {comparison.index[0]} ({comparison.iloc[0]['mean']:.2f})",
        f"- Lowest avg: {comparison.index[-1]} ({comparison.iloc[-1]['mean']:.2f})",
        f"- Range: {comparison['mean'].max() - comparison['mean'].min():.2f}",
    ]

    return "\n".join(lines)


@mcp.tool()
def hourly_pattern(name: str, value_column: str, hour_column: str = "hour",
                   group_by: str | None = None) -> str:
    """
    Analyze hourly patterns in the data.

    Args:
        name: Name of the dataframe
        value_column: Column containing values to analyze
        hour_column: Column containing hour (0-23, default: 'hour')
        group_by: Optional column to group by (e.g., 'city')

    Returns:
        Hourly pattern analysis with peak/off-peak hours.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]

    lines = [f"# Hourly Pattern: {value_column}", ""]

    if group_by and group_by in df.columns:
        for group in df[group_by].unique():
            group_df = df[df[group_by] == group]
            hourly = group_df.groupby(hour_column)[value_column].mean().round(2)
            peak_hour = hourly.idxmax()
            low_hour = hourly.idxmin()

            lines.append(f"## {group}")
            lines.append(f"- Peak hour: {peak_hour}:00 ({hourly[peak_hour]:.2f})")
            lines.append(f"- Lowest hour: {low_hour}:00 ({hourly[low_hour]:.2f})")
            lines.append(f"- Daily range: {hourly.max() - hourly.min():.2f}")
            lines.append("")
    else:
        hourly = df.groupby(hour_column)[value_column].mean().round(2)
        peak_hour = hourly.idxmax()
        low_hour = hourly.idxmin()

        lines.append(hourly.to_string())
        lines.append("")
        lines.append(f"**Peak hour:** {peak_hour}:00 ({hourly[peak_hour]:.2f})")
        lines.append(f"**Lowest hour:** {low_hour}:00 ({hourly[low_hour]:.2f})")

    return "\n".join(lines)


@mcp.tool()
def correlation_analysis(name: str, columns: list[str] | None = None,
                         target: str | None = None) -> str:
    """
    Analyze correlations between numeric columns.

    Args:
        name: Name of the dataframe
        columns: Optional list of columns to analyze. If None, uses all numeric columns.
        target: Optional target column to show correlations with (sorted by strength).

    Returns:
        Correlation matrix or target correlations.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]

    if columns:
        numeric_df = df[columns].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])

    corr = numeric_df.corr().round(3)

    lines = ["# Correlation Analysis", ""]

    if target and target in corr.columns:
        target_corr = corr[target].drop(target).sort_values(key=abs, ascending=False)
        lines.append(f"## Correlations with {target}")
        lines.append("")
        for col, val in target_corr.items():
            strength = "strong" if abs(val) > 0.7 else "moderate" if abs(val) > 0.4 else "weak"
            direction = "positive" if val > 0 else "negative"
            lines.append(f"- {col}: {val:.3f} ({strength} {direction})")
    else:
        lines.append("## Correlation Matrix")
        lines.append("")
        lines.append(corr.to_string())

    return "\n".join(lines)


@mcp.tool()
def trend_analysis(name: str, value_column: str, date_column: str = "date",
                   period: str = "daily", group_by: str | None = None) -> str:
    """
    Analyze trends over time.

    Args:
        name: Name of the dataframe
        value_column: Column containing values to analyze
        date_column: Column containing dates (default: 'date')
        period: Aggregation period - 'daily', 'weekly', 'monthly' (default: 'daily')
        group_by: Optional column to group by

    Returns:
        Trend analysis with statistics.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name].copy()
    df['_date'] = pd.to_datetime(df[date_column])

    # Set period
    if period == 'weekly':
        df['_period'] = df['_date'].dt.isocalendar().week
    elif period == 'monthly':
        df['_period'] = df['_date'].dt.month
    else:
        df['_period'] = df['_date'].dt.date

    lines = [f"# Trend Analysis: {value_column} ({period})", ""]

    if group_by and group_by in df.columns:
        for group in df[group_by].unique():
            group_df = df[df[group_by] == group]
            trend = group_df.groupby('_period')[value_column].mean()

            # Simple trend direction
            first_half = trend.iloc[:len(trend)//2].mean()
            second_half = trend.iloc[len(trend)//2:].mean()
            direction = "increasing" if second_half > first_half else "decreasing"
            change_pct = ((second_half - first_half) / first_half * 100)

            lines.append(f"## {group}")
            lines.append(f"- Trend: {direction} ({change_pct:+.1f}%)")
            lines.append(f"- Range: {trend.min():.2f} to {trend.max():.2f}")
            lines.append("")
    else:
        trend = df.groupby('_period')[value_column].mean()
        first_half = trend.iloc[:len(trend)//2].mean()
        second_half = trend.iloc[len(trend)//2:].mean()
        direction = "increasing" if second_half > first_half else "decreasing"
        change_pct = ((second_half - first_half) / first_half * 100)

        lines.append(f"**Overall trend:** {direction} ({change_pct:+.1f}%)")
        lines.append(f"**Min:** {trend.min():.2f}")
        lines.append(f"**Max:** {trend.max():.2f}")
        lines.append(f"**Volatility (std):** {trend.std():.2f}")

    return "\n".join(lines)


@mcp.tool()
def top_bottom_analysis(name: str, value_column: str, n: int = 5,
                        group_by: str | None = None) -> str:
    """
    Find top and bottom records by a value column.

    Args:
        name: Name of the dataframe
        value_column: Column to rank by
        n: Number of top/bottom records (default: 5)
        group_by: Optional column to find top/bottom within each group

    Returns:
        Top and bottom records.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]

    lines = [f"# Top/Bottom Analysis: {value_column}", ""]

    if group_by and group_by in df.columns:
        for group in df[group_by].unique():
            group_df = df[df[group_by] == group]

            lines.append(f"## {group}")
            lines.append(f"### Top {n}")
            top = group_df.nlargest(n, value_column)[[value_column]].reset_index(drop=True)
            lines.append(top.to_string())
            lines.append(f"### Bottom {n}")
            bottom = group_df.nsmallest(n, value_column)[[value_column]].reset_index(drop=True)
            lines.append(bottom.to_string())
            lines.append("")
    else:
        lines.append(f"## Top {n}")
        lines.append(df.nlargest(n, value_column).to_string())
        lines.append("")
        lines.append(f"## Bottom {n}")
        lines.append(df.nsmallest(n, value_column).to_string())

    return "\n".join(lines)


@mcp.tool()
def get_column_values(name: str, column: str, unique: bool = True, top_n: int | None = None) -> str:
    """
    Get values from a specific column.

    Args:
        name: Name of the dataframe
        column: Column name
        unique: If True, return unique values. If False, return value counts.
        top_n: Limit to top N values (useful for columns with many unique values)

    Returns:
        Column values or value counts.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]

    if column not in df.columns:
        return f"Error: Column '{column}' not found. Available columns: {', '.join(df.columns)}"

    if unique:
        values = df[column].unique()
        if top_n:
            values = values[:top_n]
        return f"Unique values in '{column}' ({len(df[column].unique())} total):\n{values}"
    else:
        counts = df[column].value_counts()
        if top_n:
            counts = counts.head(top_n)
        return f"Value counts for '{column}':\n{counts.to_string()}"


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

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


@mcp.tool()
def plot_comparison(name: str, value_column: str, group_column: str,
                    chart_type: str = "bar", title: str | None = None) -> str:
    """
    Create a comparison chart across groups.

    Args:
        name: Name of the dataframe
        value_column: Column containing values to plot
        group_column: Column containing groups (e.g., 'city')
        chart_type: Type of chart - 'bar', 'horizontal_bar', 'box' (default: 'bar')
        title: Optional chart title

    Returns:
        Base64 encoded plot image.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]
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
            ax.bar(grouped.index, grouped.values, color=colors)
            ax.set_ylabel(value_column)
            plt.xticks(rotation=45, ha='right')

    ax.set_title(title or f'{value_column} by {group_column}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    img_base64 = _fig_to_base64(fig)
    plt.close(fig)

    return f"Generated comparison plot:\ndata:image/png;base64,{img_base64}"


@mcp.tool()
def plot_time_series(name: str, value_column: str, date_column: str = "date",
                     group_by: str | None = None, title: str | None = None) -> str:
    """
    Create a time series plot.

    Args:
        name: Name of the dataframe
        value_column: Column containing values to plot
        date_column: Column containing dates (default: 'date')
        group_by: Optional column to create separate lines for each group
        title: Optional chart title

    Returns:
        Base64 encoded plot image.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name].copy()
    df['_date'] = pd.to_datetime(df[date_column])
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

    return f"Generated time series plot:\ndata:image/png;base64,{img_base64}"


@mcp.tool()
def plot_distribution(name: str, value_column: str, group_by: str | None = None,
                      bins: int = 30, title: str | None = None) -> str:
    """
    Create a distribution histogram.

    Args:
        name: Name of the dataframe
        value_column: Column containing values to plot
        group_by: Optional column to create overlaid distributions
        bins: Number of histogram bins (default: 30)
        title: Optional chart title

    Returns:
        Base64 encoded plot image.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]
    _setup_plot_style()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(10, 6))

    if group_by and group_by in df.columns:
        groups = df[group_by].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for group, color in zip(groups, colors):
            data = df[df[group_by] == group][value_column].dropna()
            ax.hist(data, bins=bins, alpha=0.6, label=group, color=color, edgecolor='white')
        ax.legend()
    else:
        ax.hist(df[value_column].dropna(), bins=bins, color='#3498db',
               alpha=0.7, edgecolor='white')

    ax.set_xlabel(value_column)
    ax.set_ylabel('Frequency')
    ax.set_title(title or f'Distribution of {value_column}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    img_base64 = _fig_to_base64(fig)
    plt.close(fig)

    return f"Generated distribution plot:\ndata:image/png;base64,{img_base64}"


@mcp.tool()
def plot_hourly_pattern(name: str, value_column: str, hour_column: str = "hour",
                        group_by: str | None = None, title: str | None = None) -> str:
    """
    Create an hourly pattern plot.

    Args:
        name: Name of the dataframe
        value_column: Column containing values to plot
        hour_column: Column containing hour (0-23, default: 'hour')
        group_by: Optional column to create separate lines for each group
        title: Optional chart title

    Returns:
        Base64 encoded plot image.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name]
    _setup_plot_style()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(12, 6))

    if group_by and group_by in df.columns:
        groups = df[group_by].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for group, color in zip(groups, colors):
            group_df = df[df[group_by] == group]
            hourly = group_df.groupby(hour_column)[value_column].mean()
            ax.plot(hourly.index, hourly.values, label=group, color=color,
                   linewidth=2, marker='o', markersize=4)
        ax.legend()
    else:
        hourly = df.groupby(hour_column)[value_column].mean()
        ax.plot(hourly.index, hourly.values, color='#3498db',
               linewidth=2, marker='o', markersize=6)

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(value_column)
    ax.set_title(title or f'{value_column} by Hour', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    img_base64 = _fig_to_base64(fig)
    plt.close(fig)

    return f"Generated hourly pattern plot:\ndata:image/png;base64,{img_base64}"


@mcp.tool()
def plot_weekday_weekend(name: str, value_column: str, day_of_week_column: str = "day_of_week",
                         group_by: str | None = None, title: str | None = None) -> str:
    """
    Create a weekday vs weekend comparison bar chart.

    Args:
        name: Name of the dataframe
        value_column: Column containing values to compare
        day_of_week_column: Column containing day names (default: 'day_of_week')
        group_by: Optional column to group by (e.g., 'city')
        title: Optional chart title

    Returns:
        Base64 encoded plot image.
    """
    if name not in _dataframes:
        available = ", ".join(_dataframes.keys()) if _dataframes else "none"
        return f"Error: Dataframe '{name}' not found. Available: {available}"

    df = _dataframes[name].copy()
    _setup_plot_style()
    plt.close('all')

    weekend_days = ['Saturday', 'Sunday']
    df['_day_type'] = df[day_of_week_column].apply(
        lambda x: 'Weekend' if x in weekend_days else 'Weekday'
    )

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

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    else:
        comparison = df.groupby('_day_type')[value_column].mean()
        colors = ['#e74c3c', '#3498db']
        ax.bar(comparison.index, comparison.values, color=colors, alpha=0.8)

    ax.set_ylabel(value_column)
    ax.set_title(title or f'Weekday vs Weekend: {value_column}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    img_base64 = _fig_to_base64(fig)
    plt.close(fig)

    return f"Generated weekday vs weekend plot:\ndata:image/png;base64,{img_base64}"


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
