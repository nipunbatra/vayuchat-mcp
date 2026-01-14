"""
VayuChat MCP Server - Natural language data analysis for air quality data.

Pre-loaded datasets:
- air_quality: Hourly PM2.5, PM10, NO2, etc. for Delhi & Bangalore
- funding: Government funding for air quality initiatives by city/year
- city_info: City metadata (population, vehicles, industries, etc.)
"""

import sys
import io
import traceback
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastmcp import FastMCP

# Import core analysis functions
from . import analysis

# Create the MCP server
mcp = FastMCP("VayuChat")


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
    return analysis.list_tables()


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
    return analysis.show_table(name, rows, columns)


@mcp.tool()
def describe_table(name: str) -> str:
    """
    Get detailed statistics for a table.

    Args:
        name: Table name

    Returns:
        Statistical summary and column info.
    """
    return analysis.describe_table(name)


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
    return analysis.query_table(name, query)


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
    return analysis.compare_weekday_weekend(value_column, group_by, table)


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
    return analysis.compare_cities(value_column, cities, table)


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
    return analysis.analyze_correlation(columns, target, table)


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
    return analysis.analyze_funding(city, year)


@mcp.tool()
def get_city_profile(city: str) -> str:
    """
    Get comprehensive profile for a city including all available data.

    Args:
        city: City name (Delhi, Bangalore, Mumbai, etc.)

    Returns:
        City profile with air quality, funding, and metadata.
    """
    return analysis.get_city_profile(city)


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
    return analysis.plot_comparison(value_column, group_column, chart_type, table, title)


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
    return analysis.plot_time_series(value_column, group_by, table, title)


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
    return analysis.plot_weekday_weekend(value_column, group_by, table, title)


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
    return analysis.plot_funding_trend(cities, title)


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
    return analysis.plot_hourly_pattern(value_column, group_by, table, title)


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
    analysis.setup_plot_style()

    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    dfs = analysis.get_dataframes()
    namespace: dict[str, Any] = {
        'pd': pd, 'np': np, 'plt': plt,
        **dfs
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
            img_base64 = analysis.fig_to_base64(fig)
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
