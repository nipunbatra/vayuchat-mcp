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


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
