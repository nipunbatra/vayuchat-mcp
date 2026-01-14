# VayuChat MCP

A FastMCP server for natural language data analysis with pandas and matplotlib.

This is an MCP (Model Context Protocol) server implementation of [VayuChat](https://github.com/nipunbatra/vayuchat-webllm), allowing you to analyze CSV data using natural language through Claude or any other MCP-compatible client.

## Features

- **Load CSV files** into pandas DataFrames
- **Explore data** with detailed column information and statistics
- **Execute Python code** with full access to pandas, numpy, and matplotlib
- **Generate visualizations** that are returned as base64-encoded images
- **Query data** using pandas query syntax

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Usage

### With Claude Code

Add to your Claude Code MCP configuration (`~/.claude/claude_desktop_config.json` or project settings):

```json
{
  "mcpServers": {
    "vayuchat": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/vayuchat-mcp", "vayuchat-mcp"]
    }
  }
}
```

Or if installed globally:

```json
{
  "mcpServers": {
    "vayuchat": {
      "command": "vayuchat-mcp"
    }
  }
}
```

### Running Standalone

```bash
# With uv
uv run vayuchat-mcp

# Or directly
vayuchat-mcp
```

## Available Tools

### `load_csv`
Load a CSV file into a pandas DataFrame.

```
load_csv(file_path="/path/to/data.csv", name="my_data")
```

### `list_dataframes`
List all currently loaded dataframes with their basic info.

### `get_dataframe_info`
Get detailed information about a specific dataframe including column types, null counts, and sample values.

```
get_dataframe_info(name="my_data")
```

### `execute_code`
Execute Python code for data analysis. Has access to:
- All loaded dataframes by their names
- `pandas` as `pd`
- `numpy` as `np`
- `matplotlib.pyplot` as `plt`

Automatically captures generated plots and returns them as base64 images.

```python
execute_code(code="""
# Calculate average by category
result = my_data.groupby('category')['value'].mean()
print(result)

# Create a visualization
plt.figure(figsize=(10, 6))
result.plot(kind='bar')
plt.title('Average Value by Category')
plt.tight_layout()
""")
```

### `query_dataframe`
Run a pandas query on a dataframe.

```
query_dataframe(name="my_data", query="value > 100 and category == 'A'")
```

### `describe_dataframe`
Get statistical summary of a dataframe.

```
describe_dataframe(name="my_data", columns=["value", "count"])
```

### `sample_dataframe`
Get a sample of rows from a dataframe.

```
sample_dataframe(name="my_data", n=10, random=True)
```

### `get_column_values`
Get unique values or value counts from a column.

```
get_column_values(name="my_data", column="category", unique=False, top_n=10)
```

### `unload_dataframe`
Unload a dataframe from memory.

## Example Conversation

```
User: Load the air quality data from ~/data/air_quality.csv