# VayuChat MCP

Natural language data analysis for air quality data using MCP (Model Context Protocol).

## Features

### Pre-loaded Datasets
- **air_quality**: Hourly PM2.5, PM10, NO2, SO2, CO, O3 readings for Delhi & Bangalore
- **funding**: Government air quality funding by city/year (2020-2024)
- **city_info**: City metadata - population, vehicles, industries, green cover

### Analysis Tools (No Code Required!)
| Function | Description |
|----------|-------------|
| `list_tables` | Show available tables |
| `show_table` | Display table data |
| `describe_table` | Detailed statistics |
| `query_table` | Filter with pandas query |
| `compare_weekday_weekend` | Weekday vs weekend analysis |
| `compare_cities` | Compare metrics across cities |
| `analyze_correlation` | Correlation analysis |
| `analyze_funding` | Funding breakdown |
| `get_city_profile` | Comprehensive city profile |

### Visualization Tools
| Function | Description |
|----------|-------------|
| `plot_comparison` | Bar/box charts |
| `plot_time_series` | Time series charts |
| `plot_weekday_weekend` | Weekday vs weekend bars |
| `plot_funding_trend` | Funding over years |
| `plot_hourly_pattern` | Hourly patterns |

## Installation

```bash
# Using uv
uv pip install -e .

# Or with pip
pip install -e .
```

## Usage

### As MCP Server (with Claude Code)

Add to your Claude Code MCP configuration:

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

### As Gradio App (HF Spaces)

```bash
# Run locally
python app.py

# Or with gradio
gradio app.py
```

Then open http://localhost:7860

### Deploy to Hugging Face Spaces

1. Create a new Space on HF (Gradio SDK)
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `src/` folder
   - `data/` folder

Or connect your GitHub repo directly to HF Spaces.

## Example Queries

```
# Data exploration
"What tables are available?"
"Show me the funding table"
"Describe the air quality data"

# Analysis
"Compare weekday vs weekend PM2.5"
"Compare cities by PM10 levels"
"Get Delhi city profile"
"Show correlation with PM2.5"

# Funding
"Show funding for Delhi"
"What's the funding trend?"

# Visualizations
"Plot weekday vs weekend PM2.5"
"Show hourly pattern for NO2"
"Plot funding trend chart"
```

## Architecture

```
NLQ (User Question)
       ↓
  Gradio Chat UI
       ↓
  Query Router (keyword-based / LLM)
       ↓
  MCP Tool Call
       ↓
  Response (Markdown + Base64 Plot)
       ↓
  Rendered in UI
```

## Data Sources

- Air quality data: Simulated based on real patterns from Indian cities
- Funding data: Mock data representing typical government allocations
- City info: Approximate real statistics

## License

MIT
