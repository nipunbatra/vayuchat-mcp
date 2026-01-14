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

## Why Predefined Functions vs LLM-Generated Code?

This project uses **predefined MCP functions** instead of letting the LLM generate arbitrary pandas/matplotlib code. Here's why:

### Comparison Table

| Aspect | Predefined Functions (This Approach) | LLM-Generated Code | Function-Calling LLM |
|--------|-------------------------------------|-------------------|---------------------|
| **Reliability** | ✅ Deterministic, always works | ❌ May hallucinate syntax | ⚠️ Better but can miss params |
| **Speed** | ✅ Instant (no code generation) | ❌ Slow (generate → parse → execute) | ⚠️ Moderate |
| **Cost** | ✅ Minimal tokens | ❌ Long prompts with schema | ⚠️ Moderate |
| **Security** | ✅ No arbitrary code execution | ❌ Code injection risk | ✅ Safe |
| **Consistency** | ✅ Same visualization style | ❌ Random styling each time | ✅ Consistent |
| **Model Size** | ✅ Works with small/cheap models | ❌ Needs capable coder model | ⚠️ Needs fine-tuned model |
| **Flexibility** | ❌ Limited to predefined queries | ✅ Infinite flexibility | ⚠️ Limited to defined functions |
| **Error Handling** | ✅ Graceful, predictable | ❌ May crash, retry loops | ✅ Structured errors |

### When to Use Each Approach

**Use Predefined Functions (this approach) when:**
- You have a known, bounded set of analysis patterns
- Users are non-technical (need consistent UX)
- Cost/latency matters (production deployment)
- You want guaranteed correct outputs
- Using smaller/cheaper models (Haiku, GPT-3.5)

**Use LLM-Generated Code when:**
- Exploratory data analysis with unknown patterns
- Power users who can debug code
- One-off analyses
- Prototype/research phase

**Use Function-Calling LLM when:**
- You have predefined functions BUT need better intent parsing
- Using OpenAI/Claude with native function calling
- Queries are ambiguous and need sophisticated NLU

### The Hybrid Approach (Best of Both)

```
User Query
     ↓
┌─────────────────────────────────────┐
│  LLM with Function Calling          │  ← Parses intent, extracts params
│  (Claude, GPT-4, etc.)              │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  MCP Predefined Functions           │  ← Executes reliably
│  (compare_cities, plot_trend, etc.) │
└─────────────────────────────────────┘
     ↓
  Structured Response + Plot
```

This gives you:
- **LLM's NLU capabilities** for parsing complex queries
- **Predefined functions' reliability** for execution
- **No code hallucination** risk
- **Consistent outputs** every time

### Example: Same Query, Different Approaches

**Query:** "Compare PM2.5 on weekdays vs weekends for Delhi and Bangalore"

**LLM-Generated Code (risky):**
```python
# LLM might generate:
df['is_weekend'] = df['day'].isin(['Sat', 'Sun'])  # Wrong column name!
df.groupby(['city', 'is_weekend'])['pm25'].mean()  # Wrong column name!
# ... errors, retries, inconsistent output
```

**Predefined Function (reliable):**
```python
# MCP calls:
compare_weekday_weekend(value_column="PM2.5", group_by="city")
# Always works, consistent format, proper column names
```

### Cost Comparison (Approximate)

| Approach | Tokens per Query | Cost (GPT-4) | Latency |
|----------|------------------|--------------|---------|
| Predefined + Keyword Router | ~100 | $0.001 | <100ms |
| Predefined + LLM Router | ~500 | $0.005 | ~500ms |
| LLM-Generated Code | ~2000+ | $0.02+ | 2-5s |

For 1000 queries/day:
- Predefined: ~$1-5/day
- LLM Code Gen: ~$20+/day

## Data Sources

- Air quality data: Simulated based on real patterns from Indian cities
- Funding data: Mock data representing typical government allocations
- City info: Approximate real statistics

## License

MIT
