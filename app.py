"""
VayuChat - Natural Language Air Quality Analysis

A Gradio app for HF Spaces that provides a chat interface
for querying air quality data using natural language.
"""

import gradio as gr
import base64
import re
from PIL import Image
import io

# Import core analysis functions (not MCP-wrapped)
from src.vayuchat_mcp.analysis import (
    get_dataframes,
    list_tables,
    show_table,
    describe_table,
    query_table,
    compare_weekday_weekend,
    compare_cities,
    get_ranking,
    analyze_correlation,
    analyze_funding,
    get_city_profile,
    plot_comparison,
    plot_time_series,
    plot_weekday_weekend,
    plot_funding_trend,
    plot_hourly_pattern,
)


def extract_image_from_response(response: str) -> tuple[str, str | None]:
    """Extract base64 image from response if present."""
    if "data:image/png;base64," in response:
        match = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', response)
        if match:
            img_data = match.group(1)
            text = re.sub(r'data:image/png;base64,[A-Za-z0-9+/=]+', '', response)
            return text.strip(), img_data
    return response, None


def decode_image(image_data: str | None):
    """Decode base64 image for display."""
    if image_data:
        img_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(img_bytes))
    return None


def process_query(query: str) -> tuple[str, str | None]:
    """Process a natural language query and return response with optional image."""
    query_lower = query.lower()

    # Detect metric from query
    def get_metric(q):
        for m in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "temperature", "humidity"]:
            if m.lower() in q.lower():
                return m
        return "PM2.5"  # default

    # Highest/lowest/max/min questions
    if any(w in query_lower for w in ["highest", "lowest", "maximum", "minimum", "max", "min", "most", "least", "best", "worst"]):
        metric = get_metric(query)
        if any(w in query_lower for w in ["highest", "maximum", "max", "most", "worst"]):
            return get_ranking(metric, "highest"), None
        else:
            return get_ranking(metric, "lowest"), None

    # Average/mean questions for specific city
    if any(w in query_lower for w in ["average", "mean", "avg"]):
        metric = get_metric(query)
        for city in ["Delhi", "Bangalore", "Mumbai", "Chennai", "Kolkata", "Hyderabad"]:
            if city.lower() in query_lower:
                return get_city_profile(city), None
        # General average across cities
        return compare_cities(metric), None

    # "What is" / "How much" questions
    if query_lower.startswith(("what is", "what's", "how much", "how high", "how bad")):
        metric = get_metric(query)
        for city in ["Delhi", "Bangalore", "Mumbai", "Chennai", "Kolkata", "Hyderabad"]:
            if city.lower() in query_lower:
                return get_city_profile(city), None
        return compare_cities(metric), None

    # Data exploration
    if any(w in query_lower for w in ["tables", "available", "what data", "datasets", "list"]):
        return list_tables(), None

    # City profile
    if "profile" in query_lower:
        for city in ["Delhi", "Bangalore", "Mumbai", "Chennai", "Kolkata", "Hyderabad"]:
            if city.lower() in query_lower:
                return get_city_profile(city), None
        return "Please specify a city (Delhi, Bangalore, Mumbai, Chennai, Kolkata, Hyderabad)", None

    # Weekday vs weekend
    if "weekday" in query_lower and "weekend" in query_lower:
        metric = "PM2.5"
        for m in ["PM10", "NO2", "SO2", "CO", "O3"]:
            if m.lower() in query_lower:
                metric = m
                break

        if any(w in query_lower for w in ["plot", "chart", "show", "visualize", "graph"]):
            response = plot_weekday_weekend(metric, group_by="city")
            return extract_image_from_response(response)
        else:
            return compare_weekday_weekend(metric, group_by="city"), None

    # Compare cities
    if "compare" in query_lower and ("cities" in query_lower or "city" in query_lower):
        metric = "PM2.5"
        for m in ["PM10", "NO2", "SO2", "CO", "O3"]:
            if m.lower() in query_lower:
                metric = m
                break

        if any(w in query_lower for w in ["plot", "chart", "show", "visualize"]):
            response = plot_comparison(metric, group_column="city")
            return extract_image_from_response(response)
        return compare_cities(metric), None

    # Funding
    if "funding" in query_lower or "budget" in query_lower:
        if any(w in query_lower for w in ["trend", "plot", "chart", "over time", "graph"]):
            response = plot_funding_trend()
            return extract_image_from_response(response)

        for city in ["Delhi", "Bangalore", "Mumbai", "Chennai", "Kolkata", "Hyderabad"]:
            if city.lower() in query_lower:
                return analyze_funding(city=city), None

        for year in [2020, 2021, 2022, 2023, 2024]:
            if str(year) in query_lower:
                return analyze_funding(year=year), None

        return analyze_funding(), None

    # Hourly pattern
    if "hourly" in query_lower or ("hour" in query_lower and "pattern" in query_lower):
        metric = "PM2.5"
        for m in ["PM10", "NO2", "SO2", "CO", "O3"]:
            if m.lower() in query_lower:
                metric = m
                break
        response = plot_hourly_pattern(metric, group_by="city")
        return extract_image_from_response(response)

    # Time series / trend
    if "time series" in query_lower or "over time" in query_lower or ("trend" in query_lower and "funding" not in query_lower):
        metric = "PM2.5"
        for m in ["PM10", "NO2", "SO2", "CO", "O3"]:
            if m.lower() in query_lower:
                metric = m
                break
        response = plot_time_series(metric, group_by="city")
        return extract_image_from_response(response)

    # Correlation
    if "correlation" in query_lower or "correlate" in query_lower:
        target = None
        for m in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "temperature", "humidity"]:
            if m.lower() in query_lower:
                target = m
                break
        return analyze_correlation(target=target), None

    # Show table
    if any(w in query_lower for w in ["show", "display", "view"]) and "table" in query_lower:
        for table in ["air_quality", "funding", "city_info"]:
            if table.replace("_", " ") in query_lower or table in query_lower:
                return show_table(table, rows=10), None
        return show_table("air_quality", rows=10), None

    # Describe table
    if "describe" in query_lower or "statistics" in query_lower or "stats" in query_lower:
        for table in ["air_quality", "funding", "city_info"]:
            if table.replace("_", " ") in query_lower or table in query_lower:
                return describe_table(table), None
        return describe_table("air_quality"), None

    # Default help
    return """I can help you analyze air quality data! Try asking:

**Data Exploration:**
- "What tables are available?"
- "Show me the funding table"
- "Describe the air quality data"

**Analysis:**
- "Compare weekday vs weekend PM2.5"
- "Compare cities by PM10 levels"
- "Show correlation with PM2.5"
- "Get Delhi city profile"

**Funding:**
- "Show funding analysis"
- "Plot funding trend"
- "Funding for Delhi"

**Visualizations:**
- "Plot weekday vs weekend comparison"
- "Show hourly PM2.5 pattern"
- "Plot PM2.5 time series"
""", None


def respond(message: str, history: list) -> tuple[list, Image.Image | None]:
    """Handle user message and return updated history + image."""
    response_text, img_data = process_query(message)

    # Gradio 6.x uses dict format for messages
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_text}
    ]

    img = decode_image(img_data)
    return history, img


# Build the Gradio interface
with gr.Blocks(title="VayuChat - Air Quality Analysis") as demo:
    gr.Markdown("""
    # VayuChat - Air Quality Analysis

    Ask questions about air quality data for Indian cities in natural language.

    **Available Data:**
    - Air Quality: Hourly PM2.5, PM10, NO2, SO2, CO, O3 for Delhi & Bangalore
    - Funding: Government air quality funding by city/year (2020-2024)
    - City Info: Population, vehicles, industries, green cover
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=450)

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about air quality... (e.g., 'Compare weekday vs weekend PM2.5')",
                    show_label=False,
                    scale=4,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "What tables are available?",
                    "Compare weekday vs weekend PM2.5",
                    "Plot hourly PM2.5 pattern",
                    "Show funding trend chart",
                    "Get Delhi city profile",
                    "Show correlation with PM2.5",
                ],
                inputs=msg,
            )

        with gr.Column(scale=1):
            image_output = gr.Image(label="Visualization", height=400)

            gr.Markdown("### Quick Stats")
            dfs = get_dataframes()
            if dfs:
                stats_md = "\n".join([f"- **{name}:** {len(df):,} rows" for name, df in dfs.items()])
                gr.Markdown(stats_md)

    # Event handlers
    msg.submit(respond, [msg, chatbot], [chatbot, image_output]).then(
        lambda: "", outputs=[msg]
    )
    submit_btn.click(respond, [msg, chatbot], [chatbot, image_output]).then(
        lambda: "", outputs=[msg]
    )


if __name__ == "__main__":
    demo.launch()
