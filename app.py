"""
VayuChat - Natural Language Air Quality Analysis

A Gradio app for HF Spaces that provides a chat interface
for querying air quality data using natural language.
"""

import gradio as gr
import base64
import re
from pathlib import Path

# Import the MCP server functions directly
from src.vayuchat_mcp.server import (
    _dataframes,
    list_tables,
    show_table,
    describe_table,
    query_table,
    compare_weekday_weekend,
    compare_cities,
    analyze_correlation,
    analyze_funding,
    get_city_profile,
    plot_comparison,
    plot_time_series,
    plot_weekday_weekend,
    plot_funding_trend,
    plot_hourly_pattern,
)


# Available functions for the LLM to call
AVAILABLE_FUNCTIONS = {
    "list_tables": list_tables,
    "show_table": show_table,
    "describe_table": describe_table,
    "query_table": query_table,
    "compare_weekday_weekend": compare_weekday_weekend,
    "compare_cities": compare_cities,
    "analyze_correlation": analyze_correlation,
    "analyze_funding": analyze_funding,
    "get_city_profile": get_city_profile,
    "plot_comparison": plot_comparison,
    "plot_time_series": plot_time_series,
    "plot_weekday_weekend": plot_weekday_weekend,
    "plot_funding_trend": plot_funding_trend,
    "plot_hourly_pattern": plot_hourly_pattern,
}


def extract_image_from_response(response: str) -> tuple[str, str | None]:
    """Extract base64 image from response if present."""
    if "data:image/png;base64," in response:
        # Extract image
        match = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', response)
        if match:
            img_data = match.group(1)
            # Remove image from text response
            text = re.sub(r'data:image/png;base64,[A-Za-z0-9+/=]+', '[Chart displayed below]', response)
            return text.strip(), img_data
    return response, None


def process_query(query: str, history: list) -> tuple[str, str | None]:
    """
    Process a natural language query and return response with optional image.

    This is a simple keyword-based router. In production, you'd use an LLM
    to parse the query and decide which function to call.
    """
    query_lower = query.lower()

    # Simple keyword-based routing
    if any(w in query_lower for w in ["tables", "available", "what data", "datasets"]):
        return list_tables(), None

    elif "profile" in query_lower:
        for city in ["Delhi", "Bangalore", "Mumbai", "Chennai", "Kolkata", "Hyderabad"]:
            if city.lower() in query_lower:
                return get_city_profile(city), None
        return "Please specify a city (Delhi, Bangalore, Mumbai, Chennai, Kolkata, Hyderabad)", None

    elif "weekday" in query_lower and "weekend" in query_lower:
        # Determine if they want a plot or stats
        if any(w in query_lower for w in ["plot", "chart", "show", "visualize", "graph"]):
            metric = "PM2.5"
            for m in ["PM10", "NO2", "SO2", "CO", "O3"]:
                if m.lower() in query_lower:
                    metric = m
                    break
            response = plot_weekday_weekend(metric, group_by="city")
            return extract_image_from_response(response)
        else:
            metric = "PM2.5"
            for m in ["PM10", "NO2", "SO2", "CO", "O3"]:
                if m.lower() in query_lower:
                    metric = m
                    break
            return compare_weekday_weekend(metric, group_by="city"), None

    elif "compare" in query_lower and "cities" in query_lower:
        metric = "PM2.5"
        for m in ["PM10", "NO2", "SO2", "CO", "O3"]:
            if m.lower() in query_lower:
                metric = m
                break
        if any(w in query_lower for w in ["plot", "chart", "show", "visualize"]):
            response = plot_comparison(metric, group_column="city")
            return extract_image_from_response(response)
        return compare_cities(metric), None

    elif "funding" in query_lower:
        if any(w in query_lower for w in ["trend", "plot", "chart", "over time"]):
            response = plot_funding_trend()
            return extract_image_from_response(response)

        # Check for specific city
        for city in ["Delhi", "Bangalore", "Mumbai", "Chennai", "Kolkata", "Hyderabad"]:
            if city.lower() in query_lower:
                return analyze_funding(city=city), None

        # Check for specific year
        for year in [2020, 2021, 2022, 2023, 2024]:
            if str(year) in query_lower:
                return analyze_funding(year=year), None

        return analyze_funding(), None

    elif "hourly" in query_lower or "hour" in query_lower:
        metric = "PM2.5"
        for m in ["PM10", "NO2", "SO2", "CO", "O3"]:
            if m.lower() in query_lower:
                metric = m
                break
        response = plot_hourly_pattern(metric, group_by="city")
        return extract_image_from_response(response)

    elif "time series" in query_lower or "over time" in query_lower or "trend" in query_lower:
        metric = "PM2.5"
        for m in ["PM10", "NO2", "SO2", "CO", "O3"]:
            if m.lower() in query_lower:
                metric = m
                break
        response = plot_time_series(metric, group_by="city")
        return extract_image_from_response(response)

    elif "correlation" in query_lower:
        target = None
        for m in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "temperature", "humidity"]:
            if m.lower() in query_lower:
                target = m
                break
        return analyze_correlation(target=target), None

    elif any(w in query_lower for w in ["show", "display"]) and "table" in query_lower:
        for table in ["air_quality", "funding", "city_info"]:
            if table.replace("_", " ") in query_lower or table in query_lower:
                return show_table(table, rows=10), None
        return show_table("air_quality", rows=10), None

    elif "describe" in query_lower:
        for table in ["air_quality", "funding", "city_info"]:
            if table.replace("_", " ") in query_lower or table in query_lower:
                return describe_table(table), None
        return describe_table("air_quality"), None

    else:
        # Default help message
        return """I can help you analyze air quality data! Try asking:

**Data Exploration:**
- "What tables are available?"
- "Show me the funding table"
- "Describe the air quality data"

**Analysis:**
- "Compare weekday vs weekend PM2.5 for Delhi and Bangalore"
- "Compare cities by PM2.5 levels"
- "Show correlation with PM2.5"
- "Get Delhi city profile"

**Funding:**
- "Show funding analysis"
- "What's the funding trend?"
- "Funding for Delhi"

**Visualizations:**
- "Plot weekday vs weekend comparison"
- "Show hourly PM2.5 pattern"
- "Plot PM2.5 time series"
- "Show funding trend chart"
""", None


def chat(message: str, history: list) -> tuple[list, str | None]:
    """Chat function for Gradio."""
    response_text, image_data = process_query(message, history)

    # Add to history
    history.append((message, response_text))

    return history, image_data


def decode_image(image_data: str | None):
    """Decode base64 image for display."""
    if image_data:
        import io
        from PIL import Image
        img_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(img_bytes))
    return None


# Custom CSS for dark theme
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.chat-message {
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
}
"""

# Build the Gradio interface
with gr.Blocks(title="VayuChat - Air Quality Analysis", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌬️ VayuChat - Air Quality Analysis

    Ask questions about air quality data for Indian cities in natural language.

    **Available Data:**
    - 📊 Air Quality: Hourly PM2.5, PM10, NO2, SO2, CO, O3 for Delhi & Bangalore
    - 💰 Funding: Government air quality funding by city/year
    - 🏙️ City Info: Population, vehicles, industries, green cover
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                show_label=False,
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about air quality... (e.g., 'Compare weekday vs weekend PM2.5')",
                    show_label=False,
                    scale=4,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            # Example queries
            gr.Examples(
                examples=[
                    "What tables are available?",
                    "Compare weekday vs weekend PM2.5 for cities",
                    "Plot hourly PM2.5 pattern",
                    "Show funding trend chart",
                    "Get Delhi city profile",
                    "Show correlation with PM2.5",
                ],
                inputs=msg,
            )

        with gr.Column(scale=1):
            image_output = gr.Image(
                label="Visualization",
                show_label=True,
                height=400,
            )

            gr.Markdown("""
            ### Quick Stats
            """)

            # Show quick stats
            if _dataframes:
                stats_text = ""
                for name, df in _dataframes.items():
                    stats_text += f"**{name}:** {len(df):,} rows\n"
                gr.Markdown(stats_text)

    # Event handlers
    def respond(message, history):
        history, img_data = chat(message, history)
        img = decode_image(img_data)
        return history, img, ""

    msg.submit(respond, [msg, chatbot], [chatbot, image_output, msg])
    submit_btn.click(respond, [msg, chatbot], [chatbot, image_output, msg])


if __name__ == "__main__":
    demo.launch()
