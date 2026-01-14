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

# Import core analysis functions
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

# Import router
from src.vayuchat_mcp.router import route_query, get_router

# Map function names to actual functions
FUNCTIONS = {
    "list_tables": list_tables,
    "show_table": show_table,
    "describe_table": describe_table,
    "get_ranking": get_ranking,
    "compare_cities": compare_cities,
    "compare_weekday_weekend": compare_weekday_weekend,
    "get_city_profile": get_city_profile,
    "analyze_funding": analyze_funding,
    "analyze_correlation": analyze_correlation,
    "plot_comparison": plot_comparison,
    "plot_time_series": plot_time_series,
    "plot_weekday_weekend": plot_weekday_weekend,
    "plot_funding_trend": plot_funding_trend,
    "plot_hourly_pattern": plot_hourly_pattern,
}

HELP_TEXT = """I can help you analyze air quality data! Try asking:

**Data Exploration:**
- "What tables are available?"
- "Show me the funding table"
- "Describe the air quality data"

**Analysis:**
- "Which city has highest PM2.5?"
- "Compare weekday vs weekend PM2.5"
- "Compare cities by PM10 levels"
- "Get Delhi city profile"
- "Average NO2 in Bangalore"

**Funding:**
- "Show funding analysis"
- "Plot funding trend"
- "Funding for Delhi"

**Visualizations:**
- "Plot weekday vs weekend comparison"
- "Show hourly PM2.5 pattern"
- "Plot PM2.5 time series"
"""


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
    """Process a natural language query using the router."""
    # Route the query
    route = route_query(query)
    func_name = route.get("function", "help")
    params = route.get("params", {})

    # Handle help/unknown
    if func_name == "help" or func_name not in FUNCTIONS:
        return HELP_TEXT, None

    # Call the function
    try:
        func = FUNCTIONS[func_name]
        # Filter out None params
        params = {k: v for k, v in params.items() if v is not None}
        result = func(**params)
        return extract_image_from_response(result)
    except Exception as e:
        return f"Error: {e}\n\n{HELP_TEXT}", None


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


# Initialize router on startup
router = get_router()

# Build the Gradio interface
with gr.Blocks(title="VayuChat - Air Quality Analysis") as demo:
    gr.Markdown(f"""
    # VayuChat - Air Quality Analysis

    Ask questions about air quality data for Indian cities in natural language.

    **Router Mode:** `{router.mode}` {'(LLM-powered)' if router.mode != 'keywords' else '(keyword matching)'}

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
                    placeholder="Ask about air quality... (e.g., 'Which city has highest PM2.5?')",
                    show_label=False,
                    scale=4,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "Which city has highest PM2.5?",
                    "Compare weekday vs weekend PM2.5",
                    "Plot hourly NO2 pattern",
                    "Show funding trend chart",
                    "Get Delhi city profile",
                    "What tables are available?",
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
