"""
Intent Router - Routes natural language queries to appropriate functions.

Supports:
1. Local LLM (LM Studio, Ollama) via OpenAI-compatible API
2. HF Inference API (free tier)
3. Local transformers (SmolLM2, Qwen2.5)
4. Fallback to keyword matching
"""

import os
import json
import re
from typing import Callable

# Try imports - graceful fallback if not available
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from huggingface_hub import InferenceClient
    HAS_HF = True
except ImportError:
    HAS_HF = False


# Available functions and their descriptions for the LLM
FUNCTION_DESCRIPTIONS = """
Available functions:
1. list_tables() - List all available data tables
2. show_table(name) - Show rows from a table. name: "air_quality", "funding", or "city_info"
3. describe_table(name) - Get statistics for a table
4. get_ranking(metric, rank_type) - Get city with highest/lowest metric. metric: "PM2.5", "PM10", "NO2", etc. rank_type: "highest" or "lowest"
5. compare_cities(metric) - Compare all cities by a metric
6. compare_weekday_weekend(metric, group_by) - Compare weekday vs weekend. group_by: "city" or None
7. get_city_profile(city) - Get full profile for a city. city: "Delhi", "Bangalore", "Mumbai", "Chennai", "Kolkata", "Hyderabad"
8. analyze_funding(city, year) - Analyze funding data. city and year are optional filters
9. analyze_correlation(target) - Show correlations with a target metric
10. plot_comparison(metric) - Bar chart comparing cities
11. plot_weekday_weekend(metric) - Weekday vs weekend bar chart
12. plot_hourly_pattern(metric) - Hourly pattern line chart
13. plot_time_series(metric) - Time series line chart
14. plot_funding_trend() - Funding trend over years
"""

ROUTER_PROMPT = """You are a query router. Given a user question about air quality data, determine which function to call.

{functions}

User query: "{query}"

Respond with ONLY a JSON object (no markdown, no explanation):
{{"function": "function_name", "params": {{"param1": "value1"}}}}

Examples:
- "which city has highest PM2.5" -> {{"function": "get_ranking", "params": {{"metric": "PM2.5", "rank_type": "highest"}}}}
- "show me the funding table" -> {{"function": "show_table", "params": {{"name": "funding"}}}}
- "compare weekday weekend NO2" -> {{"function": "compare_weekday_weekend", "params": {{"metric": "NO2", "group_by": "city"}}}}
- "Delhi profile" -> {{"function": "get_city_profile", "params": {{"city": "Delhi"}}}}
- "plot PM10 by hour" -> {{"function": "plot_hourly_pattern", "params": {{"metric": "PM10"}}}}

JSON response:"""


class IntentRouter:
    """Routes queries to functions using LLM or keyword fallback."""

    def __init__(self, mode: str = "auto"):
        """
        Initialize router.

        Args:
            mode: "local" (LM Studio/Ollama), "hf" (HF Inference), "keywords" (no LLM), "auto"
        """
        self.mode = mode
        self.client = None

        if mode == "auto":
            self._auto_detect()
        elif mode == "local":
            self._setup_local()
        elif mode == "hf":
            self._setup_hf()

    def _auto_detect(self):
        """Auto-detect available LLM backend."""
        # Try local first (LM Studio default port)
        if HAS_OPENAI:
            try:
                client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
                # Quick test
                client.models.list()
                self.client = client
                self.mode = "local"
                print("Router: Using local LLM (LM Studio)")
                return
            except:
                pass

        # Try HF Inference API
        if HAS_HF:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if hf_token:
                self.client = InferenceClient(token=hf_token)
                self.mode = "hf"
                print("Router: Using HF Inference API")
                return
            else:
                # Try without token (some models work without)
                self.client = InferenceClient()
                self.mode = "hf"
                print("Router: Using HF Inference API (no token)")
                return

        # Fallback to keywords
        self.mode = "keywords"
        print("Router: Using keyword matching (no LLM available)")

    def _setup_local(self):
        """Setup local LLM client (LM Studio/Ollama)."""
        if not HAS_OPENAI:
            raise ImportError("openai package required for local LLM. Run: pip install openai")

        base_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:1234/v1")
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

    def _setup_hf(self):
        """Setup HF Inference client."""
        if not HAS_HF:
            raise ImportError("huggingface_hub required. Run: pip install huggingface_hub")

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.client = InferenceClient(token=token)

    def route(self, query: str) -> dict:
        """
        Route a query to a function.

        Returns:
            {"function": "name", "params": {...}} or {"function": "help", "params": {}}
        """
        if self.mode == "keywords":
            return self._keyword_route(query)

        try:
            if self.mode == "local":
                return self._local_route(query)
            elif self.mode == "hf":
                return self._hf_route(query)
        except Exception as e:
            print(f"LLM routing failed: {e}, falling back to keywords")
            return self._keyword_route(query)

    def _local_route(self, query: str) -> dict:
        """Route using local LLM."""
        prompt = ROUTER_PROMPT.format(functions=FUNCTION_DESCRIPTIONS, query=query)

        response = self.client.chat.completions.create(
            model="local-model",  # LM Studio ignores this
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )

        return self._parse_response(response.choices[0].message.content)

    def _hf_route(self, query: str) -> dict:
        """Route using HF Inference API."""
        prompt = ROUTER_PROMPT.format(functions=FUNCTION_DESCRIPTIONS, query=query)

        # Try different models - some may not be available
        models_to_try = [
            "microsoft/Phi-3-mini-4k-instruct",
            "HuggingFaceH4/zephyr-7b-beta",
            "mistralai/Mistral-7B-Instruct-v0.2",
        ]

        for model in models_to_try:
            try:
                response = self.client.text_generation(
                    prompt,
                    model=model,
                    max_new_tokens=150,
                    temperature=0.1,
                )
                return self._parse_response(response)
            except Exception:
                continue

        # If all models fail, use keyword fallback
        return self._keyword_route(query)

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response to extract function call."""
        # Try to find JSON in response
        try:
            # Clean up response
            response = response.strip()

            # Find JSON object
            match = re.search(r'\{[^{}]*\}', response)
            if match:
                result = json.loads(match.group())
                if "function" in result:
                    return result
        except json.JSONDecodeError:
            pass

        # Fallback to keywords if parsing fails
        return {"function": "help", "params": {}}

    def _keyword_route(self, query: str) -> dict:
        """Fallback keyword-based routing."""
        q = query.lower()

        # Helper to detect metric
        def get_metric():
            for m in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
                if m.lower() in q:
                    return m
            return "PM2.5"

        # Helper to detect city
        def get_city():
            for c in ["Delhi", "Bangalore", "Mumbai", "Chennai", "Kolkata", "Hyderabad"]:
                if c.lower() in q:
                    return c
            return None

        # Ranking questions
        if any(w in q for w in ["highest", "lowest", "maximum", "minimum", "max", "min", "most", "least"]):
            rank_type = "highest" if any(w in q for w in ["highest", "max", "most"]) else "lowest"
            return {"function": "get_ranking", "params": {"metric": get_metric(), "rank_type": rank_type}}

        # City profile
        if "profile" in q:
            city = get_city()
            if city:
                return {"function": "get_city_profile", "params": {"city": city}}

        # Weekday/weekend
        if "weekday" in q and "weekend" in q:
            if any(w in q for w in ["plot", "chart", "graph", "show"]):
                return {"function": "plot_weekday_weekend", "params": {"metric": get_metric()}}
            return {"function": "compare_weekday_weekend", "params": {"metric": get_metric(), "group_by": "city"}}

        # Compare cities
        if "compare" in q and ("city" in q or "cities" in q):
            if any(w in q for w in ["plot", "chart"]):
                return {"function": "plot_comparison", "params": {"metric": get_metric()}}
            return {"function": "compare_cities", "params": {"metric": get_metric()}}

        # Hourly
        if "hourly" in q or "hour" in q:
            return {"function": "plot_hourly_pattern", "params": {"metric": get_metric()}}

        # Time series
        if "time series" in q or "over time" in q or "trend" in q:
            if "funding" in q:
                return {"function": "plot_funding_trend", "params": {}}
            return {"function": "plot_time_series", "params": {"metric": get_metric()}}

        # Funding
        if "funding" in q or "budget" in q:
            if any(w in q for w in ["plot", "chart", "trend"]):
                return {"function": "plot_funding_trend", "params": {}}
            city = get_city()
            return {"function": "analyze_funding", "params": {"city": city}}

        # Correlation
        if "correlation" in q:
            return {"function": "analyze_correlation", "params": {"target": get_metric()}}

        # Tables
        if any(w in q for w in ["tables", "available", "what data", "list"]):
            return {"function": "list_tables", "params": {}}

        # Show table
        if "show" in q and "table" in q:
            for t in ["funding", "city_info", "air_quality"]:
                if t.replace("_", " ") in q or t in q:
                    return {"function": "show_table", "params": {"name": t}}
            return {"function": "show_table", "params": {"name": "air_quality"}}

        # Describe
        if "describe" in q or "statistics" in q:
            return {"function": "describe_table", "params": {"name": "air_quality"}}

        # Average/mean for city
        if any(w in q for w in ["average", "mean", "avg"]):
            city = get_city()
            if city:
                return {"function": "get_city_profile", "params": {"city": city}}
            return {"function": "compare_cities", "params": {"metric": get_metric()}}

        # What is questions
        if q.startswith(("what is", "what's", "how much", "how high")):
            city = get_city()
            if city:
                return {"function": "get_city_profile", "params": {"city": city}}
            return {"function": "compare_cities", "params": {"metric": get_metric()}}

        # Default
        return {"function": "help", "params": {}}


# Global router instance
_router = None

def get_router() -> IntentRouter:
    """Get or create the global router instance."""
    global _router
    if _router is None:
        _router = IntentRouter(mode="auto")
    return _router


def route_query(query: str) -> dict:
    """Convenience function to route a query."""
    return get_router().route(query)
