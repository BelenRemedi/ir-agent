"""
Web Search Tool - fetches results from DuckDuckGo Instant Answer API.

No API key required. For richer results, users can configure a
SerpAPI or Brave Search key via environment variables.
"""

import os
import urllib.parse
import urllib.request
import json


class WebSearchTool:
    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the web for current information. "
                        "Use this for recent events, facts you don't know, or to verify information."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query.",
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return (default 5, max 10).",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    def web_search(self, query: str, num_results: int = 5) -> dict:
        # Try Brave Search if API key is available
        brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if brave_key:
            return self._brave_search(query, num_results, brave_key)

        # Fallback: DuckDuckGo Instant Answer API
        return self._ddg_search(query, num_results)

    def _brave_search(self, query: str, num_results: int, api_key: str) -> dict:
        encoded = urllib.parse.quote(query)
        url = f"https://api.search.brave.com/res/v1/web/search?q={encoded}&count={num_results}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            results = []
            for item in data.get("web", {}).get("results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                })
            return {"results": results, "query": query, "source": "brave"}
        except Exception as e:
            return {"error": str(e), "query": query}

    def _ddg_search(self, query: str, num_results: int) -> dict:
        """DuckDuckGo Instant Answer API - provides abstract/related topics."""
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "IR-Agent/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            results = []

            # Main abstract
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", "Abstract"),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data["AbstractText"],
                    "source": data.get("AbstractSource", ""),
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:num_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                    })

            if not results:
                return {
                    "results": [],
                    "query": query,
                    "message": (
                        "No instant results found. "
                        "Set BRAVE_SEARCH_API_KEY env var for better web search. "
                        f"Try: https://search.brave.com/search?q={encoded}"
                    ),
                }

            return {"results": results[:num_results], "query": query, "source": "duckduckgo"}
        except Exception as e:
            return {"error": str(e), "query": query}
