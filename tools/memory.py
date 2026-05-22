"""
Memory Tool - persistent working memory for the agent.

Stores notes as JSON. Supports write, read-all, and keyword search.
This implements a simple but effective "working memory" IR layer.
"""

import json
import re
from pathlib import Path
from datetime import datetime

from tools.query_expansion import expand_query


MEMORY_FILE = Path(".ir_agent_memory.json")


class MemoryTool:
    def __init__(self):
        self._data: dict[str, dict] = self._load()

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "memory_write",
                    "description": (
                        "Save a note or fact to persistent memory. Use this to remember "
                        "important information, user preferences, or results across conversations."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "A short identifier for this memory (e.g. 'user_name', 'project_goal')",
                            },
                            "value": {
                                "type": "string",
                                "description": "The content to remember.",
                            },
                        },
                        "required": ["key", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_read",
                    "description": "Read all saved memory notes.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "Search memory notes by keyword.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Keywords to search for in memory.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def memory_write(self, key: str, value: str) -> dict:
        self._data[key] = {
            "value": value,
            "updated_at": datetime.now().isoformat(),
        }
        self._save()
        return {"status": "saved", "key": key}

    def memory_read(self) -> dict:
        if not self._data:
            return {"memory": [], "count": 0}
        items = [
            {"key": k, "value": v["value"], "updated_at": v["updated_at"]}
            for k, v in self._data.items()
        ]
        return {"memory": items, "count": len(items)}

    def memory_search(self, query: str) -> dict:
        # Expand query to catch vocabulary mismatches
        # e.g. user asks "job" but memory key says "profession" or "occupation"
        queries = expand_query(query, n=3)

        seen_keys: set[str] = set()
        matches: list[dict] = []

        for q in queries:
            pattern = re.compile(re.escape(q), re.IGNORECASE)
            for key, entry in self._data.items():
                if key not in seen_keys:
                    if pattern.search(key) or pattern.search(entry["value"]):
                        seen_keys.add(key)
                        matches.append({
                            "key": key,
                            "value": entry["value"],
                            "updated_at": entry["updated_at"],
                            "matched_query": q,
                        })

        return {
            "matches": matches,
            "count": len(matches),
            "query": query,
            "queries_used": queries,
        }

    def _load(self) -> dict:
        if MEMORY_FILE.exists():
            try:
                return json.loads(MEMORY_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _save(self):
        MEMORY_FILE.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))
