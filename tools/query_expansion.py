"""
Query Expansion - generates alternative query phrasings using the LLM.

This tackles the vocabulary mismatch problem in IR: the words a user
uses in their query often don't match the words in relevant documents.

Example:
  User query: "how do cars work"
  Documents may say: "automobile engine", "vehicle mechanics", "motor operation"

By expanding the query into multiple phrasings before searching, we
retrieve documents that the original query alone would miss.

Technique: LLM-based query reformulation (a modern take on the classical
pseudo-relevance feedback and thesaurus-based expansion methods).
"""

import json
import os
from openai import OpenAI


EXPANSION_PROMPT = """You are a query expansion assistant for an information retrieval system.

Given a search query, generate {n} alternative phrasings that capture the same
information need using different vocabulary. These will be used to search a
document index and memory store.

Rules:
- Each alternative should use DIFFERENT words than the original (synonyms, related terms)
- Keep alternatives concise (under 10 words each)
- Stay faithful to the original intent — do not change the topic
- Output ONLY a JSON array of strings, nothing else

Example:
Input: "how do cars work"
Output: ["automobile engine operation", "vehicle mechanics explained", "motor car internal workings"]

Query: {query}
Output:"""


class QueryExpander:
    """
    Expands a query into multiple alternative phrasings using the LLM.
    Results are cached per session to avoid redundant API calls.
    """

    def __init__(self):
        self._cache: dict[str, list[str]] = {}
        self._client: OpenAI | None = None
        self._model: str | None = None

    def _get_client(self) -> tuple[OpenAI, str] | None:
        """Lazily initialise the OpenAI client from environment."""
        if self._client is not None:
            return self._client, self._model

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BERGET_API_KEY")
        if not api_key:
            return None

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        return self._client, self._model

    def expand(self, query: str, n: int = 3) -> list[str]:
        """
        Return [original_query] + up to n alternative phrasings.
        Falls back to [original_query] only if LLM is unavailable or fails.
        """
        # Always include the original
        if query in self._cache:
            return [query] + self._cache[query]

        client_model = self._get_client()
        if client_model is None:
            return [query]

        client, model = client_model
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=200,
                temperature=0.4,
                messages=[
                    {
                        "role": "user",
                        "content": EXPANSION_PROMPT.format(query=query, n=n),
                    }
                ],
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

            alternatives = json.loads(raw)
            if not isinstance(alternatives, list):
                raise ValueError("Expected a list")

            # Sanitise: strings only, drop empties, limit to n
            alternatives = [
                str(a).strip() for a in alternatives if str(a).strip()
            ][:n]

            self._cache[query] = alternatives
            return [query] + alternatives

        except Exception:
            # Fail silently — retrieval still works with the original query
            return [query]


# Module-level singleton so tools share the cache
_expander = QueryExpander()


def expand_query(query: str, n: int = 3) -> list[str]:
    """Public helper used by document_store and memory tools."""
    return _expander.expand(query, n)
