"""
Document Store Tool - local knowledge base with BM25 retrieval.

This is the core IR component. Documents are chunked and indexed.
BM25 (Best Match 25) is a classic probabilistic IR ranking function
used by search engines like Elasticsearch.

BM25 score for document D given query Q:
  score(D, Q) = sum over terms t in Q of:
    IDF(t) * (tf(t,D) * (k1+1)) / (tf(t,D) + k1*(1 - b + b*|D|/avgdl))

Where:
  - IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
  - tf(t,D) = term frequency of t in D
  - |D| = document length, avgdl = average document length
  - k1=1.5, b=0.75 are tuning parameters
"""

import json
import math
import re
import uuid
from pathlib import Path
from datetime import datetime
from collections import Counter

from tools.query_expansion import expand_query


STORE_FILE = Path(".ir_agent_docs.json")
CHUNK_SIZE = 400      # characters per chunk
CHUNK_OVERLAP = 80    # overlap between chunks


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """In-memory BM25 index over a list of text chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []   # {"id", "text", "tokens", "meta"}
        self.df: dict[str, int] = {}  # document frequency per term
        self.avgdl: float = 0.0

    def add(self, doc_id: str, text: str, meta: dict):
        tokens = tokenize(text)
        self.docs.append({"id": doc_id, "text": text, "tokens": tokens, "meta": meta})
        for term in set(tokens):
            self.df[term] = self.df.get(term, 0) + 1
        self._update_avgdl()

    def _update_avgdl(self):
        if self.docs:
            self.avgdl = sum(len(d["tokens"]) for d in self.docs) / len(self.docs)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_terms = tokenize(query)
        N = len(self.docs)
        if N == 0 or not q_terms:
            return []

        scores = []
        for doc in self.docs:
            tf_map = Counter(doc["tokens"])
            dl = len(doc["tokens"])
            score = 0.0
            for t in q_terms:
                tf = tf_map.get(t, 0)
                df = self.df.get(t, 0)
                if df == 0:
                    continue
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                )
                score += idf * tf_norm
            if score > 0:
                scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scores[:top_k]:
            results.append({
                "score": round(score, 4),
                "text": doc["text"],
                "doc_id": doc["id"],
                "meta": doc["meta"],
            })
        return results

    def to_dict(self) -> dict:
        return {
            "docs": [
                {"id": d["id"], "text": d["text"], "meta": d["meta"]}
                for d in self.docs
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        idx = cls()
        for d in data.get("docs", []):
            idx.add(d["id"], d["text"], d.get("meta", {}))
        return idx


class DocumentStoreTool:
    def __init__(self):
        raw = self._load_raw()
        self.index = BM25Index.from_dict(raw)
        self._documents: dict[str, dict] = raw.get("documents", {})

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "document_add",
                    "description": (
                        "Add a document or text to the local knowledge base. "
                        "It will be chunked and indexed for retrieval. "
                        "Use this to build up a searchable context from files, articles, or notes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The document text content to index.",
                            },
                            "title": {
                                "type": "string",
                                "description": "A short title or source label for this document.",
                            },
                        },
                        "required": ["text", "title"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "document_search",
                    "description": (
                        "Search the local knowledge base using BM25 ranking. "
                        "Returns the most relevant text chunks for the query."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (default 5).",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "document_list",
                    "description": "List all documents currently in the knowledge base.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

    def document_add(self, text: str, title: str) -> dict:
        doc_id = str(uuid.uuid4())[:8]
        chunks = self._chunk(text)
        added = 0
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}-{i}"
            meta = {"title": title, "doc_id": doc_id, "chunk_index": i}
            self.index.add(chunk_id, chunk, meta)
            added += 1

        self._documents[doc_id] = {
            "title": title,
            "chunks": added,
            "added_at": datetime.now().isoformat(),
            "preview": text[:200],
        }
        self._save()
        return {
            "status": "indexed",
            "doc_id": doc_id,
            "title": title,
            "chunks_created": added,
        }

    def document_search(self, query: str, top_k: int = 5) -> dict:
        # Expand query into alternative phrasings to reduce vocabulary mismatch
        queries = expand_query(query, n=3)

        # Search with every query variant, collect all results
        seen_ids: set[str] = set()
        merged: list[dict] = []

        for q in queries:
            for result in self.index.search(q, top_k=top_k):
                chunk_id = result["doc_id"]
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    # Tag which query variant found this result
                    result["matched_query"] = q
                    merged.append(result)

        if not merged:
            return {
                "results": [],
                "query": query,
                "queries_used": queries,
                "message": "No relevant documents found.",
            }

        # Re-rank the merged pool by BM25 score descending, return top_k
        merged.sort(key=lambda r: r["score"], reverse=True)
        top = merged[:top_k]

        return {
            "results": top,
            "query": query,
            "queries_used": queries,
            "count": len(top),
        }

    def document_list(self) -> dict:
        docs = [
            {
                "doc_id": did,
                "title": info["title"],
                "chunks": info["chunks"],
                "added_at": info["added_at"],
                "preview": info["preview"],
            }
            for did, info in self._documents.items()
        ]
        return {"documents": docs, "count": len(docs)}

    def _chunk(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _load_raw(self) -> dict:
        if STORE_FILE.exists():
            try:
                return json.loads(STORE_FILE.read_text())
            except Exception:
                pass
        return {"docs": [], "documents": {}}

    def _save(self):
        data = self.index.to_dict()
        data["documents"] = self._documents
        STORE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
