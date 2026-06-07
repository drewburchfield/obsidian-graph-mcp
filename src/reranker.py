"""
Cross-encoder reranker via OpenRouter (Cohere rerank-v3.5). No local model.

Stage 2 of the consulting-graph pipeline: takes the hybrid (dense + BM25)
candidate pool and re-scores each (query, chunk) pair for true relevance.
Verified on the consulting corpus to be the single biggest accuracy lever
(R@1 0.50 -> 0.77 end-to-end).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from loguru import logger


class CohereReranker:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, max_doc_chars: int = 1400):
        self.model = model or os.getenv("RERANK_MODEL", "cohere/rerank-v3.5")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.url = (base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/") + "/rerank"
        self.max_doc_chars = max_doc_chars
        if not self.api_key:
            logger.warning("CohereReranker: no OPENROUTER_API_KEY set; reranking will fail")

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
        """Return [(original_index, relevance_score), ...] sorted best-first."""
        if not documents:
            return []
        body = {"model": self.model, "query": query,
                "documents": [d[: self.max_doc_chars] for d in documents]}
        if top_n:
            body["top_n"] = top_n
        req = urllib.request.Request(
            self.url, json.dumps(body).encode(),
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        last = None
        for attempt in range(4):
            try:
                d = json.load(urllib.request.urlopen(req, timeout=30))
                if "results" not in d:
                    raise RuntimeError(str(d)[:200])
                return [(r["index"], r.get("relevance_score", 0.0)) for r in d["results"]]
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < 3:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"rerank failed after retries: {last}")
