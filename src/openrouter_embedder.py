"""
OpenRouter embedder (OpenAI-compatible /embeddings). Reuses OllamaEmbedder's
chunking, caching, query-instruction, and batching; only the HTTP call differs
(auth header + data[].embedding response). Default model: Qwen3-Embedding-8B
(4096d, $0.01/M, indexed reliably where the 4B rate-limited).
"""

from __future__ import annotations

import asyncio
import os

import httpx
from loguru import logger

from .exceptions import EmbeddingError
from .ollama_embedder import OllamaEmbedder, _l2_normalize


class OpenRouterEmbedder(OllamaEmbedder):
    def __init__(
        self,
        model: str = "qwen/qwen3-embedding-8b",
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int = 4096,
        **kwargs,
    ):
        base = (
            base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        ).rstrip("/")
        super().__init__(model=model, host=base, dimensions=dimensions, **kwargs)
        self.host = base
        self.endpoint = f"{base}/embeddings"
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            logger.warning("OpenRouterEmbedder: no OPENROUTER_API_KEY set")
        logger.success(f"OpenRouterEmbedder initialized: {model} @ {dimensions}d ({base})")

    async def _embed_group(self, client: httpx.AsyncClient, inputs: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await client.post(
                    self.endpoint, json={"model": self.model, "input": inputs}, headers=headers
                )
                if resp.status_code == 200:
                    data = resp.json().get("data")
                    if data:
                        return [_l2_normalize([float(v) for v in d["embedding"]]) for d in data]
                    last_error = EmbeddingError(f"OpenRouter no data: {resp.text[:160]}")
                else:  # 429 (engine_overloaded) etc. -> back off and retry
                    last_error = EmbeddingError(
                        f"OpenRouter HTTP {resp.status_code}: {resp.text[:160]}"
                    )
            except Exception as e:  # noqa: BLE001
                last_error = e
            await asyncio.sleep(min(20, 4 * (attempt + 1)))
        raise last_error or EmbeddingError("OpenRouter embed failed after retries")
