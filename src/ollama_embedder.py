"""
Local Ollama embedding client (drop-in for VoyageEmbedder/GeminiEmbedder).

Runs entirely on-device against an Ollama server (default qwen3-embedding:0.6b,
1024 dims) - free, private, and with no API quota. 1024d matches the existing
pgvector(1024) column, so no schema change is needed.

Qwen3-Embedding is asymmetric: queries get a short instruction prefix, documents
are embedded as-is (the model was trained this way). We L2-normalize so pgvector
cosine distance is correct.

Exposes the same public surface the other embedders do:
  chunk_text, embed_with_chunks, embed, embed_batch, get_cache_stats
so it is a drop-in wherever they were used.
"""

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path

import httpx
from loguru import logger

from .exceptions import EmbeddingError

EMBEDDING_DIMENSIONS = 1024
# Qwen3 has 32K context, so chunking is rarely needed; keep a generous threshold.
WHOLE_EMBED_TOKEN_THRESHOLD = 6000

# Qwen3-Embedding query instruction (documents are embedded without one).
QUERY_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant documents from a personal "
    "consulting knowledge base.\nQuery: "
)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return vec if norm == 0.0 else [v / norm for v in vec]


class OllamaEmbedder:
    """Local embedding client backed by an Ollama server."""

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        host: str | None = None,
        cache_dir: str = "./data/embeddings_cache",
        batch_size: int = 32,
        concurrency: int = 2,
        api_timeout: float = 120.0,
        max_retries: int = 3,
        dimensions: int = EMBEDDING_DIMENSIONS,
    ):
        self.model = model
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.endpoint = f"{self.host}/api/embed"
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.api_timeout = api_timeout
        self.max_retries = max_retries

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()

        logger.success(f"OllamaEmbedder initialized: {model} @ {dimensions}d ({self.host})")

    # ------------------------------------------------------------------ #
    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 0) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                chunk = text[start:end]
                bp = max(chunk.rfind(". "), chunk.rfind("\n\n"))
                if bp > chunk_size - 200:
                    end = start + bp + 1
            chunks.append(text[start:end].strip())
            start = end - overlap
        return chunks

    async def embed_with_chunks(
        self, text: str, chunk_size: int = 2000, input_type: str = "document"
    ) -> tuple[list[list[float]], int]:
        if len(text) / 4 < WHOLE_EMBED_TOKEN_THRESHOLD:
            return ([await self.embed(text, input_type=input_type)], 1)
        chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=0)
        embeddings = await self.embed_batch(chunks, input_type=input_type)
        clean = [e for e in embeddings if e is not None]
        if len(clean) != len(chunks):
            raise EmbeddingError(
                f"Chunk embedding count mismatch ({len(clean)}/{len(chunks)})",
                text_preview=text[:100],
            )
        return (clean, len(chunks))

    # ------------------------------------------------------------------ #
    def _load_cache_index(self) -> dict:
        if self.cache_index_path.exists():
            with open(self.cache_index_path) as f:
                return json.load(f)
        return {}

    def _save_cache_index(self):
        with open(self.cache_index_path, "w") as f:
            json.dump(self.cache_index, f)

    def _get_text_hash(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{self.dimensions}:{text}".encode()).hexdigest()

    def _prep(self, text: str, input_type: str) -> str:
        return f"{QUERY_INSTRUCTION}{text}" if input_type == "query" else text

    # ------------------------------------------------------------------ #
    def _embed_call_sync_payload(self, inputs: list[str]) -> dict:
        return {"model": self.model, "input": inputs}

    async def _embed_group(self, client: httpx.AsyncClient, inputs: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await client.post(self.endpoint, json=self._embed_call_sync_payload(inputs))
                if resp.status_code == 200:
                    raw = resp.json()["embeddings"]
                    return [_l2_normalize([float(v) for v in e]) for e in raw]
                last_error = EmbeddingError(f"Ollama HTTP {resp.status_code}: {resp.text[:160]}")
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = e
                await asyncio.sleep(attempt + 1)
        raise EmbeddingError(
            f"Ollama embed failed after {self.max_retries} attempts: {last_error}",
            text_preview=inputs[0][:100] if inputs else "",
            cause=last_error if isinstance(last_error, Exception) else None,
        )

    # ------------------------------------------------------------------ #
    async def embed(
        self, text: str, input_type: str = "document", use_cache: bool = True
    ) -> list[float]:
        results = await self.embed_batch([text], input_type, use_cache)
        if not results or results[0] is None:
            raise EmbeddingError("Failed to generate embedding", text_preview=text[:100])
        return results[0]

    async def embed_batch(
        self, texts: list[str], input_type: str = "document", use_cache: bool = True
    ) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        to_fetch: list[int] = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            if use_cache:
                cached = self.cache_index.get(self._get_text_hash(text))
                if cached and Path(cached).exists():
                    with open(cached) as f:
                        results[i] = json.load(f)
                    continue
            to_fetch.append(i)

        if not to_fetch:
            return results

        logger.info(f"Embedding {len(to_fetch)} texts (cached: {len(texts) - len(to_fetch)})")
        groups = [to_fetch[i : i + self.batch_size] for i in range(0, len(to_fetch), self.batch_size)]
        semaphore = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient(timeout=self.api_timeout) as client:

            async def run_group(indices: list[int]):
                async with semaphore:
                    inputs = [self._prep(texts[i], input_type) for i in indices]
                    return indices, await self._embed_group(client, inputs)

            for fut in asyncio.as_completed([run_group(g) for g in groups]):
                indices, vectors = await fut
                for i, vec in zip(indices, vectors, strict=True):
                    results[i] = vec
                    if use_cache:
                        text_hash = self._get_text_hash(texts[i])
                        cache_file = self.cache_dir / f"{text_hash}.json"
                        with open(cache_file, "w") as f:
                            json.dump(vec, f)
                        self.cache_index[text_hash] = str(cache_file)

        if use_cache:
            self._save_cache_index()
        return results

    def get_cache_stats(self) -> dict:
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            "total_cached": len(self.cache_index),
            "cache_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "model": self.model,
        }
