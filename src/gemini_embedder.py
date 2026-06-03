"""
Gemini embedding client for Obsidian Graph (drop-in for VoyageEmbedder).

Uses Google's gemini-embedding-001 truncated to 1024 dimensions (Matryoshka),
so it lands directly in the existing pgvector(1024) column with no schema change.

Differences from VoyageEmbedder:
- gemini-embedding-001 is NOT a contextualized model, so chunks are embedded
  independently (no contextualized_embed nesting). This makes the code simpler.
- Max input is ~2048 tokens per content, so the whole-embed threshold is much
  lower than Voyage's 32k. Anything larger is chunked.
- Truncated (sub-3072) embeddings are not unit-norm, so we L2-normalize them
  ourselves. pgvector cosine distance assumes this for correct similarity.

Transport: the official google-genai SDK with native batch embeddings
(:batchEmbedContents). Multiple texts go up in one request; several requests run
concurrently via a thread pool. Make sure the intended API key wins over any
stale shell-exported GEMINI_API_KEY (load_dotenv(override=True) in callers).

Exposes the same public surface VoyageEmbedder does:
  chunk_text, embed_with_chunks, embed, embed_batch, get_cache_stats
so it is a drop-in wherever VoyageEmbedder was used.
"""

import asyncio
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from loguru import logger

from .exceptions import EmbeddingError

# gemini-embedding-001 accepts ~2048 input tokens per content. Stay well under.
MAX_INPUT_TOKENS = 2048
WHOLE_EMBED_TOKEN_THRESHOLD = 1800  # embed whole below this; chunk above
EMBEDDING_DIMENSIONS = 1024

# Patterns for redacting sensitive information from logs
_SENSITIVE_PATTERNS = [
    (re.compile(r"(AIza[A-Za-z0-9_-]{20,})"), "[GEMINI_API_KEY]"),  # legacy key format
    (re.compile(r"(AQ\.[A-Za-z0-9_-]{20,})"), "[GEMINI_API_KEY]"),  # newer AQ. key format
    (re.compile(r"(GEMINI_API_KEY[=:]\s*)([^\s]+)"), r"\1[REDACTED]"),
    (re.compile(r"(GOOGLE_API_KEY[=:]\s*)([^\s]+)"), r"\1[REDACTED]"),
]


def _redact_sensitive(message: str) -> str:
    for pattern, replacement in _SENSITIVE_PATTERNS:
        message = pattern.sub(replacement, message)
    return message


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector. Truncated Gemini embeddings are not unit-norm."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def _task_type(input_type: str) -> str:
    return "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"


class GeminiEmbedder:
    """
    Gemini embedding client with caching, native batching, and concurrency.

    Generates 1024-dimensional embeddings (gemini-embedding-001, Matryoshka-
    truncated and L2-normalized) for documents and queries.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
        cache_dir: str = "./data/embeddings_cache",
        batch_size: int = 50,
        api_timeout: float = 120.0,
        max_retries: int = 4,
        dimensions: int = EMBEDDING_DIMENSIONS,
        concurrency: int = 4,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable required")

        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.api_timeout = api_timeout
        self.max_retries = max_retries
        self.concurrency = concurrency

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()

        logger.success(f"GeminiEmbedder initialized: {model} @ {dimensions}d (SDK batch)")

    # ------------------------------------------------------------------ #
    # Chunking
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
                last_period = chunk.rfind(". ")
                last_newline = chunk.rfind("\n\n")
                break_point = max(last_period, last_newline)
                if break_point > chunk_size - 200:
                    end = start + break_point + 1
            chunks.append(text[start:end].strip())
            start = end - overlap
        logger.debug(f"Split text into {len(chunks)} chunks ({chunk_size} chars)")
        return chunks

    async def embed_with_chunks(
        self, text: str, chunk_size: int = 2000, input_type: str = "document"
    ) -> tuple[list[list[float]], int]:
        estimated_tokens = len(text) / 4
        if estimated_tokens < WHOLE_EMBED_TOKEN_THRESHOLD:
            embedding = await self.embed(text, input_type=input_type)
            return ([embedding], 1)

        chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=0)
        logger.info(f"Large doc: embedding {len(chunks)} chunks independently")
        embeddings = await self.embed_batch(chunks, input_type=input_type)
        clean = [e for e in embeddings if e is not None]
        if len(clean) != len(chunks):
            raise EmbeddingError(
                f"Chunk embedding count mismatch ({len(clean)}/{len(chunks)})",
                text_preview=text[:100],
            )
        return (clean, len(chunks))

    # ------------------------------------------------------------------ #
    # Cache helpers
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

    # ------------------------------------------------------------------ #
    # API calls (SDK batch :batchEmbedContents, with retry)
    # ------------------------------------------------------------------ #
    def _embed_call_sync(self, batch: list[str], input_type: str) -> list[list[float]]:
        """Blocking batch embed with retry; returns L2-normalized vectors."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimensions,
                        task_type=_task_type(input_type),
                    ),
                )
                return [_l2_normalize([float(v) for v in e.values]) for e in result.embeddings]
            except Exception as e:  # noqa: BLE001 - normalized into EmbeddingError below
                last_error = e
                s = str(e)
                retryable = any(c in s for c in ("429", "500", "503")) or "RESOURCE_EXHAUSTED" in s
                if retryable and attempt < self.max_retries - 1:
                    backoff = 2 ** (attempt + 1)
                    logger.warning(
                        f"Embed retry in {backoff}s ({attempt + 1}/{self.max_retries}): "
                        f"{_redact_sensitive(s)[:90]}"
                    )
                    time.sleep(backoff)
                    continue
                break
        raise EmbeddingError(
            f"Batch embed failed: {_redact_sensitive(str(last_error))[:160]}",
            text_preview=batch[0][:100] if batch else "",
            cause=last_error,
        )

    # ------------------------------------------------------------------ #
    # Public embedding API
    # ------------------------------------------------------------------ #
    async def embed(
        self, text: str, input_type: str = "document", use_cache: bool = True
    ) -> list[float]:
        results = await self.embed_batch([text], input_type, use_cache)
        if not results or results[0] is None:
            raise EmbeddingError("Failed to generate embedding for text", text_preview=text[:100])
        return results[0]

    async def embed_batch(
        self, texts: list[str], input_type: str = "document", use_cache: bool = True
    ) -> list[list[float]]:
        """
        Embed many texts using native batch requests with bounded concurrency.

        Returns a list aligned to `texts`; empty strings map to None.
        """
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

        # Split into batch_size groups, run several groups concurrently.
        groups = [to_fetch[i : i + self.batch_size] for i in range(0, len(to_fetch), self.batch_size)]
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_group(indices: list[int]):
            async with semaphore:
                batch_texts = [texts[i] for i in indices]
                vectors = await asyncio.to_thread(self._embed_call_sync, batch_texts, input_type)
                return indices, vectors

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
