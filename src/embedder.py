"""
Voyage Context-3 embedding client for Obsidian notes.

Supports automatic chunking for large notes (>32k tokens):
- Small notes (<30k tokens): Embedded whole
- Large notes (>30k tokens): Split into 2000-char chunks, embedded with context preserved
- Uses voyage-context-3 contextualized_embed for multi-chunk notes
- Zero overlap (voyage-context-3 maintains context without overlap)
- JSON caching for security

Security features:
- API key redaction in logs
- Retry with exponential backoff on API errors
- Configurable timeout on API calls
"""

import asyncio
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import voyageai
from loguru import logger

from .exceptions import EmbeddingError

# Patterns for redacting sensitive information from logs
_SENSITIVE_PATTERNS = [
    (re.compile(r"(pa-[A-Za-z0-9_-]{20,})"), "[VOYAGE_API_KEY]"),  # Voyage API key
    (re.compile(r"(VOYAGE_API_KEY[=:]\s*)([^\s]+)"), r"\1[REDACTED]"),
]


def _redact_sensitive(message: str) -> str:
    """Redact API keys and other sensitive data from log messages."""
    for pattern, replacement in _SENSITIVE_PATTERNS:
        message = pattern.sub(replacement, message)
    return message


class VoyageEmbedder:
    """
    Voyage Context-3 embedding client with caching and rate limiting.

    Generates 1024-dimensional embeddings for Obsidian notes.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-context-3",
        cache_dir: str = "./data/embeddings_cache",
        batch_size: int = 128,
        requests_per_minute: int = 300,
        api_timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize Voyage embedder.

        Args:
            api_key: Voyage API key (or from VOYAGE_API_KEY env)
            model: Voyage model (default: voyage-context-3)
            cache_dir: Directory for caching embeddings
            batch_size: Texts per API batch
            requests_per_minute: Rate limit
            api_timeout: Timeout for API calls in seconds (default: 30)
            max_retries: Maximum retry attempts on API errors (default: 3)
        """
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY environment variable required")

        self.client = voyageai.Client(api_key=self.api_key)
        self.model = model
        self.batch_size = batch_size
        self.requests_per_minute = requests_per_minute
        self.api_timeout = api_timeout
        self.max_retries = max_retries

        # Cache setup
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()

        # Rate limiting (async-compatible)
        self.last_request_time = 0.0
        self.request_interval = 60.0 / requests_per_minute
        self._rate_limit_lock = asyncio.Lock()

        # Thread pool for running sync API calls with timeout
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="voyage_api")

        logger.success(f"VoyageEmbedder initialized: {model}")

    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 0) -> list[str]:
        """
        Split text into chunks (for notes >32k tokens).

        Args:
            text: Text to chunk
            chunk_size: Characters per chunk (default: 2000)
            overlap: Character overlap between chunks (default: 0 for voyage-context-3)

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending in last 200 chars
                chunk = text[start:end]
                last_period = chunk.rfind(". ")
                last_newline = chunk.rfind("\n\n")
                break_point = max(last_period, last_newline)

                if break_point > chunk_size - 200:  # Found good break point
                    end = start + break_point + 1

            chunks.append(text[start:end].strip())
            start = end - overlap

        logger.debug(
            f"Split text into {len(chunks)} chunks ({chunk_size} chars, {overlap} overlap)"
        )
        return chunks

    def embed_with_chunks(
        self, text: str, chunk_size: int = 2000, input_type: str = "document"
    ) -> tuple[list[list[float]], int]:
        """
        Embed text with automatic chunking for large content.

        If text exceeds ~30k tokens (~120k chars), splits into chunks
        and embeds with contextualized_embed to preserve context.

        Args:
            text: Text to embed
            chunk_size: Chunk size if splitting needed
            input_type: "document" or "query"

        Returns:
            Tuple of (embeddings_list, total_chunks)
            - For whole notes: ([embedding], 1)
            - For chunked notes: ([emb1, emb2, ...], n)

        Raises:
            EmbeddingError: If embedding fails
        """
        # Rough token estimate: ~4 chars per token
        estimated_tokens = len(text) / 4

        # If under limit, embed whole
        if estimated_tokens < 30000:
            embedding = self.embed(text, input_type=input_type)
            return ([embedding], 1)

        # Split into chunks
        chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=0)
        logger.info(f"Large note: splitting into {len(chunks)} chunks")

        # Embed chunks in batches (Voyage limit: ~60 chunks = 30k tokens per contextualized call)
        all_embeddings = []
        batch_size = 60  # ~30k tokens per batch

        try:
            for i in range(0, len(chunks), batch_size):
                chunk_batch = chunks[i : i + batch_size]

                # Rate limit
                self._rate_limit_sync()

                # Embed this batch of chunks with context (with retry)
                result = self._call_api_with_retry(
                    self.client.contextualized_embed,
                    inputs=[chunk_batch],  # One document's chunks
                    model=self.model,
                    input_type=input_type,
                )

                # Extract embeddings
                batch_embeddings = result.results[0].embeddings
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Embedded chunks {i+1}-{i+len(chunk_batch)} of {len(chunks)}")

            logger.success(f"Embedded {len(all_embeddings)} chunks with context preserved")
            return (all_embeddings, len(chunks))

        except Exception as e:
            error_msg = _redact_sensitive(str(e))
            logger.error(f"Chunked embedding failed: {error_msg}", exc_info=True)
            raise EmbeddingError(
                f"Failed to embed chunked text: {error_msg}", text_preview=text[:100], cause=e
            ) from e

    def _load_cache_index(self) -> dict:
        """Load cache index from disk."""
        if self.cache_index_path.exists():
            with open(self.cache_index_path) as f:
                return json.load(f)
        return {}

    def _save_cache_index(self):
        """Save cache index to disk."""
        with open(self.cache_index_path, "w") as f:
            json.dump(self.cache_index, f)

    def _get_text_hash(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    def _rate_limit_sync(self):
        """Enforce rate limiting (synchronous version for backwards compatibility)."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            time.sleep(self.request_interval - time_since_last)
        self.last_request_time = time.time()

    async def _rate_limit_async(self):
        """Enforce rate limiting (async version - non-blocking)."""
        async with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.request_interval:
                await asyncio.sleep(self.request_interval - time_since_last)
            self.last_request_time = time.time()

    def _call_api_with_retry(self, api_func, *args, **kwargs):
        """
        Call Voyage API with retry and exponential backoff.

        Args:
            api_func: The API function to call
            *args: Positional arguments for the API function
            **kwargs: Keyword arguments for the API function

        Returns:
            API response

        Raises:
            EmbeddingError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return api_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_msg = _redact_sensitive(str(e))

                # Check if it's a rate limit error (429)
                if "429" in str(e) or "rate" in str(e).lower():
                    # Exponential backoff: 2^attempt seconds (1, 2, 4, ...)
                    backoff = 2 ** (attempt + 1)
                    logger.warning(
                        f"Rate limited, retrying in {backoff}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(backoff)
                elif attempt < self.max_retries - 1:
                    # Other errors: shorter backoff
                    backoff = 1 * (attempt + 1)
                    logger.warning(
                        f"API error: {error_msg}, retrying in {backoff}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(backoff)
                else:
                    logger.error(f"API call failed after {self.max_retries} attempts: {error_msg}")

        raise EmbeddingError(
            f"API call failed after {self.max_retries} attempts: {_redact_sensitive(str(last_error))}",
            cause=last_error,
        )

    async def _call_api_with_timeout(self, api_func, *args, **kwargs):
        """
        Call Voyage API with timeout wrapper (runs in thread pool).

        Args:
            api_func: The API function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            API response

        Raises:
            EmbeddingError: If timeout or API error occurs
        """
        loop = asyncio.get_event_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    lambda: self._call_api_with_retry(api_func, *args, **kwargs),
                ),
                timeout=self.api_timeout,
            )
            return result
        except TimeoutError as e:
            raise EmbeddingError(f"API call timed out after {self.api_timeout}s", cause=e) from e

    def embed(self, text: str, input_type: str = "document", use_cache: bool = True) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            input_type: "document" or "query"
            use_cache: Whether to use cache

        Returns:
            1024-dimensional embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        results = self.embed_batch([text], input_type, use_cache)

        if not results or results[0] is None:
            raise EmbeddingError("Failed to generate embedding for text", text_preview=text[:100])

        return results[0]

    def embed_batch(
        self, texts: list[str], input_type: str = "document", use_cache: bool = True
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with caching.

        Args:
            texts: List of texts to embed
            input_type: "document" or "query"
            use_cache: Whether to use cache

        Returns:
            List of 1024-dimensional embedding vectors

        Raises:
            EmbeddingError: If any embedding fails
        """
        embeddings = []
        texts_to_embed = []
        text_indices = []

        # Check cache
        if use_cache:
            for i, text in enumerate(texts):
                text_hash = self._get_text_hash(text)
                if text_hash in self.cache_index:
                    cache_file = Path(self.cache_index[text_hash])
                    if cache_file.exists():
                        with open(cache_file) as f:
                            embedding = json.load(f)
                        embeddings.append(embedding)
                        continue

                texts_to_embed.append(text)
                text_indices.append(i)
        else:
            texts_to_embed = texts
            text_indices = list(range(len(texts)))

        # Embed uncached texts
        if texts_to_embed:
            logger.info(
                f"Embedding {len(texts_to_embed)} texts "
                f"(cached: {len(texts) - len(texts_to_embed)})"
            )

            new_embeddings = []
            for i in range(0, len(texts_to_embed), self.batch_size):
                batch = texts_to_embed[i : i + self.batch_size]

                # Rate limiting
                self._rate_limit_sync()

                try:
                    # Filter out empty strings (Voyage API rejects them)
                    filtered_batch = []
                    for text in batch:
                        if not text or not text.strip():
                            logger.error("Rejecting empty string in batch")
                            filtered_batch.append(None)  # Placeholder
                        else:
                            filtered_batch.append(text)

                    # Only embed non-empty texts
                    non_empty = [t for t in filtered_batch if t is not None]
                    if not non_empty:
                        logger.warning("Entire batch is empty strings, skipping")
                        new_embeddings.extend([None] * len(batch))
                        continue

                    # voyage-context-3 requires contextualized_embed with nested lists
                    # Each note is a single-element list (whole note, not chunked)
                    nested_inputs = [[text] for text in non_empty]

                    # Call Voyage API with retry and error handling
                    result = self._call_api_with_retry(
                        self.client.contextualized_embed,
                        inputs=nested_inputs,
                        model=self.model,
                        input_type=input_type,
                    )

                    # Extract embeddings from contextualized result
                    # result.results is a list of document results
                    # Each document has .embeddings list (one per chunk)
                    # Since we pass whole notes as single chunks, we take [0]
                    api_embeddings = [doc_result.embeddings[0] for doc_result in result.results]

                    # Map back to original batch positions (accounting for None placeholders)
                    embedding_idx = 0
                    for text in filtered_batch:
                        if text is None:
                            new_embeddings.append(None)
                        else:
                            new_embeddings.append(api_embeddings[embedding_idx])
                            embedding_idx += 1

                    # Cache results using JSON (safer than pickle)
                    if use_cache:
                        # Cache only non-None embeddings
                        for text, embedding in zip(non_empty, api_embeddings, strict=False):
                            text_hash = self._get_text_hash(text)
                            cache_file = self.cache_dir / f"{text_hash}.json"
                            with open(cache_file, "w") as f:
                                json.dump(embedding, f)
                            self.cache_index[text_hash] = str(cache_file)

                        self._save_cache_index()

                except Exception as e:
                    error_msg = _redact_sensitive(str(e))
                    logger.error(f"Embedding batch failed: {error_msg}", exc_info=True)
                    raise EmbeddingError(
                        f"Batch embedding failed: {error_msg}",
                        text_preview=batch[0][:100] if batch else "",
                        cause=e,
                    ) from e

            # Merge cached and new embeddings in correct order
            final_embeddings = [None] * len(texts)
            new_idx = 0
            cached_idx = 0

            for i in range(len(texts)):
                if i in text_indices:
                    final_embeddings[i] = new_embeddings[new_idx]
                    new_idx += 1
                else:
                    final_embeddings[i] = embeddings[cached_idx]
                    cached_idx += 1

            return final_embeddings

        return embeddings

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "total_cached": len(self.cache_index),
            "cache_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "model": self.model,
        }
