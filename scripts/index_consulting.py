#!/usr/bin/env python3
"""
Index the consulting folder into the standalone consulting_graph database.

Uses GeminiEmbedder (gemini-embedding-001 @ 1024d) + converters (markitdown +
table cards) so every supported document - markdown, PDF, DOCX, PPTX, HTML,
CSV, XLSX, TXT - lands in one searchable pgvector graph.

Prereqs:
  1. GEMINI_API_KEY set (in .env or environment)
  2. docker compose -f docker-compose.consulting.yml up -d

Usage:
  python scripts/index_consulting.py
  python scripts/index_consulting.py --root /path/to/folder
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env BEFORE importing vector_store (its EMBEDDING_DIMENSIONS constant reads
# the env at import time). override=True so .env wins over stale shell vars.
load_dotenv(REPO_ROOT / ".env", override=True)

from src.multi_format_indexer import index_root  # noqa: E402
from src.vector_store import PostgreSQLVectorStore  # noqa: E402


def make_embedder(cache_dir: str):
    """OpenRouter (default, matches the live graph's 4096d vectors) or gemini/ollama via env flag."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openrouter").lower()
    if provider == "openrouter":
        from src.openrouter_embedder import OpenRouterEmbedder

        return OpenRouterEmbedder(
            model=os.getenv("OPENROUTER_EMBED_MODEL", "qwen/qwen3-embedding-8b"),
            dimensions=int(os.getenv("OPENROUTER_EMBED_DIMS", "4096")),
            cache_dir=cache_dir,
        )
    if provider == "gemini":
        from src.gemini_embedder import GeminiEmbedder

        return GeminiEmbedder(
            cache_dir=cache_dir,
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "5")),
            concurrency=int(os.getenv("EMBEDDING_CONCURRENCY", "1")),
        )
    from src.ollama_embedder import OllamaEmbedder

    return OllamaEmbedder(
        model=os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        dimensions=int(os.getenv("OLLAMA_EMBED_DIMS", "1024")),
        concurrency=int(os.getenv("OLLAMA_EMBED_CONCURRENCY", "2")),
        cache_dir=cache_dir,
    )

DEFAULT_ROOT = "/Users/drewburchfield/dev/consulting"


async def main() -> int:
    # override=True so the intended .env key wins over any stale shell-exported
    # GEMINI_API_KEY (e.g. an old AIza key in ~/.zshrc).
    load_dotenv(REPO_ROOT / ".env", override=True)

    parser = argparse.ArgumentParser(description="Index a folder into consulting_graph")
    parser.add_argument("--root", default=os.getenv("CONSULTING_ROOT", DEFAULT_ROOT))
    parser.add_argument("--host", default=os.getenv("CONSULTING_PG_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CONSULTING_PG_PORT", "5434")))
    parser.add_argument("--db", default=os.getenv("CONSULTING_PG_DB", "consulting_graph"))
    args = parser.parse_args()

    provider = os.getenv("EMBEDDING_PROVIDER", "openrouter").lower()
    if provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        logger.error("EMBEDDING_PROVIDER=openrouter but OPENROUTER_API_KEY is not set.")
        return 1
    if provider == "gemini" and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        logger.error("EMBEDDING_PROVIDER=gemini but GEMINI_API_KEY is not set.")
        return 1
    if not os.getenv("POSTGRES_PASSWORD"):
        logger.error("POSTGRES_PASSWORD is not set (shared with the pgvector container).")
        return 1

    embedder = make_embedder(str(REPO_ROOT / "data" / "consulting_cache"))
    store = PostgreSQLVectorStore(
        host=args.host,
        port=args.port,
        database=args.db,
        user=os.getenv("POSTGRES_USER", "obsidian"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

    logger.info(f"Indexing {args.root} -> {args.db} @ {args.host}:{args.port}")
    await store.initialize()
    try:
        summary = await index_root(args.root, store, embedder)
        total = await store.get_note_count()
        stats = embedder.get_cache_stats()
        logger.success(
            f"Done. {summary}. Total chunks in DB: {total}. "
            f"Cache: {stats['total_cached']} embeddings, {stats['cache_size_mb']} MB"
        )
    finally:
        await store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
