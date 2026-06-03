#!/usr/bin/env python3
"""
Semantic search over the consulting_graph database (proves the index works).

Usage:
  python scripts/search_consulting.py "AI opportunity scoring for Divi"
  python scripts/search_consulting.py "Thompson onsite breakout sessions" --limit 8
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env BEFORE importing vector_store, whose EMBEDDING_DIMENSIONS constant
# reads the env at import time. override=True so .env wins over stale shell vars.
load_dotenv(REPO_ROOT / ".env", override=True)

from src.vector_store import PostgreSQLVectorStore  # noqa: E402


def make_embedder(cache_dir: str):
    """Must match the provider used to index (same vector space)."""
    if os.getenv("EMBEDDING_PROVIDER", "ollama").lower() == "gemini":
        from src.gemini_embedder import GeminiEmbedder

        return GeminiEmbedder(cache_dir=cache_dir)
    from src.ollama_embedder import OllamaEmbedder

    return OllamaEmbedder(
        model=os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
        dimensions=int(os.getenv("OLLAMA_EMBED_DIMS", "1024")),
        cache_dir=cache_dir,
    )


async def main() -> int:
    load_dotenv(REPO_ROOT / ".env", override=True)

    parser = argparse.ArgumentParser(description="Semantic search over consulting_graph")
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--host", default=os.getenv("CONSULTING_PG_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CONSULTING_PG_PORT", "5434")))
    parser.add_argument("--db", default=os.getenv("CONSULTING_PG_DB", "consulting_graph"))
    args = parser.parse_args()

    embedder = make_embedder(str(REPO_ROOT / "data" / "consulting_cache"))
    store = PostgreSQLVectorStore(
        host=args.host,
        port=args.port,
        database=args.db,
        user=os.getenv("POSTGRES_USER", "obsidian"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    await store.initialize()
    try:
        query_embedding = await embedder.embed(args.query, input_type="query")
        results = await store.search(query_embedding, limit=args.limit, threshold=args.threshold)
        print(f"\nTop {len(results)} results for: {args.query!r}\n" + "=" * 60)
        for i, r in enumerate(results, 1):
            snippet = " ".join(r.content.split())[:160]
            print(f"\n{i}. [{r.similarity:.3f}] {r.path}")
            print(f"   {snippet}")
        print()
    finally:
        await store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
