#!/usr/bin/env python3
"""Live test of the consulting-graph retrieval pipeline through the real
search_notes handler (dense + BM25 hybrid -> Cohere rerank -> top-N)."""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env", override=True)
os.environ.update({
    "EMBEDDING_PROVIDER": "ollama",
    "OLLAMA_EMBED_MODEL": "qwen3-embedding-4b-mlx",
    "OLLAMA_EMBED_DIMS": "2560",
    "EMBEDDING_DIMENSIONS": "2560",
    "RERANK_MODEL": "cohere/rerank-v3.5",
    "RERANK_POOL": "50",
})
sys.path.insert(0, str(REPO))
from src.server import make_embedder            # noqa: E402
from src.vector_store import PostgreSQLVectorStore  # noqa: E402
import src.tools as tools                        # noqa: E402

QUERIES = [
    "Who handles commissions and GL postings to Sage for reps in Aruba and St. Maarten?",
    "What is the non-solicitation period in the independent contractor agreement?",
    "Why is flat per-seat unlimited AI pricing starting to crack?",
]


async def run(ctx, q, rerank):
    os.environ["CONSULTING_RERANK"] = "1" if rerank else "0"
    r = await tools.search_notes(ctx, {"query": q, "limit": 5, "threshold": 0.0})
    return r["results"]


async def main():
    store = PostgreSQLVectorStore(host="localhost", port=5434, database="consulting_graph",
                                  user="obsidian", password=os.getenv("POSTGRES_PASSWORD"))
    await store.initialize()
    ctx = tools.ToolContext(store=store, embedder=make_embedder(), graph_builder=None, hub_analyzer=None)
    for q in QUERIES:
        print(f"\nQUERY: {q}")
        base = await run(ctx, q, rerank=False)
        full = await run(ctx, q, rerank=True)
        print("  dense-only top3:   " + " | ".join(x["path"].split("/")[-1][:34] for x in base[:3]))
        print("  HYBRID+RERANK top3:" + " | ".join(f"[{x['similarity']:.2f}] " + x["path"].split("/")[-1][:30] for x in full[:3]))
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
