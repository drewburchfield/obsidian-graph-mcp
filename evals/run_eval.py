#!/usr/bin/env python3
"""
Consulting-graph retrieval eval. Runs the LIVE search_notes pipeline
(MLX-4B -> dense + BM25 hybrid -> Cohere rerank -> top-20) over a golden set
of questions authored by braintrust-plugin models, and reports Recall@1/5/10/20,
MRR, nDCG@10. A regression test for the deployed system.

  cd ~/dev/projects/obsidian-graph-mcp
  .venv/bin/python evals/run_eval.py            # full pipeline
  .venv/bin/python evals/run_eval.py --baseline # dense-only (no rerank), for delta
"""
import argparse
import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env", override=True)
os.environ.update({
    "EMBEDDING_PROVIDER": "ollama", "OLLAMA_EMBED_MODEL": "qwen3-embedding-4b-mlx",
    "OLLAMA_EMBED_DIMS": "2560", "EMBEDDING_DIMENSIONS": "2560",
    "RERANK_MODEL": "cohere/rerank-v3.5", "RERANK_POOL": "50",
})
sys.path.insert(0, str(REPO))
from src.server import make_embedder            # noqa: E402
from src.vector_store import PostgreSQLVectorStore  # noqa: E402
import src.tools as tools                        # noqa: E402

_norm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="dense-only (rerank off)")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    os.environ["CONSULTING_RERANK"] = "0" if args.baseline else "1"

    golden = json.load(open(REPO / "evals" / "golden.json"))
    store = PostgreSQLVectorStore(host="localhost", port=5434, database="consulting_graph",
                                  user="obsidian", password=os.getenv("POSTGRES_PASSWORD"))
    await store.initialize()

    # full gold-chunk content from the live DB -> exact normalized match key
    gold_key = {}
    async with store.pool.acquire() as conn:
        for g in golden:
            row = await conn.fetchrow(
                "SELECT content FROM notes WHERE path=$1 AND chunk_index=$2", g["path"], g["chunk"])
            gold_key[(g["path"], g["chunk"])] = _norm(row["content"]) if row else None

    ctx = tools.ToolContext(store=store, embedder=make_embedder(), graph_builder=None, hub_analyzer=None)
    ks = [1, 5, 10, 20]
    hits = {k: 0 for k in ks}
    mrr = ndcg = 0.0
    misses = []
    for g in golden:
        res = await tools.search_notes(ctx, {"query": g["query"], "limit": args.limit, "threshold": 0.0})
        key = gold_key[(g["path"], g["chunk"])]
        rank = next((i + 1 for i, r in enumerate(res["results"])
                     if r["path"] == g["path"] and _norm(r["content"]) == key), None)
        if rank:
            for k in ks:
                hits[k] += rank <= k
            mrr += 1.0 / rank
            if rank <= 10:
                ndcg += 1.0 / math.log2(rank + 1)
        else:
            misses.append(g["query"][:60])
    await store.close()

    n = len(golden)
    mode = "DENSE-ONLY (baseline)" if args.baseline else "FULL PIPELINE (hybrid+rerank)"
    print(f"\n=== consulting-graph eval [{mode}] — {n} braintrust-authored queries ===")
    for k in ks:
        print(f"  Recall@{k:<2} {hits[k]/n:.3f}")
    print(f"  MRR       {mrr/n:.3f}")
    print(f"  nDCG@10   {ndcg/n:.3f}")
    if misses:
        print(f"  misses ({len(misses)}): " + "; ".join(misses[:5]))
    out = {"mode": mode, "n": n, **{f"R@{k}": round(hits[k]/n, 3) for k in ks},
           "MRR": round(mrr/n, 3), "nDCG@10": round(ndcg/n, 3), "misses": misses}
    json.dump(out, open(REPO / "evals" / ("results_baseline.json" if args.baseline else "results.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
