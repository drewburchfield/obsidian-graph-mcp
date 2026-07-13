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

# Matryoshka parametrization (default = the 4096 control on the live DB, so an
# un-parametrized run is byte-for-byte the original harness). EVAL_DIMS is the
# width of the index/column being searched; EVAL_DB is which database holds it.
# Query embeddings are ALWAYS generated at 4096 (OPENROUTER_EMBED_DIMS) so the
# provider cache stays width-stable across runs; the query vector is truncated +
# L2-renormalized to EVAL_DIMS in-process (see _MRLQueryEmbedder), matching how
# make_mrl_index.py derived the truncated document vectors. EMBEDDING_DIMENSIONS
# must equal EVAL_DIMS so vector_store's search/dimension validation lines up.
_EVAL_DIMS = int(os.getenv("EVAL_DIMS", "4096"))
_EVAL_DB = os.getenv("EVAL_DB", "consulting_graph")
os.environ.update(
    {
        "EMBEDDING_PROVIDER": "openrouter",
        "OPENROUTER_EMBED_MODEL": "qwen/qwen3-embedding-8b",
        "OPENROUTER_EMBED_DIMS": "4096",
        "EMBEDDING_DIMENSIONS": str(_EVAL_DIMS),
        "RERANK_MODEL": "cohere/rerank-v3.5",
        "RERANK_POOL": "50",
    }
)
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import src.tools as tools  # noqa: E402
from src.server import make_embedder  # noqa: E402
from src.vector_store import PostgreSQLVectorStore  # noqa: E402


def mrl_truncate(vec: list[float], dims: int) -> list[float]:
    """Truncate a full-width embedding to its first ``dims`` components and
    L2-renormalize, exactly as make_mrl_index.py transformed the stored
    document vectors. Renorm is ranking-irrelevant for pgvector's cosine
    ``<=>`` but kept for parity with the index build."""
    trunc = np.asarray(vec[:dims], dtype=np.float32)
    norm = float(np.linalg.norm(trunc))
    if norm > 0.0:
        trunc = trunc / norm
    return trunc.tolist()


class _MRLQueryEmbedder:
    """Wraps a full-width (4096d) embedder and returns every query vector
    truncated + renormalized to ``dims`` so a 4096d query can search a
    truncated MRL index. The inner embedder still fetches/caches at 4096, so
    the provider cache is shared across dims and never poisoned."""

    def __init__(self, inner, dims: int):
        self._inner = inner
        self._dims = dims
        self.model = getattr(inner, "model", "unknown")

    async def embed(self, text, input_type="document", use_cache=True):
        return mrl_truncate(await self._inner.embed(text, input_type, use_cache), self._dims)

    async def embed_batch(self, texts, input_type="document", use_cache=True):
        vecs = await self._inner.embed_batch(texts, input_type, use_cache)
        return [None if v is None else mrl_truncate(v, self._dims) for v in vecs]


def _norm(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="dense-only (rerank off)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--min-gold-coverage", type=float, default=0.75)
    ap.add_argument("--min-r5", type=float, default=0.65)
    ap.add_argument("--min-r20", type=float, default=0.85)
    args = ap.parse_args()
    os.environ["CONSULTING_RERANK"] = "0" if args.baseline else "1"

    golden = json.load(open(REPO / "evals" / "golden.json"))
    store = PostgreSQLVectorStore(
        host="localhost",
        port=5434,
        database=_EVAL_DB,
        user="obsidian",
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    await store.initialize()

    # full gold-chunk content from the live DB -> exact normalized match key
    gold_key = {}
    active_golden = []
    unreachable = []
    async with store.pool.acquire() as conn:
        for g in golden:
            row = await conn.fetchrow(
                "SELECT content FROM notes WHERE path=$1 AND chunk_index=$2", g["path"], g["chunk"]
            )
            if row:
                gold_key[(g["path"], g["chunk"])] = _norm(row["content"])
                active_golden.append(g)
            else:
                unreachable.append(g)

    gold_coverage = len(active_golden) / len(golden) if golden else 0.0
    if not active_golden or gold_coverage < args.min_gold_coverage:
        await store.close()
        print(
            f"QUALITY GATE FAILED: gold coverage {gold_coverage:.3f} is below "
            f"{args.min_gold_coverage:.3f}"
        )
        return 1

    embedder = make_embedder()
    if _EVAL_DIMS != 4096:
        embedder = _MRLQueryEmbedder(embedder, _EVAL_DIMS)
    ctx = tools.ToolContext(store=store, embedder=embedder, graph_builder=None, hub_analyzer=None)
    # Prime query embeddings in provider-sized batches so a full quality run
    # does not spend one network round trip per question.
    await ctx.embedder.embed_batch([g["query"] for g in active_golden], input_type="query")
    ks = [1, 5, 10, 20]
    hits = dict.fromkeys(ks, 0)
    mrr = ndcg = 0.0
    misses = []
    for g in active_golden:
        res = await tools.search_notes(
            ctx, {"query": g["query"], "limit": args.limit, "threshold": 0.0}
        )
        key = gold_key[(g["path"], g["chunk"])]
        rank = next(
            (
                i + 1
                for i, r in enumerate(res["results"])
                if r["path"] == g["path"] and _norm(r["content"]) == key
            ),
            None,
        )
        if rank:
            for k in ks:
                hits[k] += rank <= k
            mrr += 1.0 / rank
            if rank <= 10:
                ndcg += 1.0 / math.log2(rank + 1)
        else:
            misses.append(g["query"][:60])
    await store.close()

    n = len(active_golden)
    mode = "DENSE-ONLY (baseline)" if args.baseline else "FULL PIPELINE (hybrid+rerank)"
    print(
        f"\n=== consulting-graph eval [{mode}] "
        f"db={_EVAL_DB} dims={_EVAL_DIMS}: {n} reachable gold queries ==="
    )
    print(f"  Gold coverage {n}/{len(golden)} ({gold_coverage:.3f})")
    for k in ks:
        print(f"  Recall@{k:<2} {hits[k] / n:.3f}")
    print(f"  MRR       {mrr / n:.3f}")
    print(f"  nDCG@10   {ndcg / n:.3f}")
    if misses:
        print(f"  misses ({len(misses)}): " + "; ".join(misses[:5]))
    out = {
        "mode": mode,
        "db": _EVAL_DB,
        "dims": _EVAL_DIMS,
        "n": n,
        "gold_total": len(golden),
        "gold_coverage": round(gold_coverage, 3),
        **{f"R@{k}": round(hits[k] / n, 3) for k in ks},
        "MRR": round(mrr / n, 3),
        "nDCG@10": round(ndcg / n, 3),
        "misses": misses,
        "unreachable": [g["path"] for g in unreachable],
    }
    # Namespace the result file by db so parametrized runs never clobber the
    # 4096 control's results.json.
    suffix = "_baseline" if args.baseline else ""
    stem = "results" if _EVAL_DB == "consulting_graph" else f"results_{_EVAL_DB}"
    json.dump(out, open(REPO / "evals" / f"{stem}{suffix}.json", "w"), indent=2)
    if not args.baseline and (hits[5] / n < args.min_r5 or hits[20] / n < args.min_r20):
        print(
            f"  QUALITY GATE FAILED: require R@5 >= {args.min_r5:.2f} "
            f"and R@20 >= {args.min_r20:.2f}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
