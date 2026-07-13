#!/usr/bin/env python3
"""Ablate the pipeline stages on the golden set to find what helps/hurts:
dense-only | dense+rerank (no BM25) | dense+BM25 hybrid (no rerank) | full."""

import asyncio, json, math, os, re, sys
from pathlib import Path
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env", override=True)
os.environ.update(
    {
        "EMBEDDING_PROVIDER": "openrouter",
        "OPENROUTER_EMBED_MODEL": "qwen/qwen3-embedding-8b",
        "OPENROUTER_EMBED_DIMS": "4096",
        "EMBEDDING_DIMENSIONS": "4096",
        "RERANK_MODEL": "cohere/rerank-v3.5",
    }
)
sys.path.insert(0, str(REPO))
from src.server import make_embedder
from src.vector_store import PostgreSQLVectorStore
from src.reranker import CohereReranker
import src.tools as tools

_norm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower()
POOL = 50


def rrf(*lists, k=60):
    sc, obj = {}, {}
    for L in lists:
        for r, x in enumerate(L):
            sc[x.content] = sc.get(x.content, 0) + 1 / (k + r + 1)
            obj[x.content] = x
    return [obj[c] for c, _ in sorted(sc.items(), key=lambda kv: -kv[1])]


def metrics(ranks, n):
    ks = [1, 5, 10, 20]
    return {f"R@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in ks} | {
        "MRR": round(sum(1 / r for r in ranks if r) / n, 3),
        "nDCG": round(sum(1 / math.log2(r + 1) for r in ranks if r and r <= 10) / n, 3),
    }


async def main():
    golden = json.load(open(REPO / "evals" / "golden.json"))
    store = PostgreSQLVectorStore(
        host="localhost",
        port=5434,
        database="consulting_graph",
        user="obsidian",
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    await store.initialize()
    emb = make_embedder()
    rr = CohereReranker()
    gold_key = {}
    async with store.pool.acquire() as conn:
        for g in golden:
            row = await conn.fetchrow(
                "SELECT content FROM notes WHERE path=$1 AND chunk_index=$2", g["path"], g["chunk"]
            )
            gold_key[(g["path"], g["chunk"])] = _norm(row["content"]) if row else None

    def rank_of(g, ordered):
        key = gold_key[(g["path"], g["chunk"])]
        return next(
            (
                i + 1
                for i, r in enumerate(ordered)
                if r.path == g["path"] and _norm(r.content) == key
            ),
            None,
        )

    cfgs = {
        "dense": [],
        "dense+rerank": [],
        "dense FUSED rerank(RRF)": [],
        "full(hybrid+rerank)": [],
    }
    for g in golden:
        qv = await emb.embed(g["query"], input_type="query")
        dense = await store.search(qv, POOL, threshold=0.0)
        lexical = await store.lexical_search(g["query"], POOL)
        hybrid = rrf(dense, lexical)[:POOL]
        cfgs["dense"].append(rank_of(g, dense[:20]))
        # rerank dense-only pool (pure rerank order)
        o1 = await asyncio.to_thread(rr.rerank, g["query"], [r.content for r in dense[:POOL]], POOL)
        rerank_order = [dense[i] for i, _ in o1]
        cfgs["dense+rerank"].append(rank_of(g, rerank_order[:20]))
        # FUSE dense ranking with rerank ranking (RRF) -> rerank can't override a strong dense hit
        fused = rrf(dense[:POOL], rerank_order)[:20]
        cfgs["dense FUSED rerank(RRF)"].append(rank_of(g, fused))
        # production (hybrid pool, pure rerank)
        o2 = await asyncio.to_thread(rr.rerank, g["query"], [r.content for r in hybrid], 20)
        cfgs["full(hybrid+rerank)"].append(rank_of(g, [hybrid[i] for i, _ in o2]))
    await store.close()
    styles = [g.get("style", "?") for g in golden]
    print(f"\n=== ABLATION ({len(golden)} queries across styles) ===")

    def report(label, keep):
        idxs = [i for i, s in enumerate(styles) if keep(s)]
        n = len(idxs)
        print(f"\n--- {label} (n={n}) ---")
        print(f"{'config':24} {'R@1':>5} {'R@5':>5} {'R@10':>5} {'R@20':>5} {'MRR':>6} {'nDCG':>6}")
        for name, ranks in cfgs.items():
            m = metrics([ranks[i] for i in idxs], n)
            print(
                f"{name:24} {m['R@1']:>5.2f} {m['R@5']:>5.2f} {m['R@10']:>5.2f} {m['R@20']:>5.2f} {m['MRR']:>6.3f} {m['nDCG']:>6.3f}"
            )

    report("ALL", lambda s: True)
    report("HARD (adversarial)", lambda s: s == "hard")
    report("REALISTIC", lambda s: s == "realistic")


if __name__ == "__main__":
    asyncio.run(main())
