#!/usr/bin/env python3
"""Dense-stage latency benchmark for the Matryoshka comparison.

Times ONLY the dense vector-search stage (store.search at pool size 50,
threshold 0, per_document=False -- the exact call the consulting pipeline makes
to build its rerank candidate pool) over the reachable golden queries, against
one backend per invocation. Query vectors are generated once at 4096 (cache
hits) and truncated + L2-renormalized to --dims, matching make_mrl_index.py.

A single pooled connection (min=max=1) is warmed with 3 untimed queries before
timing, and one EXPLAIN (ANALYZE) is run on a representative query to prove
index usage vs seq scan.

    .venv/bin/python evals/bench_latency.py --db consulting_graph      --dims 4096
    .venv/bin/python evals/bench_latency.py --db consulting_eval_2048  --dims 2048
    .venv/bin/python evals/bench_latency.py --db consulting_eval_1024  --dims 1024
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env", override=True)

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--dims", type=int, required=True)
args = ap.parse_args()

# EMBEDDING_DIMENSIONS must be set before importing vector_store (module-level
# constant); it gates store.search's query-width validation.
os.environ.update(
    {
        "EMBEDDING_PROVIDER": "openrouter",
        "OPENROUTER_EMBED_MODEL": "qwen/qwen3-embedding-8b",
        "OPENROUTER_EMBED_DIMS": "4096",
        "EMBEDDING_DIMENSIONS": str(args.dims),
    }
)

import sys  # noqa: E402

sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

from src.server import make_embedder  # noqa: E402
from src.vector_store import PostgreSQLVectorStore  # noqa: E402

# Same chunk-level pool query vector_store.search issues for per_document=False.
POOL_SQL = """
    SELECT path, title, content, 1.0 - (embedding <=> $1::vector) AS similarity
    FROM notes
    WHERE embedding IS NOT NULL AND (embedding <=> $1::vector) <= $2
    ORDER BY embedding <=> $1::vector
    LIMIT $3
"""


def mrl(vec, dims):
    t = np.asarray(vec[:dims], dtype=np.float32)
    n = float(np.linalg.norm(t))
    return (t / n if n > 0 else t).tolist()


async def main():
    golden = json.load(open(REPO / "evals" / "golden.json"))
    store = PostgreSQLVectorStore(
        host="localhost",
        port=5434,
        database=args.db,
        user="obsidian",
        password=os.getenv("POSTGRES_PASSWORD"),
        min_connections=1,
        max_connections=1,  # one fresh connection per backend
    )
    await store.initialize()

    # Reachable golden queries in THIS db (content present) -> same 52 set.
    reachable = []
    async with store.pool.acquire() as conn:
        for g in golden:
            if await conn.fetchval(
                "SELECT 1 FROM notes WHERE path=$1 AND chunk_index=$2", g["path"], g["chunk"]
            ):
                reachable.append(g["query"])

    embedder = make_embedder()
    full = await embedder.embed_batch(reachable, input_type="query")  # 4096d, cache hits
    qvecs = [mrl(v, args.dims) for v in full if v is not None]

    # Warm the single connection with 3 untimed searches.
    for qv in qvecs[:3]:
        await store.search(qv, limit=50, threshold=0.0, per_document=False)

    # Timed dense stage, one measurement per query.
    lat = []
    for qv in qvecs:
        t0 = time.perf_counter()
        await store.search(qv, limit=50, threshold=0.0, per_document=False)
        lat.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(lat, 50))
    p95 = float(np.percentile(lat, 95))

    # EXPLAIN ANALYZE on a representative query -> node type proof.
    async with store.pool.acquire() as conn:
        plan = await conn.fetch("EXPLAIN (ANALYZE, FORMAT JSON) " + POOL_SQL, qvecs[0], 1.0, 50)
    plan_json = plan[0][0]
    if isinstance(plan_json, str):
        plan_json = json.loads(plan_json)
    top = plan_json[0]["Plan"]

    def scan_nodes(node, acc):
        acc.append((node.get("Node Type"), node.get("Index Name")))
        for c in node.get("Plans", []):
            scan_nodes(c, acc)
        return acc

    nodes = scan_nodes(top, [])
    await store.close()

    print(
        json.dumps(
            {
                "db": args.db,
                "dims": args.dims,
                "n_queries": len(qvecs),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "min_ms": round(min(lat), 2),
                "max_ms": round(max(lat), 2),
                "plan_nodes": nodes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
