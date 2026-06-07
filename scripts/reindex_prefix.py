#!/usr/bin/env python3
"""
Re-embed the consulting_graph `notes` table with the verified contextual PREFIX,
in place, via the MLX-4B endpoint on bigbot. Documents only get the prefix;
queries stay raw (matches the eval that produced R@1 0.77 / R@10 0.92).

  cd ~/dev/projects/obsidian-graph-mcp && .venv/bin/python scripts/reindex_prefix.py
"""
import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env", override=True)
MLX = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/embed"


def mlx_embed(texts, bs=50):
    out = []
    for i in range(0, len(texts), bs):
        body = json.dumps({"input": [t[:8000] for t in texts[i : i + bs]]}).encode()
        for a in range(6):
            try:
                req = urllib.request.Request(MLX, body, {"Content-Type": "application/json"})
                out.extend(json.load(urllib.request.urlopen(req, timeout=180))["embeddings"])
                break
            except Exception:
                if a == 5:
                    raise
                time.sleep(5)
        print(".", end="", flush=True)
    print()
    return out


async def main() -> int:
    # Host-side consulting DB params (the .env POSTGRES_* are docker-internal; the
    # launcher overrides them to localhost:5434/consulting_graph).
    conn = await asyncpg.connect(
        host=os.getenv("CONSULTING_PG_HOST", "localhost"),
        port=int(os.getenv("CONSULTING_PG_PORT", "5434")),
        database=os.getenv("CONSULTING_PG_DB", "consulting_graph"),
        user=os.getenv("CONSULTING_PG_USER", "obsidian"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    rows = await conn.fetch("SELECT path, title, chunk_index, content FROM notes ORDER BY path, chunk_index")
    print(f"re-embedding {len(rows)} chunks with contextual prefix via {MLX}", flush=True)

    ctx = {}
    for r in rows:
        if r["chunk_index"] == 0:
            ctx[r["path"]] = (r["title"] + ". " + r["content"])[:250].replace("\n", " ")
    for r in rows:
        ctx.setdefault(r["path"], r["title"])

    prefixed = [f'[Document: {ctx[r["path"]]}]\n{r["content"]}' for r in rows]
    embs = mlx_embed(prefixed)

    async with conn.transaction():
        for r, e in zip(rows, embs):
            await conn.execute(
                "UPDATE notes SET embedding = $1::vector WHERE path = $2 AND chunk_index = $3",
                str(e), r["path"], r["chunk_index"],
            )
    print(f"updated {len(rows)} embeddings (prefix-contextualized)")
    await conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
