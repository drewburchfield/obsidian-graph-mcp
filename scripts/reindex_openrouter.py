#!/usr/bin/env python3
"""Re-embed the consulting_graph `notes` table (contextual PREFIX) via OpenRouter
Qwen3-Embedding-8B (4096d). All-remote embedding path. 429-tolerant.

  cd ~/dev/projects/obsidian-graph-mcp && .venv/bin/python scripts/reindex_openrouter.py
"""

import asyncio
import json
import os
import time
import urllib.request
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env", override=True)
KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "qwen/qwen3-embedding-8b")


def validated_embeddings(data, expected_count):
    ordered = sorted(data, key=lambda item: item.get("index", 0))
    if len(ordered) != expected_count:
        raise RuntimeError(
            f"OpenRouter returned {len(ordered)} embeddings for {expected_count} inputs"
        )
    return [item["embedding"] for item in ordered]


def embed(texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        batch = [t[:7000] for t in texts[i : i + bs]]
        last_error = None
        for _attempt in range(8):
            try:
                req = urllib.request.Request(  # noqa: S310
                    URL,
                    json.dumps({"model": MODEL, "input": batch}).encode(),
                    {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                )
                d = json.load(urllib.request.urlopen(req, timeout=120))  # noqa: S310
                if "data" in d:
                    out.extend(validated_embeddings(d["data"], len(batch)))
                    break
            except Exception as exc:  # noqa: BLE001 - retried below and chained on failure
                last_error = exc
            time.sleep(6)
        else:
            raise RuntimeError(f"OpenRouter embed failed at batch {i}") from last_error
        print(".", end="", flush=True)
    print()
    return out


async def main() -> int:
    conn = await asyncpg.connect(
        host=os.getenv("CONSULTING_PG_HOST", "localhost"),
        port=int(os.getenv("CONSULTING_PG_PORT", "5434")),
        database=os.getenv("CONSULTING_PG_DB", "consulting_graph"),
        user=os.getenv("CONSULTING_PG_USER", "obsidian"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    rows = await conn.fetch(
        "SELECT path, title, chunk_index, content FROM notes ORDER BY path, chunk_index"
    )
    ctx = {}
    for r in rows:
        if r["chunk_index"] == 0:
            ctx[r["path"]] = (r["title"] + ". " + r["content"])[:250].replace("\n", " ")
    for r in rows:
        ctx.setdefault(r["path"], r["title"])
    prefixed = [f"[Document: {ctx[r['path']]}]\n{r['content']}" for r in rows]
    print(f"embedding {len(rows)} chunks via OpenRouter {MODEL} (4096d)", flush=True)
    embs = embed(prefixed)
    if len(embs) != len(rows):
        raise RuntimeError(f"Embedding count mismatch: {len(embs)} for {len(rows)} rows")
    async with conn.transaction():
        for r, e in zip(rows, embs, strict=True):
            await conn.execute(
                "UPDATE notes SET embedding = $1::vector WHERE path = $2 AND chunk_index = $3",
                str(e),
                r["path"],
                r["chunk_index"],
            )
    print(f"UPDATED {len(rows)} embeddings (OpenRouter 8B, prefix)")
    await conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
