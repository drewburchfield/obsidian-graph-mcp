#!/usr/bin/env python3
"""
Force a rebuild of the materialized connection_count column for the
consulting_graph database.

The MCP server refreshes connection_count lazily (only when >50% of rows are
stale). After bulk edits (deletes, renames, exclusions) the counts that feed
get_hub_notes / get_orphaned_notes drift. This script recomputes them for every
note immediately, bypassing the staleness gate.

Usage:
  python scripts/refresh_counts.py                 # threshold 0.5 (tool default)
  python scripts/refresh_counts.py --threshold 0.55
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env", override=True)

from src.hub_analyzer import HubAnalyzer  # noqa: E402
from src.vector_store import PostgreSQLVectorStore  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="Force connection_count rebuild")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--host", default=os.getenv("CONSULTING_PG_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CONSULTING_PG_PORT", "5434")))
    parser.add_argument("--db", default=os.getenv("CONSULTING_PG_DB", "consulting_graph"))
    args = parser.parse_args()

    store = PostgreSQLVectorStore(
        host=args.host,
        port=args.port,
        database=args.db,
        user=os.getenv("POSTGRES_USER", "obsidian"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    await store.initialize()
    try:
        analyzer = HubAnalyzer(store)
        # Hold the lock and call the refresh directly so it runs regardless of
        # the >50%-stale gate in _ensure_fresh_counts.
        async with analyzer._refresh_lock:
            await analyzer._do_refresh(args.threshold)
        print("connection_count rebuild complete.")
    finally:
        await store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
