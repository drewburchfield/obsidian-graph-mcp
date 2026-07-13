#!/usr/bin/env python3
"""
Standalone E2E regression tests for Obsidian Graph.

Runs inside Docker without pytest. Tests all tools, validation,
security, data integrity, and performance baselines.

Usage (the image ships only src/, so copy the script in first):
    docker cp scripts/run_e2e_tests.py obsidian-graph:/tmp/
    docker exec -w /app obsidian-graph .venv/bin/python /tmp/run_e2e_tests.py

Or locally (with postgres port-mapped):
    POSTGRES_HOST=localhost python3 scripts/run_e2e_tests.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_all_tests():
    from src.embedder import VoyageEmbedder
    from src.graph_builder import GraphBuilder
    from src.hub_analyzer import HubAnalyzer
    from src.security_utils import SecurityError
    from src.server import _FORMATTERS
    from src.tools import TOOLS, ToolContext, ToolError
    from src.tools import get_connection_graph as tool_get_connection_graph
    from src.tools import get_hub_notes as tool_get_hub_notes
    from src.tools import get_orphaned_notes as tool_get_orphaned_notes
    from src.tools import get_similar_notes as tool_get_similar_notes
    from src.tools import search_notes as tool_search_notes
    from src.validation import ValidationError
    from src.vector_store import PostgreSQLVectorStore

    store = PostgreSQLVectorStore()
    await store.initialize()
    embedder = VoyageEmbedder(cache_dir="/tmp/e2e_cache")
    graph = GraphBuilder(store)
    hub = HubAnalyzer(store)
    ctx = ToolContext(store=store, embedder=embedder, graph_builder=graph, hub_analyzer=hub)

    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name} {detail}")

    # -- search_notes --
    print("== search_notes ==")
    t = time.time()
    r = await tool_search_notes(ctx, {"query": "project management", "limit": 5, "threshold": 0.3})
    ms = (time.time() - t) * 1000
    check("returns results", len(r["results"]) > 0)
    check("latency < 2s", ms < 2000, f"{ms:.0f}ms")
    check(
        "has fields",
        all(
            "path" in x and "title" in x and "similarity" in x and "content" in x
            for x in r["results"]
        ),
    )
    check("similarity range", all(0 <= x["similarity"] <= 1 for x in r["results"]))

    r = await tool_search_notes(ctx, {"query": "xyzzy gobbledygook", "limit": 5, "threshold": 0.99})
    check("high threshold empty", len(r["results"]) == 0)

    r = await tool_search_notes(ctx, {"query": "notes", "limit": 2, "threshold": 0.1})
    check("limit respected", len(r["results"]) <= 2)

    # -- get_similar_notes --
    print("== get_similar_notes ==")
    paths = await store.get_all_paths()
    tp = paths[0]
    r = await tool_get_similar_notes(ctx, {"note_path": tp, "limit": 5, "threshold": 0.3})
    check("returns results", len(r["results"]) > 0)
    check("excludes self", tp not in [x["path"] for x in r["results"]])

    try:
        await tool_get_similar_notes(ctx, {"note_path": "nope.md", "limit": 5, "threshold": 0.3})
        check("raises on missing", False)
    except Exception as e:
        check("raises on missing", "not found" in str(e).lower())

    # -- get_connection_graph --
    print("== get_connection_graph ==")
    t = time.time()
    g = await tool_get_connection_graph(
        ctx, {"note_path": tp, "depth": 2, "max_per_level": 3, "threshold": 0.3}
    )
    ms = (time.time() - t) * 1000
    check("has root", g["root"]["path"] == tp)
    check(
        "no dupes",
        len([n["path"] for n in g["nodes"]]) == len({n["path"] for n in g["nodes"]}),
    )
    check("edge range", all(0 <= e["similarity"] <= 1 for e in g["edges"]))
    check("latency < 5s", ms < 5000, f"{ms:.0f}ms")
    print(f"    {g['stats']['total_nodes']} nodes, {g['stats']['total_edges']} edges")

    try:
        await tool_get_connection_graph(
            ctx,
            {
                "note_path": "nope.md",
                "depth": 1,
                "max_per_level": 3,
                "threshold": 0.5,
            },
        )
        check("raises ToolError", False)
    except ToolError:
        check("raises ToolError", True)
    except Exception as e:
        check("raises ToolError", False, f"{type(e).__name__}")

    # -- get_hub_notes --
    print("== get_hub_notes ==")
    t = time.time()
    r = await tool_get_hub_notes(ctx, {"min_connections": 5, "threshold": 0.3, "limit": 10})
    ms1 = (time.time() - t) * 1000
    check("returns hubs", len(r["results"]) > 0)
    check("has connection_count", all("connection_count" in h for h in r["results"]))
    check(
        "sorted desc",
        (
            all(
                r["results"][i]["connection_count"] >= r["results"][i + 1]["connection_count"]
                for i in range(len(r["results"]) - 1)
            )
            if len(r["results"]) > 1
            else True
        ),
    )

    t = time.time()
    await tool_get_hub_notes(ctx, {"min_connections": 5, "threshold": 0.3, "limit": 10})
    ms2 = (time.time() - t) * 1000
    check("cached faster", ms2 < ms1, f"first={ms1:.0f}ms cached={ms2:.0f}ms")

    # -- get_orphaned_notes --
    print("== get_orphaned_notes ==")
    r = await tool_get_orphaned_notes(ctx, {"max_connections": 3, "threshold": 0.3, "limit": 10})
    check("returns list", isinstance(r["results"], list))
    check("below max", all(o["connection_count"] <= 3 for o in r["results"]))

    # -- validation --
    print("== validation ==")
    for name, fn, args in [
        ("empty query", tool_search_notes, {"query": "", "limit": 10}),
        ("limit too high", tool_search_notes, {"query": "x", "limit": 100}),
        ("bad threshold", tool_search_notes, {"query": "x", "threshold": -1}),
        ("depth too high", tool_get_connection_graph, {"note_path": "x.md", "depth": 10}),
    ]:
        try:
            await fn(ctx, args)
            check(f"val: {name}", False)
        except ValidationError:
            check(f"val: {name}", True)
        except Exception as e:
            check(f"val: {name}", False, f"{type(e).__name__}")

    # -- security --
    print("== security ==")
    for bad in ["../../../etc/passwd", "/absolute/path.md", "a/../../../etc/shadow"]:
        try:
            await tool_get_similar_notes(ctx, {"note_path": bad, "limit": 5, "threshold": 0.5})
            check(f"sec: {bad}", False)
        except SecurityError:
            check(f"sec: {bad}", True)
        except Exception as e:
            check(f"sec: {bad}", False, f"{type(e).__name__}")

    # -- data integrity --
    print("== data integrity ==")
    rows = await store.pool.fetch(
        "SELECT path, COUNT(*) c, MAX(total_chunks) e "
        "FROM notes WHERE total_chunks > 1 GROUP BY path"
    )
    check("chunked notes exist", len(rows) > 0)
    check("chunks match expected", all(r["c"] == r["e"] for r in rows))

    check(
        "10_Shelley excluded",
        await store.pool.fetchval("SELECT COUNT(*) FROM notes WHERE path LIKE '10_Shelley/%'") == 0,
    )
    check(
        ".obsidian excluded",
        await store.pool.fetchval("SELECT COUNT(*) FROM notes WHERE path LIKE '.obsidian/%'") == 0,
    )
    check(
        "no trigger",
        await store.pool.fetchval(
            "SELECT COUNT(*) FROM pg_trigger WHERE tgrelid = 'notes'::regclass"
        )
        == 0,
    )

    # -- infrastructure --
    print("== infrastructure ==")
    s = store.get_pool_stats()
    check("pool initialized", s["initialized"])
    check("max=20", s["max_size"] == 20)
    check("TOOLS == FORMATTERS", set(TOOLS.keys()) == set(_FORMATTERS.keys()))

    await store.close()
    print(f"\nRESULTS: {passed} passed, {failed} failed / {passed + failed} total")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
