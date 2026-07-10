"""
End-to-end regression tests against a running Docker deployment.

Requires:
- Docker containers running (docker compose up)
- Vault indexed (wait for startup scan to complete)
- VOYAGE_API_KEY set in environment

Run with: pytest tests/test_e2e_docker.py -v -s
Or:       pytest -m e2e -v -s
"""

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip entire module if not configured for e2e
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.getenv("RUN_E2E_TESTS"),
        reason="RUN_E2E_TESTS not set",
    ),
    pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="VOYAGE_API_KEY not set",
    ),
    pytest.mark.skipif(
        not os.getenv("POSTGRES_PASSWORD"),
        reason="POSTGRES_PASSWORD not set (Docker not running?)",
    ),
]


@pytest.fixture(scope="module")
async def ctx():
    """Create a ToolContext connected to the running Docker database."""
    from src.embedder import VoyageEmbedder
    from src.graph_builder import GraphBuilder
    from src.hub_analyzer import HubAnalyzer
    from src.tools import ToolContext
    from src.vector_store import PostgreSQLVectorStore

    store = PostgreSQLVectorStore()
    await store.initialize()

    embedder = VoyageEmbedder(cache_dir="/tmp/e2e_test_cache")
    graph = GraphBuilder(store)
    hub = HubAnalyzer(store)

    context = ToolContext(
        store=store,
        embedder=embedder,
        graph_builder=graph,
        hub_analyzer=hub,
    )

    yield context

    await store.close()


@pytest.fixture(scope="module")
async def sample_path(ctx):
    """Get a real note path from the database for testing."""
    paths = await ctx.store.get_all_paths()
    assert len(paths) > 0, "No notes indexed. Wait for startup scan to complete."
    return paths[0]


# -- Search Notes --


class TestSearchNotes:
    async def test_basic_search_returns_results(self, ctx):
        from src.tools import search_notes

        result = await search_notes(
            ctx, {"query": "project management", "limit": 5, "threshold": 0.3}
        )
        assert len(result["results"]) > 0

    async def test_results_have_required_fields(self, ctx):
        from src.tools import search_notes

        result = await search_notes(ctx, {"query": "knowledge", "limit": 3, "threshold": 0.3})
        for r in result["results"]:
            assert "path" in r
            assert "title" in r
            assert "content" in r
            assert "similarity" in r
            assert 0.0 <= r["similarity"] <= 1.0

    async def test_search_latency_under_2s(self, ctx):
        from src.tools import search_notes

        start = time.time()
        await search_notes(ctx, {"query": "neural networks", "limit": 5, "threshold": 0.3})
        assert (time.time() - start) < 2.0

    async def test_high_threshold_returns_empty(self, ctx):
        from src.tools import search_notes

        result = await search_notes(
            ctx, {"query": "xyzzy gobbledygook", "limit": 5, "threshold": 0.99}
        )
        assert len(result["results"]) == 0

    async def test_limit_respected(self, ctx):
        from src.tools import search_notes

        result = await search_notes(ctx, {"query": "notes", "limit": 2, "threshold": 0.1})
        assert len(result["results"]) <= 2


# -- Similar Notes --


class TestSimilarNotes:
    async def test_returns_similar_notes(self, ctx, sample_path):
        from src.tools import get_similar_notes

        result = await get_similar_notes(
            ctx, {"note_path": sample_path, "limit": 5, "threshold": 0.3}
        )
        assert len(result["results"]) > 0

    async def test_excludes_source_note(self, ctx, sample_path):
        from src.tools import get_similar_notes

        result = await get_similar_notes(
            ctx, {"note_path": sample_path, "limit": 5, "threshold": 0.3}
        )
        assert sample_path not in [r["path"] for r in result["results"]]

    async def test_nonexistent_note_raises(self, ctx):
        from src.tools import get_similar_notes
        from src.vector_store import VectorStoreError

        with pytest.raises(VectorStoreError, match="[Nn]ot found"):
            await get_similar_notes(
                ctx, {"note_path": "does-not-exist.md", "limit": 5, "threshold": 0.3}
            )


# -- Connection Graph --


class TestConnectionGraph:
    async def test_builds_graph(self, ctx, sample_path):
        from src.tools import get_connection_graph

        result = await get_connection_graph(
            ctx, {"note_path": sample_path, "depth": 2, "max_per_level": 3, "threshold": 0.3}
        )
        assert result["root"]["path"] == sample_path
        assert "nodes" in result
        assert "edges" in result
        assert result["stats"]["total_nodes"] > 0

    async def test_no_duplicate_nodes(self, ctx, sample_path):
        from src.tools import get_connection_graph

        result = await get_connection_graph(
            ctx, {"note_path": sample_path, "depth": 2, "max_per_level": 3, "threshold": 0.3}
        )
        paths = [n["path"] for n in result["nodes"]]
        assert len(paths) == len(set(paths)), "Duplicate nodes detected"

    async def test_edge_similarities_in_range(self, ctx, sample_path):
        from src.tools import get_connection_graph

        result = await get_connection_graph(
            ctx, {"note_path": sample_path, "depth": 2, "max_per_level": 3, "threshold": 0.3}
        )
        for edge in result["edges"]:
            assert 0.0 <= edge["similarity"] <= 1.0

    async def test_graph_latency_under_5s(self, ctx, sample_path):
        from src.tools import get_connection_graph

        start = time.time()
        await get_connection_graph(
            ctx, {"note_path": sample_path, "depth": 2, "max_per_level": 3, "threshold": 0.3}
        )
        assert (time.time() - start) < 5.0

    async def test_nonexistent_note_raises_tool_error(self, ctx):
        from src.tools import ToolError, get_connection_graph

        with pytest.raises(ToolError, match="[Nn]ot found"):
            await get_connection_graph(
                ctx,
                {"note_path": "nonexistent.md", "depth": 1, "max_per_level": 3, "threshold": 0.5},
            )


# -- Hub Notes --


class TestHubNotes:
    async def test_returns_hubs(self, ctx):
        from src.tools import get_hub_notes

        result = await get_hub_notes(ctx, {"min_connections": 5, "threshold": 0.3, "limit": 10})
        assert len(result["results"]) > 0

    async def test_hubs_have_connection_count(self, ctx):
        from src.tools import get_hub_notes

        result = await get_hub_notes(ctx, {"min_connections": 5, "threshold": 0.3, "limit": 10})
        for hub in result["results"]:
            assert "connection_count" in hub
            assert hub["connection_count"] >= 5

    async def test_hubs_sorted_desc(self, ctx):
        from src.tools import get_hub_notes

        result = await get_hub_notes(ctx, {"min_connections": 5, "threshold": 0.3, "limit": 10})
        counts = [h["connection_count"] for h in result["results"]]
        assert counts == sorted(counts, reverse=True)

    async def test_cached_call_faster(self, ctx):
        from src.tools import get_hub_notes

        # First call may trigger refresh
        start = time.time()
        await get_hub_notes(ctx, {"min_connections": 5, "threshold": 0.3, "limit": 10})
        first_ms = (time.time() - start) * 1000

        # Second call should use cached counts
        start = time.time()
        await get_hub_notes(ctx, {"min_connections": 5, "threshold": 0.3, "limit": 10})
        second_ms = (time.time() - start) * 1000

        assert second_ms < first_ms, (
            f"Cached call not faster: first={first_ms:.0f}ms, second={second_ms:.0f}ms"
        )


# -- Orphaned Notes --


class TestOrphanedNotes:
    async def test_returns_orphans(self, ctx):
        from src.tools import get_orphaned_notes

        result = await get_orphaned_notes(
            ctx, {"max_connections": 5, "threshold": 0.3, "limit": 10}
        )
        assert isinstance(result["results"], list)

    async def test_orphans_below_max(self, ctx):
        from src.tools import get_orphaned_notes

        result = await get_orphaned_notes(
            ctx, {"max_connections": 3, "threshold": 0.3, "limit": 10}
        )
        for orphan in result["results"]:
            assert orphan["connection_count"] <= 3


# -- Input Validation --


class TestValidation:
    async def test_empty_query_rejected(self, ctx):
        from src.tools import search_notes
        from src.validation import ValidationError

        with pytest.raises(ValidationError):
            await search_notes(ctx, {"query": "", "limit": 10})

    async def test_limit_too_high_rejected(self, ctx):
        from src.tools import search_notes
        from src.validation import ValidationError

        with pytest.raises(ValidationError):
            await search_notes(ctx, {"query": "test", "limit": 100})

    async def test_negative_threshold_rejected(self, ctx):
        from src.tools import search_notes
        from src.validation import ValidationError

        with pytest.raises(ValidationError):
            await search_notes(ctx, {"query": "test", "threshold": -1.0})

    async def test_depth_too_high_rejected(self, ctx):
        from src.tools import get_connection_graph
        from src.validation import ValidationError

        with pytest.raises(ValidationError):
            await get_connection_graph(ctx, {"note_path": "x.md", "depth": 10})


# -- Security --


class TestSecurity:
    @pytest.mark.parametrize(
        "bad_path",
        [
            "../../../etc/passwd",
            "/absolute/path.md",
            "notes/../../../etc/shadow",
        ],
    )
    async def test_path_traversal_rejected(self, ctx, bad_path):
        from src.security_utils import SecurityError
        from src.tools import get_similar_notes

        with pytest.raises(SecurityError):
            await get_similar_notes(ctx, {"note_path": bad_path, "limit": 5, "threshold": 0.5})


# -- Data Integrity --


class TestDataIntegrity:
    async def test_chunked_notes_consistent(self, ctx):
        """Verify chunk_count matches actual row count for chunked notes."""
        rows = await ctx.store.pool.fetch(
            "SELECT path, COUNT(*) as actual, MAX(total_chunks) as expected "
            "FROM notes WHERE total_chunks > 1 GROUP BY path"
        )
        assert len(rows) > 0, "No chunked notes found"
        for row in rows:
            assert row["actual"] == row["expected"], (
                f"{row['path']}: {row['actual']} chunks vs {row['expected']} expected"
            )

    async def test_excluded_paths_not_indexed(self, ctx):
        """Verify exclusion config is enforced."""
        shelley = await ctx.store.pool.fetchval(
            "SELECT COUNT(*) FROM notes WHERE path LIKE '10_Shelley/%'"
        )
        assert shelley == 0, f"Found {shelley} excluded 10_Shelley notes"

        obsidian = await ctx.store.pool.fetchval(
            "SELECT COUNT(*) FROM notes WHERE path LIKE '.obsidian/%'"
        )
        assert obsidian == 0, f"Found {obsidian} excluded .obsidian notes"

    async def test_no_trigger_exists(self, ctx):
        """Verify the modified_at trigger was removed."""
        count = await ctx.store.pool.fetchval(
            "SELECT COUNT(*) FROM pg_trigger WHERE tgrelid = 'notes'::regclass"
        )
        assert count == 0, "Trigger still exists on notes table"


# -- Infrastructure --


class TestInfrastructure:
    async def test_pool_stats(self, ctx):
        stats = ctx.store.get_pool_stats()
        assert stats["initialized"]
        assert stats["size"] > 0
        assert stats["max_size"] == 20

    async def test_dispatch_tables_match(self):
        from src.server import _FORMATTERS
        from src.tools import TOOLS

        assert set(TOOLS.keys()) == set(_FORMATTERS.keys()), (
            f"Mismatch: TOOLS={set(TOOLS.keys())}, FORMATTERS={set(_FORMATTERS.keys())}"
        )
