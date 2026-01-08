"""
Integration tests for MCP tools.

Tests all 3 core MCP tools with quality baseline verification:
1. search_notes - Semantic search across vault
2. get_similar_notes - Find related notes
3. get_connection_graph - Build knowledge graph

NOTE: This is an integration test requiring:
- PostgreSQL with pgvector running
- Valid VOYAGE_API_KEY for embeddings

Marked as slow to allow skipping in quick test runs.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import VoyageEmbedder
from src.graph_builder import GraphBuilder
from src.vector_store import Note, PostgreSQLVectorStore


# Skip if no VOYAGE_API_KEY (required for embeddings)
# Also skip if RUN_INTEGRATION_TESTS env var is not set (for pytest runs)
pytestmark = [
    pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="VOYAGE_API_KEY not set - skipping integration tests",
    ),
    pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="RUN_INTEGRATION_TESTS not set - skipping (set to 'true' to run)",
    ),
    pytest.mark.slow,
]


@pytest.mark.asyncio
async def test_mcp_tools_integration(tmp_path):
    """Comprehensive integration test of all 3 MCP tools."""
    print("\n🧪 Testing MCP Tools Integration")
    print("=" * 60)

    # Use temp directory for cache to avoid permission issues
    cache_dir = tmp_path / "embeddings_cache"

    # Initialize
    embedder = VoyageEmbedder(cache_dir=str(cache_dir))
    store = PostgreSQLVectorStore()
    await store.initialize()
    graph_builder = GraphBuilder(store)

    try:
        # Setup: Create 3 test notes
        print("\n📝 Setup: Creating test notes...")
        test_notes = [
            (
                "ml/machine-learning.md",
                "Machine Learning",
                "Machine learning uses algorithms to learn from data. Neural networks and deep learning are important techniques.",
            ),
            (
                "ml/neural-networks.md",
                "Neural Networks",
                "Neural networks are inspired by biological neural networks. Deep learning uses multi-layer neural networks.",
            ),
            (
                "philosophy/mind.md",
                "The Mind",
                "The human mind is a complex system. Philosophers study consciousness and cognition.",
            ),
        ]

        texts = [content for _, _, content in test_notes]
        embeddings = embedder.embed_batch(texts, input_type="document")

        notes = []
        for (path, title, content), embedding in zip(test_notes, embeddings, strict=False):
            notes.append(
                Note(
                    path=path,
                    title=title,
                    content=content,
                    embedding=embedding,
                    modified_at=datetime.now(),
                    file_size_bytes=len(content),
                )
            )

        await store.upsert_batch(notes)
        print(f"   ✅ Created {len(notes)} test notes")

        # Test 1: search_notes
        print("\n🔍 Test 1: search_notes")
        print("-" * 60)

        query_emb = embedder.embed("neural networks deep learning", input_type="query")
        start = time.time()
        results = await store.search(query_emb, limit=5, threshold=0.0)
        latency_ms = (time.time() - start) * 1000

        print(f"   Results: {len(results)} notes found")
        print(f"   Performance: {latency_ms:.1f}ms (target: <500ms)")

        # Assertions
        assert latency_ms < 500, f"Search took {latency_ms:.1f}ms (target: <500ms)"

        for result in results:
            assert 0.0 <= result.similarity <= 1.0, f"Similarity {result.similarity} out of range"
            print(f"   - {result.title}: {result.similarity:.3f}")

        print("   ✅ search_notes passed")

        # Test 2: get_similar_notes
        print("\n🔗 Test 2: get_similar_notes")
        print("-" * 60)

        start = time.time()
        similar = await store.get_similar_notes("ml/machine-learning.md", limit=5, threshold=0.0)
        latency_ms = (time.time() - start) * 1000

        print(f"   Results: {len(similar)} similar notes")
        print(f"   Performance: {latency_ms:.1f}ms (target: <300ms)")

        # Assertions
        assert latency_ms < 300, f"Similar search took {latency_ms:.1f}ms"

        paths = [r.path for r in similar]
        assert "ml/machine-learning.md" not in paths, "Should exclude source note"

        for result in similar:
            assert 0.0 <= result.similarity <= 1.0
            print(f"   - {result.title}: {result.similarity:.3f}")

        print("   ✅ get_similar_notes passed")

        # Test 3: get_connection_graph
        print("\n🕸️  Test 3: get_connection_graph")
        print("-" * 60)

        start = time.time()
        graph = await graph_builder.build_connection_graph(
            "ml/machine-learning.md", depth=2, max_per_level=3, threshold=0.0
        )
        latency_ms = (time.time() - start) * 1000

        print(
            f"   Network: {graph['stats']['total_nodes']} nodes, {graph['stats']['total_edges']} edges"
        )
        print(f"   Performance: {latency_ms:.1f}ms (target: <2000ms)")

        # Assertions
        assert latency_ms < 2000, f"Graph building took {latency_ms:.1f}ms"

        # Check structure
        assert "root" in graph
        assert "nodes" in graph
        assert "edges" in graph
        assert graph["root"]["path"] == "ml/machine-learning.md"

        # Check no cycles
        paths = [n["path"] for n in graph["nodes"]]
        assert len(paths) == len(set(paths)), "Duplicate nodes (cycle detected)"

        # Check edge similarities
        for edge in graph["edges"]:
            assert (
                0.0 <= edge["similarity"] <= 1.0
            ), f"Edge similarity {edge['similarity']} out of range"

        print(f"   - Root: {graph['root']['title']}")
        for node in graph["nodes"][:5]:
            print(f"   - L{node['level']}: {node['title']}")

        print("   ✅ get_connection_graph passed")

        print("\n" + "=" * 60)
        print("✅ All MCP tools passed quality baselines!")
        print("=" * 60)

    finally:
        await store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
