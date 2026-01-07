"""
Unit tests for MCP tools.

Tests the 3 required tools: search_notes, get_similar_notes, get_connection_graph
"""

import asyncio

# Add src to path
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import VoyageEmbedder
from src.graph_builder import GraphBuilder
from src.vector_store import Note, PostgreSQLVectorStore


# Fixtures
@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def setup_test_data():
    """Set up test database with sample notes."""
    embedder = VoyageEmbedder()
    store = PostgreSQLVectorStore()
    await store.initialize()

    # Create test notes
    test_notes_content = [
        (
            "ml/machine-learning.md",
            "Machine Learning",
            """
Machine learning is a subset of artificial intelligence that focuses on
developing algorithms that can learn from and make predictions on data.
Neural networks and deep learning are key techniques.
        """,
        ),
        (
            "ml/neural-networks.md",
            "Neural Networks",
            """
Neural networks are computing systems inspired by biological neural networks.
They consist of layers of interconnected nodes that process information.
Deep learning uses multi-layer neural networks.
        """,
        ),
        (
            "ml/deep-learning.md",
            "Deep Learning",
            """
Deep learning is a subset of machine learning using multi-layered neural networks.
It excels at tasks like image recognition, natural language processing,
and speech recognition.
        """,
        ),
        (
            "philosophy/consciousness.md",
            "Consciousness",
            """
Consciousness is the state of being aware of one's thoughts, feelings, and surroundings.
Philosophers have debated the nature of consciousness for centuries.
        """,
        ),
        (
            "neuroscience/brain.md",
            "The Brain",
            """
The human brain is the central organ of the nervous system.
It contains billions of neurons that form complex networks.
        """,
        ),
    ]

    # Generate embeddings and insert
    texts = [content for _, _, content in test_notes_content]
    embeddings = embedder.embed_batch(texts, input_type="document")

    notes = []
    for (path, title, content), embedding in zip(test_notes_content, embeddings, strict=False):
        if embedding:
            notes.append(
                Note(
                    path=path,
                    title=title,
                    content=content.strip(),
                    embedding=embedding,
                    modified_at=datetime.now(),
                    file_size_bytes=len(content),
                )
            )

    await store.upsert_batch(notes)

    yield store, embedder

    # Cleanup
    await store.close()


# Tests
@pytest.mark.asyncio
async def test_search_notes_similarity_range(setup_test_data):
    """Ensure similarity scores are in [0.0, 1.0] range."""
    store, embedder = setup_test_data

    query_embedding = embedder.embed("machine learning algorithms", input_type="query")
    results = await store.search(query_embedding, limit=10, threshold=0.0)

    for result in results:
        assert (
            0.0 <= result.similarity <= 1.0
        ), f"Similarity {result.similarity} out of range [0.0, 1.0]"


@pytest.mark.asyncio
async def test_search_notes_performance(setup_test_data):
    """Verify search latency < 500ms."""
    store, embedder = setup_test_data

    query_embedding = embedder.embed("neural networks", input_type="query")

    start = time.time()
    results = await store.search(query_embedding, limit=10)
    latency_ms = (time.time() - start) * 1000

    assert latency_ms < 500, f"Search took {latency_ms:.1f}ms (target: <500ms)"


@pytest.mark.asyncio
async def test_search_notes_threshold(setup_test_data):
    """Verify threshold filtering works."""
    store, embedder = setup_test_data

    query_embedding = embedder.embed("machine learning", input_type="query")
    results = await store.search(query_embedding, limit=10, threshold=0.2)

    for result in results:
        assert (
            result.similarity >= 0.2
        ), f"Result {result.title} has similarity {result.similarity} < threshold 0.2"


@pytest.mark.asyncio
async def test_get_similar_notes_excludes_self(setup_test_data):
    """Verify get_similar_notes excludes the source note."""
    store, _ = setup_test_data

    results = await store.get_similar_notes("ml/machine-learning.md", limit=10)

    paths = [r.path for r in results]
    assert "ml/machine-learning.md" not in paths, "get_similar_notes should exclude the source note"


@pytest.mark.asyncio
async def test_get_similar_notes_finds_related(setup_test_data):
    """Verify get_similar_notes finds semantically related notes."""
    store, _ = setup_test_data

    results = await store.get_similar_notes("ml/machine-learning.md", limit=5, threshold=0.3)

    # Should find neural networks and deep learning (semantically related)
    paths = [r.path for r in results]
    assert any(
        "neural" in path.lower() or "deep" in path.lower() for path in paths
    ), "Should find semantically related ML notes"


@pytest.mark.asyncio
async def test_connection_graph_no_cycles(setup_test_data):
    """Verify BFS doesn't revisit nodes (no cycles)."""
    store, _ = setup_test_data

    graph_builder = GraphBuilder(store)
    graph = await graph_builder.build_connection_graph(
        "ml/machine-learning.md", depth=3, max_per_level=5, threshold=0.3
    )

    paths = [n["path"] for n in graph["nodes"]]
    assert len(paths) == len(set(paths)), "Duplicate nodes detected (cycle in graph)"


@pytest.mark.asyncio
async def test_connection_graph_structure(setup_test_data):
    """Verify graph structure is correct."""
    store, _ = setup_test_data

    graph_builder = GraphBuilder(store)
    graph = await graph_builder.build_connection_graph(
        "ml/machine-learning.md", depth=2, max_per_level=3
    )

    # Verify structure
    assert "root" in graph
    assert "nodes" in graph
    assert "edges" in graph
    assert "stats" in graph

    # Verify root
    assert graph["root"]["path"] == "ml/machine-learning.md"

    # Verify stats match actual data
    assert graph["stats"]["total_nodes"] == len(graph["nodes"])
    assert graph["stats"]["total_edges"] == len(graph["edges"])


@pytest.mark.asyncio
async def test_connection_graph_performance(setup_test_data):
    """Verify graph building < 2 seconds."""
    store, _ = setup_test_data

    graph_builder = GraphBuilder(store)

    start = time.time()
    graph = await graph_builder.build_connection_graph(
        "ml/machine-learning.md", depth=3, max_per_level=5, threshold=0.3
    )
    latency_ms = (time.time() - start) * 1000

    assert latency_ms < 2000, f"Graph building took {latency_ms:.1f}ms (target: <2000ms)"


@pytest.mark.asyncio
async def test_connection_graph_edge_similarity(setup_test_data):
    """Verify edge similarities are in valid range."""
    store, _ = setup_test_data

    graph_builder = GraphBuilder(store)
    graph = await graph_builder.build_connection_graph("ml/machine-learning.md", depth=2)

    for edge in graph["edges"]:
        assert (
            0.0 <= edge["similarity"] <= 1.0
        ), f"Edge similarity {edge['similarity']} out of range"


@pytest.mark.asyncio
async def test_connection_graph_levels(setup_test_data):
    """Verify nodes are at correct levels."""
    store, _ = setup_test_data

    graph_builder = GraphBuilder(store)
    graph = await graph_builder.build_connection_graph(
        "ml/machine-learning.md", depth=3, max_per_level=3
    )

    for node in graph["nodes"]:
        assert 0 <= node["level"] <= 3, f"Node {node['title']} at invalid level {node['level']}"

    # Root should be at level 0
    root_node = next(n for n in graph["nodes"] if n["path"] == "ml/machine-learning.md")
    assert root_node["level"] == 0, "Root should be at level 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
