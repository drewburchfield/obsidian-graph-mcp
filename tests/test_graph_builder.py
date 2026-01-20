"""
Unit tests for BFS graph traversal.

Tests:
1. Multi-hop graph building
2. Cycle prevention in BFS
3. Depth parameter clamping
4. Level-by-level traversal
5. Similarity edge computation
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_builder import GraphBuilder


@pytest.mark.asyncio
async def test_build_connection_graph_structure(mock_store):
    """Test that connection graph has correct structure."""
    builder = GraphBuilder(mock_store)

    # Mock _get_note_info to return root note
    builder._get_note_info = AsyncMock(return_value={"path": "root.md", "title": "Root Note"})

    # Mock get_similar_notes to return empty (no connections)
    mock_store.get_similar_notes = AsyncMock(return_value=[])

    # Build graph
    graph = await builder.build_connection_graph("root.md", depth=2, max_per_level=5, threshold=0.5)

    # Verify structure
    assert "root" in graph
    assert "nodes" in graph
    assert "edges" in graph
    assert "stats" in graph

    # Verify root
    assert graph["root"]["path"] == "root.md"
    assert graph["root"]["title"] == "Root Note"

    # Verify stats
    assert graph["stats"]["total_nodes"] >= 1
    assert graph["stats"]["total_edges"] >= 0


@pytest.mark.asyncio
async def test_bfs_prevents_cycles(mock_store):
    """Test that BFS traversal prevents cycles."""
    builder = GraphBuilder(mock_store)

    # Create a circular reference scenario
    # root -> A -> B -> A (cycle!)
    note_info = {
        "root.md": {"path": "root.md", "title": "Root"},
        "A.md": {"path": "A.md", "title": "Note A"},
        "B.md": {"path": "B.md", "title": "Note B"},
    }

    builder._get_note_info = AsyncMock(side_effect=lambda p: note_info.get(p))

    # Mock search results to create cycle
    from src.vector_store import SearchResult

    async def mock_similar(note_path, limit, threshold):
        if note_path == "root.md":
            return [SearchResult("A.md", "Note A", 0.9, "content")]
        elif note_path == "A.md":
            return [SearchResult("B.md", "Note B", 0.8, "content")]
        elif note_path == "B.md":
            return [SearchResult("A.md", "Note A", 0.7, "content")]  # Back to A!
        return []

    mock_store.get_similar_notes = AsyncMock(side_effect=mock_similar)
    builder._compute_similarity = AsyncMock(return_value=0.8)

    # Build graph with depth that would expose cycle
    graph = await builder.build_connection_graph(
        "root.md", depth=5, max_per_level=10, threshold=0.5
    )

    # Count how many times each node appears
    node_paths = [n["path"] for n in graph["nodes"]]

    # Each node should appear exactly once (no duplicates from cycles)
    assert len(node_paths) == len(set(node_paths)), (
        "Graph contains duplicate nodes (cycle detected!)"
    )

    # Specifically check that A.md doesn't appear twice
    assert node_paths.count("A.md") <= 1


@pytest.mark.asyncio
async def test_depth_clamping(mock_store):
    """Test that depth parameter is clamped to [1, 5]."""
    builder = GraphBuilder(mock_store)

    builder._get_note_info = AsyncMock(return_value={"path": "root.md", "title": "Root"})
    mock_store.get_similar_notes = AsyncMock(return_value=[])

    # Test depth=0 gets clamped to 1
    graph = await builder.build_connection_graph("root.md", depth=0)
    assert graph["stats"]["levels"] == 1

    # Test depth=100 gets clamped to 5
    graph = await builder.build_connection_graph("root.md", depth=100)
    assert graph["stats"]["levels"] == 5

    # Test negative depth gets clamped to 1
    graph = await builder.build_connection_graph("root.md", depth=-5)
    assert graph["stats"]["levels"] == 1


@pytest.mark.asyncio
async def test_max_per_level_clamping(mock_store):
    """Test that max_per_level parameter is clamped to [1, 10]."""
    builder = GraphBuilder(mock_store)

    builder._get_note_info = AsyncMock(return_value={"path": "root.md", "title": "Root"})

    # Create 20 similar notes
    from src.vector_store import SearchResult

    all_similar_notes = [
        SearchResult(f"note{i}.md", f"Note {i}", 0.8, "content") for i in range(20)
    ]

    # Mock that respects limit parameter
    async def mock_get_similar(note_path, limit, threshold):
        return all_similar_notes[:limit]  # Respect the limit parameter!

    mock_store.get_similar_notes = AsyncMock(side_effect=mock_get_similar)
    builder._compute_similarity = AsyncMock(return_value=0.8)

    # Test max_per_level=0 gets clamped to 1
    graph = await builder.build_connection_graph("root.md", depth=1, max_per_level=0)
    # Should have root + at most 1 neighbor
    assert graph["stats"]["total_nodes"] <= 2

    # Test max_per_level=100 gets clamped to 10
    graph = await builder.build_connection_graph("root.md", depth=1, max_per_level=100)
    # Should have root + at most 10 neighbors
    assert graph["stats"]["total_nodes"] <= 11


@pytest.mark.asyncio
async def test_graph_builder_computes_edges(mock_store):
    """Test that edges are properly computed with similarity scores."""
    builder = GraphBuilder(mock_store)

    note_info = {
        "root.md": {"path": "root.md", "title": "Root"},
        "child1.md": {"path": "child1.md", "title": "Child 1"},
        "child2.md": {"path": "child2.md", "title": "Child 2"},
    }

    builder._get_note_info = AsyncMock(side_effect=lambda p: note_info.get(p))

    from src.vector_store import SearchResult

    async def mock_similar(note_path, limit, threshold):
        if note_path == "root.md":
            return [
                SearchResult("child1.md", "Child 1", 0.9, "content"),
                SearchResult("child2.md", "Child 2", 0.8, "content"),
            ]
        return []

    mock_store.get_similar_notes = AsyncMock(side_effect=mock_similar)
    builder._compute_similarity = AsyncMock(
        side_effect=lambda p1, p2: 0.9 if "child1" in p2 else 0.8
    )

    # Build graph
    graph = await builder.build_connection_graph(
        "root.md", depth=1, max_per_level=10, threshold=0.5
    )

    # Should have 2 edges (root -> child1, root -> child2)
    assert len(graph["edges"]) == 2

    # Verify edge structure
    for edge in graph["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert "similarity" in edge
        assert edge["source"] == "root.md"
        assert edge["target"] in ["child1.md", "child2.md"]
        assert 0.0 <= edge["similarity"] <= 1.0


@pytest.mark.asyncio
async def test_graph_builder_assigns_levels_correctly(mock_store):
    """Test that nodes are assigned correct level numbers."""
    builder = GraphBuilder(mock_store)

    note_info = {
        "root.md": {"path": "root.md", "title": "Root"},
        "level1_a.md": {"path": "level1_a.md", "title": "Level 1A"},
        "level1_b.md": {"path": "level1_b.md", "title": "Level 1B"},
        "level2_a.md": {"path": "level2_a.md", "title": "Level 2A"},
    }

    builder._get_note_info = AsyncMock(side_effect=lambda p: note_info.get(p))

    from src.vector_store import SearchResult

    async def mock_similar(note_path, limit, threshold):
        if note_path == "root.md":
            return [
                SearchResult("level1_a.md", "Level 1A", 0.9, "content"),
                SearchResult("level1_b.md", "Level 1B", 0.8, "content"),
            ]
        elif note_path == "level1_a.md":
            return [SearchResult("level2_a.md", "Level 2A", 0.7, "content")]
        return []

    mock_store.get_similar_notes = AsyncMock(side_effect=mock_similar)
    builder._compute_similarity = AsyncMock(return_value=0.8)

    # Build graph with depth=2
    graph = await builder.build_connection_graph(
        "root.md", depth=2, max_per_level=10, threshold=0.5
    )

    # Find nodes by path
    nodes_by_path = {n["path"]: n for n in graph["nodes"]}

    # Verify levels
    assert nodes_by_path["root.md"]["level"] == 0
    assert nodes_by_path["level1_a.md"]["level"] == 1
    assert nodes_by_path["level1_b.md"]["level"] == 1
    assert nodes_by_path["level2_a.md"]["level"] == 2


@pytest.mark.asyncio
async def test_compute_similarity_returns_score(mock_store):
    """Test _compute_similarity calculates cosine similarity correctly using single query."""
    builder = GraphBuilder(mock_store)

    # Mock database connection
    mock_conn = AsyncMock()

    # NEW: Single query approach returns similarity directly (not separate embeddings)
    # The query computes: 1.0 - (n1.embedding <=> n2.embedding)
    mock_conn.fetchval = AsyncMock(return_value=0.95)

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

    mock_store.pool = MagicMock()
    mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

    # Compute similarity
    similarity = await builder._compute_similarity("note1.md", "note2.md")

    # Verify single query was made (not 3 separate ones)
    assert mock_conn.fetchval.call_count == 1

    # Should return float in [0.0, 1.0]
    assert isinstance(similarity, float)
    assert 0.0 <= similarity <= 1.0
    assert similarity == 0.95


@pytest.mark.asyncio
async def test_compute_similarity_handles_missing_notes(mock_store):
    """Test _compute_similarity returns 0.0 when notes are missing."""
    builder = GraphBuilder(mock_store)

    mock_conn = AsyncMock()
    # CROSS JOIN returns NULL when either note is missing
    mock_conn.fetchval = AsyncMock(return_value=None)

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

    mock_store.pool = MagicMock()
    mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

    # Compute similarity for missing note
    similarity = await builder._compute_similarity("note1.md", "missing.md")

    # Should return 0.0 for missing embeddings
    assert similarity == 0.0


@pytest.mark.asyncio
async def test_get_note_info_returns_none_for_missing_note(mock_store):
    """Test that _get_note_info returns None for non-existent notes."""
    builder = GraphBuilder(mock_store)

    # Mock database to return no results
    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=None)

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

    mock_store.pool = MagicMock()
    mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

    # Get info for non-existent note
    info = await builder._get_note_info("nonexistent.md")

    assert info is None
