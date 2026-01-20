"""
Shared test fixtures for Obsidian Graph MCP Server.

Provides reusable fixtures for:
- Event loops (module-scoped for async tests)
- Mock stores and embedders
- Temporary vaults with test data
- Server contexts for integration testing
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(scope="module")
def event_loop():
    """
    Module-scoped event loop for async tests.

    This ensures all async tests in a module share the same event loop,
    which is more efficient and avoids loop recreation overhead.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_store():
    """
    Mock PostgreSQL vector store for testing.

    Provides:
    - Mock connection pool
    - Mock async methods (search, upsert_note, get_similar_notes)
    - Configurable return values
    """
    store = MagicMock()
    store.pool = MagicMock()

    # Mock async methods
    store.initialize = AsyncMock()
    store.close = AsyncMock()
    store.search = AsyncMock(return_value=[])
    store.get_similar_notes = AsyncMock(return_value=[])
    store.upsert_note = AsyncMock(return_value=True)
    store.upsert_batch = AsyncMock(return_value=0)
    store.get_note_count = AsyncMock(return_value=0)

    # Methods for exclusion cleanup
    store.get_all_paths = AsyncMock(return_value=[])
    store.delete_notes_by_paths = AsyncMock(return_value=0)

    return store


@pytest.fixture
def mock_embedder():
    """
    Mock Voyage embedder for testing.

    Returns dummy 1024-dimensional vectors without API calls.

    Note: To test embedding failures, configure mock with side_effect:
        mock_embedder.embed = MagicMock(side_effect=EmbeddingError("API failed"))
    """
    embedder = MagicMock()

    # Return dummy 1024-dim vector (successful embedding)
    dummy_embedding = [0.1] * 1024

    embedder.embed = MagicMock(return_value=dummy_embedding)
    embedder.embed_batch = MagicMock(return_value=[dummy_embedding])
    embedder.embed_with_chunks = MagicMock(return_value=([dummy_embedding], 1))
    embedder.chunk_text = MagicMock(return_value=["chunk1"])
    embedder.get_cache_stats = MagicMock(
        return_value={
            "total_cached": 0,
            "cache_size_mb": 0.0,
            "cache_dir": "/tmp/cache",
            "model": "voyage-context-3",
        }
    )

    return embedder


@pytest.fixture
async def tmp_vault(tmp_path):
    """
    Create temporary Obsidian vault with test notes.

    Structure:
        vault/
        ├── note1.md (Machine learning basics)
        ├── note2.md (Python programming)
        ├── folder/
        │   └── note3.md (Nested note)
        └── empty.md (Empty file for edge case testing)
    """
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create test notes
    (vault / "note1.md").write_text(
        "# Machine Learning\n\n"
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn and improve from experience.\n\n"
        "Key concepts: neural networks, deep learning, supervised learning."
    )

    (vault / "note2.md").write_text(
        "# Python Programming\n\n"
        "Python is a high-level programming language known for its simplicity.\n\n"
        "Popular for: web development, data science, machine learning, automation."
    )

    # Create nested folder structure
    folder = vault / "folder"
    folder.mkdir()

    (folder / "note3.md").write_text(
        "# Nested Note\n\nThis note is in a subfolder to test path handling."
    )

    # Create empty file for edge case testing
    (vault / "empty.md").write_text("")

    yield vault


@pytest.fixture
async def server_context(mock_store, mock_embedder):
    """
    Mock ServerContext for testing tool handlers.

    Provides complete server context with mocked dependencies.

    IMPORTANT: This fixture injects the context into src.server._server_context
    so that call_tool() uses the mocked dependencies. The original context
    is restored after the test completes.
    """
    import src.server
    from src.graph_builder import GraphBuilder
    from src.hub_analyzer import HubAnalyzer
    from src.server import ServerContext

    # Save original context
    original_context = src.server._server_context

    graph_builder = GraphBuilder(mock_store)
    hub_analyzer = HubAnalyzer(mock_store)

    context = ServerContext(
        store=mock_store,
        embedder=mock_embedder,
        graph_builder=graph_builder,
        hub_analyzer=hub_analyzer,
        vault_watcher=None,
    )

    # Inject into global for call_tool()
    src.server._server_context = context

    yield context

    # Restore original context
    src.server._server_context = original_context


@pytest.fixture
def sample_search_results():
    """Sample search results for testing formatting."""
    from src.vector_store import SearchResult

    return [
        SearchResult(
            path="notes/ml.md",
            title="Machine Learning",
            content="Machine learning is a subset of AI...",
            similarity=0.95,
        ),
        SearchResult(
            path="notes/python.md",
            title="Python",
            content="Python is a programming language...",
            similarity=0.82,
        ),
    ]


@pytest.fixture
def sample_notes():
    """Sample Note objects for testing database operations."""
    from src.vector_store import Note

    return [
        Note(
            path="test1.md",
            title="Test 1",
            content="This is test note 1",
            embedding=[0.1] * 1024,
            modified_at=datetime.now(),
            file_size_bytes=100,
            chunk_index=0,
            total_chunks=1,
        ),
        Note(
            path="test2.md",
            title="Test 2",
            content="This is test note 2",
            embedding=[0.2] * 1024,
            modified_at=datetime.now(),
            file_size_bytes=150,
            chunk_index=0,
            total_chunks=1,
        ),
    ]
