"""
Error path tests for MCP server tools.

Tests error scenarios that are difficult to reproduce in normal operation:
- Embedding failures (API issues)
- Database pool exhaustion
- Server uninitialized state
- Missing notes
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exceptions import EmbeddingError
from src.vector_store import VectorStoreError


class TestEmbeddingFailures:
    """Tests for embedding generation failures."""

    @pytest.mark.asyncio
    async def test_search_notes_embedding_failure(self, server_context):
        """Verify graceful handling when embedder.embed() raises EmbeddingError."""
        from src.server import call_tool

        # Configure mock to raise EmbeddingError (simulating Voyage API failure)
        server_context.embedder.embed = MagicMock(
            side_effect=EmbeddingError("Voyage API rate limited", text_preview="test query")
        )

        # Call search_notes
        result = await call_tool(
            "search_notes", {"query": "test query", "limit": 10, "threshold": 0.5}
        )

        # Verify error response
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert "Failed to generate query embedding" in result[0]["text"]
        assert (
            "Voyage API rate limited" in result[0]["text"] or "EmbeddingError" in result[0]["text"]
        )


class TestDatabaseFailures:
    """Tests for database connection/pool issues."""

    @pytest.mark.asyncio
    async def test_database_pool_timeout(self, server_context):
        """Verify timeout handling when connection pool is exhausted."""

        from src.server import call_tool

        # Configure mock to simulate pool exhaustion timeout
        async def timeout_side_effect(*args, **kwargs):
            raise TimeoutError("Connection pool exhausted")

        server_context.store.search = AsyncMock(side_effect=timeout_side_effect)
        server_context.embedder.embed = MagicMock(return_value=[0.1] * 1024)

        # Call search_notes
        result = await call_tool("search_notes", {"query": "test", "limit": 10, "threshold": 0.5})

        # Verify error is caught and reported
        assert len(result) == 1
        assert "Error" in result[0]["text"]
        # Should contain timeout/pool information
        assert "pool" in result[0]["text"].lower() or "timeout" in result[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_get_similar_notes_note_not_found(self, server_context):
        """Verify error when source note doesn't exist in database."""
        from src.server import call_tool

        # Configure mock to raise VectorStoreError (note not found)
        async def not_found_side_effect(*args, **kwargs):
            raise VectorStoreError("Note not found: nonexistent.md")

        server_context.store.get_similar_notes = AsyncMock(side_effect=not_found_side_effect)

        # Call get_similar_notes
        result = await call_tool(
            "get_similar_notes", {"note_path": "nonexistent.md", "limit": 10, "threshold": 0.5}
        )

        # Verify error is caught and reported
        assert len(result) == 1
        assert "Error" in result[0]["text"]
        assert "Note not found" in result[0]["text"] or "nonexistent.md" in result[0]["text"]


class TestServerInitialization:
    """Tests for uninitialized server state."""

    @pytest.mark.asyncio
    async def test_all_tools_handle_uninitialized_server(self):
        """Verify all tools return proper error when server not initialized."""
        import src.server
        from src.server import call_tool

        # Save original context
        original_context = src.server._server_context

        try:
            # Temporarily set context to None (simulate initialization failure)
            src.server._server_context = None

            # Test all 5 tools
            tools = [
                ("search_notes", {"query": "test", "limit": 10, "threshold": 0.5}),
                ("get_similar_notes", {"note_path": "test.md", "limit": 10, "threshold": 0.5}),
                ("get_connection_graph", {"note_path": "test.md", "depth": 2, "max_per_level": 5}),
                ("get_hub_notes", {"min_connections": 5, "threshold": 0.5, "limit": 20}),
                ("get_orphaned_notes", {"max_connections": 2, "threshold": 0.5, "limit": 20}),
            ]

            for tool_name, args in tools:
                result = await call_tool(tool_name, args)

                # All should return initialization error
                assert len(result) == 1
                assert result[0]["type"] == "text"
                assert "Server not initialized" in result[0]["text"], (
                    f"{tool_name} should handle uninitialized server"
                )

        finally:
            # Restore context
            src.server._server_context = original_context


class TestGraphBuilderErrors:
    """Tests for graph builder error handling."""

    @pytest.mark.asyncio
    async def test_get_note_info_raises_on_db_error(self, mock_store):
        """_get_note_info should raise DatabaseError on query failure."""
        from src.exceptions import DatabaseError
        from src.graph_builder import GraphBuilder

        # Configure mock to raise exception on pool.acquire()
        class MockAcquire:
            async def __aenter__(self):
                raise Exception("DB connection failed")

            async def __aexit__(self, *args):
                pass

        mock_store.pool = MagicMock()
        mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

        builder = GraphBuilder(mock_store)

        with pytest.raises(DatabaseError) as exc_info:
            await builder._get_note_info("test.md")

        assert "DB connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_note_info_returns_none_for_missing(self, mock_store):
        """_get_note_info should return None for missing notes (not raise)."""
        from src.graph_builder import GraphBuilder

        # Configure mock to return None (note not found)
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        class MockAcquire:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                pass

        mock_store.pool = MagicMock()
        mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

        builder = GraphBuilder(mock_store)

        result = await builder._get_note_info("missing.md")

        # Should return None (not raise)
        assert result is None

    @pytest.mark.asyncio
    async def test_compute_similarity_raises_on_db_error(self, mock_store):
        """_compute_similarity should raise DatabaseError on query failure."""
        from src.exceptions import DatabaseError
        from src.graph_builder import GraphBuilder

        # Configure mock to raise exception
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=Exception("Query failed"))

        class MockAcquire:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                pass

        mock_store.pool = MagicMock()
        mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

        builder = GraphBuilder(mock_store)

        with pytest.raises(DatabaseError) as exc_info:
            await builder._compute_similarity("note1.md", "note2.md")

        assert "Similarity computation failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
