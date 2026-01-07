"""
Tests for custom exception types.

Verifies that exceptions store context correctly.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exceptions import ObsidianGraphError, EmbeddingError, DatabaseError


class TestExceptionHierarchy:
    """Test exception inheritance."""

    def test_embedding_error_is_obsidian_graph_error(self):
        """EmbeddingError should inherit from ObsidianGraphError."""
        assert issubclass(EmbeddingError, ObsidianGraphError)

    def test_database_error_is_obsidian_graph_error(self):
        """DatabaseError should inherit from ObsidianGraphError."""
        assert issubclass(DatabaseError, ObsidianGraphError)


class TestEmbeddingError:
    """Test EmbeddingError context storage."""

    def test_stores_text_preview(self):
        """EmbeddingError should store first 100 chars of text."""
        long_text = "a" * 200
        err = EmbeddingError("Test error", text_preview=long_text)

        assert err.text_preview == "a" * 100
        assert len(err.text_preview) == 100

    def test_stores_cause(self):
        """EmbeddingError should store original exception."""
        cause = ValueError("Original error")
        err = EmbeddingError("Wrapper error", cause=cause)

        assert err.cause is cause

    def test_default_empty_preview(self):
        """EmbeddingError should have empty preview by default."""
        err = EmbeddingError("Test error")

        assert err.text_preview == ""

    def test_message_accessible(self):
        """EmbeddingError should have accessible message."""
        err = EmbeddingError("Custom message")

        assert str(err) == "Custom message"


class TestDatabaseError:
    """Test DatabaseError basic functionality."""

    def test_can_raise_database_error(self):
        """DatabaseError should be raisable and catchable."""
        with pytest.raises(DatabaseError) as exc_info:
            raise DatabaseError("DB connection failed")

        assert "DB connection failed" in str(exc_info.value)

    def test_can_catch_as_obsidian_graph_error(self):
        """DatabaseError should be catchable as base exception."""
        with pytest.raises(ObsidianGraphError):
            raise DatabaseError("Test error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
