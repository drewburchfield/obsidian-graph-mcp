"""
Input validation tests for MCP tool parameters.

Tests that all parameters are properly validated for:
1. Required parameters present
2. Correct types
3. Valid ranges
4. Sensible defaults
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation import (
    ValidationError,
    validate_connection_graph_args,
    validate_hub_notes_args,
    validate_orphaned_notes_args,
    validate_search_notes_args,
    validate_similar_notes_args,
)


class TestSearchNotesValidation:
    """Tests for search_notes parameter validation."""

    def test_rejects_missing_query(self):
        """Required parameter 'query' must be present."""
        with pytest.raises(ValidationError):
            validate_search_notes_args({})

    def test_rejects_empty_query(self):
        """Query cannot be empty string."""
        with pytest.raises(ValidationError, match="query.*empty"):
            validate_search_notes_args({"query": ""})

        with pytest.raises(ValidationError, match="query.*empty"):
            validate_search_notes_args({"query": "   "})

    def test_rejects_non_string_query(self):
        """Query must be string type."""
        with pytest.raises(ValidationError, match="query.*string"):
            validate_search_notes_args({"query": 123})

        with pytest.raises(ValidationError, match="query.*string"):
            validate_search_notes_args({"query": ["list"]})

    def test_rejects_limit_out_of_range(self):
        """Limit must be in [1, 50] range."""
        with pytest.raises(ValidationError, match="limit.*1.*50"):
            validate_search_notes_args({"query": "test", "limit": 0})

        with pytest.raises(ValidationError, match="limit.*1.*50"):
            validate_search_notes_args({"query": "test", "limit": 100})

        with pytest.raises(ValidationError, match="limit.*1.*50"):
            validate_search_notes_args({"query": "test", "limit": -5})

    def test_rejects_threshold_out_of_range(self):
        """Threshold must be in [0.0, 1.0] range."""
        with pytest.raises(ValidationError, match="threshold.*0.0.*1.0"):
            validate_search_notes_args({"query": "test", "threshold": -0.1})

        with pytest.raises(ValidationError, match="threshold.*0.0.*1.0"):
            validate_search_notes_args({"query": "test", "threshold": 1.5})

        with pytest.raises(ValidationError, match="threshold.*0.0.*1.0"):
            validate_search_notes_args({"query": "test", "threshold": 2.0})

    def test_applies_defaults(self):
        """Missing optional parameters get defaults."""
        validated = validate_search_notes_args({"query": "test"})

        assert validated["query"] == "test"
        assert validated["limit"] == 10
        assert validated["threshold"] == 0.5

    def test_accepts_valid_input(self):
        """Valid input passes validation."""
        validated = validate_search_notes_args(
            {"query": "machine learning", "limit": 20, "threshold": 0.7}
        )

        assert validated["query"] == "machine learning"
        assert validated["limit"] == 20
        assert validated["threshold"] == 0.7

    def test_handles_string_numbers_for_limit(self):
        """String numbers should be converted to int."""
        # This tests graceful handling - returns default on invalid type
        validated = validate_search_notes_args(
            {"query": "test", "limit": "15"}  # String instead of int
        )
        # Should convert successfully
        assert validated["limit"] == 15


class TestSimilarNotesValidation:
    """Tests for get_similar_notes parameter validation."""

    def test_rejects_missing_note_path(self):
        """Required parameter 'note_path' must be present."""
        with pytest.raises(ValidationError):
            validate_similar_notes_args({})

    def test_rejects_empty_note_path(self):
        """note_path cannot be empty."""
        with pytest.raises(ValidationError, match="note_path.*empty"):
            validate_similar_notes_args({"note_path": ""})

    def test_accepts_valid_note_path(self):
        """Valid note_path passes validation."""
        validated = validate_similar_notes_args({"note_path": "folder/note.md"})
        assert validated["note_path"] == "folder/note.md"

    def test_applies_defaults(self):
        """Missing optional parameters get defaults."""
        validated = validate_similar_notes_args({"note_path": "test.md"})

        assert validated["note_path"] == "test.md"
        assert validated["limit"] == 10
        assert validated["threshold"] == 0.5


class TestConnectionGraphValidation:
    """Tests for get_connection_graph parameter validation."""

    def test_rejects_missing_note_path(self):
        """Required parameter 'note_path' must be present."""
        with pytest.raises(ValidationError):
            validate_connection_graph_args({})

    def test_rejects_depth_out_of_range(self):
        """Depth must be in [1, 5] range."""
        with pytest.raises(ValidationError, match="depth.*1.*5"):
            validate_connection_graph_args({"note_path": "test.md", "depth": 0})

        with pytest.raises(ValidationError, match="depth.*1.*5"):
            validate_connection_graph_args({"note_path": "test.md", "depth": 10})

        with pytest.raises(ValidationError, match="depth.*1.*5"):
            validate_connection_graph_args({"note_path": "test.md", "depth": -1})

    def test_rejects_max_per_level_out_of_range(self):
        """max_per_level must be in [1, 10] range."""
        with pytest.raises(ValidationError, match="max_per_level.*1.*10"):
            validate_connection_graph_args({"note_path": "test.md", "max_per_level": 0})

        with pytest.raises(ValidationError, match="max_per_level.*1.*10"):
            validate_connection_graph_args({"note_path": "test.md", "max_per_level": 20})

    def test_applies_defaults(self):
        """Missing optional parameters get defaults."""
        validated = validate_connection_graph_args({"note_path": "test.md"})

        assert validated["note_path"] == "test.md"
        assert validated["depth"] == 3
        assert validated["max_per_level"] == 5
        assert validated["threshold"] == 0.5

    def test_accepts_valid_input(self):
        """Valid input passes validation."""
        validated = validate_connection_graph_args(
            {"note_path": "notes/important.md", "depth": 2, "max_per_level": 8, "threshold": 0.6}
        )

        assert validated["note_path"] == "notes/important.md"
        assert validated["depth"] == 2
        assert validated["max_per_level"] == 8
        assert validated["threshold"] == 0.6


class TestHubNotesValidation:
    """Tests for get_hub_notes parameter validation."""

    def test_accepts_empty_args(self):
        """All parameters are optional."""
        validated = validate_hub_notes_args({})

        assert validated["min_connections"] == 10
        assert validated["threshold"] == 0.5
        assert validated["limit"] == 20

    def test_rejects_min_connections_out_of_range(self):
        """min_connections must be >= 1."""
        with pytest.raises(ValidationError, match="min_connections.*1.*1000"):
            validate_hub_notes_args({"min_connections": 0})

        with pytest.raises(ValidationError, match="min_connections.*1.*1000"):
            validate_hub_notes_args({"min_connections": 1500})

    def test_accepts_valid_input(self):
        """Valid input passes validation."""
        validated = validate_hub_notes_args({"min_connections": 15, "threshold": 0.7, "limit": 30})

        assert validated["min_connections"] == 15
        assert validated["threshold"] == 0.7
        assert validated["limit"] == 30


class TestOrphanedNotesValidation:
    """Tests for get_orphaned_notes parameter validation."""

    def test_accepts_empty_args(self):
        """All parameters are optional."""
        validated = validate_orphaned_notes_args({})

        assert validated["max_connections"] == 2
        assert validated["threshold"] == 0.5
        assert validated["limit"] == 20

    def test_allows_zero_max_connections(self):
        """max_connections can be 0 (truly orphaned)."""
        validated = validate_orphaned_notes_args({"max_connections": 0})
        assert validated["max_connections"] == 0

    def test_rejects_max_connections_out_of_range(self):
        """max_connections must be in [0, 100]."""
        with pytest.raises(ValidationError, match="max_connections.*0.*100"):
            validate_orphaned_notes_args({"max_connections": -1})

        with pytest.raises(ValidationError, match="max_connections.*0.*100"):
            validate_orphaned_notes_args({"max_connections": 150})

    def test_accepts_valid_input(self):
        """Valid input passes validation."""
        validated = validate_orphaned_notes_args(
            {"max_connections": 3, "threshold": 0.6, "limit": 15}
        )

        assert validated["max_connections"] == 3
        assert validated["threshold"] == 0.6
        assert validated["limit"] == 15


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_query_max_length(self):
        """Very long queries should be rejected."""
        long_query = "a" * 20000

        with pytest.raises(ValidationError, match="exceeds maximum length"):
            validate_search_notes_args({"query": long_query})

    def test_threshold_boundaries(self):
        """Test exact boundary values for threshold."""
        # Exactly 0.0 should work
        validated = validate_search_notes_args({"query": "test", "threshold": 0.0})
        assert validated["threshold"] == 0.0

        # Exactly 1.0 should work
        validated = validate_search_notes_args({"query": "test", "threshold": 1.0})
        assert validated["threshold"] == 1.0

    def test_limit_boundaries(self):
        """Test exact boundary values for limit."""
        # Exactly 1 should work
        validated = validate_search_notes_args({"query": "test", "limit": 1})
        assert validated["limit"] == 1

        # Exactly 50 should work
        validated = validate_search_notes_args({"query": "test", "limit": 50})
        assert validated["limit"] == 50
