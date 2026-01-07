"""
Custom exceptions for Obsidian Graph MCP Server.

Provides domain-specific exceptions for better error handling and diagnostics.
"""


class ObsidianGraphError(Exception):
    """Base exception for all Obsidian Graph errors."""
    pass


class EmbeddingError(ObsidianGraphError):
    """
    Raised when embedding generation fails.

    Stores additional context to aid debugging:
    - text_preview: First 100 characters of text that failed to embed
    - cause: Original exception that triggered the failure
    """
    def __init__(self, message: str, text_preview: str = "", cause: Exception = None):
        self.text_preview = text_preview[:100]  # First 100 chars for debugging
        self.cause = cause
        super().__init__(message)


class DatabaseError(ObsidianGraphError):
    """
    Raised when database operations fail.

    Used for PostgreSQL connection errors, query failures, and pgvector issues.
    """
    pass
