"""
Obsidian Graph MCP Server - Semantic knowledge graph navigation.

Provides semantic search and graph analysis for Obsidian vaults using:
- Voyage Context-3 embeddings (1024-dimensional)
- PostgreSQL + pgvector for vector storage
- Multi-hop graph traversal with BFS
- Hub and orphan note detection
- Incremental file watching with debouncing
"""

__version__ = "1.0.0"

from .embedder import VoyageEmbedder
from .exceptions import DatabaseError, EmbeddingError, ObsidianGraphError
from .file_watcher import VaultWatcher
from .graph_builder import GraphBuilder
from .hub_analyzer import HubAnalyzer
from .security_utils import SecurityError, validate_note_path_parameter
from .validation import ValidationError
from .vector_store import Note, PostgreSQLVectorStore, SearchResult, VectorStoreError

__all__ = [
    # Vector Store
    "PostgreSQLVectorStore",
    "Note",
    "SearchResult",
    "VectorStoreError",
    # Embeddings
    "VoyageEmbedder",
    "EmbeddingError",
    # Graph Analysis
    "GraphBuilder",
    "HubAnalyzer",
    # File Watching
    "VaultWatcher",
    # Security & Validation
    "validate_note_path_parameter",
    "SecurityError",
    "ValidationError",
    # Exceptions
    "DatabaseError",
    "ObsidianGraphError",
]
