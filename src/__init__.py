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

from .vector_store import PostgreSQLVectorStore, Note, SearchResult, VectorStoreError
from .embedder import VoyageEmbedder
from .graph_builder import GraphBuilder
from .hub_analyzer import HubAnalyzer
from .file_watcher import VaultWatcher
from .security_utils import validate_note_path_parameter, SecurityError
from .validation import ValidationError
from .exceptions import EmbeddingError, DatabaseError, ObsidianGraphError

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
