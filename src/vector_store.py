"""
PostgreSQL+pgvector Vector Store for Obsidian Graph MCP Server.

Adapted from oachatbot's PostgreSQL store, simplified for Obsidian notes:
- Stores whole notes (not chunked documents)
- Uses 'path' as identifier (not document_id)
- No site_id or publish_date (Obsidian-specific)
- Adds connection_count materialization for graph queries
"""

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote_plus

import asyncpg
from loguru import logger
from pgvector.asyncpg import register_vector

from .exceptions import DatabaseError

# Expected embedding dimensions for voyage-context-3
EMBEDDING_DIMENSIONS = 1024


@dataclass
class Note:
    """Represents an Obsidian note or note chunk with embedding."""

    path: str
    title: str
    content: str
    embedding: list[float]
    modified_at: datetime | None = None
    file_size_bytes: int | None = None
    chunk_index: int = 0
    total_chunks: int = 1


@dataclass
class SearchResult:
    """Result from vector similarity search."""

    path: str
    title: str
    similarity: float  # 0.0 to 1.0
    content: str


class VectorStoreError(DatabaseError):
    """
    Exception for vector store operations.

    Inherits from DatabaseError for consistency with exception hierarchy.
    This allows catching either VectorStoreError specifically or DatabaseError generally.
    """

    pass


class PostgreSQLVectorStore:
    """
    PostgreSQL+pgvector implementation for Obsidian notes.

    Uses HNSW indexing for fast cosine similarity search.
    Supports connection pooling and async operations.
    """

    def __init__(self, **kwargs):
        self.host = kwargs.get("host") or os.getenv("POSTGRES_HOST", "localhost")
        self.port = kwargs.get("port") or int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = kwargs.get("database") or os.getenv("POSTGRES_DB", "obsidian_graph")
        self.user = kwargs.get("user") or os.getenv("POSTGRES_USER", "obsidian")
        self.password = kwargs.get("password") or os.getenv("POSTGRES_PASSWORD")
        self.table_name = "notes"

        # Validate required parameters
        if not self.password:
            raise VectorStoreError(
                "PostgreSQL password is required (set POSTGRES_PASSWORD env var)"
            )

        # Connection pool configuration
        self.min_connections = kwargs.get("min_connections") or int(
            os.getenv("POSTGRES_MIN_CONNECTIONS", "5")
        )
        self.max_connections = kwargs.get("max_connections") or int(
            os.getenv("POSTGRES_MAX_CONNECTIONS", "20")
        )
        self.connection_timeout = kwargs.get("connection_timeout", 10)

        self.pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool with pgvector support."""
        try:
            # URL-encode credentials to handle special characters in passwords
            dsn = (
                f"postgresql://{quote_plus(self.user)}:{quote_plus(self.password)}"
                f"@{self.host}:{self.port}/{quote_plus(self.database)}"
            )

            self.pool = await asyncpg.create_pool(
                dsn,
                min_size=self.min_connections,
                max_size=self.max_connections,
                timeout=self.connection_timeout,
                setup=self._setup_connection,
            )

            # Verify pgvector extension
            async with self.pool.acquire() as conn:
                has_pgvector = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
                if not has_pgvector:
                    raise VectorStoreError("pgvector extension is not installed")

                # Verify notes table exists
                table_exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'notes')"
                )
                if not table_exists:
                    logger.warning("Notes table does not exist yet (will be created by schema.sql)")

            logger.info(f"PostgreSQL connected: {self.max_connections} max connections")

        except asyncpg.PostgresError as e:
            raise VectorStoreError(f"PostgreSQL connection failed: {e}") from e
        except Exception as e:
            raise VectorStoreError(f"PostgreSQL initialization failed: {e}") from e

    async def _setup_connection(self, conn):
        """Setup each connection with pgvector support."""
        await register_vector(conn)
        logger.debug(f"Registered vector type for connection {id(conn)}")

    async def close(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.debug("PostgreSQL connection pool closed")

    async def search(
        self, query_embedding: list[float], limit: int = 10, threshold: float = 0.5
    ) -> list[SearchResult]:
        """
        Semantic search using vector similarity.

        Args:
            query_embedding: 1024-dimensional query vector
            limit: Max results (1-50)
            threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of SearchResult with similarity scores
        """
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")

        if len(query_embedding) != EMBEDDING_DIMENSIONS:
            raise VectorStoreError(
                f"Query embedding must be {EMBEDDING_DIMENSIONS} dimensions, got {len(query_embedding)}"
            )

        try:
            # Convert similarity threshold to distance threshold
            # Cosine distance: 0 = identical, 2 = opposite
            # Similarity: 1 = identical, 0 = opposite
            distance_threshold = 1.0 - threshold

            query = """
                SELECT
                    path,
                    title,
                    content,
                    1.0 - (embedding <=> $1::vector) AS similarity
                FROM notes
                WHERE embedding IS NOT NULL
                    AND (embedding <=> $1::vector) <= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """

            async with self.pool.acquire() as conn:
                start_time = time.time()
                rows = await asyncio.wait_for(
                    conn.fetch(query, query_embedding, distance_threshold, limit), timeout=5.0
                )
                query_time_ms = (time.time() - start_time) * 1000

                results = [
                    SearchResult(
                        path=row["path"],
                        title=row["title"],
                        content=row["content"],
                        similarity=float(row["similarity"]),
                    )
                    for row in rows
                ]

                logger.debug(f"Search: {len(results)} results in {query_time_ms:.1f}ms")
                return results

        except TimeoutError as e:
            raise VectorStoreError("Search query timed out") from e
        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}") from e

    async def get_similar_notes(
        self, note_path: str, limit: int = 10, threshold: float = 0.5
    ) -> list[SearchResult]:
        """
        Find notes similar to the given note.

        Args:
            note_path: Path to source note
            limit: Max results (1-50)
            threshold: Minimum similarity (0.0-1.0)

        Returns:
            List of similar notes (excluding self)
        """
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")

        try:
            async with self.pool.acquire() as conn:
                # Fetch source note's embedding
                source_embedding = await conn.fetchval(
                    "SELECT embedding FROM notes WHERE path = $1", note_path
                )

                if source_embedding is None:
                    raise VectorStoreError(f"Note not found: {note_path}")

                # Search using source embedding (exclude self)
                results = await self.search(
                    query_embedding=list(source_embedding),
                    limit=limit + 1,  # +1 to account for self exclusion
                    threshold=threshold,
                )

                # Remove self from results
                results = [r for r in results if r.path != note_path]
                return results[:limit]

        except Exception as e:
            raise VectorStoreError(f"Similar notes search failed: {e}") from e

    async def upsert_note(self, note: Note) -> bool:
        """
        Insert or update a note in the database.

        Args:
            note: Note object with embedding

        Returns:
            True if successful
        """
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")

        # Validate embedding dimensions
        if len(note.embedding) != EMBEDDING_DIMENSIONS:
            raise VectorStoreError(
                f"Note embedding must be {EMBEDDING_DIMENSIONS} dimensions, "
                f"got {len(note.embedding)} for note: {note.path}"
            )

        try:
            query = """
                INSERT INTO notes (path, title, content, embedding, modified_at, file_size_bytes, chunk_index, total_chunks, last_indexed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
                ON CONFLICT (path, chunk_index) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    modified_at = EXCLUDED.modified_at,
                    file_size_bytes = EXCLUDED.file_size_bytes,
                    total_chunks = EXCLUDED.total_chunks,
                    last_indexed_at = CURRENT_TIMESTAMP,
                    connection_count = 0
            """

            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    note.path,
                    note.title,
                    note.content,
                    note.embedding,
                    note.modified_at,
                    note.file_size_bytes,
                    note.chunk_index,
                    note.total_chunks,
                )

            logger.debug(f"Upserted note: {note.path}")
            return True

        except Exception as e:
            raise VectorStoreError(f"Note upsert failed: {e}") from e

    async def upsert_batch(self, notes: list[Note]) -> int:
        """
        Insert or update multiple notes in a batch.

        Returns:
            Number of notes processed
        """
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")

        if not notes:
            return 0  # Early return for empty batches

        if len(notes) > 1000:
            raise VectorStoreError(f"Batch size {len(notes)} exceeds maximum of 1000")

        # Validate all embedding dimensions
        for note in notes:
            if len(note.embedding) != EMBEDDING_DIMENSIONS:
                raise VectorStoreError(
                    f"Note embedding must be {EMBEDDING_DIMENSIONS} dimensions, "
                    f"got {len(note.embedding)} for note: {note.path}"
                )

        try:
            query = """
                INSERT INTO notes (path, title, content, embedding, modified_at, file_size_bytes, chunk_index, total_chunks, last_indexed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
                ON CONFLICT (path, chunk_index) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    modified_at = EXCLUDED.modified_at,
                    file_size_bytes = EXCLUDED.file_size_bytes,
                    last_indexed_at = CURRENT_TIMESTAMP,
                    connection_count = 0
            """

            batch_data = [
                (
                    n.path,
                    n.title,
                    n.content,
                    n.embedding,
                    n.modified_at,
                    n.file_size_bytes,
                    n.chunk_index,
                    n.total_chunks,
                )
                for n in notes
            ]

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(query, batch_data)

            logger.info(f"Batch upserted {len(notes)} notes")
            return len(notes)

        except Exception as e:
            raise VectorStoreError(f"Batch upsert failed: {e}") from e

    async def get_note_count(self) -> int:
        """Get total number of indexed notes."""
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")

        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval("SELECT COUNT(*) FROM notes")
        except Exception as e:
            raise VectorStoreError(f"Count query failed: {e}") from e

    async def delete_notes_by_paths(self, paths: list[str]) -> int:
        """
        Delete notes by their paths.

        Args:
            paths: List of note paths to delete

        Returns:
            Number of distinct notes deleted (not chunk count)
        """
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")

        if not paths:
            return 0

        try:
            async with self.pool.acquire() as conn:
                # Use RETURNING to get distinct note count (not chunk count)
                rows = await conn.fetch(
                    "DELETE FROM notes WHERE path = ANY($1) RETURNING path", paths
                )
                # Count distinct paths (a chunked note has multiple rows with same path)
                return len({row["path"] for row in rows})
        except Exception as e:
            raise VectorStoreError(f"Delete failed: {e}") from e

    async def get_all_paths(self) -> list[str]:
        """
        Get all unique note paths in the database.

        Returns:
            List of all note paths
        """
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT DISTINCT path FROM notes")
                return [row["path"] for row in rows]
        except Exception as e:
            raise VectorStoreError(f"Get paths failed: {e}") from e

    def get_pool_stats(self) -> dict:
        """
        Get connection pool statistics for monitoring.

        Returns:
            Dictionary with pool statistics:
            - size: Current pool size
            - free_size: Available connections
            - used_size: Connections in use
            - max_size: Maximum pool size
            - min_size: Minimum pool size
        """
        if not self.pool:
            return {
                "size": 0,
                "free_size": 0,
                "used_size": 0,
                "max_size": self.max_connections,
                "min_size": self.min_connections,
                "initialized": False,
            }

        size = self.pool.get_size()
        free_size = self.pool.get_idle_size()

        return {
            "size": size,
            "free_size": free_size,
            "used_size": size - free_size,
            "max_size": self.pool.get_max_size(),
            "min_size": self.pool.get_min_size(),
            "initialized": True,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
