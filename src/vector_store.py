"""
PostgreSQL+pgvector Vector Store for Obsidian Graph.

Stores notes (whole or chunked) with vector embeddings for semantic search,
graph analysis, and hub/orphan detection. Uses connection_count materialization
for fast graph queries.
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

# Expected embedding dimensions. Defaults to 1024 (voyage-context-4 / Gemini /
# Qwen3-0.6B); override via EMBEDDING_DIMENSIONS for other models (e.g. Qwen3-8B
# at 4096). Must match the pgvector column width.
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))


def embedding_signature(embedder) -> str:
    """
    Provider:model:dimensions fingerprint of the embedding configuration.

    Written to index_metadata by full index runs and compared at server
    startup: vectors from different providers, models, or dimensions are
    not comparable, and on a multi-provider deployment any of the three
    can drift independently.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "voyage").lower()
    model = getattr(embedder, "model", "unknown")
    return f"{provider}:{model}:{EMBEDDING_DIMENSIONS}"


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

            # Assign self.pool only after verification succeeds; close the
            # local pool on any post-creation failure so it never leaks
            pool = await asyncpg.create_pool(
                dsn,
                min_size=self.min_connections,
                max_size=self.max_connections,
                timeout=self.connection_timeout,
                setup=self._setup_connection,
            )

            try:
                # Verify pgvector extension
                async with pool.acquire() as conn:
                    has_pgvector = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    )
                    if not has_pgvector:
                        raise VectorStoreError("pgvector extension is not installed")

                    # Verify notes table exists
                    table_exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                        "WHERE table_name = 'notes')"
                    )
                    if not table_exists:
                        logger.warning(
                            "Notes table does not exist yet (will be created by schema.sql)"
                        )
                    else:
                        vector_dimensions = await conn.fetchval("""
                            SELECT atttypmod
                            FROM pg_attribute
                            WHERE attrelid = 'notes'::regclass AND attname = 'embedding'
                            """)
                        if vector_dimensions != EMBEDDING_DIMENSIONS:
                            note_count = await conn.fetchval("SELECT COUNT(*) FROM notes")
                            if note_count:
                                raise VectorStoreError(
                                    f"Database contains vector({vector_dimensions}) data but "
                                    f"EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}"
                                )
                            await conn.execute("DROP INDEX IF EXISTS idx_notes_embedding_cosine")
                            await conn.execute(
                                f"ALTER TABLE notes ALTER COLUMN embedding "
                                f"TYPE vector({EMBEDDING_DIMENSIONS})"  # nosec B608
                            )
                            if EMBEDDING_DIMENSIONS <= 2000:
                                await conn.execute("""
                                    CREATE INDEX idx_notes_embedding_cosine
                                    ON notes USING hnsw (embedding vector_cosine_ops)
                                    WITH (m = 16, ef_construction = 64)
                                    """)
                        # Migration: remove trigger that overwrites file mtime
                        # (existing databases)
                        await conn.execute(
                            "DROP TRIGGER IF EXISTS trigger_update_notes_modified_at ON notes"
                        )
                        await conn.execute("DROP FUNCTION IF EXISTS update_modified_at()")

                    # Migration: corpus-level metadata (e.g. which embedding model built
                    # the index); created here so existing databases get it without
                    # re-running schema.sql
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS index_metadata (
                            key TEXT PRIMARY KEY,
                            value TEXT NOT NULL,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        )
                        """)
            except BaseException:
                await pool.close()
                raise

            self.pool = pool
            logger.info(f"PostgreSQL connected: {self.max_connections} max connections")

        except asyncpg.PostgresError as e:
            raise VectorStoreError(f"PostgreSQL connection failed: {e}") from e
        except Exception as e:
            raise VectorStoreError(f"PostgreSQL initialization failed: {e}") from e

    async def _with_timeout(self, coro, timeout=10.0):
        """Execute a coroutine with a timeout."""
        return await asyncio.wait_for(coro, timeout=timeout)

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

    async def get_metadata(self, key: str) -> str | None:
        """Read a corpus-level metadata value (e.g. 'embedding_model')."""
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")

        async with self.pool.acquire() as conn:
            return await self._with_timeout(
                conn.fetchval("SELECT value FROM index_metadata WHERE key = $1", key)
            )

    async def set_metadata(self, key: str, value: str) -> None:
        """Write a corpus-level metadata value."""
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")

        async with self.pool.acquire() as conn:
            await self._with_timeout(
                conn.execute(
                    """
                    INSERT INTO index_metadata (key, value, updated_at)
                    VALUES ($1, $2, CURRENT_TIMESTAMP)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    key,
                    value,
                )
            )

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
                f"Query embedding must be {EMBEDDING_DIMENSIONS} dimensions, "
                f"got {len(query_embedding)}"
            )

        try:
            # Convert similarity threshold to distance threshold
            # Cosine distance: 0 = identical, 2 = opposite
            # Similarity: 1 = identical, 0 = opposite
            distance_threshold = 1.0 - threshold

            # One result per note: chunked notes are collapsed to their
            # best-matching chunk so a single long note cannot fill the results
            query = """
                SELECT path, title, content, similarity FROM (
                    SELECT DISTINCT ON (path)
                        path,
                        title,
                        content,
                        1.0 - (embedding <=> $1::vector) AS similarity
                    FROM notes
                    WHERE embedding IS NOT NULL
                        AND (embedding <=> $1::vector) <= $2
                    ORDER BY path, embedding <=> $1::vector
                ) best_chunk_per_note
                ORDER BY similarity DESC, path
                LIMIT $3
            """

            async with self.pool.acquire() as conn:
                start_time = time.time()
                rows = await asyncio.wait_for(
                    conn.fetch(query, query_embedding, distance_threshold, limit),
                    timeout=5.0,
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

    async def lexical_search(self, query_text: str, limit: int = 50) -> list[SearchResult]:
        """BM25-style full-text (tsvector) candidate retrieval for the hybrid leg."""
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")
        q = """
            SELECT path, title, content,
                   ts_rank_cd(fts, websearch_to_tsquery('english', $1)) AS rank
            FROM notes
            WHERE fts @@ websearch_to_tsquery('english', $1)
            ORDER BY rank DESC
            LIMIT $2
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await asyncio.wait_for(conn.fetch(q, query_text, limit), timeout=5.0)
            return [
                SearchResult(
                    path=r["path"],
                    title=r["title"],
                    content=r["content"],
                    similarity=float(r["rank"]),
                )
                for r in rows
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Lexical search failed (continuing dense-only): {e}")
            return []

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
                exists = await self._with_timeout(
                    conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM notes "
                        "WHERE path = $1 AND embedding IS NOT NULL)",
                        note_path,
                    ),
                    timeout=5.0,
                )
                if not exists:
                    raise VectorStoreError(f"Note not found (or has no embedding): {note_path}")

                # Any-chunk-to-any-chunk comparison, collapsed to the best match
                # per target note, with self excluded in SQL. One pooled
                # connection for both statements, so a one-connection pool
                # cannot deadlock.
                query = """
                    SELECT path, title, content, similarity FROM (
                        SELECT DISTINCT ON (n2.path)
                            n2.path,
                            n2.title,
                            n2.content,
                            1.0 - (n1.embedding <=> n2.embedding) AS similarity
                        FROM notes n1
                        JOIN notes n2 ON n2.path != n1.path
                        WHERE n1.path = $1
                            AND n1.embedding IS NOT NULL
                            AND n2.embedding IS NOT NULL
                            AND (n1.embedding <=> n2.embedding) <= $2
                        ORDER BY n2.path, n1.embedding <=> n2.embedding
                    ) best_chunk_per_note
                    ORDER BY similarity DESC, path
                    LIMIT $3
                """
                rows = await self._with_timeout(
                    conn.fetch(query, note_path, 1.0 - threshold, limit)
                )
                return [
                    SearchResult(
                        path=row["path"],
                        title=row["title"],
                        content=row["content"],
                        similarity=float(row["similarity"]),
                    )
                    for row in rows
                ]

        except VectorStoreError:
            raise
        except TimeoutError as e:
            raise VectorStoreError("Similar notes search timed out") from e
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
                INSERT INTO notes (path, title, content, embedding, modified_at,
                    file_size_bytes, chunk_index, total_chunks, last_indexed_at)
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
                await self._with_timeout(
                    conn.execute(
                        query,
                        note.path,
                        note.title,
                        note.content,
                        note.embedding,
                        note.modified_at,
                        note.file_size_bytes,
                        note.chunk_index,
                        note.total_chunks,
                    ),
                    timeout=10.0,
                )

            logger.debug(f"Upserted note: {note.path}")
            return True

        except TimeoutError as e:
            raise VectorStoreError("Note upsert timed out") from e
        except Exception as e:
            raise VectorStoreError(f"Note upsert failed: {e}") from e

    async def upsert_batch(self, notes: list[Note]) -> int:
        """
        Insert or update multiple notes in a batch.

        The batch must contain the COMPLETE chunk set for every path it
        includes: in the same transaction, chunk rows at or beyond each
        path's total_chunks are deleted, so a note that shrank on re-index
        cannot keep stale chunks searchable (atomic replacement).

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

        # Enforce the complete-chunk-set contract: a partial or inconsistent
        # batch would silently delete live chunks via the stale-chunk cleanup
        by_path: dict[str, list[Note]] = {}
        for note in notes:
            by_path.setdefault(note.path, []).append(note)
        for note_path, path_notes in by_path.items():
            totals = {n.total_chunks for n in path_notes}
            indexes = {n.chunk_index for n in path_notes}
            if len(totals) != 1 or indexes != set(range(next(iter(totals)))):
                raise VectorStoreError(
                    f"upsert_batch requires the complete chunk set per path: "
                    f"'{note_path}' has chunk_index {sorted(indexes)} with "
                    f"total_chunks {sorted(totals)}"
                )

        try:
            query = """
                INSERT INTO notes (path, title, content, embedding, modified_at,
                    file_size_bytes, chunk_index, total_chunks, last_indexed_at)
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

            chunk_totals = {n.path: n.total_chunks for n in notes}

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await self._with_timeout(
                        conn.executemany(query, batch_data),
                        timeout=30.0,
                    )
                    # Same transaction: drop chunk rows beyond each path's new
                    # chunk count so replacement is atomic (no crash window
                    # between upsert and stale-chunk cleanup)
                    for note_path, chunk_total in chunk_totals.items():
                        await self._with_timeout(
                            conn.execute(
                                "DELETE FROM notes WHERE path = $1 AND chunk_index >= $2",
                                note_path,
                                chunk_total,
                            )
                        )

            logger.info(f"Batch upserted {len(notes)} notes")
            return len(notes)

        except TimeoutError as e:
            raise VectorStoreError("Batch upsert timed out") from e
        except Exception as e:
            raise VectorStoreError(f"Batch upsert failed: {e}") from e

    async def replace_file_notes(self, path: str, notes: list[Note]) -> int:
        """Atomically replace every stored chunk for one source file."""
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")
        for note in notes:
            if note.path != path:
                raise VectorStoreError("Every replacement note must match the source path")
            if len(note.embedding) != EMBEDDING_DIMENSIONS:
                raise VectorStoreError(
                    f"Note embedding must be {EMBEDDING_DIMENSIONS} dimensions, "
                    f"got {len(note.embedding)} for note: {note.path}"
                )

        query = """
            INSERT INTO notes (path, title, content, embedding, modified_at,
                file_size_bytes, chunk_index, total_chunks, last_indexed_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
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
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("DELETE FROM notes WHERE path = $1", path)
                    if batch_data:
                        await conn.executemany(query, batch_data)
            return len(notes)
        except Exception as e:
            raise VectorStoreError(f"Atomic file replacement failed: {e}") from e

    async def get_file_metadata(self) -> dict[str, tuple[datetime | None, int | None]]:
        """Return one modification time and size tuple per indexed source path."""
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT path, MAX(modified_at) AS modified_at,
                       MAX(file_size_bytes) AS file_size_bytes
                FROM notes
                GROUP BY path
                """)
        return {row["path"]: (row["modified_at"], row["file_size_bytes"]) for row in rows}

    async def get_note_count(self) -> int:
        """Get total number of indexed notes."""
        if not self.pool:
            raise VectorStoreError("PostgreSQL store not initialized")

        try:
            async with self.pool.acquire() as conn:
                return await self._with_timeout(
                    conn.fetchval("SELECT COUNT(*) FROM notes"),
                    timeout=5.0,
                )
        except TimeoutError as e:
            raise VectorStoreError("Count query timed out") from e
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
                rows = await self._with_timeout(
                    conn.fetch(
                        "DELETE FROM notes WHERE path = ANY($1) RETURNING path",
                        paths,
                    ),
                    timeout=10.0,
                )
                # Count distinct paths (a chunked note has multiple rows with same path)
                return len({row["path"] for row in rows})
        except TimeoutError as e:
            raise VectorStoreError("Delete operation timed out") from e
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
                rows = await self._with_timeout(
                    conn.fetch("SELECT DISTINCT path FROM notes"),
                    timeout=10.0,
                )
                return [row["path"] for row in rows]
        except TimeoutError as e:
            raise VectorStoreError("Get paths query timed out") from e
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
