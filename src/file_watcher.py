"""
File watching and incremental re-indexing for Obsidian vault.

Monitors markdown files for changes and triggers debounced re-indexing.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .embedder import VoyageEmbedder
from .exceptions import EmbeddingError
from .vector_store import Note, PostgreSQLVectorStore


class ObsidianFileWatcher(FileSystemEventHandler):
    """
    Watches Obsidian vault for markdown file changes.

    Uses 30-second debounce to batch rapid edits.

    Thread-Safety:
        - Uses per-file async locks to prevent concurrent re-indexing
        - Different files can be re-indexed concurrently
        - Watchdog events (on_modified/on_created) run in background thread
        - Bridges to asyncio event loop via run_coroutine_threadsafe()
        - Safe for concurrent file modifications

    Concurrency Behavior:
        - Multiple rapid edits to SAME file: debounced to single re-index
        - Edits to DIFFERENT files: processed concurrently
        - Lock cleanup prevents unbounded memory growth
    """

    def __init__(
        self,
        vault_path: str,
        store: PostgreSQLVectorStore,
        embedder: VoyageEmbedder,
        loop: asyncio.AbstractEventLoop,
        debounce_seconds: int = 30,
    ):
        """
        Initialize file watcher.

        Args:
            vault_path: Path to Obsidian vault
            store: PostgreSQL vector store
            embedder: Voyage embedder
            loop: Event loop for async operations
            debounce_seconds: Seconds to wait before re-indexing
        """
        super().__init__()
        self.vault_path = Path(vault_path)
        self.store = store
        self.embedder = embedder
        self.loop = loop
        self.debounce_seconds = debounce_seconds
        self.pending_changes: dict[str, float] = {}
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Per-file locks to prevent concurrent re-indexing of same file
        self._reindex_locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Protects _reindex_locks dict itself

        logger.info(f"File watcher initialized (debounce: {debounce_seconds}s)")

    def _handle_reindex_future_error(self, future: asyncio.Future):
        """
        Error callback for threadsafe reindex futures.

        Logs errors from file re-indexing operations without crashing the watcher.
        File will be retried on next modification or startup scan.

        Args:
            future: Completed Future to check for errors
        """
        try:
            future.result()  # Raises if coroutine failed
        except Exception as e:
            logger.error(f"File reindex task failed: {e}", exc_info=True)
            # File will be retried on next modification or startup scan

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        if not event.src_path.endswith(".md"):
            return

        file_path = event.src_path
        logger.debug(f"File modified: {file_path}")

        # Schedule debounced re-index with error handling
        self.pending_changes[file_path] = time.time()
        future = asyncio.run_coroutine_threadsafe(self._debounced_reindex(file_path), self.loop)
        future.add_done_callback(self._handle_reindex_future_error)

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return

        if not event.src_path.endswith(".md"):
            return

        file_path = event.src_path
        logger.debug(f"File created: {file_path}")

        # Schedule debounced re-index with error handling
        self.pending_changes[file_path] = time.time()
        future = asyncio.run_coroutine_threadsafe(self._debounced_reindex(file_path), self.loop)
        future.add_done_callback(self._handle_reindex_future_error)

    async def _get_lock_for_file(self, file_path: str) -> asyncio.Lock:
        """
        Get or create lock for specific file (thread-safe).

        Args:
            file_path: File path to get lock for

        Returns:
            Async lock for this file
        """
        async with self._locks_lock:
            if file_path not in self._reindex_locks:
                self._reindex_locks[file_path] = asyncio.Lock()
            return self._reindex_locks[file_path]

    async def _cleanup_lock(self, file_path: str):
        """
        Remove lock after re-indexing completes (prevent memory leak).

        Args:
            file_path: File path to cleanup lock for
        """
        async with self._locks_lock:
            # Only remove if no pending changes for this file
            if file_path not in self.pending_changes:
                self._reindex_locks.pop(file_path, None)

    async def _debounced_reindex(self, file_path: str):
        """
        Debounced re-indexing with race-condition-free lock management.

        Uses per-file locks to prevent concurrent re-indexes of the same file
        while allowing parallel processing of different files.

        Thread-Safety:
            - Per-file locking prevents concurrent re-indexes of same file
            - Different files can be re-indexed concurrently
            - Lock cleanup prevents unbounded memory growth
            - CRITICAL: pending_changes deletion synchronized with cleanup check
        """
        await asyncio.sleep(self.debounce_seconds)

        # Acquire lock for this specific file
        lock = await self._get_lock_for_file(file_path)

        async with lock:
            # CRITICAL: Check pending_changes under lock protection
            async with self._locks_lock:
                last_change = self.pending_changes.get(file_path)
                if last_change is None:
                    return  # Already processed

                time_since_change = time.time() - last_change
                if time_since_change < self.debounce_seconds:
                    return  # More recent change pending

                # Clear from pending (synchronized deletion)
                del self.pending_changes[file_path]
                # Keep lock in dict (will be cleaned up after re-index)

        # Re-index the file (outside lock to allow other debounce checks)
        try:
            await self._reindex_file(file_path)
        finally:
            # Cleanup lock (now safe - pending_changes deleted under lock protection)
            await self._cleanup_lock(file_path)

    async def _reindex_file(self, file_path: str):
        """
        Re-index a single file.

        Reads content, generates embedding(s), upserts to PostgreSQL.
        Handles large notes with automatic chunking.
        """
        try:
            # Read file
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Skip empty files
            if not content or not content.strip():
                logger.warning(f"Skipping empty file: {file_path}")
                return

            # Get file metadata
            stat = os.stat(file_path)
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            file_size = stat.st_size

            # Get vault-relative path
            rel_path = str(Path(file_path).relative_to(self.vault_path))

            # Extract title from filename
            title = Path(file_path).stem

            # Generate embedding(s) with automatic chunking for large notes
            try:
                embeddings_list, total_chunks = self.embedder.embed_with_chunks(
                    content, chunk_size=2000, input_type="document"
                )
            except EmbeddingError as e:
                logger.warning(f"Failed to generate embedding for {rel_path}: {e}")
                return

            # Create note(s) and upsert
            if total_chunks == 1:
                # Single note (not chunked)
                note = Note(
                    path=rel_path,
                    title=title,
                    content=content,
                    embedding=embeddings_list[0],
                    modified_at=modified_at,
                    file_size_bytes=file_size,
                    chunk_index=0,
                    total_chunks=1,
                )
                await self.store.upsert_note(note)
                logger.info(f"Re-indexed: {rel_path}")
            else:
                # Chunked note - create one Note per chunk
                chunks = self.embedder.chunk_text(content, chunk_size=2000, overlap=0)
                logger.info(f"Re-indexing chunked note {rel_path}: {total_chunks} chunks")

                for chunk_idx, (chunk_text, embedding) in enumerate(
                    zip(chunks, embeddings_list, strict=False)
                ):
                    note = Note(
                        path=rel_path,
                        title=title,
                        content=chunk_text,
                        embedding=embedding,
                        modified_at=modified_at,
                        file_size_bytes=file_size,
                        chunk_index=chunk_idx,
                        total_chunks=total_chunks,
                    )
                    await self.store.upsert_note(note)

                logger.info(f"Re-indexed {total_chunks} chunks: {rel_path}")

        except Exception as e:
            logger.error(f"Failed to re-index {file_path}: {e}")


class VaultWatcher:
    """
    Manages file watching for Obsidian vault.

    Starts watchdog observer and handles lifecycle.
    Includes startup scan to catch files changed while offline.
    """

    def __init__(
        self,
        vault_path: str,
        store: PostgreSQLVectorStore,
        embedder: VoyageEmbedder,
        debounce_seconds: int = 30,
    ):
        """
        Initialize vault watcher.

        Args:
            vault_path: Path to Obsidian vault
            store: PostgreSQL vector store
            embedder: Voyage embedder
            debounce_seconds: Debounce delay
        """
        self.vault_path = vault_path
        self.store = store
        self.embedder = embedder
        self.debounce_seconds = debounce_seconds

        self.observer = None
        self.event_handler = None

    async def startup_scan(self):
        """
        Scan vault on startup to re-index files changed while offline.

        Compares file mtime vs database last_indexed_at.
        """
        if not self.store.pool:
            logger.warning("Store not initialized, skipping startup scan")
            return

        try:
            vault = Path(self.vault_path)
            md_files = list(vault.rglob("*.md"))

            stale_files = []

            async with self.store.pool.acquire() as conn:
                for file_path in md_files:
                    rel_path = str(file_path.relative_to(vault))
                    # Create timezone-aware datetime to match database TIMESTAMP WITH TIME ZONE
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC)

                    # Check last indexed time from database
                    last_indexed = await conn.fetchval(
                        "SELECT MAX(last_indexed_at) FROM notes WHERE path = $1", rel_path
                    )

                    # If file changed since last index, mark for re-indexing
                    if last_indexed is None or file_mtime > last_indexed:
                        stale_files.append(file_path)

            if stale_files:
                logger.info(f"Startup scan: {len(stale_files)} files need re-indexing")
                # Re-index stale files with failure tracking
                failed_files = []
                for file_path in stale_files:
                    try:
                        await self.event_handler._reindex_file(str(file_path))
                    except EmbeddingError as e:
                        logger.error(f"Failed to re-index {file_path}: {e}")
                        failed_files.append(str(file_path))
                    except Exception as e:
                        logger.error(
                            f"Unexpected error re-indexing {file_path}: {e}", exc_info=True
                        )
                        failed_files.append(str(file_path))

                # Report summary
                if failed_files:
                    logger.warning(
                        f"Startup scan: {len(stale_files) - len(failed_files)}/{len(stale_files)} succeeded, "
                        f"{len(failed_files)} failed:\n"
                        + "\n".join(f"  - {f}" for f in failed_files[:10])
                    )
                    if len(failed_files) > 10:
                        logger.warning(f"  ... and {len(failed_files) - 10} more failures")
                else:
                    logger.success(
                        f"Startup scan: All {len(stale_files)} stale files re-indexed successfully"
                    )
            else:
                logger.info("Startup scan: All files up to date")

        except Exception as e:
            logger.error(f"Startup scan failed: {e}", exc_info=True)
            raise  # Re-raise critical failures (database unavailable, etc.)

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start watching the vault for changes."""
        if self.observer:
            logger.warning("Watcher already running")
            return

        self.event_handler = ObsidianFileWatcher(
            self.vault_path, self.store, self.embedder, loop, self.debounce_seconds
        )

        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.vault_path, recursive=True)
        self.observer.start()

        logger.success(f"Watching vault: {self.vault_path}")

    def stop(self):
        """Stop watching the vault."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Vault watcher stopped")
