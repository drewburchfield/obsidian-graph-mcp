"""
Unit tests for file watching and debounce logic.

Tests:
1. File modification detection
2. Debouncing of rapid edits
3. Empty file handling
4. Startup scan for offline changes
5. Lock cleanup for memory leak prevention
"""
import asyncio
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.file_watcher import ObsidianFileWatcher, VaultWatcher


@pytest.mark.asyncio
async def test_file_watcher_detects_modifications(tmp_vault, mock_store, mock_embedder):
    """Test that file modification events are detected."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1
    )

    # Create a test file
    test_file = tmp_vault / "test.md"
    test_file.write_text("# Test")

    # Simulate file modification event
    class MockEvent:
        def __init__(self, src_path, is_directory=False):
            self.src_path = src_path
            self.is_directory = is_directory

    event = MockEvent(str(test_file))
    watcher.on_modified(event)

    # Check that file was added to pending_changes
    assert str(test_file) in watcher.pending_changes


@pytest.mark.asyncio
async def test_file_watcher_ignores_non_markdown(tmp_vault, mock_store, mock_embedder):
    """Test that non-markdown files are ignored."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1
    )

    # Create non-markdown files
    txt_file = tmp_vault / "notes.txt"
    pdf_file = tmp_vault / "document.pdf"

    class MockEvent:
        def __init__(self, src_path):
            self.src_path = src_path
            self.is_directory = False

    # Simulate events for non-markdown files
    watcher.on_modified(MockEvent(str(txt_file)))
    watcher.on_modified(MockEvent(str(pdf_file)))

    # Should not be in pending_changes
    assert str(txt_file) not in watcher.pending_changes
    assert str(pdf_file) not in watcher.pending_changes


@pytest.mark.asyncio
async def test_file_watcher_debounces_rapid_edits(tmp_vault, mock_store, mock_embedder):
    """Test that rapid edits are debounced to a single re-index."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=0.3  # Short debounce for testing
    )

    test_file = tmp_vault / "test.md"
    test_file.write_text("# Test")

    # Track re-index calls
    reindex_count = 0

    async def track_reindex(file_path):
        nonlocal reindex_count
        reindex_count += 1

    watcher._reindex_file = track_reindex

    # Simulate 5 rapid changes
    file_path = str(test_file)
    for i in range(5):
        watcher.pending_changes[file_path] = time.time()
        asyncio.create_task(watcher._debounced_reindex(file_path))
        await asyncio.sleep(0.05)  # 50ms between edits

    # Wait for debounce period
    await asyncio.sleep(0.5)

    # Should have exactly 1 re-index
    assert reindex_count == 1, f"Expected 1 re-index, got {reindex_count}"


@pytest.mark.asyncio
async def test_file_watcher_handles_empty_files(tmp_vault, mock_store, mock_embedder):
    """Test that empty files are handled gracefully."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=0.1
    )

    # Use the existing empty.md file from tmp_vault
    empty_file = tmp_vault / "empty.md"

    # Mock embedder to raise EmbeddingError for empty content
    from src.exceptions import EmbeddingError
    mock_embedder.embed = MagicMock(side_effect=EmbeddingError("Empty content", text_preview=""))

    # Trigger re-index
    await watcher._reindex_file(str(empty_file))

    # Should not crash, and should not call upsert_note (returns early on EmbeddingError)
    mock_store.upsert_note.assert_not_called()


@pytest.mark.asyncio
async def test_lock_cleanup_prevents_memory_leak(tmp_vault, mock_store, mock_embedder):
    """Test that lock cleanup prevents unbounded memory growth."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=0.1
    )

    # Mock re-index to do nothing
    watcher._reindex_file = AsyncMock()

    # Process many files
    for i in range(100):
        file_path = str(tmp_vault / f"test_{i}.md")
        watcher.pending_changes[file_path] = time.time()
        await watcher._debounced_reindex(file_path)

    # Wait for cleanup
    await asyncio.sleep(0.2)

    # Lock dict should be small (most locks cleaned up)
    assert len(watcher._reindex_locks) < 10, \
        f"Lock dict has {len(watcher._reindex_locks)} entries (memory leak!)"


@pytest.mark.asyncio
async def test_vault_watcher_startup_scan_detects_stale_files(tmp_vault, mock_store, mock_embedder):
    """Test that startup scan detects files changed while offline."""
    # Create vault watcher
    vault_watcher = VaultWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        debounce_seconds=1
    )

    # Mock database to return old last_indexed_at times
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=datetime(2020, 1, 1, tzinfo=UTC))  # Very old timestamp (timezone-aware)

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(return_value=MockAcquire())
    mock_store.pool = mock_pool

    # Create event handler
    loop = asyncio.get_running_loop()
    vault_watcher.start(loop)

    # Mock re-index to track calls
    vault_watcher.event_handler._reindex_file = AsyncMock()

    # Run startup scan
    await vault_watcher.startup_scan()

    # Should have detected stale files (note1.md, note2.md, folder/note3.md)
    # Empty.md might be skipped
    assert vault_watcher.event_handler._reindex_file.call_count >= 3, \
        "Expected at least 3 stale files detected"


@pytest.mark.asyncio
async def test_get_lock_for_file_creates_new_locks(tmp_vault, mock_store, mock_embedder):
    """Test that _get_lock_for_file creates locks on demand."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1
    )

    # Initially no locks
    assert len(watcher._reindex_locks) == 0

    # Get lock for file
    lock1 = await watcher._get_lock_for_file("/vault/test1.md")
    assert len(watcher._reindex_locks) == 1
    assert isinstance(lock1, asyncio.Lock)

    # Get lock for same file again (should return same lock)
    lock2 = await watcher._get_lock_for_file("/vault/test1.md")
    assert lock1 is lock2

    # Get lock for different file (should create new lock)
    lock3 = await watcher._get_lock_for_file("/vault/test2.md")
    assert len(watcher._reindex_locks) == 2
    assert lock3 is not lock1


@pytest.mark.asyncio
async def test_cleanup_lock_removes_unused_locks(tmp_vault, mock_store, mock_embedder):
    """Test that _cleanup_lock removes locks when no pending changes."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1
    )

    file_path = "/vault/test.md"

    # Create lock
    await watcher._get_lock_for_file(file_path)
    assert len(watcher._reindex_locks) == 1

    # Cleanup with no pending changes
    await watcher._cleanup_lock(file_path)
    assert len(watcher._reindex_locks) == 0

    # Create lock again
    await watcher._get_lock_for_file(file_path)

    # Add pending change
    watcher.pending_changes[file_path] = time.time()

    # Cleanup should NOT remove lock (pending change exists)
    await watcher._cleanup_lock(file_path)
    assert len(watcher._reindex_locks) == 1
