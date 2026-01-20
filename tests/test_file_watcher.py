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
        debounce_seconds=1,
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
        debounce_seconds=1,
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
        debounce_seconds=0.3,  # Short debounce for testing
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
        debounce_seconds=0.1,
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
        debounce_seconds=0.1,
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
    assert (
        len(watcher._reindex_locks) < 10
    ), f"Lock dict has {len(watcher._reindex_locks)} entries (memory leak!)"


@pytest.mark.asyncio
async def test_vault_watcher_startup_scan_detects_stale_files(tmp_vault, mock_store, mock_embedder):
    """Test that startup scan detects files changed while offline."""
    # Create vault watcher
    vault_watcher = VaultWatcher(
        vault_path=str(tmp_vault), store=mock_store, embedder=mock_embedder, debounce_seconds=1
    )

    # Mock database to return old last_indexed_at times
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(
        return_value=datetime(2020, 1, 1, tzinfo=UTC)
    )  # Very old timestamp (timezone-aware)

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
    assert (
        vault_watcher.event_handler._reindex_file.call_count >= 3
    ), "Expected at least 3 stale files detected"


@pytest.mark.asyncio
async def test_get_lock_for_file_creates_new_locks(tmp_vault, mock_store, mock_embedder):
    """Test that _get_lock_for_file creates locks on demand."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1,
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
        debounce_seconds=1,
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


@pytest.mark.asyncio
async def test_file_watcher_ignores_excluded_paths(tmp_path, mock_store, mock_embedder):
    """Test that file watcher skips excluded paths."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create config to exclude trash/
    (vault / ".obsidian-graph.conf").write_text("trash/\n")

    # Create excluded and non-excluded files
    trash = vault / "trash"
    trash.mkdir()
    (trash / "deleted.md").write_text("# Deleted")
    (vault / "normal.md").write_text("# Normal")

    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1,
    )

    class MockEvent:
        def __init__(self, src_path):
            self.src_path = src_path
            self.is_directory = False

    # Simulate event for excluded file
    watcher.on_modified(MockEvent(str(trash / "deleted.md")))
    # Should not add to pending changes
    assert str(trash / "deleted.md") not in watcher.pending_changes

    # Simulate event for non-excluded file
    watcher.on_modified(MockEvent(str(vault / "normal.md")))
    # Should add to pending changes
    assert str(vault / "normal.md") in watcher.pending_changes


@pytest.mark.asyncio
async def test_file_watcher_excludes_default_patterns(tmp_path, mock_store, mock_embedder):
    """Test that default exclusions (.obsidian, .git) work."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create .obsidian folder (default exclusion)
    obsidian = vault / ".obsidian"
    obsidian.mkdir()
    (obsidian / "plugins.md").write_text("plugins")

    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1,
    )

    class MockEvent:
        def __init__(self, src_path):
            self.src_path = src_path
            self.is_directory = False

    # Simulate event for .obsidian file
    watcher.on_modified(MockEvent(str(obsidian / "plugins.md")))

    # Should not add to pending changes (default exclusion)
    assert str(obsidian / "plugins.md") not in watcher.pending_changes


@pytest.mark.asyncio
async def test_file_watcher_ignores_excluded_paths_on_created(tmp_path, mock_store, mock_embedder):
    """Test that on_created skips excluded paths."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create config to exclude trash/
    (vault / ".obsidian-graph.conf").write_text("trash/\n")

    # Create excluded folder
    trash = vault / "trash"
    trash.mkdir()
    (trash / "deleted.md").write_text("# Deleted")
    (vault / "normal.md").write_text("# Normal")

    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1,
    )

    class MockEvent:
        def __init__(self, src_path):
            self.src_path = src_path
            self.is_directory = False

    # Simulate on_created for excluded file
    watcher.on_created(MockEvent(str(trash / "deleted.md")))
    # Should not add to pending changes
    assert str(trash / "deleted.md") not in watcher.pending_changes

    # Simulate on_created for non-excluded file
    watcher.on_created(MockEvent(str(vault / "normal.md")))
    # Should add to pending changes
    assert str(vault / "normal.md") in watcher.pending_changes


@pytest.mark.asyncio
async def test_file_watcher_handles_deletion(tmp_vault, mock_store, mock_embedder):
    """Test that on_deleted removes the file from the database."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1,
    )

    # Create a test file path
    test_file = tmp_vault / "to_delete.md"
    test_file.write_text("# Will be deleted")

    class MockEvent:
        def __init__(self, src_path, is_directory=False):
            self.src_path = src_path
            self.is_directory = is_directory

    # Simulate file deletion event
    event = MockEvent(str(test_file))
    watcher.on_deleted(event)

    # Wait for async operation to complete
    await asyncio.sleep(0.1)

    # Should have called delete_notes_by_paths with the relative path
    mock_store.delete_notes_by_paths.assert_called_once()
    call_args = mock_store.delete_notes_by_paths.call_args[0][0]
    assert "to_delete.md" in call_args


@pytest.mark.asyncio
async def test_file_watcher_handles_move(tmp_vault, mock_store, mock_embedder):
    """Test that on_moved deletes old path and indexes new path."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=0.1,  # Short debounce for testing
    )

    # Create source file
    old_file = tmp_vault / "old_location.md"
    old_file.write_text("# Moving this note")

    # New location
    new_file = tmp_vault / "new_location.md"

    class MockMoveEvent:
        def __init__(self, src_path, dest_path, is_directory=False):
            self.src_path = src_path
            self.dest_path = dest_path
            self.is_directory = is_directory

    # Simulate file move event
    event = MockMoveEvent(str(old_file), str(new_file))
    watcher.on_moved(event)

    # Wait for async operations
    await asyncio.sleep(0.1)

    # Should have called delete for old path
    mock_store.delete_notes_by_paths.assert_called_once()
    delete_args = mock_store.delete_notes_by_paths.call_args[0][0]
    assert "old_location.md" in delete_args

    # Should have added new path to pending changes for re-indexing
    assert str(new_file) in watcher.pending_changes


@pytest.mark.asyncio
async def test_file_watcher_handles_move_to_excluded(tmp_path, mock_store, mock_embedder):
    """Test that moving a file to an excluded location deletes but doesn't re-index."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create config to exclude trash/
    (vault / ".obsidian-graph.conf").write_text("trash/\n")

    # Create trash folder and source file
    trash = vault / "trash"
    trash.mkdir()
    source_file = vault / "active.md"
    source_file.write_text("# Active note")

    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1,
    )

    # Target location in excluded trash folder
    dest_file = trash / "active.md"

    class MockMoveEvent:
        def __init__(self, src_path, dest_path, is_directory=False):
            self.src_path = src_path
            self.dest_path = dest_path
            self.is_directory = is_directory

    # Simulate move to trash
    event = MockMoveEvent(str(source_file), str(dest_file))
    watcher.on_moved(event)

    # Wait for async operations
    await asyncio.sleep(0.1)

    # Should delete old path from DB
    mock_store.delete_notes_by_paths.assert_called_once()

    # Should NOT add new path to pending (it's excluded)
    assert str(dest_file) not in watcher.pending_changes


@pytest.mark.asyncio
async def test_file_watcher_handles_rename_txt_to_md(tmp_vault, mock_store, mock_embedder):
    """Test that renaming a .txt file to .md indexes the new file."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=0.1,
    )

    # Source is .txt, destination is .md
    old_file = tmp_vault / "draft.txt"
    new_file = tmp_vault / "draft.md"
    new_file.write_text("# Now a markdown file")

    class MockMoveEvent:
        def __init__(self, src_path, dest_path, is_directory=False):
            self.src_path = src_path
            self.dest_path = dest_path
            self.is_directory = is_directory

    # Simulate rename from .txt to .md
    event = MockMoveEvent(str(old_file), str(new_file))
    watcher.on_moved(event)

    # Wait for async operations
    await asyncio.sleep(0.1)

    # Should NOT delete old path (it wasn't .md)
    mock_store.delete_notes_by_paths.assert_not_called()

    # Should add new path to pending for indexing
    assert str(new_file) in watcher.pending_changes


@pytest.mark.asyncio
async def test_file_watcher_handles_rename_md_to_txt(tmp_vault, mock_store, mock_embedder):
    """Test that renaming a .md file to .txt deletes from DB but doesn't re-index."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1,
    )

    # Source is .md, destination is .txt
    old_file = tmp_vault / "note.md"
    old_file.write_text("# Was a markdown file")
    new_file = tmp_vault / "note.txt"

    class MockMoveEvent:
        def __init__(self, src_path, dest_path, is_directory=False):
            self.src_path = src_path
            self.dest_path = dest_path
            self.is_directory = is_directory

    # Simulate rename from .md to .txt
    event = MockMoveEvent(str(old_file), str(new_file))
    watcher.on_moved(event)

    # Wait for async operations
    await asyncio.sleep(0.1)

    # Should delete old path from DB
    mock_store.delete_notes_by_paths.assert_called_once()
    delete_args = mock_store.delete_notes_by_paths.call_args[0][0]
    assert "note.md" in delete_args

    # Should NOT add new path to pending (it's not .md)
    assert str(new_file) not in watcher.pending_changes


@pytest.mark.asyncio
async def test_startup_scan_cleans_orphans(tmp_vault, mock_store, mock_embedder):
    """Test that startup scan removes orphan paths from database."""
    # Create vault watcher
    vault_watcher = VaultWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        debounce_seconds=1,
    )

    # Mock get_all_paths to return paths including one that doesn't exist
    mock_store.get_all_paths = AsyncMock(
        return_value=["note1.md", "deleted_note.md", "folder/note3.md"]
    )

    # Mock database connection for stale file detection
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=None)  # All files need re-indexing

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

    # Mock re-index to not actually do anything
    vault_watcher.event_handler._reindex_file = AsyncMock()

    # Run startup scan
    await vault_watcher.startup_scan()

    # Should have called delete_notes_by_paths with the orphan path
    mock_store.delete_notes_by_paths.assert_called()
    delete_args = mock_store.delete_notes_by_paths.call_args[0][0]
    assert "deleted_note.md" in delete_args
    # note1.md and folder/note3.md exist in tmp_vault, so should NOT be deleted
    assert "note1.md" not in delete_args
    assert "folder/note3.md" not in delete_args


@pytest.mark.asyncio
async def test_file_watcher_ignores_non_md_deletion(tmp_vault, mock_store, mock_embedder):
    """Test that deletion of non-markdown files is ignored."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1,
    )

    class MockEvent:
        def __init__(self, src_path, is_directory=False):
            self.src_path = src_path
            self.is_directory = is_directory

    # Simulate deletion of non-markdown file
    watcher.on_deleted(MockEvent(str(tmp_vault / "image.png")))

    # Wait for any async operations
    await asyncio.sleep(0.1)

    # Should NOT have called delete_notes_by_paths
    mock_store.delete_notes_by_paths.assert_not_called()


@pytest.mark.asyncio
async def test_file_watcher_ignores_directory_deletion(tmp_vault, mock_store, mock_embedder):
    """Test that deletion of directories is ignored."""
    loop = asyncio.get_running_loop()
    watcher = ObsidianFileWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        loop=loop,
        debounce_seconds=1,
    )

    class MockEvent:
        def __init__(self, src_path, is_directory=False):
            self.src_path = src_path
            self.is_directory = is_directory

    # Simulate deletion of a directory
    watcher.on_deleted(MockEvent(str(tmp_vault / "folder"), is_directory=True))

    # Wait for any async operations
    await asyncio.sleep(0.1)

    # Should NOT have called delete_notes_by_paths
    mock_store.delete_notes_by_paths.assert_not_called()


# =============================================================================
# Polling Mode Tests
# =============================================================================

from src.file_watcher import is_cloud_synced_path, should_use_polling
from watchdog.observers.polling import PollingObserver


def test_is_cloud_synced_path_detects_icloud():
    """Test that iCloud Drive paths are detected."""
    icloud_paths = [
        "/Users/drew/Library/Mobile Documents/com~apple~CloudDocs/Obsidian/MyVault",
        "/Users/john/Library/Mobile Documents/iCloud~md~obsidian/Documents/vault",
    ]
    for path in icloud_paths:
        assert is_cloud_synced_path(path), f"Should detect iCloud path: {path}"


def test_is_cloud_synced_path_detects_google_drive():
    """Test that Google Drive paths are detected."""
    gdrive_paths = [
        "/Users/drew/Library/CloudStorage/GoogleDrive-drew@example.com/My Drive/Obsidian",
        "/Users/john/Library/CloudStorage/GoogleDrive-john@company.com/vault",
    ]
    for path in gdrive_paths:
        assert is_cloud_synced_path(path), f"Should detect Google Drive path: {path}"


def test_is_cloud_synced_path_detects_dropbox():
    """Test that Dropbox paths are detected."""
    dropbox_paths = [
        "/Users/drew/Library/CloudStorage/Dropbox/Obsidian",
        "/Users/john/Dropbox/vault",  # Legacy location
    ]
    for path in dropbox_paths:
        assert is_cloud_synced_path(path), f"Should detect Dropbox path: {path}"


def test_is_cloud_synced_path_detects_onedrive():
    """Test that OneDrive paths are detected."""
    onedrive_paths = [
        "/Users/drew/Library/CloudStorage/OneDrive-Personal/Obsidian",
        "/Users/john/Library/CloudStorage/OneDrive-Company/Documents/vault",
    ]
    for path in onedrive_paths:
        assert is_cloud_synced_path(path), f"Should detect OneDrive path: {path}"


def test_is_cloud_synced_path_returns_false_for_local():
    """Test that local paths are not detected as cloud-synced."""
    local_paths = [
        "/Users/drew/Documents/Obsidian",
        "/tmp/vault",
        "/home/user/obsidian-vault",
        "/vault",  # Docker default path
    ]
    for path in local_paths:
        assert not is_cloud_synced_path(path), f"Should NOT detect as cloud: {path}"


def test_should_use_polling_respects_env_true(monkeypatch):
    """Test that OBSIDIAN_WATCH_USE_POLLING=true forces polling."""
    monkeypatch.setenv("OBSIDIAN_WATCH_USE_POLLING", "true")
    assert should_use_polling("/local/path") is True


def test_should_use_polling_respects_env_false(monkeypatch):
    """Test that OBSIDIAN_WATCH_USE_POLLING=false disables polling."""
    monkeypatch.setenv("OBSIDIAN_WATCH_USE_POLLING", "false")
    # Even for cloud paths, env var should override
    assert should_use_polling("/Users/drew/Library/Mobile Documents/vault") is False


def test_should_use_polling_auto_detects_cloud_path(monkeypatch):
    """Test that polling is auto-enabled for cloud-synced paths."""
    monkeypatch.delenv("OBSIDIAN_WATCH_USE_POLLING", raising=False)
    # Mock is_running_in_docker to return False
    import src.file_watcher as fw

    original_func = fw.is_running_in_docker
    fw.is_running_in_docker = lambda: False
    try:
        assert should_use_polling("/Users/drew/Library/Mobile Documents/vault") is True
        assert should_use_polling("/Users/drew/Library/CloudStorage/GoogleDrive-x/vault") is True
    finally:
        fw.is_running_in_docker = original_func


@pytest.mark.asyncio
async def test_vault_watcher_uses_polling_observer_when_configured(
    tmp_vault, mock_store, mock_embedder, monkeypatch
):
    """Test that VaultWatcher uses PollingObserver when polling is enabled."""
    monkeypatch.setenv("OBSIDIAN_WATCH_USE_POLLING", "true")

    vault_watcher = VaultWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        debounce_seconds=1,
    )

    assert vault_watcher.use_polling is True

    # Start the watcher
    loop = asyncio.get_running_loop()
    vault_watcher.start(loop)

    try:
        # Should be using PollingObserver
        assert isinstance(vault_watcher.observer, PollingObserver)
    finally:
        vault_watcher.stop()


@pytest.mark.asyncio
async def test_vault_watcher_uses_native_observer_when_not_polling(
    tmp_vault, mock_store, mock_embedder, monkeypatch
):
    """Test that VaultWatcher uses native Observer when polling is disabled."""
    monkeypatch.setenv("OBSIDIAN_WATCH_USE_POLLING", "false")

    vault_watcher = VaultWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        debounce_seconds=1,
    )

    assert vault_watcher.use_polling is False

    # Start the watcher
    loop = asyncio.get_running_loop()
    vault_watcher.start(loop)

    try:
        # Should be using native Observer (not PollingObserver)
        assert not isinstance(vault_watcher.observer, PollingObserver)
    finally:
        vault_watcher.stop()


def test_vault_watcher_polling_interval_from_env(tmp_vault, mock_store, mock_embedder, monkeypatch):
    """Test that polling interval is read from environment variable."""
    monkeypatch.setenv("OBSIDIAN_WATCH_POLLING_INTERVAL", "60")

    vault_watcher = VaultWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        debounce_seconds=1,
    )

    assert vault_watcher.polling_interval == 60


def test_vault_watcher_polling_interval_from_param(tmp_vault, mock_store, mock_embedder):
    """Test that polling interval parameter overrides environment variable."""
    vault_watcher = VaultWatcher(
        vault_path=str(tmp_vault),
        store=mock_store,
        embedder=mock_embedder,
        debounce_seconds=1,
        polling_interval=45,
    )

    assert vault_watcher.polling_interval == 45
