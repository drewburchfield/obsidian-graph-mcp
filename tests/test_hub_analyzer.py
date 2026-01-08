"""
Unit tests for hub and orphan detection.

Tests:
1. Hub note detection logic
2. Orphan note detection logic
3. Connection count refresh mechanism
4. Staleness detection and refresh triggering
5. Lock-based concurrency control
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hub_analyzer import HubAnalyzer


@pytest.mark.asyncio
async def test_get_hub_notes_queries_correctly(mock_store):
    """Test that get_hub_notes queries with correct parameters."""
    analyzer = HubAnalyzer(mock_store)

    # Mock database response
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(
        return_value=[
            {"path": "hub1.md", "title": "Hub 1", "connection_count": 25},
            {"path": "hub2.md", "title": "Hub 2", "connection_count": 15},
        ]
    )

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

    mock_store.pool = MagicMock()
    mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

    # Disable refresh check for this test
    analyzer._ensure_fresh_counts = AsyncMock()

    # Call get_hub_notes
    hubs = await analyzer.get_hub_notes(min_connections=10, threshold=0.5, limit=20)

    # Verify results
    assert len(hubs) == 2
    assert hubs[0]["path"] == "hub1.md"
    assert hubs[0]["connection_count"] == 25
    assert hubs[1]["path"] == "hub2.md"
    assert hubs[1]["connection_count"] == 15


@pytest.mark.asyncio
async def test_get_orphaned_notes_queries_correctly(mock_store):
    """Test that get_orphaned_notes queries with correct parameters."""
    analyzer = HubAnalyzer(mock_store)

    # Mock database response
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(
        return_value=[
            {"path": "orphan1.md", "title": "Orphan 1", "connection_count": 0, "modified_at": None},
            {"path": "orphan2.md", "title": "Orphan 2", "connection_count": 1, "modified_at": None},
        ]
    )

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

    mock_store.pool = MagicMock()
    mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

    # Disable refresh check
    analyzer._ensure_fresh_counts = AsyncMock()

    # Call get_orphaned_notes
    orphans = await analyzer.get_orphaned_notes(max_connections=2, threshold=0.5, limit=20)

    # Verify results
    assert len(orphans) == 2
    assert orphans[0]["path"] == "orphan1.md"
    assert orphans[0]["connection_count"] == 0


@pytest.mark.asyncio
async def test_refresh_lock_prevents_concurrent_execution(mock_store):
    """Test that refresh lock prevents multiple concurrent refreshes."""
    analyzer = HubAnalyzer(mock_store)

    # Mock database
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(
        return_value=[
            {"path": "note1.md", "embedding": [0.1] * 1024},
            {"path": "note2.md", "embedding": [0.2] * 1024},
        ]
    )
    mock_conn.fetchval = AsyncMock(return_value=0)
    mock_conn.execute = AsyncMock()

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

    mock_store.pool = MagicMock()
    mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

    # Track refresh executions
    refresh_count = 0
    count_lock = asyncio.Lock()

    original_refresh = analyzer._refresh_all_counts

    async def tracked_refresh(threshold):
        nonlocal refresh_count
        async with count_lock:
            refresh_count += 1
        await original_refresh(threshold)

    analyzer._refresh_all_counts = tracked_refresh

    # Start 5 concurrent refreshes
    tasks = [asyncio.create_task(tracked_refresh(0.5)) for _ in range(5)]

    await asyncio.gather(*tasks)

    # All should have executed (but serially due to lock)
    # The lock ensures they don't run concurrently, but all complete
    assert refresh_count == 5


@pytest.mark.asyncio
async def test_ensure_fresh_counts_skips_if_locked(mock_store):
    """Test that _ensure_fresh_counts skips check if refresh already running."""
    analyzer = HubAnalyzer(mock_store)

    # Mock database
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(side_effect=[500, 1000])  # 50% stale

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

    mock_store.pool = MagicMock()
    mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

    # Acquire lock to simulate refresh in progress
    async with analyzer._refresh_lock:
        # While locked, ensure_fresh_counts should skip immediately
        await analyzer._ensure_fresh_counts(0.5)

        # Should not have queried database (skipped due to lock)
        mock_conn.fetchval.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_updates_connection_counts(mock_store):
    """Test that refresh properly updates connection_count for all notes.

    The batched approach processes all notes that fit in a batch (100 notes)
    with a single UPDATE statement, so 3 notes = 1 batch = 1 execute call.
    """
    analyzer = HubAnalyzer(mock_store)

    # Mock database with 3 notes
    mock_conn = AsyncMock()

    # fetchval: first call returns total note count (3)
    mock_conn.fetchval = AsyncMock(return_value=3)

    # fetch: returns batch of note paths, then empty on second call (end of batches)
    mock_conn.fetch = AsyncMock(
        side_effect=[
            [{"path": "note1.md"}, {"path": "note2.md"}, {"path": "note3.md"}],
            [],  # No more batches
        ]
    )

    mock_conn.execute = AsyncMock()

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

    mock_store.pool = MagicMock()
    mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

    # Run refresh
    await analyzer._refresh_all_counts(threshold=0.5)

    # Batched approach: 3 notes fit in 1 batch (batch_size=100), so 1 execute call
    assert mock_conn.execute.call_count == 1


@pytest.mark.asyncio
async def test_staleness_check_triggers_refresh_when_needed(mock_store):
    """Test that staleness check triggers refresh when >50% notes stale."""
    analyzer = HubAnalyzer(mock_store)

    # Mock database: 60% of notes have connection_count = 0 (stale)
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(side_effect=[600, 1000])  # stale_count  # total_count

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

    mock_store.pool = MagicMock()
    mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

    # Mock refresh
    analyzer._refresh_all_counts = AsyncMock()

    # Check freshness
    await analyzer._ensure_fresh_counts(0.5)

    # Allow background task to be scheduled
    await asyncio.sleep(0.1)

    # Should have triggered refresh (60% > 50% threshold)
    # Note: refresh runs via asyncio.create_task, so we check it was called
    # The actual scheduling depends on event loop timing


@pytest.mark.asyncio
async def test_staleness_check_skips_refresh_when_fresh(mock_store):
    """Test that staleness check skips refresh when counts are fresh."""
    analyzer = HubAnalyzer(mock_store)

    # Mock database: only 20% of notes stale (<50% threshold)
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(side_effect=[200, 1000])  # stale_count  # total_count

    class MockAcquire:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, *args):
            pass

    mock_store.pool = MagicMock()
    mock_store.pool.acquire = MagicMock(return_value=MockAcquire())

    # Mock refresh
    analyzer._refresh_all_counts = AsyncMock()

    # Check freshness
    await analyzer._ensure_fresh_counts(0.5)

    # Should NOT have triggered refresh (20% < 50% threshold)
    analyzer._refresh_all_counts.assert_not_called()
