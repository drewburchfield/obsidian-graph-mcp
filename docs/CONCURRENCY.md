# Thread-Safety and Concurrency - Obsidian Graph MCP Server

## Overview

This document describes the concurrency model, thread-safety guarantees, and synchronization mechanisms used in the Obsidian Graph MCP server.

**Concurrency Model**: Single-threaded asyncio event loop with watchdog file observer running in background thread.

---

## Component Thread-Safety

### File Watcher

**Component**: `src/file_watcher.py::ObsidianFileWatcher`

**Thread-Safety Guarantees:**
- ✅ Multiple concurrent modifications to SAME file: Safe (debounced to single re-index)
- ✅ Concurrent modifications to DIFFERENT files: Safe (processed in parallel)
- ✅ Watchdog events from background thread: Safe (bridges to asyncio via run_coroutine_threadsafe)

**Synchronization Mechanism**: Per-file async locks

```python
class ObsidianFileWatcher:
    def __init__(self, ...):
        self._reindex_locks: Dict[str, asyncio.Lock] = {}  # One lock per file
        self._locks_lock = asyncio.Lock()  # Protects the locks dict
```

**Concurrency Behavior:**

| Scenario | Behavior | Performance |
|----------|----------|-------------|
| 10 edits to `note.md` in 5 seconds | Single re-index after debounce | Optimal |
| Edits to `note1.md` and `note2.md` simultaneously | Both re-indexed in parallel | Optimal |
| 100 files modified | 100 concurrent re-indexes (limited by thread pool) | Scales well |

**Lock Lifecycle:**
1. First modification to file creates lock
2. Lock acquired during debounce check
3. Lock released after debounce logic completes
4. Lock cleaned up after re-index finishes (prevents memory leak)

**Event Loop Ownership:**
- Main thread: Asyncio event loop (runs server, handles MCP tools)
- Watchdog thread: File system observer (runs `on_modified`, `on_created`)
- Bridge: `asyncio.run_coroutine_threadsafe()` schedules async work from watchdog thread

---

### Hub Analyzer

**Component**: `src/hub_analyzer.py::HubAnalyzer`

**Thread-Safety Guarantees:**
- ✅ Concurrent calls to `get_hub_notes()`: Safe (read-only queries)
- ✅ Concurrent calls to `get_orphaned_notes()`: Safe (read-only queries)
- ✅ Multiple refresh requests: Only ONE refresh runs (others skip)

**Synchronization Mechanism**: Global async lock for refresh operations

```python
class HubAnalyzer:
    def __init__(self, store):
        self._refresh_lock = asyncio.Lock()  # Serializes refresh operations
```

**Concurrency Behavior:**

| Scenario | Behavior | Performance |
|----------|----------|-------------|
| 20 concurrent `get_hub_notes()` calls | All execute in parallel (read-only) | Optimal |
| 20 concurrent requests trigger refresh | Only 1 refresh runs, others skip | Optimal |
| Refresh running, new `get_hub_notes()` call | Query executes immediately (no blocking) | Optimal |

**Refresh Logic:**
```python
async def _ensure_fresh_counts(self, threshold):
    if self._refresh_lock.locked():  # Non-blocking check
        return  # Skip if refresh already running

    # Check staleness, schedule refresh if needed
    asyncio.create_task(self._refresh_all_counts(threshold))

async def _refresh_all_counts(self, threshold):
    async with self._refresh_lock:  # Blocking - waits for lock
        # Scan entire vault, update connection_count for all notes
        ...
```

**Lock Granularity**: Coarse-grained (one lock for entire refresh operation)

**Rationale**: Refresh is inherently global (updates all notes), so fine-grained locking provides no benefit.

---

### PostgreSQL Vector Store

**Component**: `src/vector_store.py::PostgreSQLVectorStore`

**Thread-Safety Guarantees:**
- ✅ Concurrent searches: Safe (read-only, connection pooling)
- ✅ Concurrent upserts: Safe (connection pooling, transactions)
- ✅ Batch operations: Atomic (wrapped in transactions)

**Synchronization Mechanism**: asyncpg connection pooling

```python
self.pool = await asyncpg.create_pool(
    dsn,
    min_size=5,   # Minimum connections
    max_size=20,  # Maximum connections
    timeout=10    # Connection acquisition timeout
)
```

**Concurrency Behavior:**
- Up to 20 concurrent database operations
- Operations queue if all connections busy
- 10-second timeout prevents deadlocks
- Each operation gets exclusive connection from pool

---

### Voyage Embedder

**Component**: `src/embedder.py::VoyageEmbedder`

**Concurrency Model**: Single-threaded (called exclusively from asyncio event loop)

**Thread-Safety**: ✅ Safe in current architecture (no multi-threaded access)
- Watchdog file events are bridged to event loop using `asyncio.run_coroutine_threadsafe()`
- All embedding calls happen on the same event loop thread
- Not designed for multi-threaded access, but this is not required in current deployment

**Rate Limiting**: Built-in rate limiter (300 requests/minute)

```python
def _rate_limit(self):
    current_time = time.time()
    time_since_last = current_time - self.last_request_time
    if time_since_last < self.request_interval:
        time.sleep(self.request_interval - time_since_last)
    self.last_request_time = time.time()
```

**Concurrency Model**: All embedding calls are serialized (intentional)

**Rationale**: Voyage AI API has strict rate limits; concurrent requests would hit limit faster.

---

## Testing Concurrency

### Run Race Condition Tests

```bash
# All concurrency tests
pytest tests/test_race_conditions.py -v

# Stress tests (high concurrency)
pytest tests/test_race_conditions.py -m stress -v

# File watcher specific
pytest tests/test_race_conditions.py::test_file_watcher_concurrent_debounce_race -v

# Hub analyzer specific
pytest tests/test_race_conditions.py::test_hub_analyzer_concurrent_refresh_race -v
```

### Stress Test Parameters

| Test | Concurrency Level | Expected Behavior |
|------|------------------|-------------------|
| File watcher same file | 10 rapid edits | 1 re-index |
| File watcher different files | 50 concurrent files | 50 parallel re-indexes |
| Hub analyzer refresh | 20 concurrent requests | 1 refresh execution |

---

## Debugging Concurrency Issues

### Detecting Deadlocks

**Symptom**: Server hangs, no progress

**Diagnosis:**
```bash
# Check if any locks are held
docker exec -it mcp-obsidian-graph python3 -c "
import asyncio
# Print all running tasks
for task in asyncio.all_tasks():
    print(task)
"

# Check container CPU usage
docker stats mcp-obsidian-graph
```

**Common Causes:**
- Circular lock dependency (should be impossible with current design)
- Lock not released (should be prevented by `async with` context manager)
- Database connection timeout (10s timeout configured)

---

### Detecting Race Conditions

**Symptom**: Duplicate operations, inconsistent state, unexpected behavior

**Diagnosis:**
1. Enable debug logging: `LOG_LEVEL=DEBUG` in `.env.instance`
2. Look for duplicate log messages:
   ```bash
   docker logs mcp-obsidian-graph | grep "Re-indexed:" | sort | uniq -c
   ```
3. Run race condition tests: `pytest tests/test_race_conditions.py -v`

**Common Patterns:**
- Multiple "Re-indexed: same-file.md" messages → File watcher race
- Multiple "Starting refresh" messages → Hub analyzer race
- Duplicate embeddings API calls → Check rate limiting

---

### Performance Under Concurrency

**Benchmarking:**
```bash
# Run performance tests
pytest tests/test_tools.py::test_search_notes_performance -v
pytest tests/test_tools.py::test_connection_graph_performance -v

# With concurrency stress
pytest tests/test_race_conditions.py -m stress -v
```

**Expected Performance:**
- Search latency: <500ms (even with concurrent requests)
- Graph building: <2000ms (for depth=3, max_per_level=5)
- Hub detection: <1000ms (uses materialized column)
- File re-indexing: <2000ms per file (embedding generation is bottleneck)

---

## Lock Hierarchy

**Current Design**: No lock hierarchy (no circular dependencies possible)

**Lock Independence:**
- File watcher locks: Per-file (independent)
- Hub analyzer lock: Global (single lock)
- No dependencies between these locks

**Why This Works:**
- File watcher never calls hub analyzer
- Hub analyzer never calls file watcher
- Both use separate database connections (from pool)

---

## Event Loop Architecture

```
Main Thread
├── asyncio.run(main())
│   ├── initialize_server()
│   │   ├── Creates components (store, embedder, etc.)
│   │   └── Starts watchdog observer (background thread)
│   └── app.run() - MCP server loop
│       └── call_tool() handlers
│
└── Watchdog Thread (Observer)
    ├── on_modified() events
    ├── on_created() events
    └── run_coroutine_threadsafe() → Main thread event loop
```

**Thread Safety:**
- Main thread: Owns asyncio event loop
- Watchdog thread: Only calls `run_coroutine_threadsafe()`
- No shared mutable state between threads (except `pending_changes` dict, protected by async lock)

---

## Future Improvements

**Under Consideration:**
- [ ] Add timeout to all lock acquisitions (detect deadlocks)
- [ ] Metrics for lock contention (Prometheus)
- [ ] Configurable concurrency limits
- [ ] Separate embedding queue (async worker pattern)
- [ ] Connection pool monitoring and alerts

---

## References

- [asyncio Locks Documentation](https://docs.python.org/3/library/asyncio-sync.html#asyncio.Lock)
- [Watchdog Library](https://python-watchdog.readthedocs.io/)
- [asyncpg Connection Pooling](https://magicstack.github.io/asyncpg/current/api/index.html#connection-pools)
