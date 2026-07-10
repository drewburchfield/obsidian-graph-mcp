# Consulting Graph Continuous Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an isolated containerized `consulting-graph` MCP that continuously reconciles every supported consulting document format while the separate Obsidian graph remains unchanged.

**Architecture:** Generalize the existing corpus scanner and watcher so both startup reconciliation and polling use one multi-format document pipeline. Restore the core two-container application-plus-PostgreSQL pattern in `docker-compose.consulting.yml`, with a read-only `/corpus` mount and consulting-specific configuration.

**Tech Stack:** Python 3.11, asyncio, watchdog polling, MarkItDown, pandas, OpenRouter embeddings, MCP stdio, PostgreSQL 15, pgvector, Docker Compose, pytest.

## Global Constraints

- Preserve the existing Obsidian Compose stack and its Markdown-only Voyage defaults.
- Run consulting and Obsidian instances concurrently with separate names, networks, volumes, databases, caches, mounts, and MCP registrations.
- Mount `/Users/drewburchfield/dev/consulting` read-only at `/corpus` only in the consulting application container.
- Support Markdown, PDF, DOCX, XLSX, XLS, PPTX, HTML, CSV, and TXT in the consulting instance.
- Preserve unchanged 4096-dimension embeddings.
- Apply one exclusion policy during discovery, startup reconciliation, and live polling.
- Never depend on an active MCP client for synchronization.

---

### Task 1: Generalize corpus discovery and exclusions

**Files:**
- Modify: `src/multi_format_indexer.py`
- Modify: `src/exclusion.py`
- Test: `tests/test_multi_format_indexer.py`
- Test: `tests/test_exclusion.py`

**Interfaces:**
- Produces: `scan_documents(root_path: str, enabled_extensions: set[str] | None = None, exclusion_filter: ExclusionFilter | None = None) -> list[Path]`.
- Produces: one `ExclusionFilter` used by batch indexing and the watcher.

- [ ] Write failing tests proving configured extensions and consulting exclusions apply to nested files, `.tmp-*`, `.superpowers`, Office lock files, archives, and agent tooling.
- [ ] Run `pytest tests/test_multi_format_indexer.py tests/test_exclusion.py -q` and confirm the new tests fail.
- [ ] Replace the indexer's private exclusion rules with `ExclusionFilter`, add configured extension filtering, and retain current consulting exclusions.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Add atomic multi-format file indexing and reconciliation

**Files:**
- Create: `src/corpus_sync.py`
- Modify: `src/vector_store.py`
- Modify: `src/multi_format_indexer.py`
- Test: `tests/test_corpus_sync.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `CorpusSynchronizer.reindex_file(path: Path) -> bool`.
- Produces: `CorpusSynchronizer.reconcile() -> ReconcileSummary`.
- Produces: `PostgreSQLVectorStore.replace_file_notes(path: str, notes: list[Note]) -> int` with one database transaction.

- [ ] Write failing tests for new, changed, unchanged, deleted, newly excluded, shortened multi-chunk, conversion-failure, and embedding-failure cases.
- [ ] Run `pytest tests/test_corpus_sync.py tests/test_tools.py -q` and confirm the new tests fail.
- [ ] Implement atomic chunk replacement and metadata-based reconciliation using relative path, modification time, and size.
- [ ] Keep the last valid database version when conversion or embedding fails.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Generalize the polling watcher and server lifecycle

**Files:**
- Modify: `src/file_watcher.py`
- Modify: `src/server.py`
- Test: `tests/test_file_watcher.py`

**Interfaces:**
- Consumes: `CorpusSynchronizer` from Task 2.
- Produces: a watcher that accepts `enabled_extensions: set[str]`, reconciles at startup, and handles create, modify, move, and delete events for those extensions.

- [ ] Replace the Markdown-only watcher tests with parameterized enabled-format tests while retaining a Markdown-only configuration test.
- [ ] Add failing tests proving startup uses reconciliation and move/delete events remove old paths.
- [ ] Run `pytest tests/test_file_watcher.py -q` and confirm the new tests fail.
- [ ] Route watcher events through `CorpusSynchronizer`, parse `CORPUS_EXTENSIONS`, and keep `OBSIDIAN_WATCH_ENABLED` backward compatible.
- [ ] Run the focused tests and confirm they pass.

### Task 4: Restore the core Docker application pattern for consulting

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.consulting.yml`
- Modify: `scripts/run_consulting_mcp.sh`
- Create: `.consulting-graph.conf`
- Test: `tests/test_e2e_docker.py`

**Interfaces:**
- Produces: `consulting-graph` application container and `consulting-graph-pgvector` database container on a dedicated `consulting-graph` network.
- Produces: separate MCP command `docker exec -i consulting-graph .venv/bin/python -m src.server`.

- [ ] Add assertions for the application service, `/corpus:ro` mount, multi-format dependencies, OpenRouter configuration, enabled polling, isolated names, and isolated volumes.
- [ ] Run the Compose configuration and focused tests to confirm the assertions fail.
- [ ] Install the `multiformat` optional dependency group in the image and add the application service beside PostgreSQL.
- [ ] Configure `/corpus`, OpenRouter 4096d, consulting database host, polling, cache, health check, and restart policy.
- [ ] Run `docker compose -f docker-compose.consulting.yml config` and focused tests.

### Task 5: Start, migrate, and verify the live consulting instance

**Files:**
- Modify: `README.md`
- Modify: `semantic-graph-playbook.md` in the consulting workspace if its operational instructions are stale.

**Interfaces:**
- Consumes: the complete consulting Compose stack.
- Produces: a live `consulting-graph` MCP and continuously maintained database.

- [ ] Run the full unit suite and lint checks.
- [ ] Build and start `docker compose -f docker-compose.consulting.yml up -d --build`.
- [ ] Confirm both consulting containers are healthy and the Obsidian stack remains separate.
- [ ] Wait for startup reconciliation and compare filesystem paths, modification metadata, and database paths until missing, stale, and extra counts are zero.
- [ ] Create, modify, move, and delete a disposable supported file under the consulting corpus, waiting only for the configured polling and embedding interval between checks.
- [ ] Verify each state transition directly in PostgreSQL and verify current content through the consulting MCP search path.
- [ ] Remove the disposable file and confirm the database removes it.
- [ ] Update usage documentation with the separate MCP client command and health commands.
- [ ] Run the completion audit against every design success criterion.
