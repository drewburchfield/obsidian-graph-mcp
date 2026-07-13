# Changelog

All notable changes to Obsidian Graph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-17

### Added
- Initial release of Obsidian Graph MCP Server
- 5 MCP tools for semantic knowledge graph navigation:
  - `search_notes`: Semantic search across vault
  - `get_similar_notes`: Find semantically similar notes
  - `get_connection_graph`: Multi-hop BFS graph traversal
  - `get_hub_notes`: Identify highly connected notes
  - `get_orphaned_notes`: Find isolated notes
- Voyage Context-3 integration (1024-dimensional embeddings)
- PostgreSQL+pgvector vector store with HNSW indexing
- Automatic file watching with 30-second debounce
- Incremental re-indexing on file changes
- Docker-based deployment with security hardening
- Comprehensive documentation (README, CONTRIBUTING)
- Unit and integration tests
- Docker Compose deployment with PostgreSQL+pgvector

### Performance
- Search latency: 0.9ms (555x better than <500ms target)
- Graph building: <2s for depth=3, max_per_level=5
- Hub/orphan queries: <100ms with materialized connection_count
- Similarity scores: Validated [0.0-1.0] range

### Security
- Non-root Docker user (mcpuser)
- JSON caching (not unsafe serialization formats)
- Parameterized SQL queries
- .gitignore for credential files
- Security hardening (cap_drop, no-new-privileges)

## [Unreleased]

### Changed
- **Embedding model upgraded to `voyage-context-4`** (from `voyage-context-3`): same 1024-dim vectors and `contextualized_embed` API, better retrieval quality (+2.08% chunk-level NDCG@10 per Voyage), lower price ($0.12/1M vs $0.18/1M), and its own 200M free-token tier
  - New `VOYAGE_MODEL` env var selects the model (default: `voyage-context-4`); resolved inside `VoyageEmbedder` so the server and indexer always agree
  - **Migration**: embeddings from different models are not comparable. After upgrading, run a full re-index: `docker exec -i obsidian-graph .venv/bin/python -m src.indexer`. The embedding cache is keyed by model name and invalidates itself.
  - **Model-mismatch detection**: a new `index_metadata` table records which model built the index (written by the indexer, table auto-created for existing databases); the server logs a loud error at startup when its configured model differs from the stored one, instead of silently returning meaningless similarity scores
  - **Indexer honesty**: the indexer now reports how many chunks it actually upserted this run, logs an error instead of a success line when files failed (failed files keep their previous embeddings), and exits non-zero so partial migrations are visible
  - **Fail-fast on bad model names**: invalid-model API errors (e.g. a typo'd `VOYAGE_MODEL`) are no longer retried with backoff; empty or whitespace `VOYAGE_MODEL` values fall back to the default
- Docs: `docker exec` examples now use the image's `.venv/bin/python` (the system `python` has no dependencies installed)
- Docs: MCP client config now disables the file watcher (`OBSIDIAN_WATCH_ENABLED=false`) for exec'd stdio sessions; the main container process already owns watching
- Added `scripts/run_vault_mcp.sh` launcher for MCP clients (wraps the docker exec invocation)
- Added `.dockerignore` so host `__pycache__`, `.venv`, and `.env` never leak into the image

### Added
- **Cloud Sync Support**: Automatic polling mode for iCloud, Google Drive, Dropbox, and OneDrive vaults
  - Auto-detection of cloud-synced paths on macOS
  - Auto-enabled in Docker for reliable file watching
  - Configurable polling interval via `OBSIDIAN_WATCH_POLLING_INTERVAL`
  - Override with `OBSIDIAN_WATCH_USE_POLLING=true|false`
- **File Deletion Handling**: `on_deleted` handler removes notes from database when files are deleted
- **File Move Handling**: `on_moved` handler updates database when files are renamed or moved
- **Orphan Cleanup**: Startup scan removes stale database entries for files that no longer exist
- **Folder Exclusion**: Custom `.obsidian-graph.conf` file for excluding folders from indexing

### Fixed
- Stale database entries no longer persist after file deletions (Issue #2)
- File moves now update paths correctly instead of creating duplicates
- Embedding token limit errors on large/dense notes: dynamic batch sizing with retry-halving (#7)
- Hub notes returning empty on first call: inline-await refresh instead of fire-and-forget (#8)
- Event loop blocking during embedding API calls: async embedder methods (#9)
- Missing database timeouts on 6 vector store methods (#9)
- Schema trigger overwriting file modification times (#10)
- Hub analyzer raising wrong exception type (#10)
- Dead code cleanup and weak test assertions (#10)

### Changed
- File watcher now defaults to polling mode in Docker (native filesystem events unreliable)
- Startup scan now cleans up orphan paths before re-indexing stale files
- Renamed project from "MCP Server" to "Obsidian Graph" (semantic knowledge graph engine)
- Container names: obsidian-graph (app), obsidian-graph-pgvector (db)

### Planned
- Separate src/ into engine/ and mcp/ packages
- Additional embedding provider support
- Cluster analysis tool (community detection)
- Performance optimizations for large vaults (>10k notes)
