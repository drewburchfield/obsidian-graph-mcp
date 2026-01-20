# Changelog

All notable changes to the Obsidian Graph MCP Server will be documented in this file.

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

### Changed
- File watcher now defaults to polling mode in Docker (native filesystem events unreliable)
- Startup scan now cleans up orphan paths before re-indexing stale files

## [Unreleased - Planned]

### Planned
- Additional embedding provider support (OpenAI, SentenceTransformers)
- Cluster analysis tool (community detection)
- Temporal statistics tool
- Embedding validation tool
- Performance optimizations for large vaults (>10k notes)
- GitHub Actions CI/CD
- Code coverage reporting
