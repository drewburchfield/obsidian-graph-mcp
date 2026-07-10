# Consulting Graph Continuous Sync Design

## Goal

Run `consulting-graph` as an isolated dogfood instance of the core graph MCP architecture. The instance must keep `/Users/drewburchfield/dev/consulting` synchronized with its graph without depending on Claude, Codex, or an MCP client session.

## Product Boundary

The consulting instance differs from the core Obsidian offering in two material ways:

1. It indexes every supported consulting document format, not only Markdown.
2. It selects OpenRouter embeddings instead of Voyage through the existing provider configuration.

The implementation must generalize core components. It must not introduce a separate consulting-only indexing architecture.

## Architecture

The consulting deployment follows the core two-container pattern:

- `consulting-graph`: the long-running graph application, MCP server, startup reconciler, and filesystem watcher.
- `consulting-graph-pgvector`: the PostgreSQL and pgvector database.

The application container mounts the consulting folder read-only at `/corpus`. It never mounts or scans the Obsidian vault. Both containers use consulting-specific names, a dedicated Docker network, a dedicated database volume, and a dedicated embedding cache volume.

The existing Obsidian deployment remains unchanged and can run at the same time. It retains its own application container, database container, network, volumes, MCP registration, vault mount, and Voyage configuration.

## Generalized Corpus Pipeline

One corpus pipeline owns document discovery, exclusions, conversion, chunking, embedding, and database updates. Initial indexing, startup reconciliation, and live change handling must use that pipeline.

The pipeline supports Markdown, PDF, DOCX, XLSX, XLS, PPTX, HTML, CSV, and TXT. A configuration value may restrict formats for an instance. The Obsidian configuration remains Markdown-only. The consulting configuration enables every supported format.

The same exclusion policy applies during every scan and event. The consulting policy excludes source-control metadata, dependency folders, build output, agent tooling, archives, temporary work folders, lock files, and known non-consulting corpora. A deleted file or a file that becomes excluded must be removed from the graph.

## Synchronization

The application container performs a full reconciliation when it starts:

1. Scan eligible files under the mounted corpus.
2. Compare each file's relative path, modification time, and size with stored metadata.
3. Embed new and changed files.
4. Leave unchanged files and embeddings intact.
5. Remove database rows for deleted, moved, or newly excluded files.

After startup, the existing Docker polling watcher checks the mounted folder at a configurable interval. It handles creates, modifications, moves, and deletions for every enabled format. Event handling uses the same conversion and indexing path as startup reconciliation.

Each file update replaces the file's complete chunk set atomically. This prevents stale trailing chunks when a document becomes shorter. A failed conversion or embedding leaves the last valid indexed version intact and records the failure for retry.

## Embedding Providers

The graph application builds its embedder from configuration:

- Core Obsidian default: Voyage.
- Consulting dogfood instance: OpenRouter `qwen/qwen3-embedding-8b` with 4096 dimensions.

Provider configuration must match the database vector dimension. The application must fail startup with a clear error when required credentials are missing or dimensions conflict.

## MCP Clients

The consulting MCP client has its own registration and executes the server in the `consulting-graph` container. Its server name is `consulting-graph`, and it connects only to `consulting-graph-pgvector`.

The Obsidian MCP client remains separate. Either client can run without changing or indexing the other instance's corpus.

## Operations and Health

Docker Compose starts both consulting containers and restarts them unless stopped. The database health check gates application startup. The application logs:

- corpus path and enabled formats;
- watcher mode and polling interval;
- startup reconciliation counts;
- successful file additions, updates, and removals;
- failed files and retry results.

Container health must distinguish a running process from a working synchronizer. Health includes database connectivity, a mounted corpus, a running watcher, and the time and outcome of the latest reconciliation.

## Migration

The existing consulting database and embeddings remain available during the code change. Before enabling continuous sync, the implementation will:

1. Apply corrected exclusions so temporary and tooling files cannot enter the graph.
2. Start the consulting application against the existing database.
3. Run incremental startup reconciliation.
4. Confirm that new and changed files enter the graph and stale paths leave it.

Unchanged 4096-dimension embeddings remain intact. The migration must not rebuild the entire corpus unless metadata or provider incompatibility makes a rebuild necessary.

## Verification

Automated tests cover discovery and exclusions for every supported format, incremental startup reconciliation, atomic chunk replacement, and create, change, move, delete event handling.

An end-to-end local test will run the consulting Compose stack alongside the Obsidian stack. The test will add, modify, move, and delete disposable files inside an eligible consulting test directory, then verify the corresponding database state and MCP search behavior. It will also confirm that the consulting container has no Obsidian vault mount.

## Success Criteria

- Both graph instances run concurrently without shared containers, networks, databases, caches, corpus mounts, or MCP names.
- The consulting graph catches up on container startup.
- Supported consulting file changes appear automatically within the polling interval plus embedding time.
- Deleted, moved, and excluded files disappear automatically.
- Unchanged files are not re-embedded.
- Synchronization continues with no MCP client connected.
- The consulting MCP returns current results from the continuously maintained database.
