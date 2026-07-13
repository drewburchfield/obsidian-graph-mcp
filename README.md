[![Obsidian Graph](https://ghrb.waren.build/banner?header=obsidian-graph%20![obsidian]&subheader=Semantic%20knowledge%20graph%20engine%20for%20markdown%20vaults&bg=0a1628&secondaryBg=1e3a5f&color=e8f0fe&subheaderColor=7eb8da&headerFont=Inter&subheaderFont=Inter&support=false)](https://github.com/drewburchfield/obsidian-graph)

[![CI](https://github.com/drewburchfield/obsidian-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/drewburchfield/obsidian-graph/actions/workflows/ci.yml) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/drewburchfield/obsidian-graph)

Semantic knowledge graph engine for markdown vaults. Discovers hidden connections between notes using AI-powered vector embeddings and PostgreSQL+pgvector. Accessible to any AI app or harness compatible with MCP.

## Overview

Obsidian Graph builds a semantic knowledge graph of your markdown vault, discovering relationships between notes that go beyond keywords and explicit links. It embeds your notes as vectors using Voyage Context-4, stores them in PostgreSQL+pgvector, and provides tools for semantic search, multi-hop graph traversal, hub detection, and orphan analysis.

Designed for Obsidian vaults but works with any folder of markdown files. Connects to any AI app or harness compatible with the Model Context Protocol (MCP).

## Features

- **Semantic Search**: Find notes by meaning, not just keywords
- **Connection Discovery**: Multi-hop BFS graph traversal to map note relationships
- **Hub Analysis**: Identify highly connected conceptual anchors (MOC candidates)
- **Orphan Detection**: Find isolated insights that need integration
- **Auto-Indexing**: Automatic file watching with 30-second debounce
- **Superior Quality**: Voyage Context-4 (1024d) vs typical 384d embeddings

## Architecture

```
┌─ obsidian-graph container ─────────────────┐
│                                            │
│  MCP Client ◄──stdio──► server.py          │
│                            │               │
│                     ┌──────┴──────┐        │
│                     ▼             ▼        │
│              graph_builder   hub_analyzer  │
│              embedder.py     file_watcher  │
│                  │               │         │
│                  │ HTTPS         │ watch   │
│                  ▼               ▼         │
│            Voyage AI API    /vault (ro)    │
│                  │                         │
│                  │ 1024d vectors           │
│                  ▼                         │
│           vector_store.py                  │
│                  │                         │
└──────────────────┼─────────────────────────┘
                   │ SQL
                   ▼
┌─ obsidian-graph-pgvector container ────────┐
│  PostgreSQL 15 + pgvector (HNSW index)     │
└────────────────────────────────────────────┘
```

- **Embeddings**: Voyage Context-4 (1024 dimensions, contextualized)
- **Vector Store**: PostgreSQL 15+ with pgvector HNSW indexing
- **Performance**: 0.9ms search (555x better than target), <2s graph building
- **File Watching**: Watchdog with polling mode for cloud sync compatibility
- **Transport**: Docker stdio for MCP communication

## MCP Tools

### Overview

All tools use **semantic similarity** via 1024-dimensional Voyage Context-4 embeddings. Similarity scores range from 0.0 (unrelated) to 1.0 (identical). Default threshold is 0.5 (clear connection).

**How it works:**
1. Notes are embedded as vectors in 1024-dimensional space
2. Cosine similarity measures semantic closeness between vectors
3. HNSW index enables sub-millisecond vector search
4. Results ranked by similarity score (0.0-1.0)

### Tool Reference

| Tool | Purpose | Method | Performance | Use Case |
|------|---------|--------|------------|----------|
| `search_notes` | Semantic search across vault | Query embedding → vector search | <1ms | Find notes by concept |
| `get_similar_notes` | Find notes similar to given note | Note embedding → vector search | <300ms | Discover related ideas |
| `get_connection_graph` | Multi-hop BFS graph traversal | Recursive similarity search | <2s | Map knowledge networks |
| `get_hub_notes` | Identify highly connected notes | Materialized connection counts | <100ms | Find conceptual anchors |
| `get_orphaned_notes` | Find isolated notes | Materialized connection counts | <100ms | Unintegrated insights |

### Methodology Details

**search_notes:**
- Generates query embedding using Voyage Context-4
- Performs cosine similarity search against all note embeddings
- Returns top-k most similar notes above threshold
- HNSW index enables O(log n) search complexity

**get_similar_notes:**
- Fetches source note's embedding from database
- Searches for notes with similar embeddings
- Excludes source note from results
- Useful for exploring conceptual neighborhoods

**get_connection_graph:**
- Uses Breadth-First Search (BFS) for level-by-level exploration
- Prevents cycles by tracking visited nodes
- Builds multi-hop network (depth 1-5 levels)
- Each level: finds top-k most similar notes from previous level
- Returns: nodes (with level), edges (with similarity), stats

**get_hub_notes:**
- Uses materialized `connection_count` column (O(1) query)
- Connection count = # of notes above threshold similarity
- Background refresh when >50% of counts are stale
- Identifies notes with many semantic connections
- High hub scores → good MOC (Map of Content) candidates

**get_orphaned_notes:**
- Uses materialized `connection_count` column
- Finds notes with few semantic connections
- Sorted by: connection count (ASC), modified date (DESC)
- Shows recent notes first (likely new insights)
- Helps identify notes needing integration

### Chunking Support

**For large notes (>30k tokens):**
- Automatically split into sentence-aligned chunks (target: ~2000 characters, 0 overlap)
- Chunking algorithm breaks at sentence boundaries (`. ` or `\n\n`) for readability
- Chunk sizes vary (1800-2200 chars) to preserve sentence integrity
- Embedded in batches of 60 chunks (preserves context)
- Voyage Context-4 maintains semantic coherence across chunks
- Each chunk stored separately with `chunk_index`
- Search returns individual chunks (can aggregate by path)

**Example:** 168k-char note → ~87 variable-sized chunks → 2 batches (60+27) → context preserved

Most Obsidian notes are <10k tokens and embedded whole (single chunk).

## Prerequisites

### Voyage AI Account Setup

This server requires a Voyage AI API key for generating embeddings:

1. **Create account**: Sign up at https://www.voyageai.com/
2. **Get API key**: Visit https://dashboard.voyageai.com/ → API Keys → Create new key
3. **Add payment method** (Important!):
   - Go to https://dashboard.voyageai.com/billing
   - Add a payment method (credit card)
   - **Why**: Without payment, rate limit is only 3 RPM (unusable)
   - **With payment**: 300 RPM rate limit unlocked
4. **Free tier**: Voyage Context-4 includes 200M free tokens (one-time per account, per model):
   - First 200M tokens are FREE
   - Sufficient for indexing ~50,000 notes
   - After free tier: ~$0.12 per 1M tokens

**Cost estimate**: Indexing 1,000 notes ≈ 4M tokens ≈ **$0.48** (or free if within 200M token limit)

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/drewburchfield/obsidian-graph.git
cd obsidian-graph
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your settings:
#   - VOYAGE_API_KEY (from https://dashboard.voyageai.com/)
#   - OBSIDIAN_VAULT_PATH (absolute path to your vault)
#   - POSTGRES_PASSWORD (generate with: openssl rand -base64 36)
```

3. **Start services**:
```bash
docker-compose up -d
```

4. **Initial indexing** (first time only):
```bash
docker exec -i obsidian-graph .venv/bin/python -m src.indexer
```
Indexes entire vault (30-60 min for large vaults). After this, file watching handles incremental updates.

5. **Add to MCP client** (`~/.mcp.json`):
```json
{
  "mcpServers": {
    "obsidian-graph": {
      "command": "docker",
      "args": ["exec", "-i", "-e", "OBSIDIAN_WATCH_ENABLED=false", "obsidian-graph", ".venv/bin/python", "-m", "src.server"],
      "disabled": false
    }
  }
}
```
The watcher is disabled for MCP sessions because the main container process already watches the vault; without this, every connected client would spawn a duplicate watcher. `scripts/run_vault_mcp.sh` wraps the same command if you prefer pointing your client at a script.

## Configuration

### Required Environment Variables

```bash
# Voyage AI
VOYAGE_API_KEY=your_key_here  # Get from https://www.voyageai.com/

# PostgreSQL (POSTGRES_HOST is set by docker-compose.yml, no need to set in .env)
POSTGRES_PASSWORD=your_secure_password_here  # Generate with: openssl rand -base64 36

# Obsidian Vault
OBSIDIAN_VAULT_PATH=/path/to/your/vault  # Absolute path on your system
```

### Optional Tuning

```bash
# File watching
OBSIDIAN_WATCH_ENABLED=true
OBSIDIAN_DEBOUNCE_SECONDS=30

# Polling mode (auto-enabled for Docker and cloud-synced vaults)
# OBSIDIAN_WATCH_USE_POLLING=       # true | false (unset = auto-detect)
# OBSIDIAN_WATCH_POLLING_INTERVAL=30  # seconds between polls (default: 30)

# Performance
POSTGRES_MIN_CONNECTIONS=5
POSTGRES_MAX_CONNECTIONS=20
EMBEDDING_BATCH_SIZE=128
EMBEDDING_REQUESTS_PER_MINUTE=300
```

### Cloud Sync Support (iCloud, Google Drive, Dropbox, OneDrive)

If your Obsidian vault is stored in a cloud-synced folder, the file watcher automatically uses **polling mode** for reliable change detection. This is because Docker's filesystem events don't propagate reliably through cloud sync virtualization layers.

**Auto-detection:** Polling mode is automatically enabled when:
- Running inside Docker (always uses polling for reliability)
- Vault path contains cloud sync patterns (`Library/Mobile Documents`, `Library/CloudStorage`, etc.)

**How it works:**
- Polling mode compares directory snapshots every 30 seconds (configurable)
- Detects file creates, modifications, moves, and deletions
- Slightly higher CPU than native filesystem events, but works reliably everywhere

**Mobile workflow:** Edit notes on mobile (iOS/Android) via Obsidian's iCloud/Google Drive sync. Changes sync to your Mac, and the polling watcher detects them within the polling interval.

**Override behavior:**
```bash
# Force polling on (for edge cases)
OBSIDIAN_WATCH_USE_POLLING=true

# Force native events (may miss changes with cloud sync)
OBSIDIAN_WATCH_USE_POLLING=false

# Faster detection (higher CPU)
OBSIDIAN_WATCH_POLLING_INTERVAL=15
```

### Excluding Folders from Indexing

By default, the indexer excludes common system and tool folders:
- `.obsidian/` / `.trash/` / `.Trash/` (Obsidian system)
- `.git/` / `.github/` (version control)
- `.vscode/` / `.cursor/` (editor config)
- `.claude/` / `.aider/` / `.smart-env/` (AI tools)

**Custom Exclusions**: To exclude additional folders (like a soft-delete folder), create `.obsidian-graph.conf` in your vault root:

```conf
# Exclude soft delete folder
07_Archive/Trash/

# Exclude drafts
drafts/
```

See [`docs/obsidian-graph.conf.example`](docs/obsidian-graph.conf.example) for more patterns and examples.

**Pattern Syntax:**

| Pattern | Matches |
|---------|---------|
| `folder/` | All files in `folder/` and subfolders |
| `drafts/*` | All files directly in `drafts/` |
| `*.tmp.md` | All files ending in `.tmp.md` |

## Security

This server implements multiple security layers to protect your vault:

- **Path Traversal Protection**: Validates all file paths stay within vault (`src/security_utils.py`)
- **Input Validation**: All parameters validated before processing (`src/validation.py`)
- **Secure Credentials**: Random generated database passwords (`scripts/generate-db-password.sh`)
- **Container Isolation**: Read-only vault mount, dropped capabilities, non-root user

**Concurrency**: See [docs/CONCURRENCY.md](docs/CONCURRENCY.md) for thread-safety guarantees and race condition prevention.

### Running Security Tests

```bash
# Security tests
pytest tests/test_security*.py -v

# Input validation tests
pytest tests/test_validation.py -v

# Race condition tests
pytest tests/test_race_conditions.py -v

# All tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## Usage Examples

### Semantic Search
```
search_notes(query="neural networks and consciousness", limit=10, threshold=0.5)

Returns notes semantically related to the query, even if they don't contain
the exact keywords.
```

### Find Similar Notes
```
get_similar_notes(note_path="neuroscience/dopamine.md", limit=10, threshold=0.6)

Discovers notes conceptually similar to dopamine note (might find:
reward-systems.md, motivation.md, decision-making.md)
```

### Build Connection Graph
```
get_connection_graph(
  note_path="philosophy/free-will.md",
  depth=3,
  max_per_level=5,
  threshold=0.65
)

Maps 3-level network showing how free-will connects to neuroscience,
psychology, and ethics notes through semantic similarity.
```

### Identify Hubs
```
get_hub_notes(min_connections=10, threshold=0.5, limit=20)

Finds notes with >=10 connections - candidates for Maps of Content (MOCs).
Example: "decision-making.md" might connect to psychology, neuroscience,
economics, and philosophy notes.
```

### Find Orphans
```
get_orphaned_notes(max_connections=2, limit=20)

Identifies isolated notes that need integration into knowledge graph.
Sorted by modification date to surface recent unconnected insights.
```

## Performance

Validated metrics:

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Search latency | <500ms | 0.9ms | ✅ 555x better |
| Graph building (depth=3) | <2s | <2s | ✅ On target |
| Hub/orphan queries | <100ms | <100ms | ✅ Materialized |
| Similarity range | [0.0-1.0] | [0.0-1.0] | ✅ Validated |
| Embedding quality | 1024-dim | 1024-dim | ✅ Voyage Context-4 |

**Performance Note**: Metrics measured on development vault (~500 notes, M1 MacBook Pro). Actual performance depends on vault size, hardware (CPU/RAM/SSD), and database configuration. HNSW indexing provides O(log n) search, so performance degrades gracefully with vault size.

## Troubleshooting

### "Reduced rate limits of 3 RPM"
- **Cause**: No payment method on Voyage account
- **Solution**: Add payment method at https://dashboard.voyageai.com/
- **Note**: 200M free tokens still apply

### "PostgreSQL connection failed"
```bash
# Check postgres container
docker ps | grep obsidian-graph-pgvector
docker logs obsidian-graph-pgvector

# Verify credentials
grep POSTGRES_ .env
```

### "Note not found" errors
- Ensure initial indexing completed: `docker exec -i obsidian-graph .venv/bin/python -m src.indexer`
- Check vault path is mounted: `docker exec -i obsidian-graph ls /vault`

### File changes not detected
- Verify `OBSIDIAN_WATCH_ENABLED=true`
- Check logs: `docker logs obsidian-graph`
- Look for: `Watching vault: /vault [polling (interval: 30s)]`
- File watcher starts after PostgreSQL connection
- **Cloud sync users**: Changes take up to polling interval (default 30s) plus cloud sync time
- **Reduce detection time**: Set `OBSIDIAN_WATCH_POLLING_INTERVAL=15` in `.env`

## Development

### Running Tests

Tests run on the host (the Docker image ships without dev dependencies or the `tests/` directory):

```bash
# Full test suite
uv sync --extra dev
uv run pytest tests/ -v

# Docker end-to-end tests (requires the stack running and VOYAGE_API_KEY)
uv run pytest tests/test_e2e_docker.py -v -s

# Standalone E2E regression suite against the running stack (read-only, 35 checks)
docker cp scripts/run_e2e_tests.py obsidian-graph:/tmp/
docker exec -w /app obsidian-graph .venv/bin/python /tmp/run_e2e_tests.py
```

### Rebuilding

```bash
docker-compose build obsidian-graph
docker-compose restart obsidian-graph
```

### Debugging

```bash
# View logs
docker logs -f obsidian-graph

# Interactive shell
docker exec -it obsidian-graph /bin/bash

# Check database
docker exec -it obsidian-graph-pgvector psql -U obsidian -d obsidian_graph
```

## Comparison to mcp-obsidian

| Feature | mcp-obsidian | obsidian-graph |
|---------|--------------|----------------|
| Embeddings | 384-dim (all-MiniLM-L6-v2) | 1024-dim (Voyage Context-4) |
| Vector Store | ChromaDB | PostgreSQL+pgvector |
| Tools | 2 (search, reindex) | 5 (search, similar, graph, hubs, orphans) |
| Search perf | Unknown | 0.9ms validated |
| Graph traversal | ❌ No | ✅ BFS with cycle prevention |
| Hub detection | ❌ No | ✅ Materialized stats |

## License

MIT License - Copyright (c) 2025 Drew Burchfield

See LICENSE file for details.

## Links

- **Voyage AI**: https://www.voyageai.com/
- **pgvector**: https://github.com/pgvector/pgvector
- **MCP Protocol**: https://modelcontextprotocol.io/
