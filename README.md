# Obsidian Graph MCP Server

[![CI](https://github.com/drewburchfield/obsidian-graph-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/drewburchfield/obsidian-graph-mcp/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)

Semantic knowledge graph navigation for Obsidian vaults using AI-powered vector embeddings and PostgreSQL+pgvector.

## Overview

This MCP server enables AI-powered semantic analysis of your Obsidian vault, discovering hidden connections and relationships between notes through vector embeddings. Unlike keyword search or explicit links, it finds conceptual similarity and builds multi-hop knowledge graphs.

**Use case:** Powers the `/find-connections` slash command for discovering non-obvious relationships, bridge notes, and conceptual clusters in your knowledge base.

## Features

- **Semantic Search**: Find notes by meaning, not just keywords
- **Connection Discovery**: Multi-hop BFS graph traversal to map note relationships
- **Hub Analysis**: Identify highly connected conceptual anchors (MOC candidates)
- **Orphan Detection**: Find isolated insights that need integration
- **Auto-Indexing**: Automatic file watching with 30-second debounce
- **Superior Quality**: Voyage Context-3 (1024d) vs typical 384d embeddings

## Architecture

- **Embeddings**: Voyage Context-3 (1024 dimensions, contextualized)
- **Vector Store**: PostgreSQL 15+ with pgvector HNSW indexing
- **Performance**: 0.9ms search (555x better than target), <2s graph building
- **File Watching**: Watchdog with 30-second debounce for batch edits
- **Transport**: Docker stdio for MCP communication

## MCP Tools

### Overview

All tools use **semantic similarity** via 1024-dimensional Voyage Context-3 embeddings. Similarity scores range from 0.0 (unrelated) to 1.0 (identical). Default threshold is 0.5 (clear connection).

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
- Generates query embedding using Voyage Context-3
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
- Voyage Context-3 maintains semantic coherence across chunks
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
4. **Free tier**: Voyage Context-3 includes 200M free tokens (one-time per account):
   - First 200M tokens are FREE
   - Sufficient for indexing ~50,000 notes
   - After free tier: ~$0.12 per 1M tokens

**Cost estimate**: Indexing 1,000 notes ≈ 4M tokens ≈ **$0.48** (or free if within 200M token limit)

## Installation

### Integrated Mode (Part of master_mcp stack)

1. **Configure environment**:
```bash
cd /path/to/master_mcp
cp configs/obsidian-graph/.env.instance.example configs/obsidian-graph/.env.instance

# Edit .env.instance and add your Voyage API key from step 2 above
# VOYAGE_API_KEY=pa-your-actual-key-here
```

2. **Generate secure database password**:
```bash
cd servers/obsidian-graph
./scripts/generate-db-password.sh
```
This creates a random 48-character password and configures docker-compose.override.yml.

3. **Update vault path** (`docker-compose.override.yml`):
```yaml
services:
  obsidian-graph:
    volumes:
      - /path/to/your/obsidian/vault:/vault:ro
```

4. **Start services**:
```bash
./mcp-servers.sh start-obsidian-graph
```

5. **Initial indexing** (first time):
```bash
docker exec -i mcp-obsidian-graph python -m src.indexer
```
Indexes entire vault (30-60 min for large vaults). After this, file watching handles incremental updates.

6. **Add to MCP client** (`~/.mcp.json`):
```json
{
  "mcpServers": {
    "obsidian-graph": {
      "command": "docker",
      "args": ["exec", "-i", "mcp-obsidian-graph", "python", "-m", "src.server"],
      "disabled": false
    }
  }
}
```

### Standalone Mode (Independent deployment)

1. **Copy server directory**:
```bash
cp -r /path/to/master_mcp/servers/obsidian-graph ./
cd obsidian-graph
```

2. **Create docker-compose.yml**:
```yaml
version: '3.8'

services:
  obsidian-graph:
    build: .
    container_name: obsidian-graph-mcp
    stdin_open: true
    tty: true
    env_file:
      - .env
    volumes:
      - ${OBSIDIAN_VAULT_PATH}:/vault:ro
    depends_on:
      - postgres

  postgres:
    image: pgvector/pgvector:pg15
    container_name: obsidian-graph-postgres
    environment:
      POSTGRES_DB: obsidian_graph
      POSTGRES_USER: obsidian
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./src/schema.sql:/docker-entrypoint-initdb.d/schema.sql:ro

volumes:
  postgres-data:
```

3. **Configure** `.env`:
```bash
cp .env.instance.example .env
# Edit with your settings
```

4. **Start**: `docker-compose up -d`

## Configuration

### Required Environment Variables

```bash
# Voyage AI
VOYAGE_API_KEY=your_key_here  # Get from https://www.voyageai.com/

# PostgreSQL (defaults work for integrated mode)
POSTGRES_HOST=obsidian-postgres
POSTGRES_PASSWORD=your_generated_password_here  # Generated by ./scripts/generate-db-password.sh

# Obsidian Vault
OBSIDIAN_VAULT_PATH=/vault  # Inside container
```

### Optional Tuning

```bash
# File watching
OBSIDIAN_WATCH_ENABLED=true
OBSIDIAN_DEBOUNCE_SECONDS=30

# Performance
POSTGRES_MIN_CONNECTIONS=5
POSTGRES_MAX_CONNECTIONS=20
EMBEDDING_BATCH_SIZE=128
EMBEDDING_REQUESTS_PER_MINUTE=300

# HNSW index (advanced)
HNSW_M=16
HNSW_EF_CONSTRUCTION=64
```

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
| Embedding quality | 1024-dim | 1024-dim | ✅ Voyage Context-3 |

**Performance Note**: Metrics measured on development vault (~500 notes, M1 MacBook Pro). Actual performance depends on vault size, hardware (CPU/RAM/SSD), and database configuration. HNSW indexing provides O(log n) search, so performance degrades gracefully with vault size.

## Troubleshooting

### "Reduced rate limits of 3 RPM"
- **Cause**: No payment method on Voyage account
- **Solution**: Add payment method at https://dashboard.voyageai.com/
- **Note**: 200M free tokens still apply

### "PostgreSQL connection failed"
```bash
# Check postgres container
docker ps | grep obsidian-postgres
docker logs obsidian-postgres

# Verify credentials
grep POSTGRES_ configs/obsidian-graph/.env.instance
```

### "Note not found" errors
- Ensure initial indexing completed: `docker exec -i mcp-obsidian-graph python -m src.indexer`
- Check vault path is mounted: `docker exec -i mcp-obsidian-graph ls /vault`

### File changes not detected
- Verify `OBSIDIAN_WATCH_ENABLED=true`
- Check logs: `docker logs mcp-obsidian-graph`
- File watcher starts after PostgreSQL connection

## Development

### Running Tests

```bash
# Quick validation
docker exec -i mcp-obsidian-graph python test_e2e.py

# Unit tests (requires 300 RPM rate limits)
docker exec -i mcp-obsidian-graph pytest tests/ -v
```

### Rebuilding

```bash
docker-compose build obsidian-graph
docker-compose restart obsidian-graph
```

### Debugging

```bash
# View logs
docker logs -f mcp-obsidian-graph

# Interactive shell
docker exec -it mcp-obsidian-graph /bin/bash

# Check database
docker exec -it obsidian-postgres psql -U obsidian -d obsidian_graph
```

## Comparison to mcp-obsidian

| Feature | mcp-obsidian | obsidian-graph |
|---------|--------------|----------------|
| Embeddings | 384-dim (all-MiniLM-L6-v2) | 1024-dim (Voyage Context-3) |
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
