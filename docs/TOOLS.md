# MCP Tools Reference

Complete documentation for all 5 Obsidian Graph MCP tools.

## Overview

These tools enable semantic knowledge graph navigation through your Obsidian vault using AI-powered vector embeddings. They discover hidden connections, identify conceptual hubs, and map multi-hop relationships between notes.

---

## 1. search_notes

**Purpose:** Semantic search across your entire vault using natural language queries.

**Use Case:** Find notes by concept/meaning rather than exact keyword matching.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | (required) | Natural language search query |
| `limit` | integer | 10 | Maximum results (1-50) |
| `threshold` | number | 0.5 | Minimum similarity score (0.0-1.0) |

### Returns

```json
[
  {
    "path": "neuroscience/dopamine.md",
    "title": "Dopamine",
    "similarity": 0.847,
    "snippet": "Dopamine is a neurotransmitter that plays a role in reward and motivation..."
  }
]
```

### Examples

```javascript
// Find notes about a concept
search_notes({
  query: "reward systems and motivation",
  limit: 10,
  threshold: 0.5
})

// Broad exploration (lower threshold)
search_notes({
  query: "decision making",
  limit: 20,
  threshold: 0.3
})
```

### Performance
- **Target:** <500ms
- **Actual:** <2ms with HNSW indexing
- **Scales to:** 10,000+ notes

---

## 2. get_similar_notes

**Purpose:** Find notes semantically similar to a specific note.

**Use Case:** Discover related ideas, explore conceptual neighborhoods, find notes to link together.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `note_path` | string | (required) | Vault-relative path to source note |
| `limit` | integer | 10 | Maximum results (1-50) |
| `threshold` | number | 0.5 | Minimum similarity score (0.0-1.0) |

### Returns

```json
[
  {
    "path": "psychology/reward-systems.md",
    "title": "Reward Systems",
    "similarity": 0.812,
    "connection_type": "semantic"
  }
]
```

### Examples

```javascript
// Find notes related to a specific note
get_similar_notes({
  note_path: "neuroscience/dopamine.md",
  limit: 10,
  threshold: 0.6
})

// Discover weak connections (exploration)
get_similar_notes({
  note_path: "philosophy/free-will.md",
  limit: 20,
  threshold: 0.4
})
```

### Behavior
- **Excludes:** Source note itself
- **Sorted by:** Similarity score (descending)
- **Performance:** <300ms

---

## 3. get_connection_graph

**Purpose:** Build multi-hop connection graph using BFS traversal to map knowledge networks.

**Use Case:** Discover how notes connect through multiple degrees of separation, find bridge notes connecting disparate clusters.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `note_path` | string | (required) | Starting note path |
| `depth` | integer | 3 | Maximum levels to traverse (1-5) |
| `max_per_level` | integer | 5 | Maximum nodes per level (1-10) |
| `threshold` | number | 0.5 | Minimum similarity score (0.0-1.0) |

### Returns

```json
{
  "root": {
    "path": "neuroscience/dopamine.md",
    "title": "Dopamine"
  },
  "nodes": [
    {
      "path": "psychology/reward-systems.md",
      "title": "Reward Systems",
      "level": 1,
      "parent_path": "neuroscience/dopamine.md"
    }
  ],
  "edges": [
    {
      "source": "neuroscience/dopamine.md",
      "target": "psychology/reward-systems.md",
      "similarity": 0.812
    }
  ],
  "stats": {
    "total_nodes": 15,
    "total_edges": 14,
    "levels": 3
  }
}
```

### Examples

```javascript
// Map 3-level network
get_connection_graph({
  note_path": "philosophy/free-will.md",
  depth: 3,
  max_per_level: 5,
  threshold: 0.65
})

// Quick 2-level exploration
get_connection_graph({
  note_path: "ml/neural-networks.md",
  depth: 2,
  max_per_level: 7
})
```

### Algorithm
- **Method:** Breadth-First Search (BFS)
- **Cycle Prevention:** Tracks visited nodes
- **Level-by-level:** Prevents deep rabbit holes
- **Performance:** <2s for depth=3, max_per_level=5

---

## 4. get_hub_notes

**Purpose:** Identify highly connected notes that serve as conceptual anchors.

**Use Case:** Find MOC (Map of Content) candidates, discover central themes, identify knowledge graph hubs.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_connections` | integer | 10 | Minimum connection count to qualify as hub |
| `threshold` | number | 0.5 | Similarity threshold for counting connections |
| `limit` | integer | 20 | Maximum results (1-50) |

### Returns

```json
[
  {
    "path": "concepts/decision-making.md",
    "title": "Decision Making",
    "connection_count": 28
  }
]
```

### Examples

```javascript
// Find major hubs
get_hub_notes({
  min_connections: 10,
  threshold: 0.5,
  limit: 20
})

// Find moderate hubs (looser connections)
get_hub_notes({
  min_connections: 5,
  threshold: 0.4
})
```

### Notes
- Uses materialized `connection_count` column for O(1) performance
- Background refresh triggered when >50% counts are stale
- Sorted by connection count (descending)

---

## 5. get_orphaned_notes

**Purpose:** Find isolated notes with few connections to the knowledge graph.

**Use Case:** Discover unintegrated insights, identify notes that need linking, find recent ideas not yet connected.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_connections` | integer | 2 | Maximum connections to qualify as orphan |
| `threshold` | number | 0.5 | Similarity threshold for counting |
| `limit` | integer | 20 | Maximum results (1-50) |

### Returns

```json
[
  {
    "path": "ideas/new-framework.md",
    "title": "New Framework Idea",
    "connection_count": 1,
    "modified_at": "2025-12-17T18:30:00Z"
  }
]
```

### Examples

```javascript
// Find truly isolated notes
get_orphaned_notes({
  max_connections: 0,
  limit: 10
})

// Find weakly connected notes
get_orphaned_notes({
  max_connections: 3,
  threshold: 0.6
})
```

### Notes
- Sorted by: connection_count ASC, then modified_at DESC
- Shows recent orphans first (likely new insights)
- Uses materialized stats for performance

---

## Chunking Behavior

Notes exceeding 32,000 tokens are automatically chunked:
- **Chunk size:** 2000 characters
- **Overlap:** 0 (voyage-context-3 maintains context)
- **Storage:** Each chunk as separate row with `chunk_index`
- **Search:** Returns individual chunks
- **Graph:** Chunks treated as separate nodes

Most Obsidian notes are <10k tokens and stored whole.

---

## Similarity Scores

All tools return similarity scores in `[0.0, 1.0]` range:

| Score | Meaning | Use Case |
|-------|---------|----------|
| 0.85-1.0 | Nearly identical | Duplicates, very closely related |
| 0.75-0.85 | Highly related | Strong conceptual connection |
| 0.65-0.75 | Clear connection | Related topics, shared themes |
| 0.50-0.65 | Interesting relationship | Worth exploring |
| <0.50 | Weak/spurious | Likely not meaningful |

---

## Integration with /find-connections

The `/find-connections` slash command uses these tools in sequence:

1. **Phase 1:** `search_notes` - Find anchor note
2. **Phase 2:** `get_similar_notes` - Immediate network mapping
3. **Phase 3:** `get_connection_graph` - Deep network analysis
4. **Phase 4:** Pattern analysis using hub/orphan data
5. **Phase 5:** Synthesis and insights

All tools return results compatible with the command's expected format.

---

## Performance Benchmarks

Tested with 88-note vault:

| Tool | Operations | Latency | Status |
|------|-----------|---------|--------|
| search_notes | Vector search | <2ms | ✅ 250x better than target |
| get_similar_notes | Vector lookup + search | <2ms | ✅ 150x better |
| get_connection_graph | BFS (depth=2, max=3) | <1s | ✅ 2x better |
| get_hub_notes | Materialized query | <100ms | ✅ On target |
| get_orphaned_notes | Materialized query | <100ms | ✅ On target |

---

## Troubleshooting

### "No results found"
- Check threshold (try lowering to 0.3)
- Verify note exists: `docker exec obsidian-graph-pgvector psql -U obsidian -d obsidian_graph -c "SELECT path FROM notes LIMIT 10"`

### "Hub detection returns 0 results"
- Connection counts may be stale (first query triggers background refresh)
- Wait 10 seconds and query again
- Lower min_connections parameter

### "Connection graph seems incomplete"
- Increase `max_per_level` (default: 5)
- Lower `threshold` for more connections
- Check that related notes are indexed

---

## API Examples (Direct MCP Calls)

### Using with Claude Desktop

Add to `~/.mcp.json`:
```json
{
  "mcpServers": {
    "obsidian-graph": {
      "command": "docker",
      "args": ["exec", "-i", "mcp-obsidian-graph", "python", "-m", "src.server"]
    }
  }
}
```

Then use in conversation:
```
Can you search my vault for notes about reward systems?
→ Uses search_notes tool

What notes are similar to my dopamine note?
→ Uses get_similar_notes tool

Map the connection network around free will
→ Uses get_connection_graph tool

Which notes are hubs in my knowledge graph?
→ Uses get_hub_notes tool

What recent notes haven't been connected yet?
→ Uses get_orphaned_notes tool
```

---

## Advanced Usage

### Combining Tools

```javascript
// Find hubs, then map their networks
const hubs = get_hub_notes({min_connections: 10})
for (const hub of hubs) {
  const graph = get_connection_graph({
    note_path: hub.path,
    depth: 2
  })
  // Analyze hub's neighborhood
}

// Find orphans, then discover what they should connect to
const orphans = get_orphaned_notes({max_connections: 1})
for (const orphan of orphans) {
  const similar = get_similar_notes({
    note_path: orphan.path,
    threshold: 0.7
  })
  // Suggest connections for orphan
}
```

### Threshold Tuning

Start high, go lower:
1. **0.75+** - Only very strong connections (precise)
2. **0.60-0.75** - Clear connections (balanced)
3. **0.50-0.60** - Interesting relationships (exploratory)
4. **<0.50** - Weak connections (noisy, use sparingly)

---

## Limitations

1. **Oversized notes:** Notes >32k tokens (~120k chars) currently skipped
   - Planned: Automatic chunking for these notes
   - Workaround: Manually split large notes

2. **Empty notes:** Skipped during indexing
   - Warning logged with file path

3. **First query latency:** Hub/orphan tools trigger background refresh on first use
   - Subsequent queries use cached counts
   - <100ms after initial refresh

---

## See Also

- [README.md](../README.md) - Installation and setup
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guidelines
- [CHANGELOG.md](../CHANGELOG.md) - Version history
- `/find-connections` command in knowledge vault
