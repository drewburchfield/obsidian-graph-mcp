-- Obsidian Graph MCP Server - PostgreSQL Schema
--
-- This schema is designed for storing whole Obsidian notes (not chunked documents)
-- with vector embeddings for semantic search and graph analysis.

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main notes table with vector embeddings
-- Supports chunking for large notes (voyage-context-3 pattern)
CREATE TABLE IF NOT EXISTS notes (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL,                      -- Vault-relative path (can have multiple chunks)
    title TEXT NOT NULL,                     -- Note title (from filename or frontmatter)
    content TEXT NOT NULL,                   -- Chunk content (or full note if unchunked)
    embedding vector(1024),                  -- Voyage Context-3 embedding (1024 dimensions)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE,
    file_size_bytes INTEGER,

    -- Chunking support (for notes >32k tokens)
    chunk_index INTEGER DEFAULT 0,           -- Chunk number within note (0 for whole notes)
    total_chunks INTEGER DEFAULT 1,          -- Total chunks for this note (1 for whole notes)

    -- Materialized statistics for performance optimization
    connection_count INTEGER DEFAULT 0,      -- Cached count for hub/orphan queries
    last_indexed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Composite unique constraint for path + chunk
    UNIQUE(path, chunk_index)
);

-- HNSW index for fast cosine similarity search
-- Configuration: m=16 (connections per layer), ef_construction=64 (build-time accuracy)
CREATE INDEX IF NOT EXISTS idx_notes_embedding_cosine
    ON notes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Standard B-tree indexes for common queries
CREATE INDEX IF NOT EXISTS idx_notes_path ON notes(path);
CREATE INDEX IF NOT EXISTS idx_notes_modified_at ON notes(modified_at);
CREATE INDEX IF NOT EXISTS idx_notes_connection_count ON notes(connection_count DESC);
CREATE INDEX IF NOT EXISTS idx_notes_last_indexed_at ON notes(last_indexed_at);
CREATE INDEX IF NOT EXISTS idx_notes_chunk_index ON notes(chunk_index);

-- Function to update modified_at timestamp automatically
CREATE OR REPLACE FUNCTION update_modified_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.modified_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update modified_at on note updates
CREATE TRIGGER trigger_update_notes_modified_at
    BEFORE UPDATE ON notes
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_at();
