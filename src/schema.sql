-- Obsidian Graph - PostgreSQL Schema
--
-- Stores notes (whole or chunked) with vector embeddings for
-- semantic search, graph analysis, and hub/orphan detection.

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main notes table with vector embeddings
-- Supports chunking for large notes (voyage-context-4 pattern)
CREATE TABLE IF NOT EXISTS notes (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL,                      -- Vault-relative path (can have multiple chunks)
    title TEXT NOT NULL,                     -- Note title (from filename or frontmatter)
    content TEXT NOT NULL,                   -- Chunk content (or full note if unchunked)
    embedding vector(1024),                  -- Voyage Context-4 embedding (1024 dimensions)
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

-- Migration: remove trigger that overwrites file mtime with DB timestamp
-- Must be after CREATE TABLE so the table exists on fresh databases
DROP TRIGGER IF EXISTS trigger_update_notes_modified_at ON notes;
DROP FUNCTION IF EXISTS update_modified_at();

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
