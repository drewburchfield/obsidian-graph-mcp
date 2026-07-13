#!/usr/bin/env python3
"""Build a Matryoshka-truncated copy of the consulting_graph notes table.

Reads every row from the live ``consulting_graph.notes`` table (qwen3-embedding-8b
4096-dim vectors), truncates each embedding to its first ``--dims`` components,
L2-renormalizes it, and writes the result into a scratch database with a
``vector(--dims)`` column. Because qwen3-embedding is MRL-trained, a truncated +
renormalized prefix is itself a valid lower-dimensional embedding.

No embedding API is ever called: every scratch vector is derived purely from the
stored 4096-dim vector. The live database is only ever read, never written.

Usage (inside the ``consulting-graph`` app container venv):

    /app/.venv/bin/python make_mrl_index.py --dims 1024 --target-db consulting_eval_1024 --build-hnsw
    /app/.venv/bin/python make_mrl_index.py --dims 2048 --target-db consulting_eval_2048 --build-hnsw

Connection params come from the standard POSTGRES_* env vars (same pgvector
instance hosts source and scratch DBs). ``--build-hnsw`` attempts an HNSW index;
pgvector refuses vector columns above 2000 dims, which is captured and reported.
"""

from __future__ import annotations

import argparse
import asyncio
import os

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

SOURCE_DB = os.environ.get("SOURCE_DB", "consulting_graph")
BATCH = int(os.environ.get("COPY_BATCH", "1000"))

# Columns copied verbatim from the source (id preserved so rows line up across
# databases for validation). ``fts`` is a generated column and regenerates
# itself on insert; ``embedding`` is truncated + renormalized below.
COPY_COLS = [
    "id",
    "path",
    "title",
    "content",
    "created_at",
    "modified_at",
    "file_size_bytes",
    "chunk_index",
    "total_chunks",
    "connection_count",
    "last_indexed_at",
]

CREATE_NOTES = """
CREATE TABLE notes (
    id               integer PRIMARY KEY,
    path             text NOT NULL,
    title            text NOT NULL,
    content          text NOT NULL,
    created_at       timestamptz DEFAULT CURRENT_TIMESTAMP,
    modified_at      timestamptz,
    file_size_bytes  integer,
    chunk_index      integer DEFAULT 0,
    total_chunks     integer DEFAULT 1,
    connection_count integer DEFAULT 0,
    last_indexed_at  timestamptz DEFAULT CURRENT_TIMESTAMP,
    fts              tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding        vector({dims}),
    CONSTRAINT notes_path_chunk_index_key UNIQUE (path, chunk_index)
);
"""

CREATE_META = """
CREATE TABLE index_metadata (
    key        text PRIMARY KEY,
    value      text NOT NULL,
    updated_at timestamptz DEFAULT CURRENT_TIMESTAMP
);
"""


def conn_kwargs(database: str) -> dict:
    return dict(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        database=database,
    )


async def ensure_database(target_db: str) -> None:
    admin = await asyncpg.connect(**conn_kwargs("postgres"))
    try:
        exists = await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", target_db)
        if not exists:
            # CREATE DATABASE cannot run inside a transaction block.
            await admin.execute(f'CREATE DATABASE "{target_db}"')
            print(f"created database {target_db}")
        else:
            print(f"database {target_db} already exists")
    finally:
        await admin.close()


async def build_schema(target_db: str, dims: int) -> None:
    conn = await asyncpg.connect(**conn_kwargs(target_db))
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("DROP TABLE IF EXISTS notes")
        await conn.execute("DROP TABLE IF EXISTS index_metadata")
        await conn.execute(CREATE_NOTES.format(dims=dims))
        await conn.execute(CREATE_META)
        print(f"created notes(vector({dims})) + index_metadata in {target_db}")
    finally:
        await conn.close()


async def copy_rows(target_db: str, dims: int) -> int:
    src = await asyncpg.connect(**conn_kwargs(SOURCE_DB))
    dst = await asyncpg.connect(**conn_kwargs(target_db))
    await register_vector(src)
    await register_vector(dst)
    total = 0
    try:
        select_sql = (
            f"SELECT {', '.join(COPY_COLS)}, embedding FROM notes "
            f"WHERE id > $1 ORDER BY id ASC LIMIT {BATCH}"
        )
        insert_sql = (
            f"INSERT INTO notes ({', '.join(COPY_COLS)}, embedding) VALUES ("
            + ", ".join(f"${i}" for i in range(1, len(COPY_COLS) + 2))
            + ")"
        )
        last_id = 0
        while True:
            rows = await src.fetch(select_sql, last_id)
            if not rows:
                break
            records = []
            for r in rows:
                emb = np.asarray(r["embedding"], dtype=np.float32)
                trunc = emb[:dims]
                norm = float(np.linalg.norm(trunc))
                if norm > 0.0:
                    trunc = trunc / norm
                records.append(tuple(r[c] for c in COPY_COLS) + (trunc,))
            await dst.executemany(insert_sql, records)
            total += len(rows)
            last_id = rows[-1]["id"]
            print(f"  copied {total} rows (last id {last_id})")
        return total
    finally:
        await src.close()
        await dst.close()


async def create_secondary_indexes(target_db: str) -> None:
    conn = await asyncpg.connect(**conn_kwargs(target_db))
    try:
        await conn.execute("CREATE INDEX idx_notes_fts ON notes USING gin (fts)")
        await conn.execute("CREATE INDEX idx_notes_path ON notes USING btree (path)")
        print("created idx_notes_fts (gin), idx_notes_path (btree)")
    finally:
        await conn.close()


async def build_hnsw(target_db: str) -> None:
    conn = await asyncpg.connect(**conn_kwargs(target_db))
    try:
        await conn.execute(
            "CREATE INDEX idx_notes_embedding_hnsw ON notes "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
        print("HNSW index built successfully")
    except Exception as exc:  # noqa: BLE001 - we want the exact pgvector error text
        print(f"HNSW index FAILED: {type(exc).__name__}: {exc}")
    finally:
        await conn.close()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, required=True)
    ap.add_argument("--target-db", required=True)
    ap.add_argument("--build-hnsw", action="store_true")
    args = ap.parse_args()

    await ensure_database(args.target_db)
    await build_schema(args.target_db, args.dims)
    n = await copy_rows(args.target_db, args.dims)
    await create_secondary_indexes(args.target_db)
    print(f"copied {n} rows into {args.target_db}")
    if args.build_hnsw:
        await build_hnsw(args.target_db)


if __name__ == "__main__":
    asyncio.run(main())
