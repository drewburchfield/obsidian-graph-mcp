#!/usr/bin/env bash
# Launch the MCP server bound to the CONSULTING graph (separate from the vault):
#   - embeddings from bigbot's MLX server (Qwen3-4B, 2560d) over the tailnet
#   - vectors in the consulting_graph Postgres (this Mac, :5434)
# Registered in Claude Code as the "consulting-graph" MCP server.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

# Secrets stay in .env (gitignored); everything else is explicit below.
POSTGRES_PASSWORD="$(grep -E '^POSTGRES_PASSWORD=' .env | head -1 | cut -d= -f2-)"
OPENROUTER_API_KEY="$(grep -E '^OPENROUTER_API_KEY=' .env | head -1 | cut -d= -f2-)"
export POSTGRES_PASSWORD OPENROUTER_API_KEY

export MCP_SERVER_NAME="consulting-graph"
# All-remote compute on OpenRouter (DB stays local). Qwen3-Embedding-8B (4096d)
# for embeds + Cohere rerank below. bigbot MLX no longer used.
export EMBEDDING_PROVIDER="openrouter"
export OPENROUTER_EMBED_MODEL="qwen/qwen3-embedding-8b"
export OPENROUTER_EMBED_DIMS="4096"
export EMBEDDING_DIMENSIONS="4096"                    # must match the vector(4096) column
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5434"
export POSTGRES_DB="consulting_graph"
export POSTGRES_USER="obsidian"
export OBSIDIAN_WATCH_ENABLED="false"                # query-only; reindex via index_consulting.py

# Verified retrieval pipeline: dense + BM25 hybrid (RRF) -> Cohere rerank -> top-N.
# Docs are prefix-contextualized at index time (scripts/reindex_prefix.py).
export CONSULTING_RERANK="1"
export RERANK_MODEL="cohere/rerank-v3.5"              # via OpenRouter (no local model)
export RERANK_POOL="50"                               # candidate pool before rerank

exec .venv/bin/python -m src.server
