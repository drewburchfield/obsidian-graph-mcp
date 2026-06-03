#!/usr/bin/env bash
# Launch the MCP server bound to the CONSULTING graph (separate from the vault):
#   - embeddings from bigbot's MLX server (Qwen3-4B, 2560d) over the tailnet
#   - vectors in the consulting_graph Postgres (this Mac, :5434)
# Registered in Claude Code as the "consulting-graph" MCP server.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

# Secret stays in .env (gitignored); everything else is explicit below.
POSTGRES_PASSWORD="$(grep -E '^POSTGRES_PASSWORD=' .env | head -1 | cut -d= -f2-)"
export POSTGRES_PASSWORD

export MCP_SERVER_NAME="consulting-graph"
export EMBEDDING_PROVIDER="ollama"
export OLLAMA_HOST="http://100.117.32.59:11436"      # bigbot MLX server over tailnet
export OLLAMA_EMBED_MODEL="qwen3-embedding-4b-mlx"
export OLLAMA_EMBED_DIMS="2560"
export EMBEDDING_DIMENSIONS="2560"                    # must match the vector(2560) column
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5434"
export POSTGRES_DB="consulting_graph"
export POSTGRES_USER="obsidian"
export OBSIDIAN_WATCH_ENABLED="false"                # query-only; reindex via index_consulting.py

exec .venv/bin/python -m src.server
