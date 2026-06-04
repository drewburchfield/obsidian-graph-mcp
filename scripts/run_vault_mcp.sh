#!/usr/bin/env bash
# Launch the MCP server bound to the OBSIDIAN VAULT graph:
#   - Voyage embeddings (voyage-context-3, 1024d) — matches the stored vectors
#   - obsidian_graph Postgres (this Mac, :5435)
# Registered in Claude Code / Desktop as the "obsidian-vault" MCP server.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

# Secrets stay in .env (gitignored).
POSTGRES_PASSWORD="$(grep -E '^POSTGRES_PASSWORD=' .env | head -1 | cut -d= -f2-)"
VOYAGE_API_KEY="$(grep -E '^VOYAGE_API_KEY=' .env | head -1 | cut -d= -f2-)"
export POSTGRES_PASSWORD VOYAGE_API_KEY

export MCP_SERVER_NAME="obsidian-vault"
export EMBEDDING_PROVIDER="voyage"
export EMBEDDING_DIMENSIONS="1024"
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5435"
export POSTGRES_DB="obsidian_graph"
export POSTGRES_USER="obsidian"
export OBSIDIAN_WATCH_ENABLED="false"   # query-only from the MCP

exec .venv/bin/python -m src.server
