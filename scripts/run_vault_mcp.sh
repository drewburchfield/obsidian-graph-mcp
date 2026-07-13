#!/usr/bin/env bash
# Launch an MCP stdio session inside the running obsidian-graph container.
# The main container process owns file watching; MCP sessions are query-only,
# so the watcher is disabled to avoid duplicate indexing.
set -euo pipefail
exec docker exec -i -e OBSIDIAN_WATCH_ENABLED=false obsidian-graph \
  .venv/bin/python -m src.server
