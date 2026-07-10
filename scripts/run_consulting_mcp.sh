#!/usr/bin/env bash
# Launch an MCP stdio session inside the continuously running consulting container.
set -euo pipefail
exec docker exec -i -e OBSIDIAN_WATCH_ENABLED=false consulting-graph \
  .venv/bin/python -m src.server
