"""Container health check for the continuously synchronized graph application."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    corpus = Path(os.getenv("CORPUS_PATH", os.getenv("OBSIDIAN_VAULT_PATH", "/vault")))
    state_path = Path(
        os.getenv(
            "CORPUS_SYNC_STATE_PATH",
            str(Path.home() / ".obsidian-graph" / "sync-state.json"),
        )
    )
    if not corpus.is_dir() or not state_path.is_file():
        return 1
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 1
    if state.get("status") == "ready":
        return 0
    if state.get("status") == "syncing":
        try:
            updated_at = datetime.fromisoformat(state["updated_at"])
        except (KeyError, TypeError, ValueError):
            return 1
        age = datetime.now(UTC) - updated_at
        return 0 if age.total_seconds() <= 300 else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
