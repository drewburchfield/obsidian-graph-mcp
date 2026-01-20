# Folder Exclusion Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow users to exclude folders/patterns from indexing via an optional config file in the vault.

**Architecture:** Create `src/exclusion.py` module that parses `.obsidian-graph.conf` from the vault root. The `scan_vault()` and file watcher filter paths through this module. Sensible defaults (`.obsidian/`, `.git/`, `.trash/`) are always excluded.

**Tech Stack:** Python pathlib, fnmatch for glob patterns

---

## Task 1: Create Exclusion Module with Tests

**Files:**
- Create: `src/exclusion.py`
- Create: `tests/test_exclusion.py`

**Step 1: Write the failing tests**

```python
# tests/test_exclusion.py
"""
Unit tests for path exclusion logic.

Tests:
1. Default patterns always excluded
2. Config file parsing (comments, blank lines)
3. Pattern matching (exact, glob, nested)
4. Missing config file uses defaults only
5. Invalid patterns are skipped with warning
"""

from pathlib import Path

import pytest

from src.exclusion import ExclusionFilter, load_exclusion_filter


class TestExclusionFilter:
    """Test ExclusionFilter pattern matching."""

    def test_default_patterns_excluded(self):
        """Built-in patterns should always be excluded."""
        ef = ExclusionFilter(custom_patterns=[])

        assert ef.should_exclude(".obsidian/plugins/test.md")
        assert ef.should_exclude(".git/config")
        assert ef.should_exclude(".trash/deleted.md")
        assert ef.should_exclude(".Trash/old-note.md")

    def test_custom_pattern_exact_match(self):
        """Exact folder paths should match."""
        ef = ExclusionFilter(custom_patterns=["07_Archive/Trash/"])

        assert ef.should_exclude("07_Archive/Trash/deleted.md")
        assert ef.should_exclude("07_Archive/Trash/subfolder/note.md")
        assert not ef.should_exclude("07_Archive/active.md")

    def test_custom_pattern_glob(self):
        """Glob patterns should work."""
        ef = ExclusionFilter(custom_patterns=["drafts/*", "*.tmp.md"])

        assert ef.should_exclude("drafts/wip.md")
        assert ef.should_exclude("notes/idea.tmp.md")
        assert not ef.should_exclude("published/final.md")

    def test_normal_files_not_excluded(self):
        """Regular vault files should not be excluded."""
        ef = ExclusionFilter(custom_patterns=[])

        assert not ef.should_exclude("notes/my-note.md")
        assert not ef.should_exclude("02_Permanent/ideas.md")
        assert not ef.should_exclude("folder/subfolder/deep.md")


class TestLoadExclusionFilter:
    """Test config file loading."""

    def test_missing_config_uses_defaults(self, tmp_path):
        """No config file should use defaults only."""
        ef = load_exclusion_filter(str(tmp_path))

        # Defaults still work
        assert ef.should_exclude(".obsidian/test.md")
        # No custom exclusions
        assert not ef.should_exclude("07_Archive/Trash/note.md")

    def test_parses_config_file(self, tmp_path):
        """Config file should be parsed correctly."""
        config = tmp_path / ".obsidian-graph.conf"
        config.write_text("""
# This is a comment
07_Archive/Trash/

# Another comment
drafts/
""")

        ef = load_exclusion_filter(str(tmp_path))

        assert ef.should_exclude("07_Archive/Trash/note.md")
        assert ef.should_exclude("drafts/wip.md")

    def test_ignores_comments_and_blanks(self, tmp_path):
        """Comments and blank lines should be ignored."""
        config = tmp_path / ".obsidian-graph.conf"
        config.write_text("""
# Comment line
   # Indented comment

07_Archive/Trash/


""")

        ef = load_exclusion_filter(str(tmp_path))

        # Should only have 1 custom pattern
        assert len(ef.custom_patterns) == 1
        assert ef.custom_patterns[0] == "07_Archive/Trash/"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/drewburchfield/dev/projects/obsidian-graph-mcp && python -m pytest tests/test_exclusion.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.exclusion'"

**Step 3: Write the exclusion module**

```python
# src/exclusion.py
"""
Path exclusion filtering for Obsidian vault indexing.

Supports:
- Built-in default exclusions (.obsidian, .git, .trash)
- Custom patterns from .obsidian-graph.conf in vault root
- Glob pattern matching (*, **)
"""

import fnmatch
from pathlib import Path

from loguru import logger

# Always excluded, regardless of config
DEFAULT_EXCLUSIONS = [
    ".obsidian/",
    ".obsidian/*",
    ".git/",
    ".git/*",
    ".trash/",
    ".trash/*",
    ".Trash/",
    ".Trash/*",
    ".smart-env/",
    ".smart-env/*",
]

CONFIG_FILENAME = ".obsidian-graph.conf"


class ExclusionFilter:
    """
    Filters paths based on exclusion patterns.

    Combines built-in defaults with user-configured patterns.
    """

    def __init__(self, custom_patterns: list[str]):
        """
        Initialize filter with custom patterns.

        Args:
            custom_patterns: User-defined exclusion patterns from config file
        """
        self.custom_patterns = custom_patterns
        self.all_patterns = DEFAULT_EXCLUSIONS + custom_patterns

    def should_exclude(self, rel_path: str) -> bool:
        """
        Check if a path should be excluded from indexing.

        Args:
            rel_path: Path relative to vault root (e.g., "folder/note.md")

        Returns:
            True if path matches any exclusion pattern
        """
        # Normalize path separators
        rel_path = rel_path.replace("\\", "/")

        for pattern in self.all_patterns:
            # Direct prefix match for folder patterns (ending with /)
            if pattern.endswith("/"):
                if rel_path.startswith(pattern) or rel_path.startswith(pattern.rstrip("/")):
                    return True
                # Also check if any path component matches
                if rel_path.startswith(pattern.rstrip("/") + "/"):
                    return True

            # Glob pattern match
            if fnmatch.fnmatch(rel_path, pattern):
                return True

            # Check if pattern matches any parent directory
            if fnmatch.fnmatch(rel_path, f"**/{pattern}"):
                return True
            if fnmatch.fnmatch(rel_path, f"{pattern}/**"):
                return True

        return False


def load_exclusion_filter(vault_path: str) -> ExclusionFilter:
    """
    Load exclusion filter from vault config file.

    Looks for .obsidian-graph.conf in vault root.
    If not found, uses defaults only.

    Args:
        vault_path: Path to Obsidian vault

    Returns:
        ExclusionFilter configured with defaults + custom patterns
    """
    config_path = Path(vault_path) / CONFIG_FILENAME
    custom_patterns = []

    if config_path.exists():
        logger.info(f"Loading exclusion config: {config_path}")
        try:
            with open(config_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue
                    custom_patterns.append(line)

            logger.info(f"Loaded {len(custom_patterns)} custom exclusion patterns")

        except Exception as e:
            logger.warning(f"Error reading exclusion config: {e}")
    else:
        logger.debug(
            f"No {CONFIG_FILENAME} found in vault. Using defaults. "
            f"See docs to customize exclusions."
        )

    return ExclusionFilter(custom_patterns)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/drewburchfield/dev/projects/obsidian-graph-mcp && python -m pytest tests/test_exclusion.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/exclusion.py tests/test_exclusion.py
git commit -m "feat: add exclusion filter module for path filtering"
```

---

## Task 2: Integrate Exclusion into Indexer

**Files:**
- Modify: `src/indexer.py:19-35` (scan_vault function)
- Modify: `tests/test_indexer.py`

**Step 1: Add test for exclusion filtering**

Add to `tests/test_indexer.py`:

```python
def test_scan_vault_excludes_configured_paths(tmp_path):
    """Test that scan_vault respects exclusion config."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create normal note
    (vault / "note.md").write_text("# Note")

    # Create excluded folders
    trash = vault / "07_Archive" / "Trash"
    trash.mkdir(parents=True)
    (trash / "deleted.md").write_text("# Deleted")

    obsidian = vault / ".obsidian"
    obsidian.mkdir()
    (obsidian / "config.md").write_text("config")

    # Create config file
    (vault / ".obsidian-graph.conf").write_text("07_Archive/Trash/\n")

    md_files = scan_vault(str(vault))

    # Should only find the non-excluded note
    paths = [str(f.relative_to(vault)) for f in md_files]
    assert "note.md" in paths
    assert "07_Archive/Trash/deleted.md" not in paths
    assert ".obsidian/config.md" not in paths
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/drewburchfield/dev/projects/obsidian-graph-mcp && python -m pytest tests/test_indexer.py::test_scan_vault_excludes_configured_paths -v`
Expected: FAIL (exclusion not integrated yet)

**Step 3: Modify scan_vault to use exclusion filter**

Update `src/indexer.py`:

```python
# Add import at top (after line 12)
from .exclusion import load_exclusion_filter

# Replace scan_vault function (lines 19-35)
def scan_vault(vault_path: str) -> list[Path]:
    """
    Scan Obsidian vault for all markdown files.

    Respects exclusion patterns from .obsidian-graph.conf.

    Args:
        vault_path: Path to Obsidian vault

    Returns:
        List of markdown file paths (excluding filtered paths)
    """
    vault = Path(vault_path)
    if not vault.exists():
        raise FileNotFoundError(f"Vault not found: {vault_path}")

    # Load exclusion filter
    exclusion_filter = load_exclusion_filter(vault_path)

    # Find all markdown files
    all_md_files = list(vault.rglob("*.md"))

    # Filter out excluded paths
    md_files = []
    excluded_count = 0
    for file_path in all_md_files:
        rel_path = str(file_path.relative_to(vault))
        if exclusion_filter.should_exclude(rel_path):
            excluded_count += 1
            continue
        md_files.append(file_path)

    logger.info(
        f"Found {len(md_files)} markdown files in vault "
        f"({excluded_count} excluded by filters)"
    )
    return md_files
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/drewburchfield/dev/projects/obsidian-graph-mcp && python -m pytest tests/test_indexer.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/indexer.py tests/test_indexer.py
git commit -m "feat: integrate exclusion filter into vault scanning"
```

---

## Task 3: Integrate Exclusion into File Watcher

**Files:**
- Modify: `src/file_watcher.py:91-121` (on_modified, on_created)
- Modify: `src/file_watcher.py:268-298` (VaultWatcher.__init__)
- Modify: `src/file_watcher.py:300-365` (startup_scan)

**Step 1: Add test for file watcher exclusion**

Add to `tests/test_file_watcher.py` (or create if needed):

```python
@pytest.mark.asyncio
async def test_file_watcher_ignores_excluded_paths(tmp_path, mock_store, mock_embedder):
    """Test that file watcher skips excluded paths."""
    from src.file_watcher import ObsidianFileWatcher
    from src.exclusion import load_exclusion_filter

    vault = tmp_path / "vault"
    vault.mkdir()

    # Create config
    (vault / ".obsidian-graph.conf").write_text("trash/\n")

    loop = asyncio.get_event_loop()
    watcher = ObsidianFileWatcher(
        str(vault), mock_store, mock_embedder, loop, debounce_seconds=1
    )

    # Simulate event for excluded file
    class FakeEvent:
        is_directory = False
        src_path = str(vault / "trash" / "deleted.md")

    watcher.on_modified(FakeEvent())

    # Should not add to pending changes
    assert FakeEvent.src_path not in watcher.pending_changes
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/drewburchfield/dev/projects/obsidian-graph-mcp && python -m pytest tests/test_file_watcher.py::test_file_watcher_ignores_excluded_paths -v`
Expected: FAIL

**Step 3: Modify file watcher to use exclusion filter**

Update `src/file_watcher.py`:

```python
# Add import at top (after line 20)
from .exclusion import load_exclusion_filter

# Modify ObsidianFileWatcher.__init__ (add after line 66)
        self.exclusion_filter = load_exclusion_filter(vault_path)

# Modify on_modified (replace lines 91-105)
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        if not event.src_path.endswith(".md"):
            return

        file_path = event.src_path

        # Check exclusion filter
        try:
            rel_path = str(Path(file_path).relative_to(self.vault_path))
            if self.exclusion_filter.should_exclude(rel_path):
                logger.debug(f"Ignoring excluded file: {rel_path}")
                return
        except ValueError:
            pass  # File outside vault, let it proceed

        logger.debug(f"File modified: {file_path}")

        # Schedule debounced re-index with error handling
        self.pending_changes[file_path] = time.time()
        future = asyncio.run_coroutine_threadsafe(self._debounced_reindex(file_path), self.loop)
        future.add_done_callback(self._handle_reindex_future_error)

# Modify on_created similarly (replace lines 107-121)
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return

        if not event.src_path.endswith(".md"):
            return

        file_path = event.src_path

        # Check exclusion filter
        try:
            rel_path = str(Path(file_path).relative_to(self.vault_path))
            if self.exclusion_filter.should_exclude(rel_path):
                logger.debug(f"Ignoring excluded file: {rel_path}")
                return
        except ValueError:
            pass  # File outside vault, let it proceed

        logger.debug(f"File created: {file_path}")

        # Schedule debounced re-index with error handling
        self.pending_changes[file_path] = time.time()
        future = asyncio.run_coroutine_threadsafe(self._debounced_reindex(file_path), self.loop)
        future.add_done_callback(self._handle_reindex_future_error)
```

**Step 4: Update startup_scan to use exclusion filter**

In `VaultWatcher.startup_scan()` (around line 311):

```python
    async def startup_scan(self):
        """
        Scan vault on startup to re-index files changed while offline.

        Compares file mtime vs database last_indexed_at.
        Respects exclusion patterns.
        """
        if not self.store.pool:
            logger.warning("Store not initialized, skipping startup scan")
            return

        try:
            vault = Path(self.vault_path)

            # Load exclusion filter
            exclusion_filter = load_exclusion_filter(self.vault_path)

            all_md_files = list(vault.rglob("*.md"))

            # Filter excluded files
            md_files = []
            for file_path in all_md_files:
                rel_path = str(file_path.relative_to(vault))
                if not exclusion_filter.should_exclude(rel_path):
                    md_files.append(file_path)

            stale_files = []
            # ... rest of method unchanged
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/drewburchfield/dev/projects/obsidian-graph-mcp && python -m pytest tests/test_file_watcher.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/file_watcher.py tests/test_file_watcher.py
git commit -m "feat: integrate exclusion filter into file watcher"
```

---

## Task 4: Create Example Config File

**Files:**
- Create: `.obsidian-graph.conf.example`

**Step 1: Create the example file**

```conf
# Obsidian Graph MCP - Exclusion Patterns
# ========================================
#
# This file controls which folders/files are excluded from indexing.
# Copy to your vault as: .obsidian-graph.conf
#
# How it works:
# - Lines starting with # are comments (ignored)
# - Blank lines are ignored
# - Each line is a pattern to exclude
# - Patterns are matched against paths relative to vault root
#
# Pattern types:
# - Folder path:  07_Archive/Trash/     (excludes folder and all contents)
# - Glob pattern: drafts/*              (excludes all files in drafts/)
# - File pattern: *.tmp.md              (excludes all .tmp.md files)
#
# === DEFAULTS (always excluded, no need to add) ===
# .obsidian/
# .git/
# .trash/
# .Trash/
# .smart-env/
#
# === COMMON EXCLUSIONS (uncomment what you need) ===
#
# Soft delete / trash folder
# 07_Archive/Trash/
#
# Work in progress
# drafts/
# _drafts/
#
# Templates (if you don't want them searchable)
# templates/
# _templates/
#
# Daily notes (if too noisy in search results)
# 00_Inbox/Daily/
#
# Attachments folder (usually non-markdown anyway)
# attachments/
# 08_Attachments/
#
# === ADD YOUR CUSTOM EXCLUSIONS BELOW ===

```

**Step 2: Commit**

```bash
git add .obsidian-graph.conf.example
git commit -m "docs: add example exclusion config file"
```

---

## Task 5: Update README Documentation

**Files:**
- Modify: `README.md`

**Step 1: Add exclusion documentation section**

Add a new section to README.md (find appropriate location, likely after configuration):

```markdown
## Excluding Folders from Indexing

By default, the indexer excludes common system folders:
- `.obsidian/` (Obsidian app config)
- `.git/` (version control)
- `.trash/` / `.Trash/` (Obsidian trash)
- `.smart-env/` (Smart Connections plugin)

### Custom Exclusions

To exclude additional folders (like a soft-delete folder), create `.obsidian-graph.conf` in your vault root:

```conf
# Exclude soft delete folder
07_Archive/Trash/

# Exclude drafts
drafts/
```

See `.obsidian-graph.conf.example` for more patterns and examples.

### Pattern Syntax

| Pattern | Matches |
|---------|---------|
| `folder/` | All files in `folder/` and subfolders |
| `drafts/*` | All files directly in `drafts/` |
| `*.tmp.md` | All files ending in `.tmp.md` |
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add exclusion configuration documentation"
```

---

## Task 6: Run Full Test Suite

**Step 1: Run all tests**

Run: `cd /Users/drewburchfield/dev/projects/obsidian-graph-mcp && python -m pytest -v`
Expected: All tests PASS

**Step 2: Run linting**

Run: `cd /Users/drewburchfield/dev/projects/obsidian-graph-mcp && python -m ruff check src/ tests/`
Expected: No errors (or fix any that appear)

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address any linting issues"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Create exclusion module | `src/exclusion.py`, `tests/test_exclusion.py` |
| 2 | Integrate into indexer | `src/indexer.py`, `tests/test_indexer.py` |
| 3 | Integrate into file watcher | `src/file_watcher.py`, `tests/test_file_watcher.py` |
| 4 | Create example config | `.obsidian-graph.conf.example` |
| 5 | Update documentation | `README.md` |
| 6 | Run full test suite | - |

Total: ~6 commits, feature complete with tests and docs.
