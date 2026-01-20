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
    # Obsidian system
    ".obsidian/",
    ".trash/",
    ".Trash/",
    # Version control
    ".git/",
    ".github/",
    # Editor/IDE config
    ".vscode/",
    ".cursor/",
    # AI tool folders
    ".claude/",
    ".aider/",
    ".smart-env/",
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
                folder = pattern.rstrip("/")
                # Check if path starts with this folder
                if rel_path.startswith(folder + "/") or rel_path == folder:
                    return True
                # Check if first path component matches (for dotfiles like .obsidian)
                first_component = rel_path.split("/")[0]
                if first_component == folder:
                    return True

            # Glob pattern match
            if fnmatch.fnmatch(rel_path, pattern):
                return True

            # Check basename match for simple patterns
            if "/" not in pattern and fnmatch.fnmatch(Path(rel_path).name, pattern):
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
    custom_patterns: list[str] = []

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


async def cleanup_excluded_notes(store, vault_path: str) -> int:
    """
    Remove notes from database that match exclusion patterns.

    Called automatically at startup to clean up previously indexed
    notes that are now excluded.

    Args:
        store: PostgreSQLVectorStore instance (must be initialized)
        vault_path: Path to Obsidian vault (for loading exclusion config)

    Returns:
        Number of notes removed
    """
    # Load exclusion filter
    exclusion_filter = load_exclusion_filter(vault_path)

    # Get all paths from database
    all_paths = await store.get_all_paths()

    if not all_paths:
        return 0

    # Find paths that should be excluded
    paths_to_delete = [
        path for path in all_paths if exclusion_filter.should_exclude(path)
    ]

    if not paths_to_delete:
        logger.debug("No excluded paths found in database")
        return 0

    # Delete excluded paths
    deleted_count = await store.delete_notes_by_paths(paths_to_delete)

    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} excluded notes from index")
        # Log first few paths for visibility
        for path in paths_to_delete[:5]:
            logger.debug(f"  Removed: {path}")
        if len(paths_to_delete) > 5:
            logger.debug(f"  ... and {len(paths_to_delete) - 5} more")

    return deleted_count
