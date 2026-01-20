"""
Unit tests for path exclusion logic.

Tests:
1. Default patterns always excluded
2. Config file parsing (comments, blank lines)
3. Pattern matching (exact, glob, nested)
4. Missing config file uses defaults only
5. Invalid patterns are skipped with warning
6. Cleanup of excluded notes from database
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exclusion import ExclusionFilter, cleanup_excluded_notes, load_exclusion_filter


class TestExclusionFilter:
    """Test ExclusionFilter pattern matching."""

    def test_default_patterns_excluded(self):
        """Built-in patterns should always be excluded."""
        ef = ExclusionFilter(custom_patterns=[])

        # Obsidian system
        assert ef.should_exclude(".obsidian/plugins/test.md")
        assert ef.should_exclude(".obsidian/config.json")
        assert ef.should_exclude(".trash/deleted.md")
        assert ef.should_exclude(".Trash/old-note.md")

        # Version control
        assert ef.should_exclude(".git/config")
        assert ef.should_exclude(".git/objects/abc123")
        assert ef.should_exclude(".github/workflows/ci.yml")

        # Editor/IDE config
        assert ef.should_exclude(".vscode/settings.json")
        assert ef.should_exclude(".cursor/rules.md")

        # AI tool folders
        assert ef.should_exclude(".claude/sessions/abc.json")
        assert ef.should_exclude(".aider/history.md")
        assert ef.should_exclude(".smart-env/data.json")

    def test_custom_pattern_exact_match(self):
        """Exact folder paths should match."""
        ef = ExclusionFilter(custom_patterns=["07_Archive/Trash/"])

        assert ef.should_exclude("07_Archive/Trash/deleted.md")
        assert ef.should_exclude("07_Archive/Trash/subfolder/note.md")
        assert not ef.should_exclude("07_Archive/active.md")
        assert not ef.should_exclude("07_Archive/Other/note.md")

    def test_custom_pattern_glob(self):
        """Glob patterns should work."""
        ef = ExclusionFilter(custom_patterns=["*.tmp.md"])

        assert ef.should_exclude("notes/idea.tmp.md")
        assert ef.should_exclude("idea.tmp.md")
        assert not ef.should_exclude("notes/idea.md")
        assert not ef.should_exclude("published/final.md")

    def test_normal_files_not_excluded(self):
        """Regular vault files should not be excluded."""
        ef = ExclusionFilter(custom_patterns=[])

        assert not ef.should_exclude("notes/my-note.md")
        assert not ef.should_exclude("02_Permanent/ideas.md")
        assert not ef.should_exclude("folder/subfolder/deep.md")
        assert not ef.should_exclude("README.md")

    def test_nested_default_exclusions(self):
        """Default exclusions should work for nested paths."""
        ef = ExclusionFilter(custom_patterns=[])

        # Deeply nested .obsidian paths
        assert ef.should_exclude(".obsidian/plugins/dataview/main.js")
        assert ef.should_exclude(".obsidian/themes/minimal.css")

    def test_similar_names_not_excluded(self):
        """Folders with similar names should not be excluded."""
        ef = ExclusionFilter(custom_patterns=["trash/"])

        # 'trash/' is excluded but 'my-trash/' is not
        assert ef.should_exclude("trash/note.md")
        assert not ef.should_exclude("my-trash/note.md")
        assert not ef.should_exclude("trashed/note.md")


class TestLoadExclusionFilter:
    """Test config file loading."""

    def test_missing_config_uses_defaults(self, tmp_path):
        """No config file should use defaults only."""
        ef = load_exclusion_filter(str(tmp_path))

        # Defaults still work
        assert ef.should_exclude(".obsidian/test.md")
        assert ef.should_exclude(".git/config")
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
        # Defaults still work
        assert ef.should_exclude(".obsidian/config.json")

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

    def test_multiple_custom_patterns(self, tmp_path):
        """Multiple custom patterns should all work."""
        config = tmp_path / ".obsidian-graph.conf"
        config.write_text("""
07_Archive/Trash/
drafts/
templates/
*.tmp.md
""")

        ef = load_exclusion_filter(str(tmp_path))

        assert len(ef.custom_patterns) == 4
        assert ef.should_exclude("07_Archive/Trash/old.md")
        assert ef.should_exclude("drafts/wip.md")
        assert ef.should_exclude("templates/daily.md")
        assert ef.should_exclude("notes/scratch.tmp.md")
        # But normal files are fine
        assert not ef.should_exclude("notes/important.md")


class TestCleanupExcludedNotes:
    """Test cleanup_excluded_notes function."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_excluded_paths(self, tmp_path):
        """Excluded paths in database should be deleted."""
        # Create config with exclusion
        config = tmp_path / ".obsidian-graph.conf"
        config.write_text("07_Archive/Trash/\n")

        # Mock store with excluded path
        mock_store = MagicMock()
        mock_store.get_all_paths = AsyncMock(
            return_value=[
                "notes/keep.md",
                "07_Archive/Trash/deleted.md",
                "07_Archive/Trash/old.md",
                "other/note.md",
            ]
        )
        mock_store.delete_notes_by_paths = AsyncMock(return_value=2)

        count = await cleanup_excluded_notes(mock_store, str(tmp_path))

        assert count == 2
        mock_store.delete_notes_by_paths.assert_called_once()
        # Check only excluded paths were passed for deletion
        deleted_paths = mock_store.delete_notes_by_paths.call_args[0][0]
        assert "07_Archive/Trash/deleted.md" in deleted_paths
        assert "07_Archive/Trash/old.md" in deleted_paths
        assert "notes/keep.md" not in deleted_paths
        assert "other/note.md" not in deleted_paths

    @pytest.mark.asyncio
    async def test_cleanup_with_empty_database(self, tmp_path):
        """Empty database should return 0."""
        mock_store = MagicMock()
        mock_store.get_all_paths = AsyncMock(return_value=[])

        count = await cleanup_excluded_notes(mock_store, str(tmp_path))

        assert count == 0
        mock_store.delete_notes_by_paths.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_with_no_excluded_paths(self, tmp_path):
        """Database with no excluded paths should return 0."""
        mock_store = MagicMock()
        mock_store.get_all_paths = AsyncMock(
            return_value=["notes/a.md", "notes/b.md", "other/c.md"]
        )

        count = await cleanup_excluded_notes(mock_store, str(tmp_path))

        assert count == 0
        mock_store.delete_notes_by_paths.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_removes_default_exclusions(self, tmp_path):
        """Default exclusions should be cleaned up too."""
        mock_store = MagicMock()
        mock_store.get_all_paths = AsyncMock(
            return_value=[
                "notes/keep.md",
                ".obsidian/plugins/test.md",
                ".git/config",
                ".trash/deleted.md",
            ]
        )
        mock_store.delete_notes_by_paths = AsyncMock(return_value=3)

        count = await cleanup_excluded_notes(mock_store, str(tmp_path))

        assert count == 3
        deleted_paths = mock_store.delete_notes_by_paths.call_args[0][0]
        assert ".obsidian/plugins/test.md" in deleted_paths
        assert ".git/config" in deleted_paths
        assert ".trash/deleted.md" in deleted_paths
        assert "notes/keep.md" not in deleted_paths
