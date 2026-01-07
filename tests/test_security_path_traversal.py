"""
Path traversal security tests.

Tests that note_path parameters are properly validated to prevent:
1. Directory traversal attacks (../)
2. Absolute path injection
3. Null byte injection
4. Access to files outside vault
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security_utils import SecurityError, sanitize_path, validate_vault_path


class TestPathTraversalAttacks:
    """Test defenses against path traversal attacks."""

    def test_rejects_parent_directory_traversal(self):
        """Prevent ../../../etc/passwd style attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "../../sensitive-data.md",
            "vault/../../../secrets.txt",
            "notes/../../outside-vault.md",
            "../etc/passwd",
        ]

        for path in malicious_paths:
            with pytest.raises(SecurityError, match="(Path traversal|escapes vault)"):
                validate_vault_path(path, vault_root="/vault")

    def test_rejects_absolute_paths(self):
        """Prevent /etc/passwd style attacks."""
        malicious_paths = [
            "/etc/passwd",
            "/home/user/.ssh/id_rsa",
            "/var/log/secrets.log",
            "/vault/note.md",  # Even absolute paths to vault should be rejected
        ]

        for path in malicious_paths:
            with pytest.raises(SecurityError, match="Absolute paths not allowed"):
                validate_vault_path(path, vault_root="/vault")

    def test_rejects_null_bytes(self):
        """Prevent null byte injection (file.md\x00.txt)."""
        malicious_paths = [
            "valid-note.md\x00../../../etc/passwd",
            "note\x00.md",
            "\x00etc/passwd",
        ]

        for path in malicious_paths:
            with pytest.raises(SecurityError, match="Null byte"):
                validate_vault_path(path, vault_root="/vault")

    def test_accepts_valid_vault_relative_paths(self):
        """Ensure legitimate paths still work."""
        valid_paths = [
            "note.md",
            "folder/note.md",
            "deep/nested/structure/note.md",
            "projects/2024/Q4/report.md",
            "folder/subfolder/document.md",
        ]

        for path in valid_paths:
            # Should not raise exception
            validated = validate_vault_path(path, vault_root="/vault")
            assert validated is not None
            assert isinstance(validated, str)


class TestPathSanitization:
    """Test path sanitization functions."""

    def test_sanitize_removes_relative_components(self):
        """Test that ./ components are normalized."""
        test_cases = [
            ("./note.md", "note.md"),
            ("folder/./note.md", "folder/note.md"),
            ("./folder/note.md", "folder/note.md"),
        ]

        for input_path, expected in test_cases:
            sanitized = sanitize_path(input_path)
            assert sanitized == expected

    def test_sanitize_removes_duplicate_slashes(self):
        """Test that duplicate slashes are normalized."""
        test_cases = [
            ("folder//note.md", "folder/note.md"),
            ("folder///subfolder//note.md", "folder/subfolder/note.md"),
        ]

        for input_path, expected in test_cases:
            sanitized = sanitize_path(input_path)
            assert sanitized == expected

    def test_sanitize_rejects_null_bytes(self):
        """Test null byte detection."""
        with pytest.raises(SecurityError, match="Null byte"):
            sanitize_path("note.md\x00")


class TestVaultBoundaryValidation:
    """Test that paths are constrained to vault boundaries."""

    def test_complex_traversal_attempt(self):
        """Test sophisticated path traversal attempts."""
        # Even if parts normalize, should still reject
        attacks = [
            "folder/../../../etc/passwd",
            "valid/../../.../../etc/passwd",
            "./../../etc/passwd",
        ]

        for attack in attacks:
            with pytest.raises(SecurityError):
                validate_vault_path(attack, vault_root="/vault")

    def test_validates_within_real_vault(self, tmp_path):
        """Test validation with actual filesystem."""
        # Create temporary vault
        vault = tmp_path / "test_vault"
        vault.mkdir()

        # Create a test note
        (vault / "test.md").write_text("# Test")

        # Valid path should work
        validated = validate_vault_path("test.md", str(vault))
        assert validated == "test.md"

        # Traversal should fail
        with pytest.raises(SecurityError):
            validate_vault_path("../outside.md", str(vault))


class TestNotePathParameter:
    """Test the convenience wrapper for MCP tools."""

    def test_uses_env_var_for_vault_path(self, monkeypatch):
        """Test that vault_path defaults to OBSIDIAN_VAULT_PATH."""
        from src.security_utils import validate_note_path_parameter

        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", "/custom/vault")

        # Should use env var
        validated = validate_note_path_parameter("note.md")
        assert validated == "note.md"

    def test_accepts_explicit_vault_path(self):
        """Test that explicit vault_path can be provided."""
        from src.security_utils import validate_note_path_parameter

        validated = validate_note_path_parameter("note.md", vault_path="/explicit/vault")
        assert validated == "note.md"

    def test_rejects_traversal_attacks(self):
        """Test that wrapper properly validates paths."""
        from src.security_utils import validate_note_path_parameter

        with pytest.raises(SecurityError):
            validate_note_path_parameter("../../../etc/passwd", vault_path="/vault")
