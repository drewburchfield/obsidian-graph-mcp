"""
Security utilities for path validation and sanitization.

Implements defense-in-depth against:
- Path traversal attacks (../)
- Absolute path injection
- Null byte injection
- Symbolic link exploitation
"""
import os
from pathlib import Path, PurePosixPath

from loguru import logger


class SecurityError(Exception):
    """Raised when security validation fails."""
    pass


def sanitize_path(user_path: str) -> str:
    """
    Sanitize user-provided path by normalizing and removing dangerous components.

    Args:
        user_path: User-provided path string

    Returns:
        Sanitized path string

    Raises:
        SecurityError: If path contains dangerous components
    """
    # Check for null bytes (common in path injection attacks)
    if '\x00' in user_path:
        raise SecurityError("Null byte detected in path")

    # Normalize path (resolve . and .., remove duplicate slashes)
    normalized = os.path.normpath(user_path)

    # Convert to PurePosixPath for consistent handling
    path = PurePosixPath(normalized)

    return str(path)


def validate_vault_path(user_path: str, vault_root: str) -> str:
    """
    Validate that a user-provided path is safe and within vault boundaries.

    This implements multiple security checks:
    1. Reject absolute paths
    2. Reject paths with parent directory traversal
    3. Reject null bytes
    4. Ensure resolved path stays within vault

    Args:
        user_path: User-provided path (should be vault-relative)
        vault_root: Absolute path to vault root directory

    Returns:
        Validated vault-relative path (sanitized)

    Raises:
        SecurityError: If path validation fails

    Examples:
        >>> validate_vault_path("notes/todo.md", "/vault")
        'notes/todo.md'

        >>> validate_vault_path("../../../etc/passwd", "/vault")
        SecurityError: Path traversal detected
    """
    # 1. Sanitize first
    try:
        sanitized = sanitize_path(user_path)
    except SecurityError:
        raise  # Re-raise with original message

    # 2. Reject absolute paths
    if os.path.isabs(sanitized):
        raise SecurityError(f"Absolute paths not allowed: {user_path}")

    # 3. Check for parent directory traversal in components
    path_parts = Path(sanitized).parts
    if '..' in path_parts:
        raise SecurityError(f"Path traversal detected: {user_path}")

    # 4. Resolve against vault root and ensure it stays within bounds
    vault_root_resolved = Path(vault_root).resolve()
    full_path = (vault_root_resolved / sanitized).resolve()

    # Check if resolved path is still within vault
    try:
        full_path.relative_to(vault_root_resolved)
    except ValueError as e:
        raise SecurityError(
            f"Path escapes vault boundaries: {user_path} "
            f"resolves to {full_path}"
        ) from e

    logger.debug(f"Path validated: {user_path} -> {sanitized}")
    return sanitized


def validate_note_path_parameter(
    note_path: str,
    vault_path: str | None = None
) -> str:
    """
    Convenience function to validate note_path parameters from MCP tools.

    Args:
        note_path: User-provided note path from tool arguments
        vault_path: Optional vault root (defaults to OBSIDIAN_VAULT_PATH env var)

    Returns:
        Validated path

    Raises:
        SecurityError: If validation fails
    """
    if vault_path is None:
        vault_path = os.getenv("OBSIDIAN_VAULT_PATH", "/vault")

    return validate_vault_path(note_path, vault_path)
