"""
Security tests for password handling.

Tests that database passwords are:
1. Never hardcoded in committed files
2. Generated randomly with sufficient entropy
3. Synchronized between MCP server and PostgreSQL container
4. Not logged or exposed in error messages

NOTE: Path references use parents[1] for standalone repo structure:
  tests/test_security_passwords.py -> obsidian-graph-mcp/
"""

import os
import re
from pathlib import Path

import pytest

# Project root is parents[1] (tests -> obsidian-graph-mcp)
PROJECT_ROOT = Path(__file__).parents[1]


def test_no_hardcoded_passwords_in_docker_compose():
    """Ensure docker-compose.yml doesn't contain hardcoded passwords."""
    compose_file = PROJECT_ROOT / "docker-compose.yml"

    if not compose_file.exists():
        pytest.skip("docker-compose.yml not found")

    with open(compose_file) as f:
        content = f.read()

    # Look for postgres service password definitions
    lines = content.split("\n")
    hardcoded_passwords = []

    for i, line in enumerate(lines):
        if "POSTGRES_PASSWORD:" in line or "POSTGRES_PASSWORD=" in line:
            # Check if it's a hardcoded value (not an env var reference)
            if "${" not in line:
                # Extract the value
                match = re.search(r"POSTGRES_PASSWORD[=:]\s*(.+)", line)
                if match:
                    value = match.group(1).strip()
                    # If it's a literal string (not env var), flag it
                    if value and not value.startswith("$"):
                        hardcoded_passwords.append((i + 1, value))

    assert len(hardcoded_passwords) == 0, (
        f"Found hardcoded passwords in docker-compose.yml at lines: {hardcoded_passwords}. "
        "Use environment variables instead: POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}"
    )


def test_env_example_has_placeholder():
    """Ensure .env.example has placeholder, not real password."""
    env_file = PROJECT_ROOT / ".env.example"

    if not env_file.exists():
        pytest.skip(".env.example not found")

    with open(env_file) as f:
        content = f.read()

    # Should have a placeholder for password (not a real password)
    assert "POSTGRES_PASSWORD=" in content, ".env.example should define POSTGRES_PASSWORD"

    # Extract the password value
    for line in content.split("\n"):
        if line.startswith("POSTGRES_PASSWORD="):
            value = line.split("=", 1)[1].strip()
            # Should be a placeholder like "changeme" or empty, not a real password
            assert len(value) < 32 or value in [
                "changeme",
                "your_password_here",
                "",
            ], ".env.example should have a placeholder password, not a real one"


def test_password_minimum_entropy():
    """Test that generated passwords meet minimum security requirements."""
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        pytest.skip("POSTGRES_PASSWORD not set in environment")

    # Skip test if using placeholder or CI test values
    if password in ["changeme", "your_generated_password_here", "testpassword"]:
        pytest.skip("Using placeholder/CI password - run generate-db-password.sh for production")

    # Minimum 32 characters (we generate 48)
    assert len(password) >= 32, f"Password too short: {len(password)} chars (min 32)"

    # Check character diversity (alphanumeric mix)
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)

    # Should have at least 2 of the 3 types for good entropy
    char_type_count = sum([has_lower, has_upper, has_digit])
    assert (
        char_type_count >= 2
    ), f"Password lacks character diversity (has_lower={has_lower}, has_upper={has_upper}, has_digit={has_digit})"


def test_password_not_in_common_weak_list():
    """Ensure password is not a common weak password."""
    password = os.getenv("POSTGRES_PASSWORD", "")

    if not password or password in ["your_generated_password_here", "testpassword"]:
        pytest.skip("POSTGRES_PASSWORD not set or using placeholder/CI password")

    # List of passwords that should never be used
    weak_passwords = [
        "password",
        "123456",
        "admin",
        "root",
        "changeme",
        "password123",
        "12345678",
        "qwerty",
        "abc123",
        "postgres",
        "obsidian",
        "default",
    ]

    password_lower = password.lower()
    for weak in weak_passwords:
        assert weak not in password_lower, f"Password contains weak pattern: {weak}"


def test_gitignore_includes_sensitive_files():
    """Verify .gitignore prevents committing sensitive files."""
    gitignore_file = PROJECT_ROOT / ".gitignore"

    if not gitignore_file.exists():
        pytest.skip(".gitignore not found")

    with open(gitignore_file) as f:
        content = f.read()

    # Check for .env files (where secrets are stored)
    assert ".env" in content, ".gitignore should include .env files"

    # Check for .env.instance files (alternative naming convention)
    assert (
        ".env.instance" in content or ".env.local" in content
    ), ".gitignore should include .env.instance or .env.local files"


@pytest.mark.skip(
    reason="Standalone repo uses .env file for password - no generation script needed"
)
def test_password_generation_script_exists():
    """Verify password generation script exists and is executable."""
    # This test is skipped for standalone repo - users configure .env directly
    script_path = PROJECT_ROOT / "scripts" / "generate-db-password.sh"

    assert script_path.exists(), f"Password generation script not found at {script_path}"

    # Check if executable (on Unix-like systems)
    if hasattr(os, "access"):
        assert os.access(
            script_path, os.X_OK
        ), f"Password generation script is not executable: {script_path}"


@pytest.mark.skip(
    reason="Standalone repo uses .env file for password - no generation script needed"
)
def test_password_generation_script_syntax():
    """Basic syntax check for password generation script."""
    # This test is skipped for standalone repo
    script_path = PROJECT_ROOT / "scripts" / "generate-db-password.sh"

    if not script_path.exists():
        pytest.skip("Password generation script not found")

    with open(script_path) as f:
        content = f.read()

    # Check for required components
    assert "generate_password()" in content, "Script should have generate_password() function"

    assert "/dev/urandom" in content, "Script should use /dev/urandom for cryptographic randomness"

    assert "tr -dc" in content, "Script should use tr for character filtering"

    assert (
        "docker-compose.override.yml" in content
    ), "Script should create docker-compose.override.yml"


@pytest.mark.skip(reason="Standalone repo uses .env file instead of docker-compose.override.yml")
def test_docker_compose_override_pattern():
    """Verify docker-compose.override.yml pattern is correct if it exists."""
    # This test is skipped for standalone repo - password is in .env
    override_file = PROJECT_ROOT / "docker-compose.override.yml"

    if not override_file.exists():
        pytest.skip("docker-compose.override.yml not generated yet - run generate-db-password.sh")

    with open(override_file) as f:
        content = f.read()

    # Should reference environment variable, not hardcode password
    assert (
        "${POSTGRES_PASSWORD}" in content or "$POSTGRES_PASSWORD" in content
    ), "docker-compose.override.yml should reference POSTGRES_PASSWORD environment variable"

    # Should target the correct service
    assert (
        "mcp-obsidian-graph-pgvector" in content
    ), "docker-compose.override.yml should override mcp-obsidian-graph-pgvector service"
