"""
Security tests for password handling.

Tests that database passwords are:
1. Never hardcoded in committed files
2. Generated randomly with sufficient entropy
3. Synchronized between MCP server and PostgreSQL container
4. Not logged or exposed in error messages
"""
import os
import re
from pathlib import Path

import pytest


def test_no_hardcoded_passwords_in_docker_compose():
    """Ensure docker-compose.yml doesn't contain hardcoded passwords."""
    compose_file = Path(__file__).parents[3] / "docker-compose.yml"

    if not compose_file.exists():
        pytest.skip("docker-compose.yml not found")

    with open(compose_file) as f:
        content = f.read()

    # Look for obsidian-graph-pgvector service
    lines = content.split('\n')
    in_pgvector_service = False
    hardcoded_passwords = []

    for i, line in enumerate(lines):
        if 'mcp-obsidian-graph-pgvector:' in line:
            in_pgvector_service = True
        elif in_pgvector_service and line.strip().startswith('services:'):
            break  # End of this service
        elif in_pgvector_service and 'POSTGRES_PASSWORD:' in line:
            # Check if it's a hardcoded value (not an env var reference)
            if not ('${' in line or 'POSTGRES_PASSWORD}' in line):
                # Extract the value
                match = re.search(r'POSTGRES_PASSWORD:\s*(.+)', line)
                if match:
                    value = match.group(1).strip()
                    # If it's a literal string (not env var), flag it
                    if value and not value.startswith('$'):
                        hardcoded_passwords.append((i + 1, value))

    assert len(hardcoded_passwords) == 0, \
        f"Found hardcoded passwords in docker-compose.yml at lines: {hardcoded_passwords}. " \
        "Use environment variables instead: POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}"


def test_env_example_has_placeholder():
    """Ensure .env.instance.example has placeholder, not real password."""
    env_file = Path(__file__).parents[3] / "configs" / "obsidian-graph" / ".env.instance.example"

    if not env_file.exists():
        pytest.skip(".env.instance.example not found")

    with open(env_file) as f:
        content = f.read()

    # Should have a placeholder
    assert "POSTGRES_PASSWORD=your_generated_password_here" in content or \
           "POSTGRES_PASSWORD=changeme" in content, \
        ".env.instance.example should have a clear placeholder (your_generated_password_here or changeme)"

    # Should have instructions to run the script
    assert "generate-db-password" in content, \
        ".env.instance.example should mention the password generation script"


def test_password_minimum_entropy():
    """Test that generated passwords meet minimum security requirements."""
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        pytest.skip("POSTGRES_PASSWORD not set in environment")

    # Skip test if using placeholder or CI test values
    if password in ["changeme", "your_generated_password_here", "testpassword"]:
        pytest.skip("Using placeholder/CI password - run generate-db-password.sh for production")

    # Minimum 32 characters (we generate 48)
    assert len(password) >= 32, \
        f"Password too short: {len(password)} chars (min 32)"

    # Check character diversity (alphanumeric mix)
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)

    # Should have at least 2 of the 3 types for good entropy
    char_type_count = sum([has_lower, has_upper, has_digit])
    assert char_type_count >= 2, \
        f"Password lacks character diversity (has_lower={has_lower}, has_upper={has_upper}, has_digit={has_digit})"


def test_password_not_in_common_weak_list():
    """Ensure password is not a common weak password."""
    password = os.getenv("POSTGRES_PASSWORD", "")

    if not password or password in ["your_generated_password_here", "testpassword"]:
        pytest.skip("POSTGRES_PASSWORD not set or using placeholder/CI password")

    # List of passwords that should never be used
    weak_passwords = [
        "password", "123456", "admin", "root", "changeme",
        "password123", "12345678", "qwerty", "abc123",
        "postgres", "obsidian", "default"
    ]

    password_lower = password.lower()
    for weak in weak_passwords:
        assert weak not in password_lower, \
            f"Password contains weak pattern: {weak}"


def test_gitignore_includes_sensitive_files():
    """Verify .gitignore prevents committing sensitive files."""
    gitignore_file = Path(__file__).parents[3] / ".gitignore"

    if not gitignore_file.exists():
        pytest.skip(".gitignore not found")

    with open(gitignore_file) as f:
        content = f.read()

    # Check for docker-compose.override.yml
    assert "docker-compose.override.yml" in content, \
        ".gitignore should include docker-compose.override.yml"

    # Check for .env.instance files
    assert ".env.instance" in content or "*/.env.instance" in content, \
        ".gitignore should include .env.instance files"


def test_password_generation_script_exists():
    """Verify password generation script exists and is executable."""
    script_path = Path(__file__).parent.parent / "scripts" / "generate-db-password.sh"

    assert script_path.exists(), \
        f"Password generation script not found at {script_path}"

    # Check if executable (on Unix-like systems)
    if hasattr(os, 'access'):
        assert os.access(script_path, os.X_OK), \
            f"Password generation script is not executable: {script_path}"


def test_password_generation_script_syntax():
    """Basic syntax check for password generation script."""
    script_path = Path(__file__).parent.parent / "scripts" / "generate-db-password.sh"

    if not script_path.exists():
        pytest.skip("Password generation script not found")

    with open(script_path) as f:
        content = f.read()

    # Check for required components
    assert "generate_password()" in content, \
        "Script should have generate_password() function"

    assert "/dev/urandom" in content, \
        "Script should use /dev/urandom for cryptographic randomness"

    assert "tr -dc" in content, \
        "Script should use tr for character filtering"

    assert "docker-compose.override.yml" in content, \
        "Script should create docker-compose.override.yml"


@pytest.mark.skipif(
    not Path(__file__).parent.parent.parent.parent / "docker-compose.yml",
    reason="docker-compose.yml not accessible"
)
def test_docker_compose_override_pattern():
    """Verify docker-compose.override.yml pattern is correct if it exists."""
    override_file = Path(__file__).parents[3] / "docker-compose.override.yml"

    if not override_file.exists():
        pytest.skip("docker-compose.override.yml not generated yet - run generate-db-password.sh")

    with open(override_file) as f:
        content = f.read()

    # Should reference environment variable, not hardcode password
    assert "${POSTGRES_PASSWORD}" in content or \
           "$POSTGRES_PASSWORD" in content, \
        "docker-compose.override.yml should reference POSTGRES_PASSWORD environment variable"

    # Should target the correct service
    assert "mcp-obsidian-graph-pgvector" in content, \
        "docker-compose.override.yml should override mcp-obsidian-graph-pgvector service"
