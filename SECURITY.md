# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainer directly with details of the vulnerability
3. Include steps to reproduce the issue
4. Allow reasonable time for a fix before public disclosure

## Security Measures

This project implements several security measures:

### Input Validation
- All user inputs are validated before processing
- Path parameters are checked for traversal attacks (`../`, absolute paths)
- Query parameters have length limits and type checking

### Path Traversal Protection
Multi-layer defense in `src/security_utils.py`:
1. Null byte detection
2. Path normalization
3. Absolute path rejection
4. Parent directory traversal blocking
5. Symlink resolution boundary checking

### Docker Security
- Non-root user execution (`mcpuser`, uid 1000)
- `CAP_DROP: ALL` capability restrictions
- `no-new-privileges` security option
- Read-only vault mount (`:ro`)

### Credential Management
- All secrets via environment variables
- No hardcoded credentials in source code
- `.gitignore` excludes credential files

## Security Testing

Run security tests:
```bash
# Unit tests for path traversal
pytest tests/test_security_path_traversal.py -v

# Password/credential safety tests
pytest tests/test_security_passwords.py -v

# Bandit security scan
bandit -r src/ -ll
```

## Dependencies

Dependencies are intentionally minimal to reduce attack surface:
- `mcp>=1.0.0` - MCP protocol (Anthropic)
- `asyncpg>=0.29.0` - PostgreSQL driver (parameterized queries)
- `pgvector>=0.3.0` - Vector operations
- `voyageai>=0.3.0` - Embedding API client
- `watchdog>=4.0.0` - File system events
- `loguru>=0.7.2` - Logging

All dependencies are from well-maintained, reputable sources.
