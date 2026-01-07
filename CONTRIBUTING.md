# Contributing to Obsidian Graph MCP Server

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, constructive, and collaborative. We're building tools to help people think better.

## Getting Started

### Development Setup

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   docker-compose build obsidian-graph
   ```
3. **Configure** `.env.instance` with your Voyage API key
4. **Start development environment**:
   ```bash
   docker-compose up -d
   ```

### Running Tests

```bash
# Quick validation
docker exec -i mcp-obsidian-graph python test_e2e.py

# Full test suite
docker exec -i mcp-obsidian-graph pytest tests/ -v
```

## How to Contribute

### Reporting Bugs

**Before submitting:**
- Check existing issues to avoid duplicates
- Test with latest version
- Gather logs: `docker logs mcp-obsidian-graph`

**Bug report should include:**
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Docker version, vault size)
- Relevant logs/error messages

### Suggesting Features

**Feature requests should describe:**
- Problem you're trying to solve
- Proposed solution
- Example use cases
- Impact on existing functionality

### Pull Requests

**Before submitting:**
1. Create feature branch: `git checkout -b feature/your-feature-name`
2. Write tests for new functionality
3. Update documentation (README, docstrings)
4. Ensure tests pass
5. Follow commit message conventions

**Commit message format:**
```
feat(component): Add feature description
fix(component): Fix issue description
docs(component): Update documentation
test(component): Add tests
```

**PR checklist:**
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows project style
- [ ] No hardcoded credentials or personal paths
- [ ] Backwards compatible (or migration path provided)

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8
- **Type hints**: Use for all public functions
- **Docstrings**: Google style for modules and functions
- **Logging**: Use loguru with appropriate levels
- **Error handling**: Specific exceptions, clear error messages

### Testing

- **Unit tests**: For individual components
- **Integration tests**: For cross-component workflows
- **Performance tests**: Verify quality baselines
- **Coverage target**: >80% for new code

### Performance Expectations

New features should meet these baselines:
- Search operations: <500ms
- Graph operations: <2s for depth=3
- Hub/orphan queries: <100ms
- Similarity scores: Always in [0.0, 1.0] range

### Security Best Practices

- **Never commit credentials**: Use .env files (gitignored)
- **Validate inputs**: Check user-provided paths, parameters
- **Safe serialization**: Use JSON for caching (this project uses JSON, not unsafe formats)
- **Sanitize SQL**: Use parameterized queries (never string concatenation)

## Project Structure

```
servers/obsidian-graph/
├── src/
│   ├── server.py          # MCP server entry point
│   ├── vector_store.py    # PostgreSQL operations
│   ├── embedder.py        # Voyage Context-3 client
│   ├── graph_builder.py   # BFS graph traversal
│   ├── hub_analyzer.py    # Hub/orphan detection
│   ├── file_watcher.py    # Watchdog integration
│   ├── indexer.py         # Initial vault indexing
│   └── schema.sql         # Database schema
├── tests/
│   ├── test_tools.py      # Unit tests
│   └── test_simple.py     # Integration tests
├── docs/                   # User documentation
├── examples/               # Example configurations
└── README.md
```

## Common Tasks

### Adding a New Tool

1. **Implement tool function** in appropriate module
2. **Add tool definition** to `server.py::list_tools()`
3. **Add tool handler** to `server.py::call_tool()`
4. **Write tests** in `tests/`
5. **Update README** with usage example

### Optimizing Performance

- **Profile first**: Use timing logs to identify bottlenecks
- **Index strategically**: Add PostgreSQL indexes for common queries
- **Batch operations**: Minimize round-trips to database/API
- **Cache intelligently**: Use materialized columns for expensive computations

### Adding Embedding Providers

Currently supports Voyage Context-3 only. To add alternatives:

1. Create abstraction layer (see research in OBSIDIAN_GAP_ANALYSIS.md)
2. Implement provider interface for new model
3. Update configuration to support selection
4. Document dimension requirements
5. Provide migration path for existing indexes

## Release Process

1. **Update version** in README, pyproject.toml, CHANGELOG.md
2. **Run full test suite**
3. **Update CHANGELOG.md** with changes
4. **Create git tag**: `git tag v1.x.x`
5. **Push**: `git push origin main --tags`
6. **Create GitHub release** with changelog

## Getting Help

- **Issues**: https://github.com/yourusername/obsidian-graph-mcp/issues
- **Discussions**: Use GitHub Discussions for questions
- **MCP Community**: https://modelcontextprotocol.io/community

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for their contributions
- README.md contributors section (future)
- GitHub insights

Thank you for making knowledge tools better!
