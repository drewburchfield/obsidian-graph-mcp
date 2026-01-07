FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 mcpuser

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and tests and set ownership
COPY --chown=mcpuser:mcpuser src/ ./src/
COPY --chown=mcpuser:mcpuser tests/ ./tests/

# Copy configuration files for testing
COPY --chown=mcpuser:mcpuser pyproject.toml mypy.ini ./

# Create directories for data and cache
RUN mkdir -p /home/mcpuser/.obsidian-graph/cache && \
    chown -R mcpuser:mcpuser /home/mcpuser/.obsidian-graph

# Switch to non-root user
USER mcpuser

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command - run MCP server
CMD ["python", "-m", "src.server"]
