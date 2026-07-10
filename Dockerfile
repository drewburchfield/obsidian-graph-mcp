# Aikido safe-chain installs first so every subsequent pip/uv install is
# routed through Aikido Intel's malware feed. Pinned to 1.5.3 so a future
# safe-chain regression cannot silently break this image build. The PATH
# update places shims ahead of system pip/python so even bare `pip` calls
# get intercepted.
FROM python:3.11-slim

# Default RUN shell uses pipefail so `curl ... | sh` fails the build if curl
# fails partway through, instead of executing a truncated installer.
SHELL ["/bin/sh", "-eo", "pipefail", "-c"]

# curl: needed by the safe-chain installer.
# ca-certificates: HTTPS to GitHub and PyPI.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Aikido safe-chain: blocks known-malicious pip/uv packages at install time
# and suppresses packages younger than the configured age (default 48h).
RUN curl -fsSL https://github.com/AikidoSec/safe-chain/releases/download/1.5.3/install-safe-chain.sh \
    | sh -s -- --ci

ENV PATH="/root/.safe-chain/shims:/root/.safe-chain/bin:${PATH}"

# Asserts the pip wrapper is active before any install runs. Without this, a
# broken safe-chain install would silently fall back to unprotected pip and
# the build would succeed with no malware checks happening. `which pip` must
# resolve to a shim path; otherwise fail the build loudly.
RUN safe-chain --version && \
    case "$(command -v pip)" in /root/.safe-chain/shims/*) ;; *) echo "safe-chain pip shim not on PATH" >&2; exit 1;; esac

WORKDIR /app

# Install uv from the official distroless image. uv runs `uv sync --frozen`
# against uv.lock for reproducible installs with hash verification, replacing
# the previous unpinned pip-install-from-pyproject hack.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

# Copy lockfile + pyproject so the install step caches independent of source.
COPY pyproject.toml uv.lock ./

# `--frozen` installs from the lockfile without re-resolving, so Docker
# builds are bit-for-bit reproducible against the committed uv.lock. CI
# uses `--locked` instead (which fails if pyproject.toml has drifted from
# uv.lock); the Dockerfile deliberately does not, because rebuilding a
# pinned image should never depend on pyproject.toml. `--no-dev` skips dev
# extras (pytest, ruff, etc.) for a slim runtime image;
# `--no-install-project` because src/ has not been copied yet at this layer.
RUN uv sync --frozen --no-dev --no-install-project --extra converters

# Create non-root user
RUN useradd -m -u 1000 mcpuser

# Copy source code and set ownership
COPY --chown=mcpuser:mcpuser src/ ./src/

# Create directories for data and cache
RUN mkdir -p /home/mcpuser/.obsidian-graph/cache && \
    chown -R mcpuser:mcpuser /home/mcpuser/.obsidian-graph

# Switch to non-root user
USER mcpuser

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command - run MCP server via the uv-managed virtualenv
CMD [".venv/bin/python", "-m", "src.server"]
