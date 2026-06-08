"""
Obsidian Graph MCP Server

MCP transport layer. Defines tool schemas, delegates to tools.py for
execution, and formats results as MCP-compatible text responses.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from .embedder import VoyageEmbedder
from .file_watcher import VaultWatcher
from .graph_builder import GraphBuilder
from .hub_analyzer import HubAnalyzer
from .security_utils import SecurityError
from .tools import TOOLS, ToolContext, ToolError
from .validation import ValidationError
from .vector_store import PostgreSQLVectorStore

# Global tool context (initialized once at startup)
_tool_context: ToolContext | None = None

# Global vault watcher (separate from tool context, MCP-specific lifecycle)
_vault_watcher: VaultWatcher | None = None

# Create MCP server (name configurable so one codebase can serve multiple graphs)
app = Server(os.getenv("MCP_SERVER_NAME", "obsidian-graph"))


def make_embedder():
    """
    Build the embedder for the configured provider.

    Defaults to Voyage (the vault's setup) for backward compatibility; set
    EMBEDDING_PROVIDER=ollama (local/bigbot) or gemini for other graphs. The
    embedder MUST match whatever produced the stored vectors.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "voyage").lower()
    cache_dir = os.getenv("CACHE_DIR", str(Path.home() / ".obsidian-graph" / "cache"))
    if provider == "ollama":
        from .ollama_embedder import OllamaEmbedder

        return OllamaEmbedder(
            model=os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            dimensions=int(os.getenv("OLLAMA_EMBED_DIMS", "1024")),
            cache_dir=cache_dir,
        )
    if provider == "openrouter":
        from .openrouter_embedder import OpenRouterEmbedder

        return OpenRouterEmbedder(
            model=os.getenv("OPENROUTER_EMBED_MODEL", "qwen/qwen3-embedding-8b"),
            dimensions=int(os.getenv("OPENROUTER_EMBED_DIMS", "4096")),
            cache_dir=cache_dir,
        )
    if provider == "gemini":
        from .gemini_embedder import GeminiEmbedder

        return GeminiEmbedder(cache_dir=cache_dir)
    return VoyageEmbedder(
        model="voyage-context-3",
        cache_dir=cache_dir,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "128")),
        requests_per_minute=int(os.getenv("EMBEDDING_REQUESTS_PER_MINUTE", "300")),
    )


async def initialize_server():
    """Initialize server context with all components."""
    global _tool_context, _vault_watcher

    logger.info("Initializing Obsidian Graph MCP Server...")

    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", "/vault")

    # Initialize embedder (provider-configurable: voyage / ollama / gemini)
    embedder = make_embedder()

    # Initialize PostgreSQL vector store
    store = PostgreSQLVectorStore(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "obsidian_graph"),
        user=os.getenv("POSTGRES_USER", "obsidian"),
        password=os.getenv("POSTGRES_PASSWORD"),
        min_connections=int(os.getenv("POSTGRES_MIN_CONNECTIONS", "5")),
        max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "20")),
    )

    await store.initialize()

    # Initialize graph builder and hub analyzer
    graph_builder = GraphBuilder(store)
    hub_analyzer = HubAnalyzer(store)

    # Create tool context
    _tool_context = ToolContext(
        store=store,
        embedder=embedder,
        graph_builder=graph_builder,
        hub_analyzer=hub_analyzer,
        vault_path=vault_path,
    )

    # Start file watching if enabled
    watch_enabled = os.getenv("OBSIDIAN_WATCH_ENABLED", "true").lower() == "true"

    if watch_enabled and os.path.exists(vault_path):
        _vault_watcher = VaultWatcher(
            vault_path,
            store,
            embedder,
            debounce_seconds=int(os.getenv("OBSIDIAN_DEBOUNCE_SECONDS", "30")),
        )

        loop = asyncio.get_running_loop()
        _vault_watcher.start(loop)
        await _vault_watcher.startup_scan()

        logger.success(f"File watching enabled: {vault_path}")
    else:
        logger.info("File watching disabled")

    logger.success("Server initialized successfully")


# -- MCP Tool Schemas --


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="search_notes",
            description="Semantic search across Obsidian vault using natural language queries",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (1-50)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_similar_notes",
            description="Find notes semantically similar to a given note",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_path": {
                        "type": "string",
                        "description": "Path to the source note (vault-relative)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (1-50)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["note_path"],
            },
        ),
        Tool(
            name="get_connection_graph",
            description=(
                "Build multi-hop connection graph using BFS traversal" " to discover relationships"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "note_path": {
                        "type": "string",
                        "description": "Starting note path (vault-relative)",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Maximum levels to traverse (1-5)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "max_per_level": {
                        "type": "integer",
                        "description": "Maximum nodes per level (1-10)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["note_path"],
            },
        ),
        Tool(
            name="get_hub_notes",
            description="Identify highly connected notes (conceptual hubs/anchors)",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_connections": {
                        "type": "integer",
                        "description": "Minimum connection count to qualify as hub",
                        "default": 10,
                        "minimum": 1,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Similarity threshold for counting connections (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (1-50)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
            },
        ),
        Tool(
            name="get_orphaned_notes",
            description="Find isolated notes with few connections",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_connections": {
                        "type": "integer",
                        "description": "Maximum connection count to qualify as orphan",
                        "default": 2,
                        "minimum": 0,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Similarity threshold for counting connections (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (1-50)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
            },
        ),
    ]


# -- Response Formatters --


def _format_search_results(data: dict) -> str:
    results = data["results"]
    response = f"Found {len(results)} notes:\n\n"
    for i, r in enumerate(results, 1):
        snippet = r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"]
        response += f"{i}. **{r['title']}** (similarity: {r['similarity']:.3f})\n"
        response += f"   Path: `{r['path']}`\n"
        response += f"   {snippet}\n\n"
    return response


def _format_similar_notes(data: dict) -> str:
    response = f"Notes similar to `{data['note_path']}`:\n\n"
    for i, r in enumerate(data["results"], 1):
        response += f"{i}. **{r['title']}** (similarity: {r['similarity']:.3f})\n"
        response += f"   Path: `{r['path']}`\n\n"
    return response


def _format_connection_graph(graph: dict) -> str:
    response = f"# Connection Graph: {graph['root']['title']}\n\n"
    response += f"**Starting note:** `{graph['root']['path']}`\n"
    response += (
        f"**Network size:** {graph['stats']['total_nodes']} nodes, "
        f"{graph['stats']['total_edges']} edges\n\n"
    )

    nodes_by_level: dict[int, list] = {}
    for node in graph["nodes"]:
        nodes_by_level.setdefault(node["level"], []).append(node)

    for level in sorted(nodes_by_level.keys()):
        response += f"\n## Level {level}\n"
        for node in nodes_by_level[level]:
            response += f"- **{node['title']}** (`{node['path']}`)\n"
            if node["parent_path"]:
                edge = next(
                    (e for e in graph["edges"] if e["target"] == node["path"]),
                    None,
                )
                if edge:
                    response += (
                        f"  Connected from: `{node['parent_path']}` "
                        f"(similarity: {edge['similarity']:.3f})\n"
                    )

    return response


def _format_hub_notes(data: dict) -> str:
    hubs = data["results"]
    if not hubs:
        return (
            f"No hub notes found with >={data['min_connections']} connections "
            f"at threshold {data['threshold']}"
        )

    response = "# Hub Notes (Highly Connected)\n\n"
    response += f"Found {len(hubs)} notes with >={data['min_connections']} connections:\n\n"
    for i, hub in enumerate(hubs, 1):
        response += f"{i}. **{hub['title']}** ({hub['connection_count']} connections)\n"
        response += f"   Path: `{hub['path']}`\n\n"
    return response


def _format_orphaned_notes(data: dict) -> str:
    orphans = data["results"]
    if not orphans:
        return f"No orphaned notes found with <={data['max_connections']} connections"

    response = "# Orphaned Notes (Isolated)\n\n"
    response += f"Found {len(orphans)} notes with <={data['max_connections']} connections:\n\n"
    for i, orphan in enumerate(orphans, 1):
        response += f"{i}. **{orphan['title']}** ({orphan['connection_count']} connections)\n"
        response += f"   Path: `{orphan['path']}`\n"
        if orphan.get("modified_at"):
            response += f"   Modified: {orphan['modified_at']}\n"
        response += "\n"
    return response


_FORMATTERS = {
    "search_notes": _format_search_results,
    "get_similar_notes": _format_similar_notes,
    "get_connection_graph": _format_connection_graph,
    "get_hub_notes": _format_hub_notes,
    "get_orphaned_notes": _format_orphaned_notes,
}


# -- MCP Tool Dispatch --


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Dispatch MCP tool calls to handlers in tools.py."""
    logger.info(f"Tool called: {name} with args: {list(arguments.keys())}")

    ctx = _tool_context
    if ctx is None:
        logger.error("Server context not initialized")
        return [{"type": "text", "text": "Error: Server not initialized"}]

    handler = TOOLS.get(name)
    if not handler:
        return [{"type": "text", "text": f"Unknown tool: {name}"}]

    try:
        result = await handler(ctx, arguments)
        formatted = _FORMATTERS[name](result)
        return [{"type": "text", "text": formatted}]

    except ValidationError as e:
        logger.warning(f"Validation error in {name}: {e}")
        return [{"type": "text", "text": f"Validation Error: {str(e)}"}]
    except SecurityError as e:
        logger.warning(f"Security validation failed for {name}: {e}")
        return [{"type": "text", "text": f"Security Error: {str(e)}"}]
    except ToolError as e:
        logger.error(f"Tool error in {name}: {e}")
        return [{"type": "text", "text": f"Error: {str(e)}"}]
    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return [{"type": "text", "text": f"Error: {str(e)}"}]


async def main():
    """Run the MCP server."""
    await initialize_server()

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
