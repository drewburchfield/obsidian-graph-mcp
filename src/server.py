"""
Obsidian Graph MCP Server

Provides semantic knowledge graph navigation for Obsidian vaults.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool
from loguru import logger

from .vector_store import PostgreSQLVectorStore, Note, SearchResult
from .embedder import VoyageEmbedder
from .graph_builder import GraphBuilder
from .hub_analyzer import HubAnalyzer
from .file_watcher import VaultWatcher
from .security_utils import validate_note_path_parameter, SecurityError
from .validation import (
    validate_search_notes_args,
    validate_similar_notes_args,
    validate_connection_graph_args,
    validate_hub_notes_args,
    validate_orphaned_notes_args,
    ValidationError
)
from .exceptions import EmbeddingError, DatabaseError


@dataclass
class ServerContext:
    """
    Encapsulates server dependencies for dependency injection and testing.

    Benefits:
        - Makes dependencies explicit
        - Easier unit testing (inject mock context)
        - Foundation for future full DI migration
        - No breaking changes (internal refactor)
    """
    store: PostgreSQLVectorStore
    embedder: VoyageEmbedder
    graph_builder: GraphBuilder
    hub_analyzer: HubAnalyzer
    vault_watcher: Optional[VaultWatcher] = None


# Global server context (initialized once at startup)
_server_context: Optional[ServerContext] = None

# Create MCP server
app = Server("obsidian-graph")


async def initialize_server():
    """Initialize server context with all components."""
    global _server_context

    logger.info("Initializing Obsidian Graph MCP Server...")

    # Initialize embedder
    embedder = VoyageEmbedder(
        model="voyage-context-3",
        cache_dir=os.getenv("CACHE_DIR", str(Path.home() / ".obsidian-graph" / "cache")),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "128")),
        requests_per_minute=int(os.getenv("EMBEDDING_REQUESTS_PER_MINUTE", "300"))
    )

    # Initialize PostgreSQL vector store
    store = PostgreSQLVectorStore(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "obsidian_graph"),
        user=os.getenv("POSTGRES_USER", "obsidian"),
        password=os.getenv("POSTGRES_PASSWORD"),
        min_connections=int(os.getenv("POSTGRES_MIN_CONNECTIONS", "5")),
        max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "20"))
    )

    await store.initialize()

    # Initialize graph builder and hub analyzer
    graph_builder = GraphBuilder(store)
    hub_analyzer = HubAnalyzer(store)

    # Start file watching if enabled
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", "/vault")
    watch_enabled = os.getenv("OBSIDIAN_WATCH_ENABLED", "true").lower() == "true"

    vault_watcher = None
    if watch_enabled and os.path.exists(vault_path):
        vault_watcher = VaultWatcher(
            vault_path,
            store,
            embedder,
            debounce_seconds=int(os.getenv("OBSIDIAN_DEBOUNCE_SECONDS", "30"))
        )

        # Start file watching first (creates event_handler)
        loop = asyncio.get_event_loop()
        vault_watcher.start(loop)

        # Run startup scan to catch files changed while offline
        await vault_watcher.startup_scan()

        logger.success(f"File watching enabled: {vault_path}")
    else:
        logger.info("File watching disabled")

    # Create server context
    _server_context = ServerContext(
        store=store,
        embedder=embedder,
        graph_builder=graph_builder,
        hub_analyzer=hub_analyzer,
        vault_watcher=vault_watcher
    )

    logger.success("Server initialized successfully")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="search_notes",
            description="Semantic search across Obsidian vault using natural language queries",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (1-50)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name= "get_similar_notes",
            description= "Find notes semantically similar to a given note",
            inputSchema= {
                "type": "object",
                "properties": {
                    "note_path": {
                        "type": "string",
                        "description": "Path to the source note (vault-relative)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (1-50)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["note_path"]
            }
        ),
        Tool(
            name="get_connection_graph",
            description= "Build multi-hop connection graph using BFS traversal to discover relationships",
            inputSchema= {
                "type": "object",
                "properties": {
                    "note_path": {
                        "type": "string",
                        "description": "Starting note path (vault-relative)"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Maximum levels to traverse (1-5)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 5
                    },
                    "max_per_level": {
                        "type": "integer",
                        "description": "Maximum nodes per level (1-10)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["note_path"]
            }
        ),
        Tool(
            name="get_hub_notes",
            description= "Identify highly connected notes (conceptual hubs/anchors)",
            inputSchema= {
                "type": "object",
                "properties": {
                    "min_connections": {
                        "type": "integer",
                        "description": "Minimum connection count to qualify as hub",
                        "default": 10,
                        "minimum": 1
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Similarity threshold for counting connections (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (1-50)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 50
                    }
                }
            }
        ),
        Tool(
            name="get_orphaned_notes",
            description= "Find isolated notes with few connections",
            inputSchema= {
                "type": "object",
                "properties": {
                    "max_connections": {
                        "type": "integer",
                        "description": "Maximum connection count to qualify as orphan",
                        "default": 2,
                        "minimum": 0
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Similarity threshold for counting connections (0.0-1.0)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (1-50)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 50
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Handle tool calls with comprehensive input validation.

    Security features:
    - All parameters validated before processing
    - Path traversal protection for note_path parameters
    - Type checking and range validation
    - Graceful error handling with descriptive messages
    """
    # Log tool call for debugging
    logger.info(f"Tool called: {name} with args: {list(arguments.keys())}")

    # Get server context
    ctx = _server_context
    if ctx is None:
        logger.error("Server context not initialized")
        return [{"type": "text", "text": "Error: Server not initialized"}]

    if name == "search_notes":
        try:
            # Validate arguments
            validated = validate_search_notes_args(arguments)
            query = validated["query"]
            limit = validated["limit"]
            threshold = validated["threshold"]

            # Generate query embedding
            try:
                query_embedding = ctx.embedder.embed(query, input_type="query")
            except EmbeddingError as e:
                logger.error(f"Query embedding failed: {e}", exc_info=True)
                return [{"type": "text", "text": f"Error: Failed to generate query embedding: {e}"}]

            # Search
            results = await ctx.store.search(query_embedding, limit, threshold)

            # Format results
            response = f"Found {len(results)} notes:\n\n"
            for i, result in enumerate(results, 1):
                snippet = result.content[:200] + "..." if len(result.content) > 200 else result.content
                response += f"{i}. **{result.title}** (similarity: {result.similarity:.3f})\n"
                response += f"   Path: `{result.path}`\n"
                response += f"   {snippet}\n\n"

            return [{"type": "text", "text": response}]

        except ValidationError as e:
            logger.warning(f"Validation error in search_notes: {e}")
            return [{"type": "text", "text": f"Validation Error: {str(e)}"}]
        except Exception as e:
            logger.error(f"Error in search_notes: {e}", exc_info=True)
            return [{"type": "text", "text": f"Error: {str(e)}"}]

    elif name == "get_similar_notes":
        try:
            # Validate arguments
            validated = validate_similar_notes_args(arguments)

            # SECURITY: Validate note_path before processing
            note_path = validate_note_path_parameter(
                validated["note_path"],
                vault_path=os.getenv("OBSIDIAN_VAULT_PATH", "/vault")
            )
            limit = validated["limit"]
            threshold = validated["threshold"]

            # Get similar notes
            results = await ctx.store.get_similar_notes(note_path, limit, threshold)

            # Format results
            response = f"Notes similar to `{note_path}`:\n\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. **{result.title}** (similarity: {result.similarity:.3f})\n"
                response += f"   Path: `{result.path}`\n\n"

            return [{"type": "text", "text": response}]

        except ValidationError as e:
            logger.warning(f"Validation error in get_similar_notes: {e}")
            return [{"type": "text", "text": f"Validation Error: {str(e)}"}]
        except SecurityError as e:
            logger.warning(f"Security validation failed for get_similar_notes: {e}")
            return [{"type": "text", "text": f"Security Error: {str(e)}"}]
        except Exception as e:
            logger.error(f"Error in get_similar_notes: {e}", exc_info=True)
            return [{"type": "text", "text": f"Error: {str(e)}"}]

    elif name == "get_connection_graph":
        try:
            # Validate arguments
            validated = validate_connection_graph_args(arguments)

            # SECURITY: Validate note_path before processing
            note_path = validate_note_path_parameter(
                validated["note_path"],
                vault_path=os.getenv("OBSIDIAN_VAULT_PATH", "/vault")
            )
            depth = validated["depth"]
            max_per_level = validated["max_per_level"]
            threshold = validated["threshold"]

            # Build connection graph
            graph = await ctx.graph_builder.build_connection_graph(
                note_path, depth, max_per_level, threshold
            )

            # Format results
            response = f"# Connection Graph: {graph['root']['title']}\n\n"
            response += f"**Starting note:** `{graph['root']['path']}`\n"
            response += f"**Network size:** {graph['stats']['total_nodes']} nodes, {graph['stats']['total_edges']} edges\n\n"

            # Group nodes by level
            nodes_by_level = {}
            for node in graph['nodes']:
                level = node['level']
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)

            # Display by level
            for level in sorted(nodes_by_level.keys()):
                response += f"\n## Level {level}\n"
                for node in nodes_by_level[level]:
                    response += f"- **{node['title']}** (`{node['path']}`)\n"
                    if node['parent_path']:
                        # Find edge to get similarity
                        edge = next((e for e in graph['edges'] if e['target'] == node['path']), None)
                        if edge:
                            response += f"  Connected from: `{node['parent_path']}` (similarity: {edge['similarity']:.3f})\n"

            return [{"type": "text", "text": response}]

        except ValidationError as e:
            logger.warning(f"Validation error in get_connection_graph: {e}")
            return [{"type": "text", "text": f"Validation Error: {str(e)}"}]
        except SecurityError as e:
            logger.warning(f"Security validation failed for get_connection_graph: {e}")
            return [{"type": "text", "text": f"Security Error: {str(e)}"}]
        except Exception as e:
            logger.error(f"Error in get_connection_graph: {e}", exc_info=True)
            return [{"type": "text", "text": f"Error: {str(e)}"}]

    elif name == "get_hub_notes":
        try:
            # Validate arguments
            validated = validate_hub_notes_args(arguments)
            min_connections = validated["min_connections"]
            threshold = validated["threshold"]
            limit = validated["limit"]

            # Get hub notes
            hubs = await ctx.hub_analyzer.get_hub_notes(min_connections, threshold, limit)

            # Format results
            if not hubs:
                response = f"No hub notes found with >={min_connections} connections at threshold {threshold}"
            else:
                response = f"# Hub Notes (Highly Connected)\n\n"
                response += f"Found {len(hubs)} notes with >={min_connections} connections:\n\n"
                for i, hub in enumerate(hubs, 1):
                    response += f"{i}. **{hub['title']}** ({hub['connection_count']} connections)\n"
                    response += f"   Path: `{hub['path']}`\n\n"

            return [{"type": "text", "text": response}]

        except ValidationError as e:
            logger.warning(f"Validation error in get_hub_notes: {e}")
            return [{"type": "text", "text": f"Validation Error: {str(e)}"}]
        except Exception as e:
            logger.error(f"Error in get_hub_notes: {e}", exc_info=True)
            return [{"type": "text", "text": f"Error: {str(e)}"}]

    elif name == "get_orphaned_notes":
        try:
            # Validate arguments
            validated = validate_orphaned_notes_args(arguments)
            max_connections = validated["max_connections"]
            threshold = validated["threshold"]
            limit = validated["limit"]

            # Get orphaned notes
            orphans = await ctx.hub_analyzer.get_orphaned_notes(max_connections, threshold, limit)

            # Format results
            if not orphans:
                response = f"No orphaned notes found with <={max_connections} connections"
            else:
                response = f"# Orphaned Notes (Isolated)\n\n"
                response += f"Found {len(orphans)} notes with <={max_connections} connections:\n\n"
                for i, orphan in enumerate(orphans, 1):
                    response += f"{i}. **{orphan['title']}** ({orphan['connection_count']} connections)\n"
                    response += f"   Path: `{orphan['path']}`\n"
                    if orphan['modified_at']:
                        response += f"   Modified: {orphan['modified_at']}\n"
                    response += "\n"

            return [{"type": "text", "text": response}]

        except ValidationError as e:
            logger.warning(f"Validation error in get_orphaned_notes: {e}")
            return [{"type": "text", "text": f"Validation Error: {str(e)}"}]
        except Exception as e:
            logger.error(f"Error in get_orphaned_notes: {e}", exc_info=True)
            return [{"type": "text", "text": f"Error: {str(e)}"}]

    else:
        return [{"type": "text", "text": f"Unknown tool: {name}"}]


async def main():
    """Run the MCP server."""
    await initialize_server()

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
