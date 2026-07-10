"""
Tool handlers for Obsidian Graph.

Transport-agnostic tool implementations that validate inputs, call engine
components, and return structured results. Used by server.py (MCP) and
can be reused by future transports (CLI, REST).
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .embedder import VoyageEmbedder
from .exceptions import EmbeddingError
from .graph_builder import GraphBuilder
from .hub_analyzer import HubAnalyzer
from .reranker import CohereReranker
from .security_utils import validate_note_path_parameter
from .validation import (
    validate_connection_graph_args,
    validate_hub_notes_args,
    validate_orphaned_notes_args,
    validate_search_notes_args,
    validate_similar_notes_args,
)
from .vector_store import PostgreSQLVectorStore

# --- consulting-graph retrieval pipeline (verified: hybrid + Cohere rerank) ---
_RERANKER: CohereReranker | None = None
_POOL_SIZE = int(os.getenv("RERANK_POOL", "50"))


def _rerank_enabled() -> bool:
    return os.getenv("CONSULTING_RERANK", "").lower() in ("1", "true", "yes")


def _reranker() -> CohereReranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CohereReranker()
    return _RERANKER


def _rrf(*result_lists, k: int = 60):
    """Reciprocal Rank Fusion over SearchResult lists, keyed by chunk content."""
    scores: dict[str, float] = {}
    obj: dict[str, Any] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            key = r.content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            obj[key] = r
    return [obj[key] for key, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


@dataclass
class ToolContext:
    """Dependencies needed by tool handlers."""

    store: PostgreSQLVectorStore
    embedder: VoyageEmbedder
    graph_builder: GraphBuilder
    hub_analyzer: HubAnalyzer
    vault_path: str = "/vault"


class ToolError(Exception):
    """Raised when a tool handler fails. Contains a user-facing message."""

    pass


async def search_notes(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Semantic search across vault.

    Returns:
        {"results": [{"path", "title", "content", "similarity"}, ...]}
    """
    validated = validate_search_notes_args(arguments)
    query = validated["query"]
    limit = validated["limit"]

    try:
        query_embedding = await ctx.embedder.embed(query, input_type="query")
    except EmbeddingError as e:
        raise ToolError(f"Failed to generate query embedding: {e}") from e

    if not _rerank_enabled():
        results = await ctx.store.search(query_embedding, limit, validated["threshold"])
    else:
        # Consulting pipeline: dense + BM25 hybrid candidate pool, then FUSE the
        # hybrid ranking with the Cohere rerank ranking (RRF). Fusing (not pure
        # rerank order) keeps the reranker's signal without letting it bury a
        # strong dense hit -- on the braintrust eval this is the robust choice
        # (best R@10), since results feed an LLM that reads the whole top-N.
        dense = await ctx.store.search(query_embedding, _POOL_SIZE, threshold=0.0)
        lexical = await ctx.store.lexical_search(query, _POOL_SIZE)
        pool = _rrf(dense, lexical)[:_POOL_SIZE]
        try:
            order = await asyncio.to_thread(
                _reranker().rerank, query, [r.content for r in pool], len(pool)
            )
            rerank_score = {pool[idx].content: float(s) for idx, s in order}
            rerank_ranked = [pool[idx] for idx, _ in order]
            fused = _rrf(pool, rerank_ranked)[:limit]
            results = [
                type(r)(
                    path=r.path,
                    title=r.title,
                    content=r.content,
                    similarity=rerank_score.get(r.content, r.similarity),
                )
                for r in fused
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Rerank failed, falling back to hybrid order: {e}")
            results = pool[:limit]

    return {
        "results": [
            {
                "path": r.path,
                "title": r.title,
                "content": r.content,
                "similarity": r.similarity,
            }
            for r in results
        ]
    }


async def get_similar_notes(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Find notes similar to a given note.

    Returns:
        {"note_path": str, "results": [{"path", "title", "similarity"}, ...]}
    """
    validated = validate_similar_notes_args(arguments)
    note_path = validate_note_path_parameter(validated["note_path"], vault_path=ctx.vault_path)

    results = await ctx.store.get_similar_notes(
        note_path, validated["limit"], validated["threshold"]
    )

    return {
        "note_path": note_path,
        "results": [
            {
                "path": r.path,
                "title": r.title,
                "similarity": r.similarity,
            }
            for r in results
        ],
    }


async def get_connection_graph(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Build multi-hop connection graph from a starting note.

    Returns:
        The graph dict from GraphBuilder (root, nodes, edges, stats).
    """
    validated = validate_connection_graph_args(arguments)
    note_path = validate_note_path_parameter(validated["note_path"], vault_path=ctx.vault_path)

    try:
        return await ctx.graph_builder.build_connection_graph(
            note_path, validated["depth"], validated["max_per_level"], validated["threshold"]
        )
    except ValueError as e:
        raise ToolError(str(e)) from e


async def get_hub_notes(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Identify highly connected hub notes.

    Returns:
        {"min_connections": int, "threshold": float, "results": [{"path", "title", "connection_count"}, ...]}
    """
    validated = validate_hub_notes_args(arguments)

    hubs = await ctx.hub_analyzer.get_hub_notes(
        validated["min_connections"], validated["threshold"], validated["limit"]
    )

    return {
        "min_connections": validated["min_connections"],
        "threshold": validated["threshold"],
        "results": hubs,
    }


async def get_orphaned_notes(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Find isolated notes with few connections.

    Returns:
        {"max_connections": int, "results": [{"path", "title", "connection_count", "modified_at"}, ...]}
    """
    validated = validate_orphaned_notes_args(arguments)

    orphans = await ctx.hub_analyzer.get_orphaned_notes(
        validated["max_connections"], validated["threshold"], validated["limit"]
    )

    return {
        "max_connections": validated["max_connections"],
        "results": orphans,
    }


# Tool dispatch table
TOOLS = {
    "search_notes": search_notes,
    "get_similar_notes": get_similar_notes,
    "get_connection_graph": get_connection_graph,
    "get_hub_notes": get_hub_notes,
    "get_orphaned_notes": get_orphaned_notes,
}
