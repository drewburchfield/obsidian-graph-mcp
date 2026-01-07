"""
Graph builder for multi-hop connection discovery.

Uses Breadth-First Search (BFS) to build connection graphs from a starting note.
"""

from collections import deque

from loguru import logger

from .exceptions import DatabaseError
from .vector_store import PostgreSQLVectorStore


class GraphBuilder:
    """
    Builds multi-hop connection graphs using BFS traversal.

    Ensures level-by-level exploration without cycles.
    """

    def __init__(self, store: PostgreSQLVectorStore):
        """
        Initialize graph builder.

        Args:
            store: PostgreSQL vector store instance
        """
        self.store = store

    async def build_connection_graph(
        self, note_path: str, depth: int = 3, max_per_level: int = 5, threshold: float = 0.5
    ) -> dict:
        """
        Build multi-hop connection graph using BFS.

        Args:
            note_path: Starting note path
            depth: Maximum levels to traverse (1-5)
            max_per_level: Maximum nodes per level (1-10)
            threshold: Minimum similarity score (0.0-1.0)

        Returns:
            {
                "root": {"path": str, "title": str},
                "nodes": [{"path": str, "title": str, "level": int, "parent_path": str}],
                "edges": [{"source": str, "target": str, "similarity": float}],
                "stats": {"total_nodes": int, "total_edges": int, "levels": int}
            }
        """
        # Validate parameters
        depth = max(1, min(5, depth))  # Clamp to [1, 5]
        max_per_level = max(1, min(10, max_per_level))  # Clamp to [1, 10]

        # Initialize BFS
        visited: set[str] = set()
        queue = deque([(note_path, 0, None)])  # (path, level, parent_path)
        nodes = {}
        edges = []

        # Get root note info
        root_note = await self._get_note_info(note_path)
        if not root_note:
            raise ValueError(f"Root note not found: {note_path}")

        logger.debug(f"Building graph from: {root_note['title']} (depth={depth})")

        # BFS traversal
        while queue:
            current_path, level, parent_path = queue.popleft()

            # Skip if already visited or depth exceeded
            if current_path in visited or level > depth:
                continue

            visited.add(current_path)

            # Get note info
            try:
                note_info = await self._get_note_info(current_path)
                if not note_info:
                    continue
            except DatabaseError as e:
                logger.error(f"Database error fetching {current_path}: {e}")
                continue  # Skip this node, continue BFS

            # Add to nodes
            nodes[current_path] = {
                "path": note_info["path"],
                "title": note_info["title"],
                "level": level,
                "parent_path": parent_path,
            }

            # Add edge if not root
            if parent_path and parent_path in nodes:
                try:
                    similarity = await self._compute_similarity(parent_path, current_path)
                except DatabaseError as e:
                    logger.warning(f"Could not compute similarity, using 0.0: {e}")
                    similarity = 0.0

                edges.append(
                    {
                        "source": parent_path,
                        "target": current_path,
                        "similarity": round(similarity, 4),
                    }
                )

            # Expand neighbors for next level
            if level < depth:
                neighbors = await self.store.get_similar_notes(
                    current_path, limit=max_per_level, threshold=threshold
                )

                for neighbor in neighbors:
                    if neighbor.path not in visited:
                        queue.append((neighbor.path, level + 1, current_path))

        logger.info(f"Graph built: {len(nodes)} nodes, {len(edges)} edges, " f"{depth} levels")

        return {
            "root": {"path": root_note["path"], "title": root_note["title"]},
            "nodes": list(nodes.values()),
            "edges": edges,
            "stats": {"total_nodes": len(nodes), "total_edges": len(edges), "levels": depth},
        }

    async def _get_note_info(self, note_path: str) -> dict | None:
        """
        Get basic note information (path and title).

        Args:
            note_path: Path to note

        Returns:
            Dict with path and title, or None if not found

        Raises:
            DatabaseError: If database query fails (not "not found")
        """
        if not self.store.pool:
            raise DatabaseError("Store not initialized")

        try:
            async with self.store.pool.acquire() as conn:
                result = await conn.fetchrow(
                    "SELECT path, title FROM notes WHERE path = $1", note_path
                )

                if result:
                    return {"path": result["path"], "title": result["title"]}
                return None  # Not found is OK (returns None)

        except Exception as e:
            logger.error(f"Error fetching note info for {note_path}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to fetch note info: {e}") from e

    async def _compute_similarity(self, path1: str, path2: str) -> float:
        """
        Compute similarity between two notes.

        Args:
            path1: First note path
            path2: Second note path

        Returns:
            Similarity score (0.0-1.0), or 0.0 if embeddings missing

        Raises:
            DatabaseError: If similarity computation fails
        """
        if not self.store.pool:
            raise DatabaseError("Store not initialized")

        try:
            async with self.store.pool.acquire() as conn:
                # Get both embeddings and compute similarity in single query
                # This avoids asyncpg connection state issues with multiple vector queries
                result = await conn.fetchval(
                    """
                    SELECT 1.0 - (n1.embedding <=> n2.embedding) AS similarity
                    FROM
                        (SELECT embedding FROM notes WHERE path = $1 ORDER BY chunk_index LIMIT 1) n1,
                        (SELECT embedding FROM notes WHERE path = $2 ORDER BY chunk_index LIMIT 1) n2
                    """,
                    path1,
                    path2,
                )

                if result is None:
                    # CROSS JOIN returns no rows when either note is missing
                    logger.warning(
                        f"Cannot compute similarity - missing embedding(s) for {path1} or {path2}"
                    )
                    return 0.0

                return float(result)

        except Exception as e:
            logger.error(f"Error computing similarity: {e}", exc_info=True)
            raise DatabaseError(f"Similarity computation failed: {e}") from e
