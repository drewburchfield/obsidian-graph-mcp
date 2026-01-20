"""
Initial vault indexing for Obsidian Graph MCP Server.

Scans Obsidian vault, generates embeddings, and populates PostgreSQL.
"""

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .embedder import VoyageEmbedder
from .exceptions import EmbeddingError
from .exclusion import ExclusionFilter, cleanup_excluded_notes, load_exclusion_filter
from .vector_store import Note, PostgreSQLVectorStore


def scan_vault(vault_path: str, exclusion_filter: ExclusionFilter | None = None) -> list[Path]:
    """
    Scan Obsidian vault for all markdown files.

    Respects exclusion patterns from .obsidian-graph.conf.

    Args:
        vault_path: Path to Obsidian vault
        exclusion_filter: Optional pre-loaded filter (avoids double-loading)

    Returns:
        List of markdown file paths (excluding filtered paths)
    """
    vault = Path(vault_path)
    if not vault.exists():
        raise FileNotFoundError(f"Vault not found: {vault_path}")

    # Use provided filter or load one
    if exclusion_filter is None:
        exclusion_filter = load_exclusion_filter(vault_path)

    # Find all markdown files
    all_md_files = list(vault.rglob("*.md"))

    # Filter out excluded paths
    md_files = []
    excluded_count = 0
    for file_path in all_md_files:
        rel_path = str(file_path.relative_to(vault))
        if exclusion_filter.should_exclude(rel_path):
            excluded_count += 1
            continue
        md_files.append(file_path)

    logger.info(
        f"Found {len(md_files)} markdown files in vault ({excluded_count} excluded by filters)"
    )
    return md_files


def extract_title(file_path: Path) -> str:
    """
    Extract note title from filename or frontmatter.

    Args:
        file_path: Path to markdown file

    Returns:
        Note title
    """
    # Use filename without extension as title
    # TODO: Could parse frontmatter for custom titles
    return file_path.stem


async def index_vault(vault_path: str, batch_size: int = 100):
    """
    Index entire Obsidian vault.

    Args:
        vault_path: Path to Obsidian vault
        batch_size: Number of notes to process per batch
    """
    logger.info(f"Starting vault indexing: {vault_path}")

    # Initialize components
    embedder = VoyageEmbedder(
        cache_dir=os.getenv("CACHE_DIR", str(Path.home() / ".obsidian-graph" / "cache"))
    )
    store = PostgreSQLVectorStore()
    await store.initialize()

    try:
        # Load exclusion filter once for both cleanup and scanning
        exclusion_filter = load_exclusion_filter(vault_path)

        # Clean up any previously indexed notes that are now excluded
        await cleanup_excluded_notes(store, vault_path, exclusion_filter)

        # Scan vault
        md_files = scan_vault(vault_path, exclusion_filter)
        vault_root = Path(vault_path)

        # Track failures across all batches
        all_failed_notes = []

        # Process in batches
        for i in range(0, len(md_files), batch_size):
            batch_files = md_files[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(md_files) + batch_size - 1) // batch_size}"
            )

            # Read files
            notes_data = []
            for file_path in batch_files:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read().strip()

                    # Skip empty files (Voyage API rejects empty strings)
                    if not content:
                        logger.warning(f"Skipping empty file: {file_path}")
                        continue

                    # Get file stats
                    stat = file_path.stat()
                    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
                    file_size = stat.st_size

                    # Get vault-relative path
                    rel_path = str(file_path.relative_to(vault_root))

                    notes_data.append(
                        {
                            "path": rel_path,
                            "title": extract_title(file_path),
                            "content": content,
                            "modified_at": modified_at,
                            "file_size_bytes": file_size,
                        }
                    )

                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

            # Filter out notes with empty content
            # (Large notes will be auto-chunked by embed_with_chunks)
            valid_notes = []
            for note in notes_data:
                if not note["content"] or len(note["content"].strip()) == 0:
                    logger.warning(f"Skipping empty note: {note['path']}")
                    continue

                valid_notes.append(note)

            if not valid_notes:
                logger.warning(f"No valid notes in batch {i // batch_size + 1}")
                continue

            logger.info(f"Batch: {len(valid_notes)} valid notes")

            # Process each note with automatic chunking
            notes = []
            batch_failed_notes = []
            for note_data in valid_notes:
                # embed_with_chunks handles both small (whole) and large (chunked) notes
                try:
                    embeddings_list, total_chunks = embedder.embed_with_chunks(
                        note_data["content"],
                        chunk_size=2000,  # oachatbot standard
                        input_type="document",
                    )
                except EmbeddingError as e:
                    logger.error(f"Failed to embed {note_data['path']}: {e}")
                    batch_failed_notes.append({"path": note_data["path"], "error": str(e)})
                    all_failed_notes.append({"path": note_data["path"], "error": str(e)})
                    continue

                # Create Note object(s)
                if total_chunks == 1:
                    # Whole note (not chunked)
                    notes.append(
                        Note(
                            path=note_data["path"],
                            title=note_data["title"],
                            content=note_data["content"],
                            embedding=embeddings_list[0],
                            modified_at=note_data["modified_at"],
                            file_size_bytes=note_data["file_size_bytes"],
                            chunk_index=0,
                            total_chunks=1,
                        )
                    )
                else:
                    # Chunked note - create one Note per chunk
                    chunks = embedder.chunk_text(note_data["content"], chunk_size=2000, overlap=0)
                    logger.info(f"Chunked {note_data['path']}: {total_chunks} chunks")

                    for chunk_idx, (chunk_text, embedding) in enumerate(
                        zip(chunks, embeddings_list, strict=False)
                    ):
                        notes.append(
                            Note(
                                path=note_data["path"],
                                title=note_data["title"],
                                content=chunk_text,
                                embedding=embedding,
                                modified_at=note_data["modified_at"],
                                file_size_bytes=note_data["file_size_bytes"],
                                chunk_index=chunk_idx,
                                total_chunks=total_chunks,
                            )
                        )

            # Insert into PostgreSQL
            if notes:
                count = await store.upsert_batch(notes)
                logger.info(f"Indexed {count} note chunks")

        # Final stats
        total_notes = await store.get_note_count()
        cache_stats = embedder.get_cache_stats()

        # Report failures if any
        if all_failed_notes:
            logger.warning(
                f"Indexing completed with {len(all_failed_notes)} failures "
                f"(out of {len(md_files)} total files):\n"
                + "\n".join(f"  - {n['path']}: {n['error']}" for n in all_failed_notes[:10])
            )
            if len(all_failed_notes) > 10:
                logger.warning(f"  ... and {len(all_failed_notes) - 10} more failures")
            logger.info("Run indexer again to retry failed files, or check Voyage API key/quota")

        logger.success(
            f"Indexing complete! {total_notes} notes indexed successfully. "
            f"Cache: {cache_stats['total_cached']} embeddings, {cache_stats['cache_size_mb']} MB"
        )

    finally:
        await store.close()


async def main():
    """Run initial indexing."""
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", "/vault")
    await index_vault(vault_path)


if __name__ == "__main__":
    asyncio.run(main())
