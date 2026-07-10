"""
Multi-format indexer: index every supported document under a root folder.

Generalizes the markdown-only indexer to the full document set (PDF, DOCX,
PPTX, HTML, CSV, XLSX, TXT) via converters.convert_file(), embedding with any
embedder that exposes embed_with_chunks()/chunk_text() (Gemini or Voyage).

Self-contained directory exclusion: a personal/consulting folder is full of
vendored junk (.venv, node_modules, site-packages, build output) that dwarfs
the real documents. We hard-exclude those by directory name and by path
fragment so the index stays to the work that was actually authored.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .converters import SUPPORTED_EXTS, convert_file
from .exceptions import EmbeddingError
from .exclusion import ExclusionFilter, load_exclusion_filter
from .vector_store import Note, PostgreSQLVectorStore

# Directory names that never contain authored documents.
EXCLUDED_DIR_NAMES = {
    ".git",
    ".github",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "site-packages",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".wrangler",
    ".playwright-mcp",
    ".agents",
    ".braintrust",
    "dist",
    "build",
    ".next",
    ".cache",
    "data",
    # Archive convention across the consulting tree: directories named "_archive"
    # (superseded work, any depth) and top-level "archive" (former clients /
    # lost prospects, added 2026-06-25) hold dead context. Never index either.
    "_archive",
    "archive",
    # Agent/skill tooling config, not consulting work product.
    ".claude",
}

# Exact filenames that are tooling config, not authored consulting content.
EXCLUDED_FILENAMES = {
    "AGENTS.md",
}

# Path fragments (substring match on the relative path) to exclude.
EXCLUDED_PATH_FRAGMENTS = (
    "dist-info",
    ".egg-info",
    # The UPROXX codebase is a separate corpus, not consulting documents. Match
    # without a leading slash so it hits the root-relative path "uproxx/code/...".
    "uproxx/code/",
    # Demo/sample artifact using the placeholder client "Acme Logistics" (the
    # discovery memo is literally labeled a template). Not a real engagement.
    "acme-logistics/",
)


def _is_excluded(rel_path: str, parts: tuple[str, ...]) -> bool:
    if any(p in EXCLUDED_DIR_NAMES for p in parts):
        return True
    if parts and parts[-1] in EXCLUDED_FILENAMES:
        return True
    return any(frag in rel_path for frag in EXCLUDED_PATH_FRAGMENTS)


def scan_documents(
    root_path: str,
    enabled_extensions: set[str] | None = None,
    exclusion_filter: ExclusionFilter | None = None,
) -> list[Path]:
    """Find every supported, non-excluded document under root_path."""
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root_path}")

    enabled_extensions = enabled_extensions or SUPPORTED_EXTS
    exclusion_filter = exclusion_filter or load_exclusion_filter(root_path)
    found: list[Path] = []
    excluded = 0
    for current_root, dir_names, file_names in os.walk(root):
        current = Path(current_root)
        kept_dirs = []
        for directory in dir_names:
            directory_path = current / directory
            rel_directory = str(directory_path.relative_to(root))
            if exclusion_filter.should_exclude(f"{rel_directory}/placeholder"):
                excluded += 1
            else:
                kept_dirs.append(directory)
        dir_names[:] = kept_dirs

        for file_name in file_names:
            path = current / file_name
            if path.suffix.lower() not in enabled_extensions:
                continue
            rel = str(path.relative_to(root))
            if exclusion_filter.should_exclude(rel):
                excluded += 1
                continue
            found.append(path)

    by_ext: dict[str, int] = {}
    for p in found:
        by_ext[p.suffix.lower()] = by_ext.get(p.suffix.lower(), 0) + 1
    logger.info(
        f"Found {len(found)} documents under {root_path} ({excluded} excluded). By type: {by_ext}"
    )
    return found


async def index_root(
    root_path: str,
    store: PostgreSQLVectorStore,
    embedder,
    *,
    batch_size: int = 50,
) -> dict:
    """
    Index every supported document under root_path into the vector store.

    Returns a summary dict: {indexed, files, chunks, failed}.
    """
    files = scan_documents(root_path)
    root = Path(root_path)

    total_chunks_indexed = 0
    failed: list[dict] = []

    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        logger.info(
            f"Batch {i // batch_size + 1}/{(len(files) + batch_size - 1) // batch_size} "
            f"({len(batch)} files)"
        )

        notes: list[Note] = []
        for file_path in batch:
            content = convert_file(file_path)
            if not content:
                continue

            try:
                stat = file_path.stat()
                modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
                rel_path = str(file_path.relative_to(root))

                embeddings_list, total_chunks = await embedder.embed_with_chunks(
                    content, chunk_size=2000, input_type="document"
                )

                if total_chunks == 1:
                    notes.append(
                        Note(
                            path=rel_path,
                            title=file_path.stem,
                            content=content,
                            embedding=embeddings_list[0],
                            modified_at=modified_at,
                            file_size_bytes=stat.st_size,
                            chunk_index=0,
                            total_chunks=1,
                        )
                    )
                else:
                    chunks = embedder.chunk_text(content, chunk_size=2000, overlap=0)
                    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings_list, strict=False)):
                        notes.append(
                            Note(
                                path=rel_path,
                                title=file_path.stem,
                                content=chunk,
                                embedding=emb,
                                modified_at=modified_at,
                                file_size_bytes=stat.st_size,
                                chunk_index=idx,
                                total_chunks=total_chunks,
                            )
                        )
            except EmbeddingError as e:
                logger.error(f"Failed to embed {file_path.name}: {e}")
                failed.append({"path": str(file_path), "error": str(e)})
            except Exception as e:  # noqa: BLE001 - keep going on a single bad file
                logger.error(f"Error indexing {file_path.name}: {e}")
                failed.append({"path": str(file_path), "error": str(e)})

        if notes:
            count = await store.upsert_batch(notes)
            total_chunks_indexed += count
            logger.info(f"Indexed {count} chunks (running total: {total_chunks_indexed})")

    # Prune stale rows: anything in the DB whose path is no longer a scanned file
    # under root (moved, deleted, or now-excluded). Without this, re-indexing only
    # upserts present files and leaves orphaned entries for old paths.
    pruned = 0
    valid_paths = {str(f.relative_to(root)) for f in files}
    if store.pool is not None and valid_paths:  # never prune against an empty scan
        async with store.pool.acquire() as conn:
            db_paths = {r["path"] for r in await conn.fetch("SELECT DISTINCT path FROM notes")}
        stale = sorted(db_paths - valid_paths)
        if stale:
            pruned = await store.delete_notes_by_paths(stale)
            logger.info(f"Pruned {pruned} stale paths (moved/deleted/excluded)")

    summary = {
        "indexed": total_chunks_indexed,
        "files": len(files),
        "chunks": total_chunks_indexed,
        "failed": len(failed),
        "pruned": pruned,
    }
    if failed:
        logger.warning(
            f"{len(failed)} files failed:\n"
            + "\n".join(f"  - {Path(f['path']).name}: {f['error']}" for f in failed[:10])
        )
    logger.success(f"Index complete: {summary}")
    return summary
