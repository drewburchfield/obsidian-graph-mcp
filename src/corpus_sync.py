"""Incrementally reconcile a mounted document corpus with the vector store."""

from __future__ import annotations

import json
import os
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .converters import SUPPORTED_EXTS, convert_file
from .exclusion import ExclusionFilter, load_exclusion_filter
from .multi_format_indexer import scan_documents
from .vector_store import Note, PostgreSQLVectorStore


@dataclass(frozen=True)
class ReconcileSummary:
    scanned: int
    added: int
    updated: int
    unchanged: int
    removed: int
    failed: int


class CorpusSynchronizer:
    """Own conversion, embedding, atomic replacement, and corpus reconciliation."""

    def __init__(
        self,
        root_path: str,
        store: PostgreSQLVectorStore,
        embedder,
        *,
        enabled_extensions: set[str] | None = None,
        exclusion_filter: ExclusionFilter | None = None,
        state_path: str | None = None,
    ):
        self.root = Path(root_path)
        self.store = store
        self.embedder = embedder
        self.enabled_extensions = enabled_extensions or set(SUPPORTED_EXTS)
        self.exclusion_filter = exclusion_filter or load_exclusion_filter(root_path)
        self.state_path = Path(
            state_path
            or os.getenv(
                "CORPUS_SYNC_STATE_PATH",
                str(Path.home() / ".obsidian-graph" / "sync-state.json"),
            )
        )
        self._reconciling = False
        self._reconcile_lock = asyncio.Lock()

    def _write_state(self, status: str, **details) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
            "corpus": str(self.root),
            **details,
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    def is_eligible(self, path: str | Path) -> bool:
        candidate = Path(path)
        try:
            rel_path = str(candidate.relative_to(self.root))
        except ValueError:
            return False
        return (
            candidate.suffix.lower() in self.enabled_extensions
            and not self.exclusion_filter.should_exclude(rel_path)
        )

    async def remove_file(self, path: str | Path) -> int:
        try:
            rel_path = str(Path(path).relative_to(self.root))
        except ValueError:
            return 0
        removed = await self.store.delete_notes_by_paths([rel_path])
        if self._reconciling:
            self._write_state("syncing", last_event="remove", path=rel_path)
        else:
            self._write_state("ready", last_event="remove", path=rel_path)
        return removed

    async def reindex_file(self, path: str | Path) -> bool:
        file_path = Path(path)
        if not self.is_eligible(file_path) or not file_path.is_file():
            return False
        try:
            content = convert_file(file_path, raise_errors=True)
        except Exception as exc:  # noqa: BLE001 - preserve the last valid indexed version
            logger.warning(f"Failed to convert {file_path}: {exc}")
            if self._reconciling:
                self._write_state("syncing", last_event="conversion_failed", path=str(file_path))
            else:
                self._write_state("degraded", last_event="conversion_failed", path=str(file_path))
            return False
        if not content:
            if file_path.stat().st_size == 0:
                logger.warning(f"Skipping empty file: {file_path}")
                return False
            rel_path = str(file_path.relative_to(self.root))
            content = (
                f"# {file_path.stem}\n\n"
                f"File: {rel_path}\n"
                f"Format: {file_path.suffix.lower()}\n\n"
                "This file contains no extractable text."
            )
            logger.info(f"Indexing metadata fallback for {rel_path}")

        rel_path = str(file_path.relative_to(self.root))
        try:
            embeddings, total_chunks = await self.embedder.embed_with_chunks(
                content, chunk_size=2000, input_type="document"
            )
        except Exception as exc:  # noqa: BLE001 - one provider failure must not stop reconciliation
            logger.warning(f"Failed to embed {rel_path}: {exc}")
            if self._reconciling:
                self._write_state("syncing", last_event="embedding_failed", path=rel_path)
            else:
                self._write_state("degraded", last_event="embedding_failed", path=rel_path)
            return False

        stat = file_path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        chunks = (
            [content]
            if total_chunks == 1
            else self.embedder.chunk_text(content, chunk_size=2000, overlap=0)
        )
        notes = [
            Note(
                path=rel_path,
                title=file_path.stem,
                content=chunk,
                embedding=embedding,
                modified_at=modified_at,
                file_size_bytes=stat.st_size,
                chunk_index=index,
                total_chunks=total_chunks,
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False))
        ]
        if len(notes) != total_chunks:
            logger.error(
                f"Chunk count mismatch for {rel_path}: expected {total_chunks}, got {len(notes)}"
            )
            return False
        await self.store.replace_file_notes(rel_path, notes)
        if self._reconciling:
            self._write_state("syncing", last_event="index", path=rel_path)
        else:
            self._write_state("ready", last_event="index", path=rel_path)
        logger.info(f"Indexed {rel_path} ({len(notes)} chunks)")
        return True

    async def reconcile(self) -> ReconcileSummary:
        if self._reconcile_lock.locked():
            logger.info("Corpus reconciliation already running; skipping overlapping poll")
            return ReconcileSummary(0, 0, 0, 0, 0, 0)
        async with self._reconcile_lock:
            return await self._reconcile_once()

    async def _reconcile_once(self) -> ReconcileSummary:
        self._write_state("syncing")
        self._reconciling = True
        files = scan_documents(
            str(self.root),
            enabled_extensions=self.enabled_extensions,
            exclusion_filter=self.exclusion_filter,
        )
        metadata = await self.store.get_file_metadata()
        current = {str(path.relative_to(self.root)): path for path in files}
        stale_paths = sorted(set(metadata) - set(current))
        removed = await self.store.delete_notes_by_paths(stale_paths) if stale_paths else 0

        added = updated = unchanged = failed = 0
        for rel_path, path in current.items():
            stat = path.stat()
            file_mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            stored = metadata.get(rel_path)
            if stored is not None:
                stored_mtime, stored_size = stored
                if stored_size == stat.st_size and stored_mtime is not None:
                    delta = abs((file_mtime - stored_mtime).total_seconds())
                    if delta <= 1:
                        unchanged += 1
                        continue
            try:
                success = await self.reindex_file(path)
            except Exception as exc:  # noqa: BLE001 - continue with the remaining corpus
                logger.exception(f"Unexpected failure indexing {rel_path}: {exc}")
                success = False
            if not success:
                failed += 1
            elif stored is None:
                added += 1
            else:
                updated += 1

        summary = ReconcileSummary(
            scanned=len(files),
            added=added,
            updated=updated,
            unchanged=unchanged,
            removed=removed,
            failed=failed,
        )
        logger.info(f"Corpus reconciliation complete: {summary}")
        self._reconciling = False
        self._write_state(
            "ready" if failed == 0 else "degraded",
            summary=summary.__dict__,
        )
        return summary
