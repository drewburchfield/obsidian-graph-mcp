import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus_sync import CorpusSynchronizer


@pytest.mark.asyncio
async def test_reconcile_adds_new_file(tmp_path, mock_store, mock_embedder):
    path = tmp_path / "new.md"
    path.write_text("new content")
    sync = CorpusSynchronizer(str(tmp_path), mock_store, mock_embedder, enabled_extensions={".md"})

    summary = await sync.reconcile()

    assert summary.added == 1
    mock_store.replace_file_notes.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_leaves_unchanged_file_unembedded(tmp_path, mock_store, mock_embedder):
    path = tmp_path / "same.md"
    path.write_text("same content")
    stat = path.stat()
    mock_store.get_file_metadata.return_value = {
        "same.md": (datetime.fromtimestamp(stat.st_mtime, tz=UTC), stat.st_size)
    }
    sync = CorpusSynchronizer(str(tmp_path), mock_store, mock_embedder, enabled_extensions={".md"})

    summary = await sync.reconcile()

    assert summary.unchanged == 1
    mock_embedder.embed_with_chunks.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_removes_missing_and_excluded_paths(tmp_path, mock_store, mock_embedder):
    mock_store.get_file_metadata.return_value = {
        "deleted.md": (datetime.now(UTC), 10),
        ".tmp-work/old.md": (datetime.now(UTC), 10),
    }
    mock_store.delete_notes_by_paths.return_value = 2
    sync = CorpusSynchronizer(str(tmp_path), mock_store, mock_embedder, enabled_extensions={".md"})

    summary = await sync.reconcile()

    assert summary.removed == 2
    mock_store.delete_notes_by_paths.assert_awaited_once_with([".tmp-work/old.md", "deleted.md"])


@pytest.mark.asyncio
async def test_reindex_replaces_all_chunks_atomically(tmp_path, mock_store, mock_embedder):
    path = tmp_path / "large.txt"
    path.write_text("content")
    mock_embedder.embed_with_chunks.return_value = ([[0.1] * 1024, [0.2] * 1024], 2)
    mock_embedder.chunk_text.return_value = ["first", "second"]
    sync = CorpusSynchronizer(str(tmp_path), mock_store, mock_embedder, enabled_extensions={".txt"})

    assert await sync.reindex_file(path)

    rel_path, notes = mock_store.replace_file_notes.await_args.args
    assert rel_path == "large.txt"
    assert [note.chunk_index for note in notes] == [0, 1]
    assert all(note.total_chunks == 2 for note in notes)


@pytest.mark.asyncio
async def test_conversion_failure_preserves_existing_version(
    tmp_path, mock_store, mock_embedder, monkeypatch
):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"broken")

    def fail_conversion(*args, **kwargs):
        raise RuntimeError("cannot convert")

    monkeypatch.setattr("src.corpus_sync.convert_file", fail_conversion)
    sync = CorpusSynchronizer(str(tmp_path), mock_store, mock_embedder, enabled_extensions={".pdf"})

    assert not await sync.reindex_file(path)
    mock_store.replace_file_notes.assert_not_awaited()
    mock_store.delete_notes_by_paths.assert_not_awaited()


@pytest.mark.asyncio
async def test_nonempty_document_without_extractable_text_gets_metadata_fallback(
    tmp_path, mock_store, mock_embedder, monkeypatch
):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"image-only")
    monkeypatch.setattr("src.corpus_sync.convert_file", lambda *args, **kwargs: None)
    sync = CorpusSynchronizer(str(tmp_path), mock_store, mock_embedder, enabled_extensions={".pdf"})

    assert await sync.reindex_file(path)
    content = mock_store.replace_file_notes.await_args.args[1][0].content
    assert "scan.pdf" in content
    assert "no extractable text" in content


@pytest.mark.asyncio
async def test_provider_failure_does_not_abort_reconciliation(tmp_path, mock_store, mock_embedder):
    (tmp_path / "broken.md").write_text("content")
    mock_embedder.embed_with_chunks.side_effect = RuntimeError("provider unavailable")
    sync = CorpusSynchronizer(str(tmp_path), mock_store, mock_embedder, enabled_extensions={".md"})

    summary = await sync.reconcile()

    assert summary.failed == 1
    assert summary.scanned == 1
    mock_store.replace_file_notes.assert_not_awaited()


def test_healthcheck_reflects_sync_state(tmp_path, monkeypatch):
    import json

    from src.healthcheck import main

    state = tmp_path / "state.json"
    monkeypatch.setenv("CORPUS_PATH", str(tmp_path))
    monkeypatch.setenv("CORPUS_SYNC_STATE_PATH", str(state))

    assert main() == 1
    state.write_text(json.dumps({"status": "syncing"}))
    assert main() == 1
    state.write_text(json.dumps({"status": "ready"}))
    assert main() == 0
