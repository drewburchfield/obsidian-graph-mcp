"""
Unit tests for vault indexing.

Tests:
1. Vault scanning finds all markdown files
2. Title extraction from filenames
3. Batch processing logic
4. Empty file handling
5. Large file chunking
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexer import extract_title, scan_vault


def test_scan_vault_finds_markdown_files(tmp_vault):
    """Test that scan_vault finds all .md files."""
    # tmp_vault has note1.md, note2.md, folder/note3.md, empty.md
    md_files = scan_vault(str(tmp_vault))

    # Should find at least 4 markdown files
    assert len(md_files) >= 4

    # All should be Path objects
    assert all(isinstance(f, Path) for f in md_files)

    # All should end with .md
    assert all(f.suffix == ".md" for f in md_files)


def test_scan_vault_excludes_non_markdown(tmp_path):
    """Test that scan_vault excludes non-markdown files."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create mixed file types
    (vault / "note.md").write_text("# Note")
    (vault / "readme.txt").write_text("Text file")
    (vault / "image.png").write_bytes(b"fake image")

    md_files = scan_vault(str(vault))

    # Should only find the .md file
    assert len(md_files) == 1
    assert md_files[0].name == "note.md"


def test_scan_vault_raises_on_missing_vault():
    """Test that scan_vault raises error if vault doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Vault not found"):
        scan_vault("/nonexistent/vault")


def test_extract_title_from_filename():
    """Test title extraction from filename."""
    test_cases = [
        (Path("/vault/note.md"), "note"),
        (Path("/vault/my-todo-list.md"), "my-todo-list"),
        (Path("/vault/folder/nested-note.md"), "nested-note"),
        (Path("/vault/2024-12-18-daily-note.md"), "2024-12-18-daily-note"),
    ]

    for file_path, expected_title in test_cases:
        title = extract_title(file_path)
        assert title == expected_title


@pytest.mark.asyncio
async def test_index_vault_processes_batches(tmp_vault, mock_store, mock_embedder):
    """Test that index_vault processes notes in batches."""
    from src.indexer import index_vault

    # Mock the store and embedder
    with (
        patch("src.indexer.VoyageEmbedder", return_value=mock_embedder),
        patch("src.indexer.PostgreSQLVectorStore", return_value=mock_store),
    ):

        # Run indexing with small batch size
        await index_vault(str(tmp_vault), batch_size=2)

        # Should have called upsert_batch at least once
        assert mock_store.upsert_batch.call_count >= 1


@pytest.mark.asyncio
async def test_index_vault_skips_empty_files(tmp_path, mock_store, mock_embedder):
    """Test that empty files are skipped during indexing."""
    from src.indexer import index_vault

    vault = tmp_path / "vault"
    vault.mkdir()

    # Create one valid and one empty file
    (vault / "valid.md").write_text("# Valid Note")
    (vault / "empty.md").write_text("")

    with (
        patch("src.indexer.VoyageEmbedder", return_value=mock_embedder),
        patch("src.indexer.PostgreSQLVectorStore", return_value=mock_store),
    ):

        await index_vault(str(vault), batch_size=10)

        # Check upsert_batch was called with only valid notes
        if mock_store.upsert_batch.called:
            call_args = mock_store.upsert_batch.call_args
            notes = call_args[0][0]  # First positional argument

            # Should have processed only the valid note
            assert len(notes) >= 1
            # Should not include empty.md
            assert not any(n.path == "empty.md" for n in notes)


@pytest.mark.asyncio
async def test_index_vault_handles_large_notes_with_chunking(tmp_path, mock_store, mock_embedder):
    """Test that large notes are properly chunked."""
    from src.indexer import index_vault

    vault = tmp_path / "vault"
    vault.mkdir()

    # Create a large note (>120k chars to trigger chunking)
    large_content = "This is a test sentence. " * 10000  # ~250k chars
    (vault / "large.md").write_text(large_content)

    # Mock embedder to return multiple chunks
    mock_embedder.embed_with_chunks = MagicMock(
        return_value=([[0.1] * 1024, [0.2] * 1024, [0.3] * 1024], 3)  # 3 chunks  # total_chunks
    )
    mock_embedder.chunk_text = MagicMock(
        return_value=[large_content[:2000], large_content[2000:4000], large_content[4000:6000]]
    )

    with (
        patch("src.indexer.VoyageEmbedder", return_value=mock_embedder),
        patch("src.indexer.PostgreSQLVectorStore", return_value=mock_store),
    ):

        await index_vault(str(vault), batch_size=10)

        # Should have called upsert_batch
        assert mock_store.upsert_batch.called

        # Verify chunks were created
        if mock_store.upsert_batch.called:
            notes = mock_store.upsert_batch.call_args[0][0]
            # Should have 3 Note objects for the chunked note
            large_notes = [n for n in notes if "large" in n.path]
            assert len(large_notes) == 3
            assert all(n.total_chunks == 3 for n in large_notes)
