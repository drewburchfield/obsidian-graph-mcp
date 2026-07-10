import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ollama_embedder import OllamaEmbedder


@pytest.mark.asyncio
async def test_cache_separates_query_and_document_embeddings(tmp_path):
    embedder = OllamaEmbedder(cache_dir=str(tmp_path), dimensions=3)
    embedder._embed_group = AsyncMock(
        side_effect=lambda client, inputs: [[float(len(text)), 0.0, 0.0] for text in inputs]
    )

    document = await embedder.embed("same text", input_type="document")
    query = await embedder.embed("same text", input_type="query")

    assert embedder._embed_group.await_count == 2
    assert document != query
    assert len(embedder.cache_index) == 2
