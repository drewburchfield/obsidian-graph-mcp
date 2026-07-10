import importlib.util
from pathlib import Path

import pytest


def load_script(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    path = Path(__file__).parent.parent / "scripts" / "reindex_openrouter.py"
    spec = importlib.util.spec_from_file_location("reindex_openrouter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reindex_rejects_partial_embedding_response(monkeypatch):
    module = load_script(monkeypatch)

    with pytest.raises(RuntimeError, match="1 embeddings for 2 inputs"):
        module.validated_embeddings([{"index": 0, "embedding": [0.1]}], 2)


def test_reindex_orders_embeddings_by_provider_index(monkeypatch):
    module = load_script(monkeypatch)
    data = [
        {"index": 1, "embedding": [0.2]},
        {"index": 0, "embedding": [0.1]},
    ]

    assert module.validated_embeddings(data, 2) == [[0.1], [0.2]]
