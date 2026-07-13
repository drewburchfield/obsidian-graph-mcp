"""
Unit tests for VoyageEmbedder configuration and error classification.

No API calls: voyageai.Client construction is local, so these run in the
default suite without VOYAGE_API_KEY credentials.
"""

import pytest
from src.embedder import VoyageEmbedder
from src.exceptions import EmbeddingError


@pytest.fixture
def embedder_env(monkeypatch, tmp_path):
    """Isolated env: fake API key, no VOYAGE_MODEL, temp cache dir."""
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test-key-not-real-0123456789")
    monkeypatch.delenv("VOYAGE_MODEL", raising=False)
    return tmp_path


def _make(tmp_path, **kwargs):
    return VoyageEmbedder(cache_dir=str(tmp_path / "cache"), **kwargs)


def test_model_defaults_to_voyage_context_4(embedder_env):
    assert _make(embedder_env).model == "voyage-context-4"


def test_model_env_overrides_default(embedder_env, monkeypatch):
    monkeypatch.setenv("VOYAGE_MODEL", "voyage-context-3")
    assert _make(embedder_env).model == "voyage-context-3"


def test_model_arg_overrides_env(embedder_env, monkeypatch):
    monkeypatch.setenv("VOYAGE_MODEL", "voyage-context-3")
    assert _make(embedder_env, model="custom-model").model == "custom-model"


def test_empty_env_model_falls_back_to_default(embedder_env, monkeypatch):
    monkeypatch.setenv("VOYAGE_MODEL", "")
    assert _make(embedder_env).model == "voyage-context-4"


def test_whitespace_env_model_is_stripped(embedder_env, monkeypatch):
    monkeypatch.setenv("VOYAGE_MODEL", "  voyage-context-4  ")
    assert _make(embedder_env).model == "voyage-context-4"


def test_cache_key_includes_model(embedder_env):
    """Different models must never share cache entries (vectors are incomparable)."""
    a = _make(embedder_env, model="model-a")
    b = _make(embedder_env, model="model-b")
    assert a._get_text_hash("same text") != b._get_text_hash("same text")


def test_invalid_model_error_is_not_retried(embedder_env):
    embedder = _make(embedder_env)
    calls = {"count": 0}

    def api_call():
        calls["count"] += 1
        raise Exception("400 Bad Request: model 'voyage-contxt-4' is invalid")

    with pytest.raises(EmbeddingError, match="Invalid model"):
        embedder._call_api_with_retry(api_call)

    assert calls["count"] == 1
