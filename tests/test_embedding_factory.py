from __future__ import annotations

from ragflow_orchestrator.embedding import CachedEmbedder, FallbackEmbedder, HashEmbedder, create_embedder
from ragflow_orchestrator.errors import ConfigurationError


def test_create_embedder_hash_from_options() -> None:
    embedder = create_embedder(provider="hash", options={"dimensions": 64})
    assert isinstance(embedder, HashEmbedder)
    assert embedder.dimensions == 64


def test_create_embedder_unknown_provider_raises() -> None:
    try:
        create_embedder(provider="unknown-provider")
    except ConfigurationError as exc:
        assert "Unknown embedding provider" in str(exc)
    else:
        raise AssertionError("Expected ConfigurationError")


def test_create_cached_embedder() -> None:
    embedder = create_embedder(
        provider="cached",
        options={
            "base_provider": "hash",
            "base_options": {"dimensions": 32},
            "max_items": 32,
        },
    )
    assert isinstance(embedder, CachedEmbedder)
    assert embedder.dimensions == 32
    assert len(embedder.embed("hello world")) == 32


def test_create_fallback_embedder() -> None:
    embedder = create_embedder(
        provider="fallback",
        options={
            "primary": {"provider": "hash", "options": {"dimensions": 24}},
            "secondary": {"provider": "hash", "options": {"dimensions": 24}},
        },
    )
    assert isinstance(embedder, FallbackEmbedder)
    assert embedder.dimensions == 24
    assert len(embedder.embed("fallback works")) == 24
