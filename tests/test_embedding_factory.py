from __future__ import annotations

from rag_orchestrator.embedding import HashEmbedder, create_embedder
from rag_orchestrator.errors import ConfigurationError


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
