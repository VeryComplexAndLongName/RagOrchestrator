from __future__ import annotations

import pytest

from ragflow_orchestrator.embedding import OllamaEmbedder


@pytest.mark.integration
def test_ollama_embedder_returns_non_empty_vector() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text:latest")

    try:
        vector = embedder.embed("RAG orchestration integration test")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Ollama embed endpoint unavailable: {exc}")

    assert vector
    assert isinstance(vector[0], float)
    assert embedder.dimensions == len(vector)
