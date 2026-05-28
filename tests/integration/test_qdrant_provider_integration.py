from __future__ import annotations

import pytest

from rag_orchestrator.adapters.qdrant_provider import QdrantProvider
from rag_orchestrator.errors import ProviderDependencyError
from rag_orchestrator.models import RetrievalQuery


@pytest.mark.integration
@pytest.mark.qdrant
def test_qdrant_provider_roundtrip(qdrant_url: str, sample_chunks: list, query_embedder: object) -> None:
    try:
        provider = QdrantProvider(url=qdrant_url, collection_name="rag_chunks_it_qdrant")
    except ProviderDependencyError as exc:  # pragma: no cover
        pytest.skip(str(exc))

    try:
        if not provider.healthcheck():
            raise RuntimeError("healthcheck failed")
        provider.ensure_schema(vector_dim=64)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qdrant unavailable or misconfigured: {exc}")

    provider.upsert_chunks(sample_chunks)

    query = RetrievalQuery(
        text_query="function that adds numbers",
        top_k=2,
        dense_vector=query_embedder.embed("function that adds numbers"),
        filters=[{"key": "tenant_id", "value": "t1"}],
    )
    results = provider.retrieve(query)

    assert results
    assert results[0].chunk.metadata.get("tenant_id") == "t1"
    assert results[0].chunk.id in {"math:add", "math:sub"}

    provider.delete_chunks(["math:add"], soft_delete=True)
    results_after_soft_delete = provider.retrieve(query)
    assert all(item.chunk.id != "math:add" for item in results_after_soft_delete)

    provider.delete_chunks([chunk.id for chunk in sample_chunks], soft_delete=False)
