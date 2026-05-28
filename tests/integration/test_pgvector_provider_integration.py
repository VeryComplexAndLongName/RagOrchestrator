from __future__ import annotations

import pytest

from rag_orchestrator.adapters.pgvector_provider import PGVectorProvider
from rag_orchestrator.models import RetrievalQuery


@pytest.mark.integration
@pytest.mark.pgvector
def test_pgvector_provider_roundtrip(pgvector_dsn: str, sample_chunks: list, query_embedder: object) -> None:
    provider = PGVectorProvider(connection_string=pgvector_dsn, table_name="rag_chunks_it_pg")
    try:
        if not provider.healthcheck():
            raise RuntimeError("healthcheck failed")
        provider.ensure_schema(vector_dim=64)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"pgvector unavailable or misconfigured: {exc}")

    provider.upsert_chunks(sample_chunks)

    query = RetrievalQuery(
        text_query="rag orchestration and retrieval",
        top_k=2,
        dense_vector=query_embedder.embed("rag orchestration and retrieval"),
        filters=[{"key": "doctype", "value": "note"}],
    )
    results = provider.retrieve(query)

    assert results
    assert results[0].chunk.metadata.get("doctype") == "note"

    provider.delete_chunks(["rag:overview"], soft_delete=True)
    filtered_results = provider.retrieve(query)
    assert all(item.chunk.id != "rag:overview" for item in filtered_results)

    provider.delete_chunks([chunk.id for chunk in sample_chunks], soft_delete=False)
