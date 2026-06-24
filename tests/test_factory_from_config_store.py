from __future__ import annotations

from pathlib import Path

from ragflow_orchestrator.config import ConfigStore, EmbeddingConfig, ModuleConfig, PipelineConfig, ProviderConfig
from ragflow_orchestrator.orchestrator_factory import RAGOrchestratorFactory


def test_factory_from_config_store(tmp_path: Path, require_qdrant_service: None) -> None:
    store = ConfigStore(
        ModuleConfig(
            provider=ProviderConfig(
                kind="postgres+qdrant",
                params={
                    "dsn": "postgresql://rag_user:rag_password@localhost:5432/rag_db",
                    "qdrant_url": "http://localhost:6333",
                    "qdrant_collection": "factory_chunks",
                },
            ),
            embedding=EmbeddingConfig(provider="hash", dimensions=64),
            pipeline=PipelineConfig(preset="document"),
        )
    )

    orchestrator = RAGOrchestratorFactory.from_config_store(store)

    summary = orchestrator.ingest(
        source_id="factory-doc",
        raw_text="Factory wiring should create orchestrator with consistent dependencies.",
        metadata={"doctype": "note"},
    )

    assert summary.total_chunks > 0

    results = orchestrator.search("consistent dependencies", top_k=1)
    assert results

