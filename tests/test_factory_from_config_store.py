from __future__ import annotations

from pathlib import Path

from rag_orchestrator.config import ConfigStore, EmbeddingConfig, ModuleConfig, PipelineConfig, ProviderConfig
from rag_orchestrator.orchestrator_factory import RAGOrchestratorFactory


def test_factory_from_config_store(tmp_path: Path) -> None:
    db_path = tmp_path / "factory.db"
    store = ConfigStore(
        ModuleConfig(
            provider=ProviderConfig(kind="sqlite+vec", params={"db_path": str(db_path), "table_name": "factory_chunks"}),
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
