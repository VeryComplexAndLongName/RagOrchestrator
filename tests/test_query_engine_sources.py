from __future__ import annotations

from pathlib import Path

from ragflow_orchestrator.embedding import HashEmbedder
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.presets import document_preset
from ragflow_orchestrator.query_engine import RAGQueryEngine


def _build_orchestrator(tmp_path: Path) -> RAGOrchestrator:
    provider = create_provider("sqlite+vec", db_path=str(tmp_path / "qe.db"), table_name="qe_chunks")
    preset = document_preset()
    return RAGOrchestrator(
        provider=provider,
        embedder=HashEmbedder(dimensions=64),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )


def test_answer_from_sources_filters_context(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    engine = RAGQueryEngine(orchestrator)

    orchestrator.ingest(
        source_id="c1",
        raw_text="Confluence page about incident management runbook",
        metadata={"source_type": "confluence"},
    )
    orchestrator.ingest(
        source_id="j1",
        raw_text="Jira issue about payment bug",
        metadata={"source_type": "jira"},
    )

    answer = engine.answer_from_sources(
        question="incident runbook",
        source_types=["confluence"],
        top_k=5,
    )

    assert answer.context
    assert all(item.chunk.metadata.get("source_type") == "confluence" for item in answer.context)


