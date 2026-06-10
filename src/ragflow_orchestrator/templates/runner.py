from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

from ragflow_orchestrator.config import ConfigStore
from ragflow_orchestrator.evaluation import evaluate_strategies
from ragflow_orchestrator.graph import SqlGraphStore
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.orchestrator_factory import RAGOrchestratorFactory
from ragflow_orchestrator.retrieval import (
    HybridRetriever,
    MetadataAwareHybridRetriever,
    RerankedRetriever,
    SemanticRetriever,
    WeightedSignalReranker,
)
from ragflow_orchestrator.templates.base import BaseIngestionTemplate
from ragflow_orchestrator.templates.catalog import resolve_template_class
from ragflow_orchestrator.templates.experiment_journal import append_template_run
from ragflow_orchestrator.templates.models import (
    IngestionError,
    TemplateQualityMetric,
    TemplateRunMetrics,
    TemplateRunReport,
    TemplatesConfig,
)


def _build_runtime_metrics(report: TemplateRunReport, duration_ms: float) -> TemplateRunMetrics:
    total_chunks = sum(item.total_chunks for item in report.ingested)
    duplicates = sum(item.duplicate_chunks_skipped for item in report.ingested)
    chunks_per_second = (total_chunks / (duration_ms / 1000.0)) if duration_ms > 0 else 0.0
    return TemplateRunMetrics(
        total_duration_ms=duration_ms,
        total_chunks=total_chunks,
        duplicate_chunks_skipped=duplicates,
        chunks_per_second=chunks_per_second,
    )


def _build_quality_metrics(orchestrator: RAGOrchestrator, dataset_path: str, top_k: int) -> list[TemplateQualityMetric]:
    semantic = SemanticRetriever(provider=orchestrator.provider, embedder=orchestrator.embedder)
    hybrid = HybridRetriever(provider=orchestrator.provider, embedder=orchestrator.embedder)
    metadata_hybrid = MetadataAwareHybridRetriever(provider=orchestrator.provider, embedder=orchestrator.embedder)
    reranked = RerankedRetriever(base_strategy=metadata_hybrid, reranker=WeightedSignalReranker())

    reports = evaluate_strategies(
        strategies={
            "semantic": semantic,
            "hybrid": hybrid,
            "semantic_cosine_rerank": reranked,
        },
        dataset_path=dataset_path,
        top_k=top_k,
    )
    return [
        TemplateQualityMetric(
            strategy_name=item.strategy_name,
            precision_at_k=item.precision_at_k,
            recall_at_k=item.recall_at_k,
            mrr=item.mrr,
            ndcg_at_k=item.ndcg_at_k,
        )
        for item in reports
    ]


def _create_template(
    template_cls: type[BaseIngestionTemplate],
    orchestrator: RAGOrchestrator,
    graph_store: SqlGraphStore,
) -> BaseIngestionTemplate:
    init_params = inspect.signature(template_cls.__init__).parameters
    if "graph_store" in init_params:
        return template_cls(orchestrator, graph_store=graph_store)
    return template_cls(orchestrator)


def run_template_from_json(config_path: str) -> object:
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = TemplatesConfig.model_validate(payload)

    orchestrator = RAGOrchestratorFactory.from_config_store(ConfigStore(cfg.orchestrator))
    graph_store = SqlGraphStore(cfg.graph_store.db_path)

    scenario_name = cfg.active_scenario
    scenario_payload = cfg.scenarios.get(scenario_name)
    if scenario_payload is None:
        raise ValueError(f"Active scenario '{scenario_name}' not found in scenarios")

    template_cls = resolve_template_class(scenario_name)
    config = template_cls.config_type().model_validate(scenario_payload)
    template = _create_template(template_cls, orchestrator, graph_store)

    started = time.perf_counter()
    report = template.run(config)

    duration_ms = (time.perf_counter() - started) * 1000.0
    report.run_metrics = _build_runtime_metrics(report=report, duration_ms=duration_ms)

    if cfg.evaluation.enabled:
        try:
            report.quality = _build_quality_metrics(
                orchestrator=orchestrator,
                dataset_path=cfg.evaluation.dataset_path,
                top_k=cfg.evaluation.top_k,
            )
        except Exception as exc:  # pragma: no cover
            report.failed.append(IngestionError(source="evaluation", reason=str(exc)))

    if cfg.experiment_log.enabled:
        try:
            append_template_run(
                db_path=cfg.experiment_log.db_path,
                cfg=cfg,
                scenario_name=scenario_name,
                report=report,
                config_path=str(path),
            )
        except Exception as exc:  # pragma: no cover
            report.failed.append(IngestionError(source="experiment_log", reason=str(exc)))

    return report
