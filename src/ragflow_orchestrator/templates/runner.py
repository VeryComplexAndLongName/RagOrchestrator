from __future__ import annotations

import json
import time
from pathlib import Path

from ragflow_orchestrator.config import ConfigStore
from ragflow_orchestrator.evaluation import evaluate_strategies
from ragflow_orchestrator.graph import SqlGraphStore
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.orchestrator_factory import RAGOrchestratorFactory
from ragflow_orchestrator.retrieval import CosineReranker, HybridRetriever, RerankedRetriever, SemanticRetriever
from ragflow_orchestrator.templates.api_reference import APIReferenceTemplate
from ragflow_orchestrator.templates.confluence_wiki import ConfluenceWikiTemplate
from ragflow_orchestrator.templates.document_folder import DocumentFolderTemplate
from ragflow_orchestrator.templates.email_ticket import EmailTicketTemplate
from ragflow_orchestrator.templates.experiment_journal import append_template_run
from ragflow_orchestrator.templates.github_template import GitHubTemplate
from ragflow_orchestrator.templates.gitlab_template import GitLabTemplate
from ragflow_orchestrator.templates.incremental_sync import IncrementalSyncTemplate
from ragflow_orchestrator.templates.jira import JiraTemplate
from ragflow_orchestrator.templates.models import (
    APIReferenceConfig,
    ConfluenceWikiConfig,
    DocumentFolderConfig,
    EmailTicketConfig,
    GitHubConfig,
    GitLabConfig,
    IncrementalSyncConfig,
    IngestionError,
    JiraConfig,
    PyPIConfig,
    RepoCodeConfig,
    TemplateQualityMetric,
    TemplateRunMetrics,
    TemplateRunReport,
    TemplatesConfig,
    WebCrawlConfig,
)
from ragflow_orchestrator.templates.pypi import PyPITemplate
from ragflow_orchestrator.templates.repo_code import RepoCodeTemplate
from ragflow_orchestrator.templates.web_crawl import WebCrawlTemplate


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
    reranked = RerankedRetriever(base_strategy=semantic, reranker=CosineReranker(embedder=orchestrator.embedder))

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

    started = time.perf_counter()
    report: TemplateRunReport
    if scenario_name == "web_crawl":
        report = WebCrawlTemplate(orchestrator).run(WebCrawlConfig.model_validate(scenario_payload))
    elif scenario_name == "document_folder":
        report = DocumentFolderTemplate(orchestrator).run(DocumentFolderConfig.model_validate(scenario_payload))
    elif scenario_name == "confluence_wiki":
        report = ConfluenceWikiTemplate(orchestrator).run(ConfluenceWikiConfig.model_validate(scenario_payload))
    elif scenario_name == "jira":
        report = JiraTemplate(orchestrator).run(JiraConfig.model_validate(scenario_payload))
    elif scenario_name == "api_reference":
        report = APIReferenceTemplate(orchestrator).run(APIReferenceConfig.model_validate(scenario_payload))
    elif scenario_name == "pypi":
        report = PyPITemplate(orchestrator).run(PyPIConfig.model_validate(scenario_payload))
    elif scenario_name == "github":
        report = GitHubTemplate(orchestrator, graph_store=graph_store).run(GitHubConfig.model_validate(scenario_payload))
    elif scenario_name == "gitlab":
        report = GitLabTemplate(orchestrator, graph_store=graph_store).run(GitLabConfig.model_validate(scenario_payload))
    elif scenario_name == "repo_code":
        report = RepoCodeTemplate(orchestrator).run(RepoCodeConfig.model_validate(scenario_payload))
    elif scenario_name == "email_ticket":
        report = EmailTicketTemplate(orchestrator).run(EmailTicketConfig.model_validate(scenario_payload))
    elif scenario_name == "incremental_sync":
        report = IncrementalSyncTemplate(orchestrator).run(IncrementalSyncConfig.model_validate(scenario_payload))
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

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
