from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from ragflow_orchestrator.templates.models import TemplateRunReport, TemplatesConfig


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS template_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            scenario TEXT NOT NULL,
            provider_kind TEXT NOT NULL,
            embedder_provider TEXT NOT NULL,
            embedder_model TEXT,
            pipeline_preset TEXT NOT NULL,
            evaluation_enabled INTEGER NOT NULL,
            evaluation_dataset_path TEXT,
            evaluation_top_k INTEGER,
            total_duration_ms REAL NOT NULL,
            total_chunks INTEGER NOT NULL,
            duplicate_chunks_skipped INTEGER NOT NULL,
            chunks_per_second REAL NOT NULL,
            ingested_sources INTEGER NOT NULL,
            skipped_count INTEGER NOT NULL,
            failed_count INTEGER NOT NULL,
            quality_count INTEGER NOT NULL,
            config_path TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS template_quality (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            strategy_name TEXT NOT NULL,
            precision_at_k REAL NOT NULL,
            recall_at_k REAL NOT NULL,
            mrr REAL NOT NULL,
            ndcg_at_k REAL NOT NULL,
            FOREIGN KEY(run_id) REFERENCES template_runs(id)
        )
        """
    )


def append_template_run(
    db_path: str,
    cfg: TemplatesConfig,
    scenario_name: str,
    report: TemplateRunReport,
    config_path: str,
) -> None:
    run_metrics = report.run_metrics
    if run_metrics is None:
        return

    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file))
    try:
        _ensure_schema(conn)

        cursor = conn.execute(
            """
            INSERT INTO template_runs (
                timestamp_utc,
                scenario,
                provider_kind,
                embedder_provider,
                embedder_model,
                pipeline_preset,
                evaluation_enabled,
                evaluation_dataset_path,
                evaluation_top_k,
                total_duration_ms,
                total_chunks,
                duplicate_chunks_skipped,
                chunks_per_second,
                ingested_sources,
                skipped_count,
                failed_count,
                quality_count,
                config_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                scenario_name,
                cfg.orchestrator.provider.kind,
                cfg.orchestrator.embedding.provider,
                cfg.orchestrator.embedding.model,
                cfg.orchestrator.pipeline.preset,
                1 if cfg.evaluation.enabled else 0,
                cfg.evaluation.dataset_path,
                cfg.evaluation.top_k,
                run_metrics.total_duration_ms,
                run_metrics.total_chunks,
                run_metrics.duplicate_chunks_skipped,
                run_metrics.chunks_per_second,
                len(report.ingested),
                len(report.skipped),
                len(report.failed),
                len(report.quality),
                config_path,
            ),
        )
        run_id = int(cursor.lastrowid)

        for metric in report.quality:
            conn.execute(
                """
                INSERT INTO template_quality (
                    run_id,
                    strategy_name,
                    precision_at_k,
                    recall_at_k,
                    mrr,
                    ndcg_at_k
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    metric.strategy_name,
                    metric.precision_at_k,
                    metric.recall_at_k,
                    metric.mrr,
                    metric.ndcg_at_k,
                ),
            )

        conn.commit()
    finally:
        conn.close()
