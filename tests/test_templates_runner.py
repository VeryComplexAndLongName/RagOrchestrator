from __future__ import annotations

import json
from pathlib import Path

from ragflow_orchestrator.templates.runner import run_template_from_json


def test_run_template_from_json_with_repo_code(tmp_path: Path, require_qdrant_service: None) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    cfg = {
        "orchestrator": {
            "provider": {
                "kind": "postgres+qdrant",
                "params": {
                    "dsn": "postgresql://rag_user:rag_password@localhost:5432/rag_db",
                    "qdrant_url": "http://localhost:6333",
                    "qdrant_collection": "runner_chunks",
                },
            },
            "embedding": {
                "provider": "hash",
                "options": {"dimensions": 64},
            },
            "pipeline": {"preset": "document"},
        },
        "active_scenario": "repo_code",
        "scenarios": {
            "repo_code": {
                "repos": [str(repo)],
                "recursive": True,
                "extensions": [".py"],
                "language_mode": "auto",
            }
        },
    }

    cfg_path = tmp_path / "templates.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")

    report = run_template_from_json(str(cfg_path))
    assert len(report.ingested) == 1
    assert not report.failed
    assert report.run_metrics is not None
    assert report.run_metrics.total_chunks >= 1
    assert report.run_metrics.total_duration_ms >= 0.0
    assert report.quality == []


def test_run_template_from_json_with_eval_enabled(tmp_path: Path, require_qdrant_service: None) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    cfg = {
        "orchestrator": {
            "provider": {
                "kind": "postgres+qdrant",
                "params": {
                    "dsn": "postgresql://rag_user:rag_password@localhost:5432/rag_db",
                    "qdrant_url": "http://localhost:6333",
                    "qdrant_collection": "runner_eval_chunks",
                },
            },
            "embedding": {
                "provider": "hash",
                "options": {"dimensions": 64},
            },
            "pipeline": {"preset": "document"},
        },
        "evaluation": {
            "enabled": True,
            "dataset_path": "datasets/retrieval_eval.jsonl",
            "top_k": 2,
        },
        "active_scenario": "repo_code",
        "scenarios": {
            "repo_code": {
                "repos": [str(repo)],
                "recursive": True,
                "extensions": [".py"],
                "language_mode": "auto",
            }
        },
    }

    cfg_path = tmp_path / "templates_eval.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")

    report = run_template_from_json(str(cfg_path))
    assert report.run_metrics is not None
    assert report.quality
    assert {item.strategy_name for item in report.quality} == {
        "semantic",
        "hybrid",
        "semantic_cosine_rerank",
    }

