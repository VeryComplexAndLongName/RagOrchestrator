from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag_orchestrator.quality import RetrievalEvalCase, evaluate_retrieval


@dataclass(slots=True)
class DatasetItem:
    query: str
    expected_chunk_ids: set[str]
    filters: dict[str, object]


@dataclass(slots=True)
class StrategyEvaluationReport:
    strategy_name: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


def load_dataset(dataset_path: str) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    path = Path(dataset_path)
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        items.append(
            DatasetItem(
                query=str(payload["query"]),
                expected_chunk_ids=set(payload["expected_chunk_ids"]),
                filters=dict(payload.get("filters", {})),
            )
        )
    return items


def evaluate_strategies(
    strategies: dict[str, object],
    dataset_path: str,
    top_k: int = 3,
) -> list[StrategyEvaluationReport]:
    dataset = load_dataset(dataset_path)
    reports: list[StrategyEvaluationReport] = []

    for strategy_name, strategy in strategies.items():
        cases: list[RetrievalEvalCase] = []
        for item in dataset:
            results = strategy.search(item.query, top_k=top_k, filters=item.filters)
            cases.append(
                RetrievalEvalCase(
                    expected_chunk_ids=item.expected_chunk_ids,
                    retrieved_chunk_ids=[result.chunk.id for result in results],
                )
            )

        score = evaluate_retrieval(cases, k=top_k)
        reports.append(
            StrategyEvaluationReport(
                strategy_name=strategy_name,
                precision_at_k=score.precision_at_k,
                recall_at_k=score.recall_at_k,
                mrr=score.mrr,
                ndcg_at_k=score.ndcg_at_k,
            )
        )

    reports.sort(key=lambda x: (x.mrr, x.precision_at_k, x.recall_at_k), reverse=True)
    return reports
