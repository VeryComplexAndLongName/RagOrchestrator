from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_orchestrator.evaluation.retrieval import StrategyEvaluationReport, evaluate_strategies
from rag_orchestrator.retrieval import HybridRetriever, RerankedRetriever, SemanticRetriever, create_reranker


@dataclass(slots=True)
class RerankProfileConfig:
    name: str
    reranker_provider: str
    reranker_model: str | None = None
    reranker_options: dict[str, Any] = field(default_factory=dict)


def build_profile_strategies(provider: object, embedder: object, profile: RerankProfileConfig) -> dict[str, object]:
    semantic = SemanticRetriever(provider=provider, embedder=embedder)
    hybrid = HybridRetriever(provider=provider, embedder=embedder)
    reranker = create_reranker(
        provider=profile.reranker_provider,
        embedder=embedder,
        model=profile.reranker_model,
        options=profile.reranker_options,
    )
    reranked = RerankedRetriever(base_strategy=semantic, reranker=reranker)

    return {
        "semantic": semantic,
        "hybrid": hybrid,
        f"semantic_{profile.reranker_provider}_rerank": reranked,
    }


def evaluate_rerank_profiles(
    provider: object,
    embedder: object,
    dataset_path: str,
    profiles: list[RerankProfileConfig],
    top_k: int = 3,
) -> dict[str, list[StrategyEvaluationReport]]:
    output: dict[str, list[StrategyEvaluationReport]] = {}
    for profile in profiles:
        strategies = build_profile_strategies(provider=provider, embedder=embedder, profile=profile)
        output[profile.name] = evaluate_strategies(strategies=strategies, dataset_path=dataset_path, top_k=top_k)
    return output
