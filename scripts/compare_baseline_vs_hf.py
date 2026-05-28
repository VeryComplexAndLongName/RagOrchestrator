from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag_orchestrator.embedding import HashEmbedder, create_embedder
from rag_orchestrator.evaluation import RerankProfileConfig, evaluate_rerank_profiles
from rag_orchestrator.factory import create_provider
from rag_orchestrator.orchestrator import RAGOrchestrator
from rag_orchestrator.perf import summarize_latencies
from rag_orchestrator.presets import document_preset
from rag_orchestrator.retrieval import (
    CosineReranker,
    HybridRetriever,
    RerankedRetriever,
    SemanticRetriever,
    create_reranker,
)


def _rss_mb() -> float:
    try:
        import psutil
    except ImportError:
        return 0.0
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def _vram_mb() -> float:
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


@dataclass(slots=True)
class StrategyPerf:
    profile: str
    strategy_name: str
    p50_ms: float
    p95_ms: float
    avg_ms: float
    throughput_qps: float


@dataclass(slots=True)
class ProfileComparison:
    profile: str
    quality: list[dict]
    perf: list[StrategyPerf]
    ram_mb: float
    vram_mb: float


def _build_demo_index(db_path: str, embedder: object) -> object:
    provider = create_provider("sqlite+vec", db_path=db_path, table_name="cmp_chunks")
    preset = document_preset()
    orchestrator = RAGOrchestrator(
        provider=provider,
        embedder=embedder,
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )

    orchestrator.ingest(
        source_id="math_utils.py",
        raw_text="def add(a,b): return a+b\ndef sub(a,b): return a-b",
        metadata={"language": "python", "doctype": "code"},
    )
    orchestrator.ingest(
        source_id="rag_doc",
        raw_text="RAG orchestration unifies ingestion, retrieval and evaluation.",
        metadata={"language": "en", "doctype": "note"},
    )

    return provider


def _build_strategies(provider: object, embedder: object, reranker_provider: str, reranker_model: str | None) -> dict[str, object]:
    semantic = SemanticRetriever(provider=provider, embedder=embedder)
    hybrid = HybridRetriever(provider=provider, embedder=embedder)

    if reranker_provider == "cosine":
        reranker = CosineReranker(embedder=embedder)
    else:
        reranker = create_reranker(
            provider=reranker_provider,
            embedder=embedder,
            model=reranker_model,
            options={},
        )

    reranked = RerankedRetriever(base_strategy=semantic, reranker=reranker)
    return {
        "semantic": semantic,
        "hybrid": hybrid,
        f"semantic_{reranker_provider}_rerank": reranked,
    }


def _measure_perf(strategies: dict[str, object], loops: int, top_k: int) -> list[StrategyPerf]:
    test_queries = [
        ("function that adds numbers", {"language": "python"}),
        ("subtraction helper", {"language": "python"}),
        ("rag orchestration and retrieval", {"doctype": "note"}),
    ]

    out: list[StrategyPerf] = []
    for strategy_name, strategy in strategies.items():
        latencies_ms: list[float] = []
        started = time.perf_counter()
        for _ in range(loops):
            for query_text, filters in test_queries:
                t0 = time.perf_counter()
                strategy.search(query_text=query_text, top_k=top_k, filters=filters)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        total = time.perf_counter() - started
        stats = summarize_latencies(latencies_ms)
        qps = len(latencies_ms) / total if total else 0.0
        out.append(
            StrategyPerf(
                profile="",
                strategy_name=strategy_name,
                p50_ms=stats.p50_ms,
                p95_ms=stats.p95_ms,
                avg_ms=stats.avg_ms,
                throughput_qps=qps,
            )
        )
    return out


def run_profile(
    profile_name: str,
    embedder: object,
    reranker_provider: str,
    reranker_model: str | None,
    dataset_path: str,
    loops: int,
    top_k: int,
    artifacts_dir: str,
) -> ProfileComparison:
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    db_path = str(root / f"compare_{profile_name}.sqlite")
    provider = _build_demo_index(db_path=db_path, embedder=embedder)

    quality_reports = evaluate_rerank_profiles(
        provider=provider,
        embedder=embedder,
        dataset_path=dataset_path,
        profiles=[
            RerankProfileConfig(
                name=profile_name,
                reranker_provider=reranker_provider,
                reranker_model=reranker_model,
            )
        ],
        top_k=top_k,
    )[profile_name]

    strategies = _build_strategies(
        provider=provider,
        embedder=embedder,
        reranker_provider=reranker_provider,
        reranker_model=reranker_model,
    )
    perf_rows = _measure_perf(strategies=strategies, loops=loops, top_k=top_k)
    for row in perf_rows:
        row.profile = profile_name

    return ProfileComparison(
        profile=profile_name,
        quality=[asdict(item) for item in quality_reports],
        perf=perf_rows,
        ram_mb=_rss_mb(),
        vram_mb=_vram_mb(),
    )


def _print_results(results: list[ProfileComparison]) -> None:
    print("\nQUALITY")
    print("profile | strategy | precision@k | recall@k | mrr | ndcg@k")
    print("--- | --- | --- | --- | --- | ---")
    for item in results:
        for quality_row in item.quality:
            print(
                f"{item.profile} | {quality_row['strategy_name']} | "
                f"{quality_row['precision_at_k']:.3f} | {quality_row['recall_at_k']:.3f} | {quality_row['mrr']:.3f} | {quality_row['ndcg_at_k']:.3f}"
            )

    print("\nPERF")
    print("profile | strategy | p50_ms | p95_ms | avg_ms | throughput_qps")
    print("--- | --- | --- | --- | --- | ---")
    for item in results:
        for perf_row in item.perf:
            print(
                f"{item.profile} | {perf_row.strategy_name} | "
                f"{perf_row.p50_ms:.3f} | {perf_row.p95_ms:.3f} | {perf_row.avg_ms:.3f} | {perf_row.throughput_qps:.2f}"
            )

    print("\nMEMORY")
    print("profile | ram_mb | vram_mb")
    print("--- | --- | ---")
    for item in results:
        print(f"{item.profile} | {item.ram_mb:.2f} | {item.vram_mb:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline vs Hugging Face retrieval profiles")
    parser.add_argument("--dataset", default="datasets/retrieval_eval.jsonl")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--loops", type=int, default=100)
    parser.add_argument("--hf-embedder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--hf-reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--artifacts-dir", default="loadtest")
    parser.add_argument("--json-out", default="loadtest/compare_baseline_vs_hf.json")
    args = parser.parse_args()

    results: list[ProfileComparison] = []

    baseline_embedder = HashEmbedder(dimensions=64)
    results.append(
        run_profile(
            profile_name="baseline_hash_cosine",
            embedder=baseline_embedder,
            reranker_provider="cosine",
            reranker_model=None,
            dataset_path=args.dataset,
            loops=args.loops,
            top_k=args.top_k,
            artifacts_dir=args.artifacts_dir,
        )
    )

    try:
        hf_embedder = create_embedder(
            provider="hf",
            model=args.hf_embedder_model,
            options={"normalize_embeddings": True},
        )
        results.append(
            run_profile(
                profile_name="hf_embedder_hf_reranker",
                embedder=hf_embedder,
                reranker_provider="hf",
                reranker_model=args.hf_reranker_model,
                dataset_path=args.dataset,
                loops=max(30, args.loops // 3),
                top_k=args.top_k,
                artifacts_dir=args.artifacts_dir,
            )
        )
    except Exception as exc:  # pragma: no cover
        print(f"HF profile skipped: {exc}")

    _print_results(results)

    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "top_k": args.top_k,
        "loops": args.loops,
        "results": [
            {
                "profile": item.profile,
                "quality": item.quality,
                "perf": [asdict(row) for row in item.perf],
                "ram_mb": item.ram_mb,
                "vram_mb": item.vram_mb,
            }
            for item in results
        ],
    }
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"\nJSON report: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
