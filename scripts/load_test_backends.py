from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ragflow_orchestrator import HashEmbedder, RAGOrchestrator, create_provider, document_preset
from ragflow_orchestrator.perf import summarize_latencies


@dataclass(slots=True)
class BackendResult:
    backend: str
    status: str
    reason: str
    docs: int
    queries: int
    dimensions: int
    ingest_total_s: float
    ingest_docs_per_s: float
    ingest_avg_ms: float
    ingest_p95_ms: float
    search_total_s: float
    search_qps: float
    search_avg_ms: float
    search_p50_ms: float
    search_p95_ms: float
    search_p99_ms: float


def _build_doc(i: int, words_per_doc: int, rnd: random.Random) -> str:
    topics = [
        "telegram bot architecture",
        "jira workflow automation",
        "confluence knowledge base",
        "web crawl extraction",
        "python async performance",
        "vector database retrieval",
        "rag orchestration design",
        "load testing benchmark",
    ]
    chosen = rnd.choice(topics)
    words = [chosen]
    for _ in range(max(1, words_per_doc - 8)):
        words.append(f"token{rnd.randint(1, 20000)}")
    words.append(f"doc{i}")
    return " ".join(words)


def _build_queries(count: int, rnd: random.Random) -> list[str]:
    base = [
        "telegram bot",
        "jira automation",
        "confluence notes",
        "web crawl",
        "async performance",
        "vector retrieval",
        "rag design",
    ]
    return [rnd.choice(base) for _ in range(count)]


def _provider_factory(kind: str, run_id: str, args: argparse.Namespace):
    if kind == "sqlite+vec":
        db_name = Path(args.artifacts_dir) / f"loadtest_{run_id}.sqlite"
        db_name.parent.mkdir(parents=True, exist_ok=True)
        return create_provider("sqlite+vec", db_path=str(db_name), table_name=f"chunks_{run_id}")
    if kind == "pgvector":
        if not args.pg_dsn:
            raise ValueError("PGVECTOR_DSN is missing")
        return create_provider("pgvector", connection_string=args.pg_dsn, table_name=f"bench_chunks_{run_id}")
    if kind == "qdrant":
        return create_provider("qdrant", url=args.qdrant_url, collection_name=f"bench_chunks_{run_id}")
    raise ValueError(f"Unsupported backend: {kind}")


def run_backend(kind: str, args: argparse.Namespace) -> BackendResult:
    run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    rnd = random.Random(args.seed)

    try:
        provider = _provider_factory(kind, run_id, args)
        preset = document_preset()
        orchestrator = RAGOrchestrator(
            provider=provider,
            embedder=HashEmbedder(dimensions=args.dimensions),
            chunker=preset.chunker,
            cleaner=preset.cleaner,
        )
    except Exception as exc:
        return BackendResult(
            backend=kind,
            status="skipped",
            reason=str(exc),
            docs=0,
            queries=0,
            dimensions=args.dimensions,
            ingest_total_s=0.0,
            ingest_docs_per_s=0.0,
            ingest_avg_ms=0.0,
            ingest_p95_ms=0.0,
            search_total_s=0.0,
            search_qps=0.0,
            search_avg_ms=0.0,
            search_p50_ms=0.0,
            search_p95_ms=0.0,
            search_p99_ms=0.0,
        )

    docs = [_build_doc(i, args.words_per_doc, rnd) for i in range(args.documents)]
    ingest_latencies_ms: list[float] = []

    ingest_start = time.perf_counter()
    for i, doc in enumerate(docs):
        t0 = time.perf_counter()
        orchestrator.ingest(
            source_id=f"load:{kind}:{i}",
            raw_text=doc,
            metadata={"source_type": "load_test", "backend": kind},
        )
        ingest_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    ingest_total_s = time.perf_counter() - ingest_start

    queries = _build_queries(args.queries, rnd)

    def _search_once(query_text: str) -> float:
        t0 = time.perf_counter()
        orchestrator.search(query_text=query_text, top_k=args.top_k)
        return (time.perf_counter() - t0) * 1000.0

    search_latencies_ms: list[float] = []
    search_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_search_once, q) for q in queries]
        for fut in as_completed(futures):
            search_latencies_ms.append(fut.result())
    search_total_s = time.perf_counter() - search_start

    ingest_stats = summarize_latencies(ingest_latencies_ms)
    search_stats = summarize_latencies(search_latencies_ms)

    docs_per_s = args.documents / ingest_total_s if ingest_total_s else 0.0
    qps = args.queries / search_total_s if search_total_s else 0.0

    return BackendResult(
        backend=kind,
        status="ok",
        reason="",
        docs=args.documents,
        queries=args.queries,
        dimensions=args.dimensions,
        ingest_total_s=ingest_total_s,
        ingest_docs_per_s=docs_per_s,
        ingest_avg_ms=ingest_stats.avg_ms,
        ingest_p95_ms=ingest_stats.p95_ms,
        search_total_s=search_total_s,
        search_qps=qps,
        search_avg_ms=search_stats.avg_ms,
        search_p50_ms=search_stats.p50_ms,
        search_p95_ms=search_stats.p95_ms,
        search_p99_ms=search_stats.p99_ms,
    )


def _print_table(results: list[BackendResult]) -> None:
    headers = [
        "backend",
        "status",
        "ingest_docs/s",
        "search_qps",
        "search_p50_ms",
        "search_p95_ms",
        "search_p99_ms",
        "reason",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for r in results:
        row = [
            r.backend,
            r.status,
            f"{r.ingest_docs_per_s:.2f}",
            f"{r.search_qps:.2f}",
            f"{r.search_p50_ms:.2f}",
            f"{r.search_p95_ms:.2f}",
            f"{r.search_p99_ms:.2f}",
            r.reason,
        ]
        print(" | ".join(row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Load test and compare vector backends")
    parser.add_argument("--providers", nargs="+", default=["sqlite+vec", "pgvector", "qdrant"])
    parser.add_argument("--documents", type=int, default=500)
    parser.add_argument("--words-per-doc", type=int, default=120)
    parser.add_argument("--queries", type=int, default=800)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pg-dsn", default="")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--artifacts-dir", default="loadtest")
    parser.add_argument("--json-out", default="loadtest/load_test_results.json")
    args = parser.parse_args()

    results = [run_backend(kind, args) for kind in args.providers]

    _print_table(results)

    output = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": {
            "providers": args.providers,
            "documents": args.documents,
            "words_per_doc": args.words_per_doc,
            "queries": args.queries,
            "concurrency": args.concurrency,
            "top_k": args.top_k,
            "dimensions": args.dimensions,
            "seed": args.seed,
            "qdrant_url": args.qdrant_url,
            "pg_dsn_present": bool(args.pg_dsn),
        },
        "results": [asdict(r) for r in results],
    }
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"\nJSON report: {args.json_out}")

    ok = any(r.status == "ok" for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
