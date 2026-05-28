from __future__ import annotations

import argparse
import cProfile
import pstats
import random
from pathlib import Path

from rag_orchestrator import HashEmbedder, RAGOrchestrator, create_provider, document_preset


def _build_text(i: int, words: int, rnd: random.Random) -> str:
    parts = ["profiling rag pipeline"]
    for _ in range(max(1, words - 3)):
        parts.append(f"w{rnd.randint(1, 50000)}")
    parts.append(f"doc{i}")
    return " ".join(parts)


def _build_orchestrator(args: argparse.Namespace) -> RAGOrchestrator:
    if args.provider == "sqlite+vec":
        provider = create_provider("sqlite+vec", db_path=args.sqlite_db, table_name=args.table_name)
    elif args.provider == "pgvector":
        if not args.pg_dsn:
            raise ValueError("--pg-dsn is required for pgvector")
        provider = create_provider("pgvector", connection_string=args.pg_dsn, table_name=args.table_name)
    elif args.provider == "qdrant":
        provider = create_provider("qdrant", url=args.qdrant_url, collection_name=args.table_name)
    else:
        raise ValueError(f"Unsupported provider: {args.provider}")

    preset = document_preset()
    return RAGOrchestrator(
        provider=provider,
        embedder=HashEmbedder(dimensions=args.dimensions),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )


def run_profile(args: argparse.Namespace) -> None:
    rnd = random.Random(args.seed)
    orchestrator = _build_orchestrator(args)

    docs = [_build_text(i, args.words_per_doc, rnd) for i in range(args.documents)]
    queries = ["rag profiling", "vector retrieval", "chunking pipeline", "jira issue"] * max(1, args.queries // 4)
    queries = queries[: args.queries]

    profiler = cProfile.Profile()
    profiler.enable()

    for i, doc in enumerate(docs):
        orchestrator.ingest(source_id=f"profile:{i}", raw_text=doc, metadata={"source_type": "profile"})

    for q in queries:
        orchestrator.search(query_text=q, top_k=args.top_k)

    profiler.disable()

    stats_path = Path(args.out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(args.sort)
    with stats_path.open("w", encoding="utf-8") as fh:
        stats.stream = fh
        stats.print_stats(args.limit)

    print(f"Profile report written to: {stats_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile bottlenecks in ingestion and retrieval pipeline")
    parser.add_argument("--provider", default="sqlite+vec", choices=["sqlite+vec", "pgvector", "qdrant"])
    parser.add_argument("--sqlite-db", default="loadtest/profile.sqlite")
    parser.add_argument("--pg-dsn", default="")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--table-name", default="profile_chunks")
    parser.add_argument("--documents", type=int, default=300)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--words-per-doc", type=int, default=120)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sort", default="cumulative")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--out", default="loadtest/profile_hotspots.txt")
    args = parser.parse_args()

    run_profile(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
