"""
GitHub RAG Demo
===============
Ingest public GitHub repositories into a local PostgreSQL + Qdrant store and a graph store,
then answer questions about the repositories via RAGQueryEngine backed by Ollama.

Usage
-----
# Ingest and ask a single question (non-interactive):
    python scripts/github_demo/run.py --owners fastapi psf --ask "What is FastAPI?"

# Interactive REPL (keeps asking until you type 'exit' or Ctrl-C):
    python scripts/github_demo/run.py --owners fastapi psf --interactive

# Skip ingest if DB already exists:
    python scripts/github_demo/run.py --skip-ingest --ask "What does the fastapi repository do?"

Flags
-----
--owners            list of GitHub owners to ingest              (default: fastapi psf pallets)
--db                PostgreSQL DSN (default: env RAG_POSTGRES_DSN or local dev DSN)
--graph-db          path to repository graph database           (default: scripts/github_demo/github_graph.sqlite)
--embed-model       Ollama embedding model name                 (default: nomic-embed-text:latest)
--chat-model        Ollama chat model name                      (default: llama3.1:latest)
--ollama-url        Ollama base URL                             (default: http://localhost:11434)
--top-k             number of chunks to retrieve per query      (default: 5)
--max-projects      max repositories to ingest overall          (default: 5)
--max-repos         max repositories to ingest per owner        (default: 3)
--token             GitHub token for higher rate limits
--include-readme    include README content in indexed text
--include-contributors include contributor data in the graph store
--ask               single question to answer then exit
--interactive       run interactive question-answering REPL
--skip-ingest       skip ingest step (use existing DB)
--perf              print performance report even without --ask/--interactive
"""
from __future__ import annotations

import argparse
import json as _json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request as _urllib_request

# Ensure src/ is on the path so the package is importable without install.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from ragflow_orchestrator.embedding import OllamaEmbedder  # noqa: E402
from ragflow_orchestrator.factory import create_provider  # noqa: E402
from ragflow_orchestrator.graph import SqlGraphStore  # noqa: E402
from ragflow_orchestrator.orchestrator import RAGOrchestrator  # noqa: E402
from ragflow_orchestrator.presets import document_preset  # noqa: E402
from ragflow_orchestrator.query_engine import QueryAnswer, RAGQueryEngine  # noqa: E402
from ragflow_orchestrator.templates.github_template import GitHubTemplate  # noqa: E402
from ragflow_orchestrator.templates.models import GitHubConfig  # noqa: E402


def _open_no_proxy(req: _urllib_request.Request, timeout: int) -> bytes:
    opener = _urllib_request.build_opener(_urllib_request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return response.read()


class OllamaChatGenerator:
    """AnswerGenerator backed by the Ollama /api/chat endpoint."""

    SYSTEM_PROMPT = (
        "You are an expert on GitHub repositories. "
        "Answer the user's question using ONLY the provided context chunks. "
        "If the answer is not in the context, say so clearly. "
        "Be concise and precise."
    )

    def __init__(
        self,
        model: str = "llama3.1:latest",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def generate(self, question: str, context_chunks: list[str]) -> str:
        context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no context)"
        user_message = f"Context:\n{context_text}\n\nQuestion: {question}"
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        }
        data = _json.dumps(payload).encode("utf-8")
        req = _urllib_request.Request(
            url=f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        body = _json.loads(_open_no_proxy(req, timeout=self.timeout_seconds).decode("utf-8"))
        return str(body.get("message", {}).get("content", ""))


@dataclass
class IngestPerf:
    repos_ingested: int
    ingest_summaries: int
    total_chunks: int
    duplicate_chunks_skipped: int
    total_duration_ms: float
    chunks_per_second: float
    failed: list[str]
    skipped: list[str]


@dataclass
class QueryPerf:
    question: str
    retrieve_ms: float
    generate_ms: float
    total_ms: float
    chunks_returned: int


def build_orchestrator(db_path: str, embed_model: str, ollama_url: str) -> RAGOrchestrator:
    embedder = OllamaEmbedder(model=embed_model, base_url=ollama_url)
    provider = create_provider("postgres+qdrant", dsn=db_path, qdrant_url=os.getenv("RAG_QDRANT_URL", "http://localhost:6333"), qdrant_collection=os.getenv("RAG_QDRANT_COLLECTION", "github_chunks"))
    preset = document_preset()
    return RAGOrchestrator(
        provider=provider,
        embedder=embedder,
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )


def ingest_repositories(
    orchestrator: RAGOrchestrator,
    graph_store: SqlGraphStore,
    owners: list[str],
    max_projects: int,
    max_repos_per_owner: int,
    include_readme: bool,
    include_contributors: bool,
    auth_mode: str,
    token: str | None,
) -> IngestPerf:
    config = GitHubConfig(
        owners=owners,
        max_projects=max_projects,
        max_repos_per_owner=max_repos_per_owner,
        include_readme=include_readme,
        include_contributors=include_contributors,
        auth_mode=auth_mode,
        token=token,
    )

    t0 = time.perf_counter()
    report = GitHubTemplate(orchestrator, graph_store=graph_store).run(config)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    total_chunks = sum(s.total_chunks for s in report.ingested)
    duplicates = sum(s.duplicate_chunks_skipped for s in report.ingested)
    chunks_per_second = (total_chunks / (duration_ms / 1000.0)) if duration_ms > 0 else 0.0

    return IngestPerf(
        repos_ingested=len(report.ingested),
        ingest_summaries=len(report.ingested),
        total_chunks=total_chunks,
        duplicate_chunks_skipped=duplicates,
        total_duration_ms=duration_ms,
        chunks_per_second=chunks_per_second,
        failed=[f"{e.source}: {e.reason}" for e in report.failed],
        skipped=[f"{e.source}: {e.reason}" for e in report.skipped],
    )


def _hr(char: str = "-", width: int = 70) -> None:
    print(char * width)


def print_ingest_perf(perf: IngestPerf) -> None:
    _hr("=")
    print("  INGEST PERFORMANCE")
    _hr()
    print(f"  repos ingested       : {perf.repos_ingested}")
    print(f"  ingest summaries     : {perf.ingest_summaries}")
    print(f"  total chunks stored  : {perf.total_chunks}")
    print(f"  duplicates skipped   : {perf.duplicate_chunks_skipped}")
    print(f"  total duration       : {perf.total_duration_ms:.1f} ms")
    print(f"  throughput           : {perf.chunks_per_second:.1f} chunks/s")
    if perf.failed:
        print(f"  FAILED ({len(perf.failed)}):")
        for msg in perf.failed:
            print(f"    x {msg}")
    if perf.skipped:
        print(f"  skipped ({len(perf.skipped)}):")
        for msg in perf.skipped[:5]:
            print(f"    - {msg}")
        if len(perf.skipped) > 5:
            print(f"    ... and {len(perf.skipped) - 5} more")
    _hr("=")


def print_query_perf(perf: QueryPerf) -> None:
    print(
        f"  [perf] retrieve {perf.retrieve_ms:.1f} ms  |  "
        f"generate {perf.generate_ms:.1f} ms  |  "
        f"total {perf.total_ms:.1f} ms  |  "
        f"chunks {perf.chunks_returned}"
    )


def print_graph_stats(graph_db_path: Path) -> None:
    with sqlite3.connect(graph_db_path) as conn:
        repos = int(conn.execute("SELECT COUNT(*) FROM repositories").fetchone()[0])
        contributors = int(conn.execute("SELECT COUNT(*) FROM contributors").fetchone()[0])
        edges = int(conn.execute("SELECT COUNT(*) FROM repo_contributors").fetchone()[0])
        top_repo_row = conn.execute(
            """
            SELECT full_name, stars, forks
            FROM repositories
            ORDER BY stars DESC, forks DESC
            LIMIT 1
            """
        ).fetchone()

    print("\nGRAPH STORE SUMMARY")
    _hr()
    print(f"  db file              : {graph_db_path}")
    print(f"  repositories         : {repos}")
    print(f"  contributors         : {contributors}")
    print(f"  contribution edges   : {edges}")
    if top_repo_row:
        print(f"  top repository       : {top_repo_row[0]}  (stars={top_repo_row[1]}, forks={top_repo_row[2]})")
    else:
        print("  top repository       : n/a")


def print_answer(result: QueryAnswer) -> None:
    _hr()
    print(f"Q: {result.question}")
    _hr("-")
    print(result.answer)
    _hr()
    print("Sources used:")
    seen: set[str] = set()
    for hit in result.context:
        repo_name = str(
            hit.chunk.metadata.get("full_name")
            or hit.chunk.metadata.get("repo")
            or hit.chunk.metadata.get("owner")
            or hit.chunk.source_id
        )
        if repo_name in seen:
            continue
        seen.add(repo_name)
        source_url = str(hit.chunk.metadata.get("source_url") or "")
        if source_url:
            print(f"  - {repo_name}  (score {hit.score:.4f})")
            print(f"    source_url: {source_url}")
        else:
            print(f"  - {repo_name}  (score {hit.score:.4f}, source_url: n/a)")


def ask_question(engine: RAGQueryEngine, question: str, top_k: int) -> QueryPerf:
    t0 = time.perf_counter()
    context = engine.retrieve(question=question, top_k=top_k)
    retrieve_ms = (time.perf_counter() - t0) * 1000.0

    context_chunks = [item.chunk.text for item in context]
    t1 = time.perf_counter()
    if engine.generator is None:
        answer_text = "\n\n".join(context_chunks) if context_chunks else "No context found."
        generate_ms = 0.0
    else:
        answer_text = engine.generator.generate(question=question, context_chunks=context_chunks)
        generate_ms = (time.perf_counter() - t1) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0

    result = QueryAnswer(question=question, answer=answer_text, context=context)
    print_answer(result)

    return QueryPerf(
        question=question,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        total_ms=total_ms,
        chunks_returned=len(context),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GitHub RAG demo: ingest repositories and ask questions via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--owners",
        nargs="+",
        default=["fastapi", "psf", "pallets"],
        metavar="OWNER",
        help="GitHub owners to ingest",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("RAG_POSTGRES_DSN", "postgresql://rag_user:rag_password@localhost:5432/rag_db"),
        help="PostgreSQL DSN (default: env RAG_POSTGRES_DSN or local dev DSN)",
    )
    parser.add_argument(
        "--graph-db",
        default=str(Path(__file__).parent / "github_graph.sqlite"),
        help="Repository graph DB path (default: scripts/github_demo/github_graph.sqlite)",
    )
    parser.add_argument(
        "--embed-model",
        default="nomic-embed-text:latest",
        help="Ollama embedding model (default: nomic-embed-text:latest)",
    )
    parser.add_argument(
        "--chat-model",
        default="llama3.1:latest",
        help="Ollama chat model (default: llama3.1:latest)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k (default: 5)")
    parser.add_argument("--max-projects", type=int, default=5, help="Max repositories overall")
    parser.add_argument("--max-repos", type=int, default=3, help="Max repositories per owner")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""), help="GitHub token")
    parser.add_argument(
        "--include-readme",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include README content in the indexed text",
    )
    parser.add_argument(
        "--include-contributors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include contributor data in the graph store",
    )
    parser.add_argument("--ask", default="", metavar="QUESTION", help="Single question to ask then exit")
    parser.add_argument("--interactive", action="store_true", help="Interactive question-answer REPL")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingest; use existing DB")
    parser.add_argument("--perf", action="store_true", help="Print ingest perf report even without --ask")
    args = parser.parse_args()

    pg_dsn = args.db
    graph_db_path = Path(args.graph_db)
    graph_db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding orchestrator  (embedder: {args.embed_model}, pg_dsn: {pg_dsn})")
    try:
        orchestrator = build_orchestrator(
            db_path=pg_dsn,
            embed_model=args.embed_model,
            ollama_url=args.ollama_url,
        )
    except Exception as exc:
        print(f"ERROR: Cannot connect to Ollama at {args.ollama_url}: {exc}")
        print("Make sure Ollama is running and the embedding model is pulled:")
        print(f"  ollama pull {args.embed_model}")
        return 1

    graph_store = SqlGraphStore(str(graph_db_path))

    if not args.skip_ingest:
        print(f"Ingesting {len(args.owners)} owner(s): {', '.join(args.owners)} ...")
        ingest_perf = ingest_repositories(
            orchestrator=orchestrator,
            graph_store=graph_store,
            owners=args.owners,
            max_projects=args.max_projects,
            max_repos_per_owner=args.max_repos,
            include_readme=args.include_readme,
            include_contributors=args.include_contributors,
            auth_mode="bearer" if args.token else "none",
            token=args.token or None,
        )
        print_ingest_perf(ingest_perf)
        print_graph_stats(graph_db_path)
    else:
        print("Skipping ingest (--skip-ingest).")
        ingest_perf = None

    print(f"\nConnecting chat model  ({args.chat_model}) ...")
    try:
        generator = OllamaChatGenerator(
            model=args.chat_model,
            base_url=args.ollama_url,
        )
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: chat model unavailable: {exc}. Falling back to context-only answers.")
        generator = None  # type: ignore[assignment]

    engine = RAGQueryEngine(orchestrator=orchestrator, generator=generator)

    if args.ask:
        perf = ask_question(engine=engine, question=args.ask, top_k=args.top_k)
        print_query_perf(perf)
        return 0

    if args.interactive:
        print("\nInteractive mode. Type your question and press Enter. Type 'exit' or Ctrl-C to quit.\n")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                print("Bye.")
                break
            perf = ask_question(engine=engine, question=question, top_k=args.top_k)
            print_query_perf(perf)
        return 0

    if args.perf or ingest_perf is not None:
        print("\nDone. Use --ask 'Your question' or --interactive to query the index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



