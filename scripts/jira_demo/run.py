"""
Jira RAG Demo
=============
Ingest Jira issues into a local SQLite+vec store,
then answer questions about tickets via RAGQueryEngine backed by Ollama.

Usage
-----
# Ingest and ask a single question (non-interactive):
    python scripts/jira_demo/run.py --base-url https://your-domain.atlassian.net --jql "project = DEMO order by updated desc" --ask "What are the top open issues?"

# Interactive REPL:
    python scripts/jira_demo/run.py --base-url https://your-domain.atlassian.net --interactive

# Skip ingest if DB already exists:
    python scripts/jira_demo/run.py --skip-ingest --ask "What recurring bugs do we have?"

Auth examples
-------------
# Bearer token:
    python scripts/jira_demo/run.py --base-url https://your-domain.atlassian.net --auth-mode bearer --token YOUR_TOKEN --ask "What changed recently?"

# Basic auth:
    python scripts/jira_demo/run.py --base-url https://your-domain.atlassian.net --auth-mode basic --username you@example.com --password YOUR_PASSWORD --ask "What is blocked?"

Flags
-----
--base-url           Jira base URL                               (required for real ingest)
--jql                JQL query for issue search                  (default: order by updated desc)
--max-issues         max issues to ingest                        (default: 50)
--include-comments   include issue comments in indexed text      (default: true)
--auth-mode          none | bearer | basic                       (default: none)
--username           Jira username/email for basic auth
--password           Jira password/API token for basic auth
--token              Jira bearer token
--db                 path to SQLite+vec database file            (default: scripts/jira_demo/jira.sqlite)
--embed-model        Ollama embedding model name                 (default: nomic-embed-text:latest)
--chat-model         Ollama chat model name                      (default: llama3.1:latest)
--ollama-url         Ollama base URL                             (default: http://localhost:11434)
--top-k              number of chunks to retrieve per query      (default: 5)
--ask                single question to answer then exit
--interactive        run interactive question-answering REPL
--skip-ingest        skip ingest step (use existing DB)
--perf               print ingest perf report even without --ask/--interactive
"""
from __future__ import annotations

import argparse
import json as _json
import os
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
from ragflow_orchestrator.orchestrator import RAGOrchestrator  # noqa: E402
from ragflow_orchestrator.presets import document_preset  # noqa: E402
from ragflow_orchestrator.query_engine import QueryAnswer, RAGQueryEngine  # noqa: E402
from ragflow_orchestrator.templates.jira import JiraTemplate  # noqa: E402
from ragflow_orchestrator.templates.models import JiraConfig  # noqa: E402


def _open_no_proxy(req: _urllib_request.Request, timeout: int) -> bytes:
    opener = _urllib_request.build_opener(_urllib_request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return response.read()


class OllamaChatGenerator:
    """AnswerGenerator backed by the Ollama /api/chat endpoint."""

    SYSTEM_PROMPT = (
        "You are an expert on Jira tickets. "
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
    issues_ingested: int
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
    provider = create_provider("sqlite+vec", db_path=db_path, table_name="jira_chunks")
    preset = document_preset()
    return RAGOrchestrator(
        provider=provider,
        embedder=embedder,
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )


def ingest_issues(
    orchestrator: RAGOrchestrator,
    base_url: str,
    jql: str,
    max_issues: int,
    include_comments: bool,
    auth_mode: str,
    username: str | None,
    password: str | None,
    token: str | None,
) -> IngestPerf:
    config = JiraConfig(
        base_url=base_url,
        jql=jql,
        max_issues=max_issues,
        include_comments=include_comments,
        auth_mode=auth_mode,
        username=username,
        password=password,
        token=token,
    )

    t0 = time.perf_counter()
    report = JiraTemplate(orchestrator).run(config)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    total_chunks = sum(s.total_chunks for s in report.ingested)
    duplicates = sum(s.duplicate_chunks_skipped for s in report.ingested)
    chunks_per_second = (total_chunks / (duration_ms / 1000.0)) if duration_ms > 0 else 0.0

    return IngestPerf(
        issues_ingested=len(report.ingested),
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
    print(f"  issues ingested      : {perf.issues_ingested}")
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


def print_answer(result: QueryAnswer) -> None:
    _hr()
    print(f"Q: {result.question}")
    _hr("-")
    print(result.answer)
    _hr()
    print("Sources used:")
    seen: set[str] = set()
    for hit in result.context:
        issue_key = str(hit.chunk.metadata.get("issue_key") or hit.chunk.source_id)
        if issue_key in seen:
            continue
        seen.add(issue_key)
        source_url = str(hit.chunk.metadata.get("source_url") or "")
        if source_url:
            print(f"  - {issue_key}  (score {hit.score:.4f})")
            print(f"    source_url: {source_url}")
        else:
            print(f"  - {issue_key}  (score {hit.score:.4f}, source_url: n/a)")


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
        description="Jira RAG demo: ingest issues and ask questions via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Jira base URL, e.g. https://your-domain.atlassian.net",
    )
    parser.add_argument(
        "--jql",
        default="order by updated desc",
        help="JQL query for issue search (default: order by updated desc)",
    )
    parser.add_argument("--max-issues", type=int, default=50, help="Max issues to ingest (default: 50)")
    parser.add_argument(
        "--include-comments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include issue comments in indexed text",
    )
    parser.add_argument(
        "--auth-mode",
        choices=["none", "bearer", "basic"],
        default="none",
        help="Jira auth mode (default: none)",
    )
    parser.add_argument("--username", default=os.environ.get("JIRA_USERNAME", ""), help="Jira username/email")
    parser.add_argument(
        "--password",
        default=os.environ.get("JIRA_PASSWORD", ""),
        help="Jira password/API token for basic auth",
    )
    parser.add_argument("--token", default=os.environ.get("JIRA_TOKEN", ""), help="Jira bearer token")
    parser.add_argument(
        "--db",
        default=str(Path(__file__).parent / "jira.sqlite"),
        help="SQLite+vec DB path (default: scripts/jira_demo/jira.sqlite)",
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
    parser.add_argument("--ask", default="", metavar="QUESTION", help="Single question to ask then exit")
    parser.add_argument("--interactive", action="store_true", help="Interactive question-answer REPL")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingest; use existing DB")
    parser.add_argument("--perf", action="store_true", help="Print ingest perf report even without --ask")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding orchestrator  (embedder: {args.embed_model}, db: {db_path})")
    try:
        orchestrator = build_orchestrator(
            db_path=str(db_path),
            embed_model=args.embed_model,
            ollama_url=args.ollama_url,
        )
    except Exception as exc:
        print(f"ERROR: Cannot connect to Ollama at {args.ollama_url}: {exc}")
        print("Make sure Ollama is running and the embedding model is pulled:")
        print(f"  ollama pull {args.embed_model}")
        return 1

    if not args.skip_ingest:
        if not args.base_url:
            print("ERROR: --base-url is required unless --skip-ingest is used")
            return 1
        print(f"Ingesting Jira issues from {args.base_url} ...")
        ingest_perf = ingest_issues(
            orchestrator=orchestrator,
            base_url=args.base_url,
            jql=args.jql,
            max_issues=args.max_issues,
            include_comments=args.include_comments,
            auth_mode=args.auth_mode,
            username=args.username or None,
            password=args.password or None,
            token=args.token or None,
        )
        print_ingest_perf(ingest_perf)
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
