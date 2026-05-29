"""
PyPI RAG Demo
=============
Ingest PyPI package metadata into a local SQLite+vec store using OllamaEmbedder,
then answer questions about the packages via RAGQueryEngine backed by Ollama llama3.1.

Usage
-----
# Ingest and ask a single question (non-interactive):
    python scripts/pypi_demo/run.py --packages fastapi pydantic httpx --ask "What is FastAPI?"

# Interactive REPL (keeps asking until you type 'exit' or Ctrl-C):
    python scripts/pypi_demo/run.py --packages fastapi pydantic httpx --interactive

# Skip ingest if DB already exists:
    python scripts/pypi_demo/run.py --skip-ingest --ask "What Python version does pydantic require?"

Flags
-----
--packages          list of PyPI package names to ingest        (default: fastapi pydantic httpx)
--db                path to SQLite+vec database file            (default: scripts/pypi_demo/pypi.sqlite)
--embed-model       Ollama embedding model name                 (default: nomic-embed-text:latest)
--chat-model        Ollama chat model name                      (default: llama3.1:latest)
--ollama-url        Ollama base URL                             (default: http://localhost:11434)
--top-k             number of chunks to retrieve per query      (default: 5)
--max-releases      max release history entries per package     (default: 10)
--ask               single question to answer then exit
--interactive       run interactive question-answering REPL
--skip-ingest       skip ingest step (use existing DB)
--perf              print performance report even without --ask/--interactive
"""
from __future__ import annotations

import argparse
import json as _json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request as _urllib_request

# ── Ensure src/ is on the path so the package is importable without install ──
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from ragflow_orchestrator.embedding import OllamaEmbedder  # noqa: E402
from ragflow_orchestrator.factory import create_provider  # noqa: E402
from ragflow_orchestrator.orchestrator import RAGOrchestrator  # noqa: E402
from ragflow_orchestrator.presets import document_preset  # noqa: E402
from ragflow_orchestrator.query_engine import QueryAnswer, RAGQueryEngine  # noqa: E402
from ragflow_orchestrator.templates.models import PyPIConfig  # noqa: E402
from ragflow_orchestrator.templates.pypi import PyPITemplate  # noqa: E402


def _open_no_proxy(req: _urllib_request.Request, timeout: int) -> bytes:
    opener = _urllib_request.build_opener(_urllib_request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return response.read()


class OllamaChatGenerator:
    """AnswerGenerator backed by the Ollama /api/chat endpoint (llama3.1 or any model)."""

    SYSTEM_PROMPT = (
        "You are an expert on Python packages. "
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
        user_message = (
            f"Context:\n{context_text}\n\n"
            f"Question: {question}"
        )
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


# ── Performance dataclass ─────────────────────────────────────────────────────


@dataclass
class IngestPerf:
    packages_ingested: int
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


# ── Build orchestrator ────────────────────────────────────────────────────────

def build_orchestrator(db_path: str, embed_model: str, ollama_url: str) -> RAGOrchestrator:
    embedder = OllamaEmbedder(model=embed_model, base_url=ollama_url)
    provider = create_provider("sqlite+vec", db_path=db_path, table_name="pypi_chunks")
    preset = document_preset()
    return RAGOrchestrator(
        provider=provider,
        embedder=embedder,
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )


# ── Ingest ─────────────────────────────────────────────────────────────────────

def ingest_packages(
    orchestrator: RAGOrchestrator,
    packages: list[str],
    max_releases: int,
) -> IngestPerf:
    config = PyPIConfig(
        packages=packages,
        include_release_history=True,
        max_releases_per_package=max_releases,
        include_project_urls=True,
    )

    t0 = time.perf_counter()
    report = PyPITemplate(orchestrator).run(config)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    total_chunks = sum(s.total_chunks for s in report.ingested)
    duplicates = sum(s.duplicate_chunks_skipped for s in report.ingested)
    chunks_per_second = (total_chunks / (duration_ms / 1000.0)) if duration_ms > 0 else 0.0

    return IngestPerf(
        packages_ingested=len(report.ingested),
        ingest_summaries=len(report.ingested),
        total_chunks=total_chunks,
        duplicate_chunks_skipped=duplicates,
        total_duration_ms=duration_ms,
        chunks_per_second=chunks_per_second,
        failed=[f"{e.source}: {e.reason}" for e in report.failed],
        skipped=[f"{e.source}: {e.reason}" for e in report.skipped],
    )


# ── Print helpers ──────────────────────────────────────────────────────────────

def _hr(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_ingest_perf(perf: IngestPerf) -> None:
    _hr("═")
    print("  INGEST PERFORMANCE")
    _hr()
    print(f"  packages ingested    : {perf.packages_ingested}")
    print(f"  ingest summaries     : {perf.ingest_summaries}")
    print(f"  total chunks stored  : {perf.total_chunks}")
    print(f"  duplicates skipped   : {perf.duplicate_chunks_skipped}")
    print(f"  total duration       : {perf.total_duration_ms:.1f} ms")
    print(f"  throughput           : {perf.chunks_per_second:.1f} chunks/s")
    if perf.failed:
        print(f"  FAILED ({len(perf.failed)}):")
        for msg in perf.failed:
            print(f"    ✗ {msg}")
    if perf.skipped:
        print(f"  skipped ({len(perf.skipped)}):")
        for msg in perf.skipped[:5]:
            print(f"    · {msg}")
        if len(perf.skipped) > 5:
            print(f"    … and {len(perf.skipped) - 5} more")
    _hr("═")


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
    _hr("·")
    print(result.answer)
    _hr()
    print("Sources used:")
    seen: set[str] = set()
    for hit in result.context:
        pkg = hit.chunk.metadata.get("package", hit.chunk.source_id)
        key = str(pkg)
        if key not in seen:
            seen.add(key)
            source_url = str(hit.chunk.metadata.get("source_url") or "")
            if source_url:
                print(f"  · {key}  (score {hit.score:.4f})")
                print(f"    source_url: {source_url}")
            else:
                print(f"  · {key}  (score {hit.score:.4f}, source_url: n/a)")


# ── Ask one question with perf tracking ───────────────────────────────────────

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


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PyPI RAG demo: ingest packages and ask questions via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--packages", nargs="+", default=["fastapi", "pydantic", "httpx"],
        metavar="PKG", help="PyPI package names to ingest",
    )
    parser.add_argument(
        "--db", default=str(Path(__file__).parent / "pypi.sqlite"),
        help="SQLite+vec DB path (default: scripts/pypi_demo/pypi.sqlite)",
    )
    parser.add_argument(
        "--embed-model", default="nomic-embed-text:latest",
        help="Ollama embedding model (default: nomic-embed-text:latest)",
    )
    parser.add_argument(
        "--chat-model", default="llama3.1:latest",
        help="Ollama chat model (default: llama3.1:latest)",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k (default: 5)")
    parser.add_argument("--max-releases", type=int, default=10, help="Max release history per package")
    parser.add_argument("--ask", default="", metavar="QUESTION", help="Single question to ask then exit")
    parser.add_argument("--interactive", action="store_true", help="Interactive question-answer REPL")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingest; use existing DB")
    parser.add_argument("--perf", action="store_true", help="Print ingest perf report even without --ask")
    args = parser.parse_args()

    # Ensure DB directory exists
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

    # ── Ingest ────────────────────────────────────────────────────────────────
    if not args.skip_ingest:
        print(f"Ingesting {len(args.packages)} package(s): {', '.join(args.packages)} …")
        ingest_perf = ingest_packages(
            orchestrator=orchestrator,
            packages=args.packages,
            max_releases=args.max_releases,
        )
        print_ingest_perf(ingest_perf)
    else:
        print("Skipping ingest (--skip-ingest).")
        ingest_perf = None

    # ── Build query engine ────────────────────────────────────────────────────
    print(f"\nConnecting chat model  ({args.chat_model}) …")
    try:
        generator = OllamaChatGenerator(
            model=args.chat_model,
            base_url=args.ollama_url,
        )
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: chat model unavailable: {exc}. Falling back to context-only answers.")
        generator = None  # type: ignore[assignment]

    engine = RAGQueryEngine(orchestrator=orchestrator, generator=generator)

    # ── Single question ───────────────────────────────────────────────────────
    if args.ask:
        perf = ask_question(engine=engine, question=args.ask, top_k=args.top_k)
        print_query_perf(perf)
        return 0

    # ── Interactive REPL ──────────────────────────────────────────────────────
    if args.interactive:
        print("\nInteractive mode. Type your question and press Enter. Type 'exit' or Ctrl-C to quit.\n")
        while True:
            try:
                question = input("❯ ").strip()
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

    # ── Neither --ask nor --interactive: just print perf and exit ────────────
    if args.perf or ingest_perf is not None:
        print("\nDone. Use --ask 'Your question' or --interactive to query the index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
