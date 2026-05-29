"""
Incremental Sync RAG Demo
=========================
Ingest files from folders with change tracking (mtime+size fingerprint),
then answer questions via RAGQueryEngine backed by Ollama.

Usage
-----
# First run ingests everything:
    python scripts/incremental_demo/run.py --folders d:/TEMP --ask "О чем документы?"

# Next run ingests only changed files:
    python scripts/incremental_demo/run.py --folders d:/TEMP --ask "Что нового изменилось?"

# Skip ingest and query existing index:
    python scripts/incremental_demo/run.py --skip-ingest --ask "Какие темы в базе?"
"""
from __future__ import annotations

import argparse
import json as _json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request as _urllib_request

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from ragflow_orchestrator.embedding import OllamaEmbedder  # noqa: E402
from ragflow_orchestrator.factory import create_provider  # noqa: E402
from ragflow_orchestrator.orchestrator import RAGOrchestrator  # noqa: E402
from ragflow_orchestrator.presets import document_preset  # noqa: E402
from ragflow_orchestrator.query_engine import QueryAnswer, RAGQueryEngine  # noqa: E402
from ragflow_orchestrator.templates.incremental_sync import IncrementalSyncTemplate  # noqa: E402
from ragflow_orchestrator.templates.models import IncrementalSyncConfig  # noqa: E402


def _open_no_proxy(req: _urllib_request.Request, timeout: int) -> bytes:
    opener = _urllib_request.build_opener(_urllib_request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return response.read()


class OllamaChatGenerator:
    SYSTEM_PROMPT = (
        "You are an expert on synchronized document collections. "
        "Answer the user's question using ONLY the provided context chunks. "
        "If the answer is not in the context, say so clearly. "
        "Be concise and precise."
    )

    def __init__(self, model: str = "llama3.1:latest", base_url: str = "http://localhost:11434", timeout_seconds: int = 120, temperature: float = 0.2) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def generate(self, question: str, context_chunks: list[str]) -> str:
        context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no context)"
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
        }
        req = _urllib_request.Request(
            url=f"{self.base_url}/api/chat",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        body = _json.loads(_open_no_proxy(req, timeout=self.timeout_seconds).decode("utf-8"))
        return str(body.get("message", {}).get("content", ""))


@dataclass
class IngestPerf:
    files_ingested: int
    ingest_summaries: int
    total_chunks: int
    duplicate_chunks_skipped: int
    total_duration_ms: float
    chunks_per_second: float
    failed: list[str]
    skipped: list[str]


@dataclass
class QueryPerf:
    retrieve_ms: float
    generate_ms: float
    total_ms: float
    chunks_returned: int


def build_orchestrator(db_path: str, embed_model: str, ollama_url: str) -> RAGOrchestrator:
    embedder = OllamaEmbedder(model=embed_model, base_url=ollama_url)
    provider = create_provider("sqlite+vec", db_path=db_path, table_name="incremental_chunks")
    preset = document_preset()
    return RAGOrchestrator(provider=provider, embedder=embedder, chunker=preset.chunker, cleaner=preset.cleaner)


def ingest_incremental(
    orchestrator: RAGOrchestrator,
    folders: list[str],
    recursive: bool,
    extensions: list[str],
    state_file: str,
) -> IngestPerf:
    normalized_ext = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
    config = IncrementalSyncConfig(folders=folders, recursive=recursive, extensions=normalized_ext, state_file=state_file)

    t0 = time.perf_counter()
    report = IncrementalSyncTemplate(orchestrator).run(config)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    total_chunks = sum(s.total_chunks for s in report.ingested)
    duplicates = sum(s.duplicate_chunks_skipped for s in report.ingested)
    cps = (total_chunks / (duration_ms / 1000.0)) if duration_ms > 0 else 0.0

    return IngestPerf(
        files_ingested=len(report.ingested),
        ingest_summaries=len(report.ingested),
        total_chunks=total_chunks,
        duplicate_chunks_skipped=duplicates,
        total_duration_ms=duration_ms,
        chunks_per_second=cps,
        failed=[f"{e.source}: {e.reason}" for e in report.failed],
        skipped=[f"{e.source}: {e.reason}" for e in report.skipped],
    )


def _hr(char: str = "-", width: int = 70) -> None:
    print(char * width)


def print_ingest_perf(perf: IngestPerf) -> None:
    _hr("=")
    print("  INGEST PERFORMANCE")
    _hr()
    print(f"  files ingested       : {perf.files_ingested}")
    print(f"  ingest summaries     : {perf.ingest_summaries}")
    print(f"  total chunks stored  : {perf.total_chunks}")
    print(f"  duplicates skipped   : {perf.duplicate_chunks_skipped}")
    print(f"  total duration       : {perf.total_duration_ms:.1f} ms")
    print(f"  throughput           : {perf.chunks_per_second:.1f} chunks/s")
    _hr("=")


def print_answer(result: QueryAnswer) -> None:
    _hr()
    print(f"Q: {result.question}")
    _hr("-")
    print(result.answer)
    _hr()
    print("Sources used:")
    seen: set[str] = set()
    for hit in result.context:
        file_path = str(hit.chunk.metadata.get("file_path") or hit.chunk.source_id)
        if file_path in seen:
            continue
        seen.add(file_path)
        print(f"  - {file_path}  (score {hit.score:.4f})")


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
    print_answer(QueryAnswer(question=question, answer=answer_text, context=context))
    return QueryPerf(retrieve_ms=retrieve_ms, generate_ms=generate_ms, total_ms=total_ms, chunks_returned=len(context))


def print_query_perf(perf: QueryPerf) -> None:
    print(
        f"  [perf] retrieve {perf.retrieve_ms:.1f} ms  |  "
        f"generate {perf.generate_ms:.1f} ms  |  "
        f"total {perf.total_ms:.1f} ms  |  "
        f"chunks {perf.chunks_returned}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="IncrementalSync RAG demo", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--folders", nargs="+", default=["d:/TEMP"], metavar="FOLDER", help="Folders to ingest")
    parser.add_argument("--db", default=str(Path(__file__).parent / "incremental.sqlite"), help="SQLite+vec DB path")
    parser.add_argument("--state-file", default=str(Path(__file__).parent / ".incremental_state.json"), help="Incremental state file path")
    parser.add_argument("--embed-model", default="nomic-embed-text:latest", help="Ollama embedding model")
    parser.add_argument("--chat-model", default="llama3.1:latest", help="Ollama chat model")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Include nested files recursively")
    parser.add_argument("--extensions", nargs="+", default=[".docx", ".pdf", ".xlsx", ".txt", ".md", ".html"], metavar="EXT", help="File extensions to include")
    parser.add_argument("--ask", default="", metavar="QUESTION", help="Single question to ask then exit")
    parser.add_argument("--interactive", action="store_true", help="Interactive question-answer REPL")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingest; use existing DB")
    parser.add_argument("--perf", action="store_true", help="Print ingest perf report")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding orchestrator  (embedder: {args.embed_model}, db: {db_path})")
    orchestrator = build_orchestrator(str(db_path), args.embed_model, args.ollama_url)

    if not args.skip_ingest:
        print(f"Incremental ingest for {len(args.folders)} folder(s): {', '.join(args.folders)} ...")
        ingest_perf = ingest_incremental(orchestrator, args.folders, args.recursive, args.extensions, args.state_file)
        print_ingest_perf(ingest_perf)
    else:
        print("Skipping ingest (--skip-ingest).")
        ingest_perf = None

    print(f"\nConnecting chat model  ({args.chat_model}) ...")
    try:
        generator = OllamaChatGenerator(model=args.chat_model, base_url=args.ollama_url)
    except Exception as exc:
        print(f"WARNING: chat model unavailable: {exc}. Falling back to context-only answers.")
        generator = None  # type: ignore[assignment]

    engine = RAGQueryEngine(orchestrator=orchestrator, generator=generator)

    if args.ask:
        perf = ask_question(engine, args.ask, args.top_k)
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
            perf = ask_question(engine, question, args.top_k)
            print_query_perf(perf)
        return 0

    if args.perf or ingest_perf is not None:
        print("\nDone. Use --ask 'Your question' or --interactive to query the index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
