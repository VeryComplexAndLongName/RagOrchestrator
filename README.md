# ragflow_orchestrator

Universal and extensible RAG module with standardized interfaces and storage adapters.

![Ragflow Orchestrator](RagflowOrchestrator.png)

## Authors

- Alexander Ivanov
- email: VeryComplexAndLongName@gamil.com
- Telegram: @alexander_ivan0v

## Goals

- One internal chunk contract across all vector stores.
- Pluggable providers for Qdrant, PGVector, and SQLite+vec style workflows.
- Standardized ingestion pipeline: cleaning -> chunking -> embedding -> upsert.
- Standardized retrieval APIs with semantic/hybrid strategies.
- First-class interoperability with PromptOrchestrator pipelines.
- Extensible migration framework (Alembic-like, provider-agnostic state tracking).
- Quantitative quality checks for chunks and retrieval performance.

## Core Data Model

`BaseChunk` is the canonical internal model:

- `id`: unique chunk id
- `vector`: embedding vector
- `text`: original chunk text
- `metadata`: flexible JSON metadata
- `source_id`: source document id
- `chunk_index`: chunk order in source
- `created_at`: ingestion timestamp
- `kind`, `version`, `is_deleted`: lifecycle and typing helpers

Extended chunk types are supported (`CodeChunk`, `ContractChunk`) and custom models can be added via Pydantic inheritance.

## Architecture

- `models.py`: canonical chunk/query/result models.
- `standards.py`: typed metadata standards for code/table/pdf/html/word/mixed.
- `protocols.py`: abstraction contracts (`RAGProvider`, `Chunker`, `Embedder`, `Cleaner`).
- `adapters/`: DB-specific provider implementations.
- `chunking/`, `cleaning/`, `embedding/`: pipeline strategy modules.
- `retrieval/`: retrieval strategies (semantic/hybrid).
- `migrations/`: versioned migration manager.
- `migrations/schema_evolution.py`: SQL generation helpers for add/drop/rename field workflows.
- `quality/`: chunk/retrieval quality metrics.
- `orchestrator.py`: high-level API.

## Providers

- `SQLiteVecProvider`: local SQLite with JSON vector storage + cosine fallback.
- `QdrantProvider`: native Qdrant integration.
- `PGVectorProvider`: PostgreSQL + pgvector integration.

## Production Embeddings (Ollama)

Use local Ollama embeddings in production mode:

```python
from ragflow_orchestrator.embedding import OllamaEmbedder

embedder = OllamaEmbedder(model="nomic-embed-text:latest")
print(embedder.dimensions)
```

Provider and model are configurable via settings; no provider is hardwired in orchestrator factory.

```python
from ragflow_orchestrator import (
    ConfigStore,
    EmbeddingConfig,
    ModuleConfig,
    PipelineConfig,
    PromptStyleRAGProviderAdapter,
    ProviderConfig,
    RAGOrchestratorFactory,
)

store = ConfigStore(
    ModuleConfig(
        provider=ProviderConfig(kind="sqlite+vec", params={"db_path": "rag.db", "table_name": "rag_chunks"}),
        embedding=EmbeddingConfig(
            provider="ollama",                  # switch provider here
            model="nomic-embed-text:latest",     # switch model here
            options={"base_url": "http://localhost:11434", "timeout_seconds": 60},
        ),
        pipeline=PipelineConfig(preset="document"),
    )
)

orchestrator = RAGOrchestratorFactory.from_config_store(store)
```

Discover available local models:

```python
from ragflow_orchestrator.embedding import OllamaEmbedder

print(OllamaEmbedder.list_models())
```

Recommended CPU-friendly default: `nomic-embed-text:latest`.

Use factory:

```python
from ragflow_orchestrator.factory import create_provider

provider = create_provider("sqlite+vec", db_path="rag.db")
```

## Quick Start

```python
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.embedding import HashEmbedder
from ragflow_orchestrator.presets import document_preset

provider = create_provider("sqlite+vec", db_path="rag.db")
preset = document_preset()

orchestrator = RAGOrchestrator(
    provider=provider,
    embedder=HashEmbedder(dimensions=256),
    chunker=preset.chunker,
    cleaner=preset.cleaner,
)

orchestrator.ingest(
    source_id="doc-1",
    raw_text="RAG orchestration standardizes ingestion and retrieval.",
    metadata={"tenant_id": "t1", "language": "en", "doctype": "note"},
)

hits = orchestrator.search("How does orchestration help?", top_k=3)
for hit in hits:
    print(hit.score, hit.chunk.id, hit.chunk.text)
```

## Migration Example

```python
from ragflow_orchestrator.migrations import JsonFileMigrationStore, MigrationManager, MigrationStepDef

steps = [
    MigrationStepDef(
        version=1,
        description="add tenant_id policy",
        up=lambda: print("apply v1"),
        down=lambda: print("rollback v1"),
    )
]

manager = MigrationManager(
    namespace="sqlite-main",
    store=JsonFileMigrationStore(".rag_migrations.json"),
    steps=steps,
)
manager.upgrade()
```

Schema evolution helper example:

```python
from ragflow_orchestrator.migrations import add_field_sql

sql = add_field_sql("pgvector", "rag_chunks", "tenant_id", "TEXT")
print(sql)
```

## Quality Evaluation

```python
from ragflow_orchestrator.quality import evaluate_chunks, evaluate_retrieval, RetrievalEvalCase

chunk_report = evaluate_chunks(chunks)

retrieval_report = evaluate_retrieval(
    cases=[
        RetrievalEvalCase(expected_chunk_ids={"a"}, retrieved_chunk_ids=["a", "b", "c"]),
    ],
    k=3,
)
```

## Reranking and Strategy Auto-Comparison

Offline dataset: [datasets/retrieval_eval.jsonl](datasets/retrieval_eval.jsonl)

Run comparison example:

```bash
python examples/evaluate_retrieval.py
```

Included strategies:

- semantic retrieval
- hybrid retrieval
- semantic + cosine reranker

Dual profile comparison (cosine rerank vs Ollama LLM rerank) is available via [examples/evaluate_retrieval.py](examples/evaluate_retrieval.py).

- `cosine_profile`: semantic/hybrid + cosine rerank
- `ollama_profile`: semantic/hybrid + Ollama LLM rerank

To force a specific Ollama rerank model, set `RAG_RERANK_MODEL`.

The report returns precision@k, recall@k and MRR for each strategy.

## Publishing to PyPI and GitHub

This repository is configured to publish the distribution name `ragflow_orchestrator`.

Import path stays the same:

```python
import ragflow_orchestrator
```

Install from PyPI:

```bash
pip install ragflow_orchestrator
```

If you plan to use provider-specific backends, install extras:

```bash
pip install "ragflow_orchestrator[qdrant]"
pip install "ragflow_orchestrator[pgvector]"
```

What each extra installs:

- `qdrant`: `qdrant-client>=1.9`
- `pgvector`: `sqlalchemy>=2.0`, `psycopg[binary]>=3.1`, `pgvector>=0.3`

### Local preflight before release

```bash
python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('dist','build')]"
python -m pip install --upgrade pip
python -m pip install build twine
python -m build
python -m twine check dist/*
```

### GitHub + PyPI release flow

1. Create a PyPI project named `rag-orchestrator` (PyPI normalizes `_` to `-`).
2. In GitHub repo settings:
    - Enable GitHub Actions for the repo.
3. In PyPI project settings, configure Trusted Publishing:
    - Owner: your GitHub org/user.
    - Repository: this repository.
    - Workflow: `publish.yml`.
    - Environment: `pypi`.
4. Ensure auto-tag workflow is enabled:

- [.github/workflows/auto-tag-from-version.yml](.github/workflows/auto-tag-from-version.yml)
- it watches `pyproject.toml`, reads `[project].version`, and creates `<version>` tag automatically.

5. Bump version in `pyproject.toml` and push to default branch.

6. Publish only selected version manually (recommended way with tag chooser):

- open GitHub `Releases` -> `Draft a new release`
- in `Choose a tag`, select existing tag (for example `0.1.3`)
- click `Publish release`
- workflow [.github/workflows/publish.yml](.github/workflows/publish.yml) starts automatically on `release.published`

The workflow validates that selected tag matches `[project].version` in `pyproject.toml` for that tag and only then publishes to PyPI.

Manual fallback is still available via `workflow_dispatch` input `release_tag`.

The workflow [.github/workflows/publish.yml](.github/workflows/publish.yml) will:

- build sdist and wheel
- verify metadata with Twine
- publish to PyPI using OIDC Trusted Publishing
- run for selected tag from release event (or manual fallback via `workflow_dispatch`)

## PromptOrchestrator Interoperability

RagOrchestrator can be used as a retrieval backend for PromptOrchestrator flows.

PromptOrchestrator: [https://github.com/VeryComplexAndLongName/PromptOrchestrator](https://github.com/VeryComplexAndLongName/PromptOrchestrator)

How integration works:

- RagOrchestrator keeps ingestion/storage concerns (`clean -> chunk -> embed -> upsert`).
- PromptOrchestrator consumes ready context via `retrieve(query, limit)`.
- `PromptStyleRAGProviderAdapter` bridges native retrieval output to `DocChunk` contract.

Compatibility building blocks:

- `DocChunk` model in [src/ragflow_orchestrator/context.py](src/ragflow_orchestrator/context.py)
- abstract `RAGProvider` in [src/ragflow_orchestrator/rag/base.py](src/ragflow_orchestrator/rag/base.py)
- `PromptStyleRAGProviderAdapter` bridge in [src/ragflow_orchestrator/rag/compat.py](src/ragflow_orchestrator/rag/compat.py)

This keeps ragflow_orchestrator internals storage-oriented while exposing prompt_orchestrator-style `retrieve(query, limit)` contract where needed.

Example 1: factory bootstrap (aligned with PromptOrchestrator style):

```python
from ragflow_orchestrator import (
    ConfigStore,
    EmbeddingConfig,
    ModuleConfig,
    PipelineConfig,
    PromptStyleRAGProviderAdapter,
    ProviderConfig,
    RAGOrchestratorFactory,
)

config_store = ConfigStore(
    ModuleConfig(
        provider=ProviderConfig(kind="sqlite+vec", params={"db_path": "rag.db", "table_name": "rag_chunks"}),
        embedding=EmbeddingConfig(provider="ollama", model="nomic-embed-text:latest"),
        pipeline=PipelineConfig(preset="document"),
    )
)

orchestrator = RAGOrchestratorFactory.from_config_store(config_store)

adapter = PromptStyleRAGProviderAdapter(
    provider=orchestrator.provider,
    embedder=orchestrator.embedder,
)
docs = adapter.retrieve("How does incremental sync work?", limit=3)
for doc in docs:
    print(doc.id, doc.score, doc.content[:80])
```

Example 2: explicit adapter for an existing provider/embedder pair:

```python
from ragflow_orchestrator.embedding import HashEmbedder
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.rag.compat import PromptStyleRAGProviderAdapter

provider = create_provider("sqlite+vec", db_path="rag.db")
embedder = HashEmbedder(dimensions=256)

prompt_provider = PromptStyleRAGProviderAdapter(provider=provider, embedder=embedder)
docs = prompt_provider.retrieve("What are the default integration env vars?", limit=5)
```

In PromptOrchestrator, use `docs` as context blocks for prompt assembly and answer generation.

## Integration Tests

Install test dependencies:

```bash
pip install -e .[all]
```

Run all tests:

```bash
pytest -q
```

Integration test defaults:

- `QDRANT_URL=http://localhost:6333`
- `PGVECTOR_DSN=postgresql+psycopg://postgres:N0th1ing@localhost:5432/app`

If endpoints are unavailable, integration tests are skipped automatically.

## Preflight Diagnostics

Run one command to inspect environment before integration tests:

```bash
python scripts/preflight_check.py
```

It reports:

- effective proxy detected by Python
- Qdrant default path vs direct no-proxy path
- PostgreSQL auth result
- `vector` extension presence
- ready-to-run PowerShell fix commands

Run preflight + integration tests in one command:

```bash
python scripts/run_preflight_and_integration.py
```

VS Code task is also available:

- label: RAG: Preflight + Integration
- file: .vscode/tasks.json

## Ingestion Templates (Preset Scenarios)

Ready-to-run templates are available to minimize user input:

- `WebCrawlTemplate`: crawl sites by URL list + depth, extract text from HTML, clean/chunk/embed, and ingest.
- `DocumentFolderTemplate`: scan folders for `.docx`, `.pdf`, `.xlsx`, `.txt`, `.md`, `.html`, extract and ingest.
- `ConfluenceWikiTemplate`: ingest Confluence pages by space keys or explicit page ids.
- `JiraTemplate`: ingest Jira issues by JQL (with comments support).
- `APIReferenceTemplate`: ingest OpenAPI/Swagger specs from file or URL.
- `PyPITemplate`: ingest PyPI package metadata, release history, and project URLs.
- `GitHubTemplate`: ingest public GitHub repositories by owner, enrich with contributors and README, and persist repository graph.
- `GitLabTemplate`: ingest public GitLab repositories/groups, enrich with contributors and README, and persist repository graph.
- `RepoCodeTemplate`: scan code repositories and ingest source files with repo-specific metadata.
- `EmailTicketTemplate`: ingest support tickets from `.eml`, `.jsonl`, `.csv`, `.txt`, `.md`.
- `IncrementalSyncTemplate`: ingest only changed files using a sync state file.

Demo runners (one script per template):

- `scripts/webcrawl_demo/run.py` -> `WebCrawlTemplate`
- `scripts/doc_demo/run.py` -> `DocumentFolderTemplate`
- `scripts/confluence_demo/run.py` -> `ConfluenceWikiTemplate`
- `scripts/jira_demo/run.py` -> `JiraTemplate`
- `scripts/api_demo/run.py` -> `APIReferenceTemplate`
- `scripts/pypi_demo/run.py` -> `PyPITemplate`
- `scripts/github_demo/run.py` -> `GitHubTemplate`
- `scripts/gitlab_demo/run.py` -> `GitLabTemplate`
- `scripts/repocode_demo/run.py` -> `RepoCodeTemplate`
- `scripts/email_demo/run.py` -> `EmailTicketTemplate`
- `scripts/incremental_demo/run.py` -> `IncrementalSyncTemplate`

Each demo supports the same execution pattern:

- ingest (default)
- single question mode: `--ask`
- interactive mode: `--interactive`
- query-only mode: `--skip-ingest`
- basic timing report: `--perf`

Duplicate control:

- Deduplication is enforced in `RAGOrchestrator.ingest` for all templates.
- Duplicate chunk text is fingerprinted and skipped before writing to vector DB.
- Dedup fingerprints are persisted in a sidecar SQLite store (`*.dedup.sqlite`).

## Local Generated SQLite Files

The following files are local runtime artifacts generated by examples/templates and are safe to remove:

- `.rag_dedup.sqlite`: dedup fingerprint store used by `RAGOrchestrator.ingest`.
- `.rag_graph.sqlite`: default graph DB for repository/contributor relations (`SqlGraphStore`).
- `eval_demo.db`: local SQLite+vec DB used by `examples/evaluate_retrieval.py`.
- `example_rag.db`: local SQLite+vec DB used by `examples/basic_usage.py`.

These files are recreated automatically on the next run of corresponding examples/templates.

Repository graph analytics:

- Repository and contributor graph is stored in SQLite (`graph_store.db_path`).
- Supported analytics out of the box:
    - find repositories by keyword/topic
    - count contributors for repository
    - find most popular repository (stars/forks)
- Query helper script:

```bash
python scripts/query_repo_graph.py --db rag_graph.sqlite search telegram
python scripts/query_repo_graph.py --db rag_graph.sqlite contributors microsoft/vscode
python scripts/query_repo_graph.py --db rag_graph.sqlite popular
```

Language handling modes:

- `auto`: automatic heuristic detection (`ru` / `en` / `mixed`)
- `force_ru`
- `force_en`
- `mixed`

Quick example:

```python
from ragflow_orchestrator import (
    DocumentFolderConfig,
    DocumentFolderTemplate,
    HashEmbedder,
    LanguageMode,
    RAGOrchestrator,
    WebCrawlConfig,
    WebCrawlTemplate,
    create_provider,
    document_preset,
)

provider = create_provider("sqlite+vec", db_path="rag.db", table_name="rag_chunks")
preset = document_preset()

orchestrator = RAGOrchestrator(
    provider=provider,
    embedder=HashEmbedder(dimensions=128),
    chunker=preset.chunker,
    cleaner=preset.cleaner,
)

web_report = WebCrawlTemplate(orchestrator).run(
    WebCrawlConfig(urls=["https://example.com"], max_depth=1, language_mode=LanguageMode.AUTO)
)

file_report = DocumentFolderTemplate(orchestrator).run(
    DocumentFolderConfig(folders=["docs"], recursive=True, language_mode=LanguageMode.AUTO)
)
```

## templates.json (No Code Changes)

You can switch scenarios by editing `templates.json` only.

Run:

```bash
python scripts/run_template.py templates.json
```

Runtime reporting:

- Template run report now includes `run_metrics`:
    - `total_duration_ms`
    - `total_chunks`
    - `duplicate_chunks_skipped`
    - `chunks_per_second`
- This is computed from already-available counters plus one `perf_counter` measurement, so default overhead is minimal.
- Optional quality evaluation is controlled by `evaluation.enabled` and is disabled by default.
- Append-only experiment log is controlled by `experiment_log.enabled` and is enabled by default.
- Default experiment DB path: `loadtest/experiments.sqlite`.

Switch scenario by changing only:

- `active_scenario`: `web_crawl` | `document_folder` | `confluence_wiki` | `jira` | `api_reference` | `pypi` | `github` | `gitlab` | `repo_code` | `email_ticket` | `incremental_sync`

Minimal structure:

```json
{
    "orchestrator": {
        "provider": {
            "kind": "sqlite+vec",
            "params": {"db_path": "rag_templates.db", "table_name": "rag_chunks"}
        },
        "embedding": {
            "provider": "hash",
            "options": {"dimensions": 256}
        },
        "pipeline": {"preset": "document"}
    },
    "graph_store": {
        "db_path": "rag_graph.sqlite"
    },
    "evaluation": {
        "enabled": false,
        "dataset_path": "datasets/retrieval_eval.jsonl",
        "top_k": 3
    },
    "experiment_log": {
        "enabled": true,
        "db_path": "loadtest/experiments.sqlite"
    },
    "active_scenario": "repo_code",
    "scenarios": {
        "confluence_wiki": {
            "base_url": "https://confluence.example.com",
            "space_keys": ["ENG"],
            "max_pages": 50,
            "auth_mode": "none",
            "language_mode": "auto"
        },
        "jira": {
            "base_url": "https://jira.example.com",
            "jql": "project = ENG ORDER BY updated DESC",
            "max_issues": 100,
            "include_comments": true,
            "auth_mode": "none",
            "language_mode": "auto"
        },
        "api_reference": {
            "sources": ["openapi.json"],
            "include_operations": true,
            "include_schemas": true,
            "language_mode": "auto"
        },
        "pypi": {
            "packages": ["fastapi", "pydantic"],
            "include_release_history": true,
            "max_releases_per_package": 10,
            "include_project_urls": true,
            "language_mode": "auto"
        },
        "github": {
            "owners": ["microsoft"],
            "max_projects": 20,
            "max_repos_per_owner": 10,
            "include_readme": true,
            "include_contributors": true,
            "auth_mode": "none",
            "language_mode": "auto"
        },
        "gitlab": {
            "base_url": "https://gitlab.com",
            "groups_or_users": ["gitlab-org"],
            "max_projects": 20,
            "max_repos_per_owner": 10,
            "include_readme": true,
            "include_contributors": true,
            "auth_mode": "none",
            "language_mode": "auto"
        },
        "repo_code": {"repos": ["."], "recursive": true, "language_mode": "mixed"},
        "email_ticket": {"sources": ["tickets"], "recursive": true, "language_mode": "auto"},
        "incremental_sync": {
            "folders": ["docs"],
            "recursive": true,
            "state_file": ".rag_incremental_state.json",
            "language_mode": "auto"
        }
    }
}
```

## RAG Query Interface

`RAGQueryEngine` provides a unified interface for querying indexed knowledge:

- `retrieve(question, top_k, filters)` returns retrieval hits.
- `answer(question, top_k, filters)` returns answer + used context.

If no generator is configured, it returns a deterministic context-based fallback answer.
You can plug in any LLM adapter (including prompt_orchestrator-based prompt construction) via `AnswerGenerator`.

```python
from ragflow_orchestrator.query_engine import RAGQueryEngine

engine = RAGQueryEngine(orchestrator)
result = engine.answer("Найди репозитории для Telegram-ботов", top_k=5)
print(result.answer)
```

You can scope query to specific source types (for example: `confluence`, `jira`, `repo_code`, `web_crawl`):

```python
result = engine.answer_from_sources(
        question="Какие инциденты связаны с оплатой?",
        source_types=["confluence", "jira"],
        top_k=8,
)
print(result.answer)
```

## Load Testing Across Databases

Use built-in benchmark script to compare `sqlite+vec`, `pgvector`, and `qdrant`:

```bash
python scripts/load_test_backends.py \
    --providers sqlite+vec pgvector qdrant \
    --documents 500 \
    --queries 800 \
    --concurrency 8 \
    --dimensions 256 \
    --pg-dsn "postgresql+psycopg://postgres:N0th1ing@localhost:5432/app" \
    --qdrant-url "http://localhost:6333" \
    --json-out loadtest/load_test_results.json
```

What you get:

- ingest throughput (`docs/s`)
- search throughput (`QPS`)
- latency percentiles (`p50/p95/p99`)
- JSON report for trend tracking (default: `loadtest/load_test_results.json`)

## Profiling Bottlenecks

Use cProfile-based script to detect hot functions in ingestion/retrieval pipeline:

```bash
python scripts/profile_hotspots.py \
    --provider sqlite+vec \
    --documents 300 \
    --queries 500 \
    --out loadtest/profile_hotspots.txt
```

For PGVector and Qdrant, switch provider and pass connection parameters:

```bash
python scripts/profile_hotspots.py --provider pgvector --pg-dsn "postgresql+psycopg://postgres:N0th1ing@localhost:5432/app"
python scripts/profile_hotspots.py --provider qdrant --qdrant-url "http://localhost:6333"
```

## Optional Hugging Face Layer (Embeddings + Rerank)

Hugging Face providers are optional and are not required for the base install.

- `HFEmbedder` supports sentence-transformers models (including e5/bge families by model name).
- `HFReranker` supports cross-encoder rerank models.

Install only when needed:

```bash
pip install -e .[hf]
```

Run baseline vs HF comparison on the built-in retrieval dataset:

```bash
python scripts/compare_baseline_vs_hf.py \
    --dataset datasets/retrieval_eval.jsonl \
    --top-k 2 \
    --loops 100 \
    --hf-embedder-model sentence-transformers/all-MiniLM-L6-v2 \
    --hf-reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --json-out loadtest/compare_baseline_vs_hf.json
```

Compare experiment trends from template runs:

```bash
python scripts/compare_experiment_trends.py --db loadtest/experiments.sqlite --group-by scenario --metric chunks_per_second
python scripts/compare_experiment_trends.py --db loadtest/experiments.sqlite --group-by strategy_name --metric ndcg_at_k
```

Metrics produced:

- quality: `precision@k`, `recall@k`, `MRR`, `nDCG@k`
- performance: `p50/p95 latency`, `throughput (QPS)`
- memory: `RAM MB`, `VRAM MB` (0 when CUDA is unavailable)

### Baseline vs HF Results

Environment of this run:

- dataset: `datasets/retrieval_eval.jsonl`
- top_k: `2`
- loops: `60`
- HF embedder: `sentence-transformers/all-MiniLM-L6-v2`
- HF reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Quality (`precision@k`, `recall@k`, `MRR`, `nDCG@k`):

| profile | strategy | precision@k | recall@k | MRR | nDCG@k |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline_hash_cosine | semantic | 0.500 | 1.000 | 1.000 | 1.000 |
| baseline_hash_cosine | hybrid | 0.500 | 1.000 | 1.000 | 1.000 |
| baseline_hash_cosine | semantic_cosine_rerank | 0.500 | 1.000 | 1.000 | 1.000 |
| hf_embedder_hf_reranker | semantic | 0.500 | 1.000 | 1.000 | 1.000 |
| hf_embedder_hf_reranker | hybrid | 0.500 | 1.000 | 1.000 | 1.000 |
| hf_embedder_hf_reranker | semantic_hf_rerank | 0.500 | 1.000 | 1.000 | 1.000 |

Performance (`p50/p95`, average latency, throughput):

| profile | strategy | p50 ms | p95 ms | avg ms | throughput qps |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline_hash_cosine | semantic | 0.242 | 0.433 | 0.271 | 3683.27 |
| baseline_hash_cosine | hybrid | 0.264 | 0.433 | 0.285 | 3502.76 |
| baseline_hash_cosine | semantic_cosine_rerank | 0.274 | 0.524 | 0.310 | 3221.10 |
| hf_embedder_hf_reranker | semantic | 9.197 | 10.014 | 8.569 | 116.69 |
| hf_embedder_hf_reranker | hybrid | 9.318 | 10.031 | 8.888 | 112.49 |
| hf_embedder_hf_reranker | semantic_hf_rerank | 19.336 | 21.638 | 19.113 | 52.31 |

Memory:

| profile | RAM MB | VRAM MB |
| --- | ---: | ---: |
| baseline_hash_cosine | 192.77 | 0.00 |
| hf_embedder_hf_reranker | 593.82 | 0.00 |

Result JSON: `loadtest/compare_baseline_vs_hf.json`

Interpretation:

- On this tiny evaluation dataset, quality is identical for baseline and HF profiles.
- HF profile has significantly higher overhead in latency and memory.
- Keep HF as optional layer for quality-sensitive workloads on harder datasets; keep baseline for low-latency / low-footprint paths.


## Lint and Type Checks

Install dev tooling:

```bash
pip install -e .[dev]
```

Run checks:

```bash
ruff check .
mypy src tests scripts
pytest -q
```

## Notes on Extensibility

- Add new document standards by extending `BaseChunk` and adding metadata conventions.
- Add custom chunkers for PDF, HTML, Word, tables, mixed content, AST, etc.
- Add rerankers or hybrid search backends through retrieval strategy layer.
- Add provider-specific tuning knobs without changing orchestration API.

## Install

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install -e .[qdrant]
pip install -e .[pgvector]
pip install -e .[hf]
pip install -e .[all]
```

What each extra installs:

- `qdrant`: `qdrant-client>=1.9`
- `pgvector`: `sqlalchemy>=2.0`, `psycopg[binary]>=3.1`, `pgvector>=0.3`
- `hf`: `sentence-transformers>=3.0`

## Repository Structure (What Is Required)

Required for source distribution/publication:

- `src/ragflow_orchestrator/`: package source code.
- `pyproject.toml`: build system and package metadata.
- `README.md`: project description used on PyPI.
- `LICENSE`: license text.

Useful runtime/dev content (keep in repository):

- `scripts/`: runnable demos and utility scripts.
- `tests/`: test suite.
- `examples/`: example usage.
- `datasets/`: local evaluation datasets.

Local/generated artifacts (safe to remove anytime):

- `build/`, `dist/` (can be regenerated by build).
- `src/*.egg-info/` (generated by setuptools during build/install).
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`.
- local runtime DB/state files (`*.db`, `*.sqlite`, `.rag_*`, incremental state files).

