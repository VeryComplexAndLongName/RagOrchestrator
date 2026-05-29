# GitHub Demo: max_with_prompt-orchestrator

This folder contains a standalone demo script that extends the standard GitHub RAG flow with PromptOrchestrator.

Main file:
- `max_with_prompt-orchestrator.py`

The script does all of the following in one pipeline:

1. Ingests GitHub repositories into SQLite+vec and graph storage.
2. Builds graph connectivity hints from retrieved repositories.
3. Uses PromptOrchestrator for every user turn.
4. Applies strict context and token budgets.
5. Forces summary-based compaction over multi-turn dialogue.
6. Runs LLM replies through Ollama.
7. Logs everything to console and a detailed log file:
   - full final prompt,
   - pre-fit and fitted sections,
   - all `stats` metrics,
   - all `safety` metrics,
   - compaction details (which sections were reduced),
   - summary before/after turn,
   - limit pressure ratios,
   - assistant answer.

Default bootstrap repositories:
- fastapi
- sqlalchemy
- pydantic

If you run without `--ask` and without `--interactive`, script now starts autonomous simulation by default.
It includes periodic prompt-injection and contradiction turns to trigger PromptOrchestrator safety behavior.
It also includes dedicated graph-focused turns (`graph-*`) so graph DB data actively participates in generated answers.

## Current defaults: providers and models

- Embedding provider: Ollama
- Embedding model: `nomic-embed-text:latest` (flag `--embed-model`)
- Chat provider: Ollama
- Chat model: `llama3.1:latest` (flag `--chat-model`)
- PromptOrchestrator summary provider: Ollama
- PromptOrchestrator summary model: `codellama:latest` (flags `--summary-provider`, `--summary-model`)
- Summary fallback on provider error: enabled by default (`--summary-fallback-on-error`)

You can change provider/model everywhere via CLI flags on each run.
Primary place in code with defaults: `build_parser()` in `max_with_prompt-orchestrator.py`.

## Install

From repository root:

```bash
pip install -r scripts/github_demo/max_with_prompt-orchestrator/requirements.txt
```

Notes:
- `requirements.txt` installs local `ragflow_orchestrator` from the repository root.
- `prompt-orchestrator` is installed from PyPI.

## Quick Start

### 1) Interactive mode (recommended)

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py --interactive
```

### 2) One-shot mode

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py --ask "How does FastAPI compare with Flask for async APIs?"
```

### 2.1) Autonomous simulation (default mode)

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py
```

This runs a built-in multi-turn scenario with tags:
- `graph-*`: questions explicitly about repository connectivity/contributor overlap/stars-forks in graph hints.
- `injection-*`: prompt-injection attempts.
- `contradiction-*`: conflicting instruction turns.

You can control aggressiveness of provocations:

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py --provocation-level low
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py --provocation-level medium
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py --provocation-level high
```

Flag behavior:
- `low`: fewer and softer injections/contradictions
- `medium`: balanced default set
- `high`: more frequent and stronger adversarial turns

For all levels, scenario still includes `graph-*` turns to ensure graph model data is used, not only vector RAG snippets.

### 3) Reuse existing DB (skip ingest)

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py --skip-ingest --interactive
```

To disable autonomous simulation explicitly:

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py --no-auto-simulate
```

## Why compaction happens multiple times

By default the script intentionally uses tight limits:

- `--max-prompt-chars 2600`
- `--max-prompt-tokens 520`
- `--recent-messages-limit 6`
- `--summary-trigger-messages 3`
- `--max-summary-chars 260`

In a normal multi-turn chat this typically triggers repeated fitting and summary evolution.

To increase pressure (and see more compaction events), use smaller limits, for example:

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py \
  --skip-ingest \
  --interactive \
  --max-prompt-chars 1800 \
  --max-prompt-tokens 350 \
  --recent-messages-limit 4 \
  --summary-trigger-messages 2 \
  --max-summary-chars 180
```

## PromptOrchestrator features demonstrated

The script showcases:

- RAG integration via custom `RAGProvider` adapter to `RAGOrchestrator.search(...)`.
- Context shaping into static/summary/recent/rag sections.
- Automatic token/char budget fitting.
- Summary generation and rotation across turns.
- Safety analysis per turn (`result.safety`).
- Prompt efficiency metrics per turn (`result.stats`).
- Optional prompt sanitization (`--safety-auto-rewrite/--no-safety-auto-rewrite`).
- Debug headers in prompt (`--debug-prompt-headers/--no-debug-prompt-headers`).
- Configurable section trim priority (`--section-priority rag recent summary`).
- Configurable summary provider:
  - `--summary-provider none` (deterministic local fallback)
  - `--summary-provider ollama`
  - `--summary-provider openai`

## Summary provider options

### Ollama summary model (default)

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py --interactive
```

### Local deterministic summary

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py \
  --interactive \
  --summary-provider none
```

### Ollama summary model (explicit)

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py \
  --interactive \
  --summary-provider ollama \
  --summary-model codellama:latest
```

### OpenAI summary model

```bash
set OPENAI_API_KEY=YOUR_KEY
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py \
  --interactive \
  --summary-provider openai \
  --summary-model gpt-4o-mini
```

## Logging format and where to find files

By default each run writes a timestamped file in this directory:

- `conversation_YYYYMMDD_HHMMSS.log`

You can set custom output path:

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py \
  --interactive \
  --log-file scripts/github_demo/max_with_prompt-orchestrator/my_run.log
```

In each turn, log includes:

- Explicit turn linkage block with IDs (`turn_id`, `data_ref`, `answer_ref`).
- Separate sections per turn:
  - `Txxx:QUESTION`
  - `Txxx:DATA | GRAPH HINTS`
  - `Txxx:DATA | RAG SOURCES`
  - `Txxx:ANSWER`
- Graph hints extracted from graph DB based on retrieval-aligned repositories.
- Final prompt sent to LLM.
- Each section before fitting and after fitting.
- Full `stats.model_dump()`.
- Full `safety.model_dump()`.
- Compaction analysis per section:
  - chars before,
  - chars after,
  - reduced or not,
  - how many chars removed.
- Limit pressure ratios:
  - `total_tokens / max_prompt_tokens`,
  - `total_chars / max_prompt_chars`.
- Summary state before and after the turn.
- Warnings emitted by PromptOrchestrator stats.
- Final assistant response.
- Exceptions/errors with structured block:
  - stage (`where` it happened),
  - exception type and message,
  - full traceback.

## Important runtime notes

- Ollama must be running for:
  - embedding model (`--embed-model`, default `nomic-embed-text:latest`),
  - chat model (`--chat-model`, default `llama3.1:latest`),
  - summary model by default (`--summary-provider ollama`, default `--summary-model codellama:latest`).
- If summary provider is temporarily unavailable (for example Ollama HTTP 502), script falls back to `summary-provider=none` for stability and logs fallback event.
- GitHub token is optional but strongly recommended for higher API limits:
  - pass via `--token ...` or environment `GITHUB_TOKEN`.

## Where to change provider/model

Runtime (recommended):
- Embeddings: `--embed-model ...`
- Chat: `--chat-model ...`
- Summary provider: `--summary-provider none|ollama|openai`
- Summary model: `--summary-model ...`
- Summary fallback toggle: `--summary-fallback-on-error` / `--no-summary-fallback-on-error`

In code (default values):
- `build_parser()` in `max_with_prompt-orchestrator.py`
  - `parser.add_argument("--embed-model", default="...")`
  - `parser.add_argument("--chat-model", default="...")`
  - `parser.add_argument("--summary-provider", default="...")`
  - `parser.add_argument("--summary-model", default="...")`

## Minimal command set for demo

1. First run with ingest:

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py --interactive
```

2. Follow-up runs without ingest:

```bash
python scripts/github_demo/max_with_prompt-orchestrator/max_with_prompt-orchestrator.py --skip-ingest --interactive
```

3. Ask 6-10 related questions in one session to observe repeated compaction and summary evolution.
