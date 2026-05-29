from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ragflow_orchestrator.errors import ConfigurationError
from ragflow_orchestrator.retrieval.rerankers import CosineReranker, HFReranker, OllamaReranker

RerankerBuilder = Callable[[object | None, str | None, dict[str, Any]], object]


def _build_cosine(embedder: object | None, model: str | None, options: dict[str, Any]) -> object:
    del model, options
    if embedder is None:
        raise ConfigurationError("Reranker provider 'cosine' requires an embedder")
    return CosineReranker(embedder=embedder)


def _build_ollama(embedder: object | None, model: str | None, options: dict[str, Any]) -> object:
    del embedder
    resolved_model = model or str(options.get("model") or "")
    if not resolved_model:
        raise ConfigurationError("Reranker provider 'ollama' requires a model name")
    base_url = str(options.get("base_url", "http://localhost:11434"))
    timeout_seconds = int(options.get("timeout_seconds", 60))
    return OllamaReranker(model=resolved_model, base_url=base_url, timeout_seconds=timeout_seconds)


def _build_hf(embedder: object | None, model: str | None, options: dict[str, Any]) -> object:
    del embedder
    resolved_model = model or str(options.get("model") or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    device = options.get("device")
    batch_size = int(options.get("batch_size", 32))
    return HFReranker(
        model=resolved_model,
        device=str(device) if device else None,
        batch_size=batch_size,
    )


RERANKER_BUILDERS: dict[str, RerankerBuilder] = {
    "cosine": _build_cosine,
    "ollama": _build_ollama,
    "hf": _build_hf,
    "huggingface": _build_hf,
    "cross-encoder": _build_hf,
}


def register_reranker_provider(provider: str, builder: RerankerBuilder) -> None:
    RERANKER_BUILDERS[provider.lower()] = builder


def create_reranker(
    provider: str,
    embedder: object | None = None,
    model: str | None = None,
    options: dict[str, Any] | None = None,
) -> object:
    key = provider.lower()
    builder = RERANKER_BUILDERS.get(key)
    if builder is None:
        available = ", ".join(sorted(RERANKER_BUILDERS.keys()))
        raise ConfigurationError(f"Unknown reranker provider: {provider}. Available: {available}")
    return builder(embedder, model, options or {})
