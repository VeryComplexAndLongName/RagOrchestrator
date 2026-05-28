from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rag_orchestrator.embedding.hash_embedder import HashEmbedder
from rag_orchestrator.embedding.hf_embedder import HFEmbedder
from rag_orchestrator.embedding.ollama_embedder import OllamaEmbedder
from rag_orchestrator.errors import ConfigurationError

EmbedderBuilder = Callable[[str | None, dict[str, Any]], object]


def _build_hash_embedder(model: str | None, options: dict[str, Any]) -> HashEmbedder:
    del model
    dimensions = int(options.get("dimensions", 256))
    return HashEmbedder(dimensions=dimensions)


def _build_ollama_embedder(model: str | None, options: dict[str, Any]) -> OllamaEmbedder:
    resolved_model = model or str(options.get("model") or "")
    if not resolved_model:
        raise ConfigurationError("Embedding provider 'ollama' requires a model name")

    base_url = str(options.get("base_url", "http://localhost:11434"))
    timeout_seconds = int(options.get("timeout_seconds", 60))
    return OllamaEmbedder(
        model=resolved_model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )


def _build_hf_embedder(model: str | None, options: dict[str, Any]) -> HFEmbedder:
    resolved_model = model or str(options.get("model") or "sentence-transformers/all-MiniLM-L6-v2")
    device = options.get("device")
    batch_size = int(options.get("batch_size", 32))
    normalize_embeddings = bool(options.get("normalize_embeddings", True))
    return HFEmbedder(
        model=resolved_model,
        device=str(device) if device else None,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )


EMBEDDER_BUILDERS: dict[str, EmbedderBuilder] = {
    "hash": _build_hash_embedder,
    "ollama": _build_ollama_embedder,
    "hf": _build_hf_embedder,
    "huggingface": _build_hf_embedder,
    "sentence-transformers": _build_hf_embedder,
}


def register_embedder_provider(provider: str, builder: EmbedderBuilder) -> None:
    EMBEDDER_BUILDERS[provider.lower()] = builder


def create_embedder(provider: str, model: str | None = None, options: dict[str, Any] | None = None) -> object:
    key = provider.lower()
    builder = EMBEDDER_BUILDERS.get(key)
    if builder is None:
        available = ", ".join(sorted(EMBEDDER_BUILDERS.keys()))
        raise ConfigurationError(f"Unknown embedding provider: {provider}. Available: {available}")

    return builder(model, options or {})
