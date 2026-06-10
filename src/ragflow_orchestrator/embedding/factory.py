from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ragflow_orchestrator.embedding.composite import CachedEmbedder, FallbackEmbedder
from ragflow_orchestrator.embedding.hash_embedder import HashEmbedder
from ragflow_orchestrator.embedding.hf_embedder import HFEmbedder
from ragflow_orchestrator.embedding.ollama_embedder import OllamaEmbedder
from ragflow_orchestrator.errors import ConfigurationError

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


def _build_cached_embedder(model: str | None, options: dict[str, Any]) -> CachedEmbedder:
    del model
    base_provider = str(options.get("base_provider") or "").strip()
    if not base_provider:
        raise ConfigurationError("Embedding provider 'cached' requires option 'base_provider'")

    base_model = options.get("base_model")
    base_options = dict(options.get("base_options") or {})
    max_items = int(options.get("max_items", 4096))
    base_embedder = create_embedder(
        provider=base_provider,
        model=str(base_model) if base_model else None,
        options=base_options,
    )
    return CachedEmbedder(base_embedder=base_embedder, max_items=max_items)


def _build_fallback_embedder(model: str | None, options: dict[str, Any]) -> FallbackEmbedder:
    del model
    primary = dict(options.get("primary") or {})
    secondary = dict(options.get("secondary") or {})

    primary_provider = str(primary.get("provider") or "").strip()
    secondary_provider = str(secondary.get("provider") or "").strip()
    if not primary_provider or not secondary_provider:
        raise ConfigurationError(
            "Embedding provider 'fallback' requires options.primary.provider and options.secondary.provider"
        )

    primary_embedder = create_embedder(
        provider=primary_provider,
        model=str(primary.get("model")) if primary.get("model") else None,
        options=dict(primary.get("options") or {}),
    )
    secondary_embedder = create_embedder(
        provider=secondary_provider,
        model=str(secondary.get("model")) if secondary.get("model") else None,
        options=dict(secondary.get("options") or {}),
    )
    return FallbackEmbedder(primary=primary_embedder, secondary=secondary_embedder)


EMBEDDER_BUILDERS: dict[str, EmbedderBuilder] = {
    "hash": _build_hash_embedder,
    "ollama": _build_ollama_embedder,
    "hf": _build_hf_embedder,
    "huggingface": _build_hf_embedder,
    "sentence-transformers": _build_hf_embedder,
    "cached": _build_cached_embedder,
    "fallback": _build_fallback_embedder,
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
