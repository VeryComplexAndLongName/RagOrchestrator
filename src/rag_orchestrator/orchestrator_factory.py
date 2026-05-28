from __future__ import annotations

from rag_orchestrator.config import ConfigStore
from rag_orchestrator.embedding import create_embedder
from rag_orchestrator.factory import create_provider
from rag_orchestrator.orchestrator import RAGOrchestrator
from rag_orchestrator.presets import code_preset, document_preset, markdown_preset


class RAGOrchestratorFactory:
    @staticmethod
    def from_config_store(config_store: ConfigStore) -> RAGOrchestrator:
        config = config_store.get_config()

        provider = create_provider(config.provider.kind, **config.provider.params)
        embedder = create_embedder(
            provider=config.embedding.provider,
            model=config.embedding.model,
            options=config.embedding.as_provider_options(),
        )

        preset_name = config.pipeline.preset.lower()
        if preset_name == "code":
            preset = code_preset()
        elif preset_name == "markdown":
            preset = markdown_preset()
        else:
            preset = document_preset()

        return RAGOrchestrator(
            provider=provider,
            embedder=embedder,
            chunker=preset.chunker,
            cleaner=preset.cleaner,
        )
