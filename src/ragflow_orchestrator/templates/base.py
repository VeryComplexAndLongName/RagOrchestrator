from __future__ import annotations

from typing import get_type_hints

from pydantic import BaseModel

from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.templates.models import LanguageMode, TemplateRunReport
from ragflow_orchestrator.templates.utils import detect_language


class BaseIngestionTemplate:
    template_name: str = "base"
    description: str = "Base class for ingestion templates."

    def __init__(self, orchestrator: RAGOrchestrator) -> None:
        self.orchestrator = orchestrator

    @classmethod
    def config_type(cls) -> type[BaseModel]:
        """Return the Pydantic config model expected by this template's run() method."""
        hints = get_type_hints(cls.run)
        config_type = hints.get("config")
        if not isinstance(config_type, type) or not issubclass(config_type, BaseModel):
            raise TypeError(
                f"{cls.__name__}.run() must annotate the config parameter with a Pydantic BaseModel subclass."
            )
        return config_type

    def _language_tag(self, text: str, mode: LanguageMode) -> str:
        return detect_language(text=text, mode=mode)

    @staticmethod
    def _metadata_with_source_url(metadata: dict[str, object], source_url: str | None) -> dict[str, object]:
        out = dict(metadata)
        if source_url:
            out["source_url"] = source_url
            out["source_origin"] = source_url
        return out

    @staticmethod
    def _metadata_for_url_source(
        source_type: str,
        metadata: dict[str, object],
        source_url: str | None,
    ) -> dict[str, object]:
        out = dict(metadata)
        out["source_type"] = source_type
        if source_url:
            out["source_url"] = source_url
            out["source_origin"] = source_url
        return out

    def run(self, config: object) -> TemplateRunReport:  # pragma: no cover
        raise NotImplementedError
