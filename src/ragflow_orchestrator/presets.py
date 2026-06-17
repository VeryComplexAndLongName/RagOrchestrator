from __future__ import annotations

from dataclasses import dataclass

from ragflow_orchestrator.chunking import PythonCodeChunker
from ragflow_orchestrator.cleaning import BasicTextCleaner, DocumentAwareCleaner
from ragflow_orchestrator.document_pipeline import AdaptiveDocumentChunker, MarkdownAstChunker


@dataclass(slots=True)
class PipelinePreset:
    name: str
    cleaner: object
    chunker: object


def code_preset() -> PipelinePreset:
    return PipelinePreset(
        name="code",
        cleaner=BasicTextCleaner(),
        chunker=PythonCodeChunker(),
    )


def document_preset() -> PipelinePreset:
    return PipelinePreset(
        name="document",
        cleaner=DocumentAwareCleaner(),
        chunker=AdaptiveDocumentChunker(),
    )


def markdown_preset() -> PipelinePreset:
    return PipelinePreset(
        name="markdown",
        cleaner=DocumentAwareCleaner(),
        chunker=MarkdownAstChunker(),
    )
