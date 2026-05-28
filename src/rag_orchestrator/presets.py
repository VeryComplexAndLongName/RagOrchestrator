from __future__ import annotations

from dataclasses import dataclass

from rag_orchestrator.chunking import FixedWindowChunker, MarkdownHeadingChunker, PythonCodeChunker
from rag_orchestrator.cleaning import BasicTextCleaner, MarkupAwareTextCleaner


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
        cleaner=MarkupAwareTextCleaner(),
        chunker=FixedWindowChunker(chunk_size=900, chunk_overlap=120),
    )


def markdown_preset() -> PipelinePreset:
    return PipelinePreset(
        name="markdown",
        cleaner=MarkupAwareTextCleaner(),
        chunker=MarkdownHeadingChunker(),
    )
