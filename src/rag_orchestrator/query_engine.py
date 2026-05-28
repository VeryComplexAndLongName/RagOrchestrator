from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_orchestrator.models import RetrievalResult
from rag_orchestrator.orchestrator import RAGOrchestrator


class AnswerGenerator(Protocol):
    def generate(self, question: str, context_chunks: list[str]) -> str:
        ...


@dataclass(slots=True)
class QueryAnswer:
    question: str
    answer: str
    context: list[RetrievalResult]


class RAGQueryEngine:
    def __init__(self, orchestrator: RAGOrchestrator, generator: AnswerGenerator | None = None) -> None:
        self.orchestrator = orchestrator
        self.generator = generator

    def retrieve(self, question: str, top_k: int = 5, filters: dict[str, object] | None = None) -> list[RetrievalResult]:
        return self.orchestrator.search(query_text=question, top_k=top_k, filters=filters)

    def retrieve_from_sources(self, question: str, source_types: list[str], top_k: int = 5) -> list[RetrievalResult]:
        merged: list[RetrievalResult] = []
        seen: dict[str, RetrievalResult] = {}

        for source_type in source_types:
            part = self.retrieve(question=question, top_k=top_k, filters={"source_type": source_type})
            merged.extend(part)

        for item in merged:
            previous = seen.get(item.chunk.id)
            if previous is None or item.score > previous.score:
                seen[item.chunk.id] = item

        result = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return result[:top_k]

    def answer(self, question: str, top_k: int = 5, filters: dict[str, object] | None = None) -> QueryAnswer:
        context = self.retrieve(question=question, top_k=top_k, filters=filters)
        context_chunks = [item.chunk.text for item in context]

        if self.generator is None:
            # Safe deterministic fallback when no LLM generator is configured.
            answer = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."
        else:
            answer = self.generator.generate(question=question, context_chunks=context_chunks)

        return QueryAnswer(question=question, answer=answer, context=context)

    def answer_from_sources(self, question: str, source_types: list[str], top_k: int = 5) -> QueryAnswer:
        context = self.retrieve_from_sources(question=question, source_types=source_types, top_k=top_k)
        context_chunks = [item.chunk.text for item in context]

        if self.generator is None:
            answer = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."
        else:
            answer = self.generator.generate(question=question, context_chunks=context_chunks)

        return QueryAnswer(question=question, answer=answer, context=context)
