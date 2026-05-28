from __future__ import annotations

import json
from typing import Protocol, cast
from urllib import request

from rag_orchestrator.adapters.common import cosine_similarity
from rag_orchestrator.errors import ProviderDependencyError
from rag_orchestrator.models import RetrievalResult


class Reranker(Protocol):
    def rerank(self, query_text: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        ...


class CosineReranker:
    """Fast reranker that recomputes pure cosine score by query vector."""

    def __init__(self, embedder: object) -> None:
        self.embedder = embedder

    def rerank(self, query_text: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        query_vector = self.embedder.embed(query_text)
        rescored: list[RetrievalResult] = []
        for item in candidates:
            score = cosine_similarity(query_vector, item.chunk.vector)
            rescored.append(RetrievalResult(chunk=item.chunk, score=score, provider=f"{item.provider}+cosine_rerank"))
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]


class OllamaReranker:
    """LLM-based reranker that requests a 0..1 relevance score from Ollama."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def rerank(self, query_text: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        rescored: list[RetrievalResult] = []
        for item in candidates:
            score = self._score(query_text=query_text, passage=item.chunk.text)
            rescored.append(RetrievalResult(chunk=item.chunk, score=score, provider=f"{item.provider}+ollama_rerank"))
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]

    def _score(self, query_text: str, passage: str) -> float:
        prompt = (
            "Return JSON only: {\"score\": number}. "
            "Score relevance from 0.0 to 1.0. "
            f"Query: {query_text}\n"
            f"Passage: {passage}"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))

        raw_response = body.get("response", "{}")
        try:
            parsed = json.loads(raw_response)
            score = float(parsed.get("score", 0.0))
        except json.JSONDecodeError:
            score = 0.0
        return max(0.0, min(1.0, score))


class HFReranker:
    """Cross-encoder reranker backed by sentence-transformers."""

    def __init__(self, model: str, device: str | None = None, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ProviderDependencyError(
                "HFReranker requires optional dependency 'sentence-transformers'. "
                "Install with: pip install -e .[hf]"
            ) from exc

        self.model_name = model
        self.batch_size = max(1, int(batch_size))
        model_kwargs = {"device": device} if device else {}
        self._model = CrossEncoder(model_name=model, **model_kwargs)

    def rerank(self, query_text: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not candidates:
            return []

        pairs = [(query_text, item.chunk.text) for item in candidates]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        score_list = cast(list[float], scores.tolist() if hasattr(scores, "tolist") else list(scores))

        rescored: list[RetrievalResult] = []
        for item, score in zip(candidates, score_list, strict=False):
            rescored.append(RetrievalResult(chunk=item.chunk, score=float(score), provider=f"{item.provider}+hf_rerank"))
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]
